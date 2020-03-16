
import importlib
import argparse
import os
import shutil
import sys
import math
import glob
import uuid 

import networkx as nx
import facenet.src.facenet as facenet
import tensorflow as tf
import numpy as np

from tqdm import tqdm
from random import shuffle
from argparse import ArgumentParser

import logging
logging.getLogger().setLevel(logging.INFO)


def face_distance(face_encodings, face_to_compare):
    """
        Given a list of face encodings, compare them to a known face encoding and get a euclidean distance
        for each comparison face. The distance tells you how similar the faces are.
        Inputs:
            face_encodings: List of face encodings to compare
            face_to_compare: A face encoding to compare against
        Outputs:
            A numpy ndarray with the distance for each face in the same order as the 'faces' array
    """

    if len(face_encodings) == 0:
        return np.empty((0))

    return np.sum(face_encodings * face_to_compare, axis=1)

def load_model(model_dir, meta_file, ckpt_file):
    model_dir_exp = os.path.expanduser(model_dir)
    saver = tf.compat.v1.train.import_meta_graph(os.path.join(model_dir_exp, meta_file))
    saver.restore(tf.compat.v1.get_default_session(), os.path.join(model_dir_exp, ckpt_file))

def _chinese_whispers(encoding_list, threshold=0.55, iterations=20):
    """ 
        Chinese Whispers Algorithm
        Modified from Alex Loveless' implementation,
        http://alexloveless.co.uk/data/chinese-whispers-graph-clustering-in-python/
    Inputs:
        encoding_list: a list of facial encodings from face_recognition
        threshold: facial match threshold,default 0.6
        iterations: since chinese whispers is an iterative algorithm, number of times to iterate
    Outputs:
        A list of clusters, a cluster being a list of imagepaths, sorted by largest cluster to smallest
    """
    
    # Create graph
    nodes = []
    edges = []

    image_paths, encodings = zip(*encoding_list)

    if len(encodings) <= 1:
        logging.info("No enough encodings to cluster!")
        return []

    for idx, face_encoding_to_check in enumerate(encodings):
        # Adding node of facial encoding
        node_id = idx + 1

        # Initialize 'cluster' to unique value (cluster of itself)
        node = (node_id, {"cluster": image_paths[idx], "path": image_paths[idx]})
        nodes.append(node)

        # Facial encodings to compare
        if (idx + 1) >= len(encodings):
            # Node is last element, don't create edge
            break

        compare_encodings = encodings[idx + 1:]
        distances = face_distance(compare_encodings, face_encoding_to_check)
        encoding_edges = []
        for i, distance in enumerate(distances):
            if distance > threshold:
                # Add edge if facial match
                edge_id = idx + i + 2
                encoding_edges.append((node_id, edge_id, {"weight": distance}))

        edges = edges + encoding_edges

    G = nx.Graph()
    G.add_nodes_from(nodes)
    G.add_edges_from(edges)

    # Iterate
    for _ in range(0, iterations):
        cluster_nodes = list(G.nodes())
        shuffle(cluster_nodes)
        for node in cluster_nodes:
            neighbors = G[node]
            clusters = {}

            for ne in neighbors:
                if isinstance(ne, int):
                    if G.nodes[ne]["cluster"] in clusters:
                        clusters[G.nodes[ne]['cluster']] += G[node][ne]["weight"]
                    else:
                        clusters[G.nodes[ne]["cluster"]] = G[node][ne]["weight"]

            # find the class with the highest edge weight sum
            edge_weight_sum = 0
            max_cluster = 0
            # use the max sum of neighbor weights class as current node's class
            for cluster in clusters:
                if clusters[cluster] > edge_weight_sum:
                    edge_weight_sum = clusters[cluster]
                    max_cluster = cluster

            # set the class of target node to the winning local class
            G.nodes[node]["cluster"] = max_cluster

    clusters = {}

    # Prepare cluster output
    for (_, data) in G.nodes.items():
        cluster = data["cluster"]
        path = data["path"]

        if cluster != 0:
            if cluster not in clusters:
                clusters[cluster] = []
            clusters[cluster].append(path)
        else:
            clusters[uuid.uuid4().hex[:6].upper()] = [path]
        

    # Sort cluster output
    sorted_clusters = sorted(clusters.values(), key=len, reverse=True)

    return sorted_clusters

def cluster_facial_encodings(facial_encodings):
    """ 
        Cluster facial encodings
        Intended to be an optional switch for different clustering algorithms, as of right now
        only chinese whispers is available.
        Input:
            facial_encodings: (image_path, facial_encoding) dictionary of facial encodings
        Output:
            sorted_clusters: a list of clusters, a cluster being a list of imagepaths,
                sorted by largest cluster to smallest
    """

    if len(facial_encodings) <= 1:
        logging.info("Number of facial encodings must be greater than one, can't cluster")
        return []

    sorted_clusters = _chinese_whispers(facial_encodings.items())
    return sorted_clusters

def compute_facial_encodings(sess, images_placeholder, embeddings, 
                             phase_train_placeholder, image_size, 
                             embedding_size,nrof_images, nrof_batches, 
                             emb_array, paths, batch_size=30):
    """ 
        Compute Facial Encodings
        Given a set of images, compute the facial encodings of each face detected in the images and
        return them. If no faces, or more than one face found, return nothing for that image.
        Inputs:
            image_paths: a list of image paths
        Outputs:
            facial_encodings: (image_path, facial_encoding) dictionary of facial encodings
    """

    for i in range(nrof_batches):
        start_index = i * batch_size
        end_index = min((i + 1) * batch_size, nrof_images)
        paths_batch = paths[start_index:end_index]
        images = facenet.load_data(paths_batch, False, False, image_size)
        feed_dict = { images_placeholder:images, phase_train_placeholder:False }
        emb_array[start_index:end_index, :] = sess.run(embeddings, feed_dict=feed_dict)

    facial_encodings = {}

    for i in range(nrof_images):
        facial_encodings[paths[i]] = emb_array[i, :]
    return facial_encodings

def cluster_images(model_dir, images_path, input, output, batch_size=30):
    """ 
        Given a list of images, save out facial encoding data files and copy
        images into folders of face clusters.
    """
    if not os.path.exists(images_path):
        logging.info("Path of images isn't exist: {}".format(images_path))
        return

    if os.path.exists(output):
        shutil.rmtree(output)
    os.mkdir(output)

    with tf.Graph().as_default():
        with tf.compat.v1.Session() as sess:
            image_paths = [os.path.join(input, img) for img in os.listdir(input)]
            meta_file, ckpt_file = facenet.get_model_filenames(os.path.expanduser(model_dir))
            
            logging.info("Metagraph file: {}".format(meta_file))
            logging.info("Checkpoint file: {}".format(ckpt_file))

            load_model(model_dir, meta_file, ckpt_file)
            
            # Get input and output tensors
            def_graf = tf.compat.v1.get_default_graph()
            images_placeholder = def_graf.get_tensor_by_name("input:0")
            embeddings = def_graf.get_tensor_by_name("embeddings:0")
            phase_train_placeholder = def_graf.get_tensor_by_name("phase_train:0")
            
            image_size = images_placeholder.get_shape()[1]
            logging.info("image_size: " + str(image_size))
            embedding_size = embeddings.get_shape()[1]
        
            # Run forward pass to calculate embeddings
            logging.info("Running forward pass on images") 

            nrof_images = len(image_paths)
            nrof_batches = int(math.ceil(1.0 * nrof_images / batch_size))
            emb_array = np.zeros((nrof_images, embedding_size))
            facial_encodings = compute_facial_encodings(sess, images_placeholder, embeddings,
                                                        phase_train_placeholder, image_size,
                                                        embedding_size, nrof_images, nrof_batches, 
                                                        emb_array, image_paths, batch_size)
            sorted_clusters = cluster_facial_encodings(facial_encodings)
            
            clustered = set()

            # Copy image files to cluster folders
            for idx, cluster in enumerate(tqdm(sorted_clusters)):
                cluster_dir = os.path.join(output, str(idx + 1))
                os.mkdir(cluster_dir)
                for i, path in enumerate(cluster):
                    image = os.path.basename(path).split("_")[0] + ".jpg"
                    clustered.add(image)
                    image_path = [file for file in glob.glob(os.path.join(images_path, image))][0]
                    shutil.copy(image_path, os.path.join(cluster_dir, image))
                    if i == 0:
                        shutil.copy(path, os.path.join(cluster_dir, "face.jpg"))

            # Copy not clustered image files to clusters folder
            not_clustered = [f for f in os.listdir(images_path) if f not in clustered]
            if len(not_clustered) != 0:
                cluster_dir = os.path.join(output, str(0))
                os.mkdir(cluster_dir)
                for image in tqdm(not_clustered):
                    shutil.copy(os.path.join(images_path, image), os.path.join(cluster_dir, image))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", type=str, action="store",
                        required=True, help="model dir")
    parser.add_argument("--batch_size", type=int, action="store",
                        default=30, help="batch size")
    parser.add_argument("--images", type=str, action="store",
                        dest="images_path", required=True,
                        help="Dir of images")
    parser.add_argument("--input", type=str, action="store",
                        required=True, help="Input dir of frames of faces")
    parser.add_argument("--output", type=str, action="store",
                        default="clusters", help="Output dir of clusters")
    args = parser.parse_args()
    cluster_images(args.model_dir, args.images_path, args.input, args.output, args.batch_size)
