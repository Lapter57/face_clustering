from face_detecting import faces_detect
from clustering import cluster_images

IMAGES_DIR = "../data/images"
FACE_FRAMES_DIR = "../data/face_frames"
FACENET_MODEL_DIR = "../facenet_model"
CLUSTER_DIR = "../data/clusters"
CONFIDENCE = 0.97
BATCH_SIZE = 30

faces_detect(IMAGES_DIR, FACE_FRAMES_DIR, CONFIDENCE)
cluster_images(FACENET_MODEL_DIR, IMAGES_DIR, FACE_FRAMES_DIR, CLUSTER_DIR, BATCH_SIZE)
