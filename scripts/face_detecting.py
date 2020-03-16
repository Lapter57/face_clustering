import cv2
import os
import shutil

from tqdm import tqdm
from mtcnn.mtcnn import MTCNN
from argparse import ArgumentParser

import logging
logging.getLogger().setLevel(logging.INFO)

def face_detect(image_name, images_path, dest, conf=0.95):
    image_path = os.path.join(images_path, image_name)
    if not os.path.exists(image_path):
        logging.info("Cannot find input: " + image_path)
        return

    detector = MTCNN()

    img = cv2.imread(image_path)
    faces = detector.detect_faces(img)

    if len(faces)==0:
        logging.info("No face detect: " + image_name)
        return

    for i, face in enumerate(faces):
        if face["confidence"] >= conf:
            box = face['box']
            crop_img = img[box[1]:box[1] + box[3], box[0]:box[0] + box[2]]
            crop_img = cv2.resize(crop_img, (160, 160))
            crop_img_name = "{}_{}.jpg".format(image_name.replace('.jpg', ''), i)
            cv2.imwrite(os.path.join(dest, crop_img_name), crop_img)


def faces_detect(images_path, dest, conf=0.95):
    if os.path.exists(dest):
        shutil.rmtree(dest)
    os.mkdir(dest)

    images = [f for f in os.listdir(images_path) if os.path.isfile(os.path.join(images_path, f))]
    for image in tqdm(images):
        face_detect(image, images_path, dest, conf)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--path", action="store",
                        dest="images_path", required=True,
                        help="Path to images")
    parser.add_argument("--dest", action="store",
                        dest="dest", required=True,
                        help="Path to the folder where found faces will be saved")
    parser.add_argument("--conf", action="store", type=float,
                        dest="conf", default=0.95, 
                        help="The confidence is the probability to be matching a face (default=0)")
    args = parser.parse_args()
    faces_detect(args.images_path, args.dest, args.conf)