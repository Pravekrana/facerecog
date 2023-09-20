import cv2
import os
import shutil
from tqdm import tqdm  # for progress bar (install it using pip if not installed)
import argparse 
parser = argparse.ArgumentParser()
parser.add_argument('target_dir', help='Directory of target face images')
parser.add_argument('dataset_dir', help='Directory of dataset images')
parser.add_argument('result_dir', help='Directory to store images with detected faces')
args = parser.parse_args()

from keras.models import load_model

# Load the pre-trained FaceNet model (you'll need to download the model beforehand)
face_recognition_model = load_model('path_to_face_recognition_model.h5')

import numpy as np

target_faces = []

for filename in os.listdir(args.target_dir):
    if filename.endswith(('.jpg', '.jpeg', '.png', '.bmp')):
        image_path = os.path.join(args.target_dir, filename)
        img = cv2.imread(image_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert to RGB
        img = cv2.resize(img, (160, 160))  # Resize to the input size of FaceNet
        img = img / 255.0  # Normalize pixel values
        target_faces.append(img)

target_faces = np.array(target_faces)

import numpy as np

target_faces = []

for filename in os.listdir(args.target_dir):
    if filename.endswith(('.jpg', '.jpeg', '.png', '.bmp')):
        image_path = os.path.join(args.target_dir, filename)
        img = cv2.imread(image_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert to RGB
        img = cv2.resize(img, (160, 160))  # Resize to the input size of FaceNet
        img = img / 255.0  # Normalize pixel values
        target_faces.append(img)

target_faces = np.array(target_faces)

for recognized_face in recognized_faces:
    filename, target_index = recognized_face
    target_name = target_names[target_index]  # You need to define target_names

    print(f"Detected face in {filename} belongs to: {target_name}")
