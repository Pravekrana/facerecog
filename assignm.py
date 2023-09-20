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

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

for filename in tqdm(os.listdir(args.dataset_dir)):
    if filename.endswith(('.jpg', '.jpeg', '.png', '.bmp')):
        image_path = os.path.join(args.dataset_dir, filename)
        img = cv2.imread(image_path)
        
        # Convert the image to grayscale for face detection
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Perform face detection
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        if len(faces) > 0:
            # Copy (or move) the image to the result directory
            shutil.copy(image_path, os.path.join(args.result_dir, filename))

print('Face detection completed successfully.')
