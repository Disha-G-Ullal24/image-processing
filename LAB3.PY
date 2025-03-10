import cv2
import numpy as np
from PIL import Image 

def main():
    # Read image using OpenCV
    image = cv2.imread("image1.jpg")

    if image is None:
        print("Error: Image not found.")
        return

    # Get image dimensions
    rows, cols = image.shape[:2]

    # 1. Rotation using PIL
    original_image = Image.open("image1.jpg") 
    rotated_image1 = original_image.rotate(180) 
    rotated_image2 = original_image.transpose(Image.ROTATE_90) 
    rotated_image3 = original_image.rotate(60) 

    rotated_image1.show(title="Rotated 180°") 
    rotated_image2.show(title="Rotated 90°") 
    rotated_image3.show(title="Rotated 60°") 


 
