import cv2
import numpy as np

def main():
    image = cv2.imread("input.jpg")

    if image is None:
        print("Error: Image not found.")
        return

    rows, cols = image.shape[:2]

    # 1. Rotation
    rotated = cv2.warpAffine(image, cv2.getRotationMatrix2D((cols // 2, rows // 2), 45, 1), (cols, rows))

    # 2. Scaling
    scaled = cv2.resize(image, None, fx=0.5, fy=0.5)

    # 3. Adding Border
    bordered = cv2.copyMakeBorder(image, 50, 50, 50, 50, cv2.BORDER_CONSTANT, value=(255, 0, 0))

    # 4. Flipping
    flipped = cv2.flip(image, 1)

    # 5. Converting to Grayscale
    grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 6. Blurring
    blurred = cv2.GaussianBlur(image, (15, 15), 0)

    # 7. Edge Detection
    edges = cv2.Canny(grayscale, 100, 200)

    # 8. Erosion
    kernel = np.ones((5, 5), np.uint8)
    eroded = cv2.erode(image, kernel, iterations=1)

    # 9. Dilation
    dilated = cv2.dilate(image, kernel, iterations=1)

    # 10. Histogram Equalization
    equalized = cv2.equalizeHist(grayscale)

    # 11. Adaptive Thresholding
    adaptive_thresh = cv2.adaptiveThreshold(equalized, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

    # 12. Translation
    translated = cv2.warpAffine(image, np.float32([[1, 0, 50], [0, 1, 50]]), (cols, rows))

    # 13. Perspective Transformation
    perspective_matrix = cv2.getPerspectiveTransform(
        np.float32([[0, 0], [cols - 1, 0], [0, rows - 1], [cols - 1, rows - 1]]),
        np.float32([[0, 0], [cols - 1, 50], [50, rows - 1], [cols - 51, rows - 51]])
    )
    perspective = cv2.warpPerspective(image, perspective_matrix, (cols, rows))

    # 14. Bitwise AND
    mask = np.zeros((rows, cols), dtype=np.uint8)
    cv2.circle(mask, (cols // 2, rows // 2), 100, (255), -1)
    bitwise_and = cv2.bitwise_and(image, image, mask=mask)

    # 15. Bitwise NOT
    bitwise_not = cv2.bitwise_not(image)

    # Display images one by one
    transformations = [
        ("Rotated", rotated),
        ("Scaled", scaled),
        ("Bordered", bordered),
        ("Flipped", flipped),
        ("Grayscale", grayscale),
        ("Blurred", blurred),
        ("Edges", edges),
        ("Eroded", eroded),
        ("Dilated", dilated),
        ("Equalized", equalized),
        ("Adaptive Threshold", adaptive_thresh),
        ("Translated", translated),
        ("Perspective", perspective),
        ("Bitwise AND", bitwise_and),
        ("Bitwise NOT", bitwise_not)
    ]

    for name, img in transformations:
        cv2.imshow(name, img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()