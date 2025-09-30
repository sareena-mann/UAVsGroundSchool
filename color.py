import cv2
import numpy as np


def isolate_red_stopsign(image, tolerance=30):
    """
    Isolate a red stop sign from an image and display it against a white background.

    Parameters:
    - image: Input image (BGR format)
    - tolerance: Color tolerance for matching red hues (default: 30)

    Returns:
    - Image with red stop sign isolated against a white background
    - Binary mask of the detected red regions
    """
    # Convert image to HSV color space
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Define red color ranges in HSV (red wraps around hue 0/180)
    # First range: near hue 0
    lower_red1 = np.array([0, 100, 100])
    upper_red1 = np.array([tolerance, 255, 255])

    # Second range: near hue 180
    lower_red2 = np.array([180 - tolerance, 100, 100])
    upper_red2 = np.array([180, 255, 255])

    # Create masks for both red ranges
    mask1 = cv2.inRange(hsv_image, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv_image, lower_red2, upper_red2)

    # Combine masks
    mask = cv2.bitwise_or(mask1, mask2)

    # Create white background
    white_background = np.ones_like(image, dtype=np.uint8) * 255

    # Apply mask to keep red stop sign, set rest to white
    result = np.where(mask[:, :, None] == 255, image, white_background)

    return result, mask


# Example usage
if __name__ == "__main__":
    # Load an image
    image = cv2.imread("sign.jpeg")

    # Isolate red stop sign
    result, mask = isolate_red_stopsign(image, tolerance=30)

    # Display results
    cv2.imshow("Original Image", image)
    cv2.imshow("Isolated Stop Sign", result)
    cv2.imshow("Mask", mask)
    cv2.waitKey(0)
    cv2.destroyAllWindows()