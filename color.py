import cv2
import numpy as np


def split_image_by_rgb(image, tolerance=30, smooth=True):
    """
    Split an image into three regions: heavily red, heavily green, and heavily blue.

    Parameters:
    - image: Input image (BGR format)
    - tolerance: Hue tolerance for color matching (default: 30)
    - smooth: Apply morphological operations to clean up masks (default: True)

    Returns:
    - List of three images, each isolating heavily red, green, or blue regions against a white background
    - List of corresponding masks
    """
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    # Curious if this helps:
    if smooth:
        image = cv2.GaussianBlur(image, (5, 5), 0)

    # Define hue ranges for red, green, and blue in HSV
    # Red: around hue 0 and 180 (wraparound)
    red_lower1 = np.array([0, 100, 100])
    red_upper1 = np.array([tolerance, 255, 255])
    red_lower2 = np.array([180 - tolerance, 100, 100])
    red_upper2 = np.array([180, 255, 255])

    # Green
    green_lower = np.array([120 - tolerance, 100, 100])
    green_upper = np.array([120 + tolerance, 255, 255])

    # Blue
    blue_lower = np.array([240 - tolerance, 100, 100])
    blue_upper = np.array([240 + tolerance, 255, 255])

    # Create masks for each color
    red_mask1 = cv2.inRange(hsv_image, red_lower1, red_upper1)
    red_mask2 = cv2.inRange(hsv_image, red_lower2, red_upper2)
    red_mask = cv2.bitwise_or(red_mask1, red_mask2)
    green_mask = cv2.inRange(hsv_image, green_lower, green_upper)
    blue_mask = cv2.inRange(hsv_image, blue_lower, blue_upper)

    #Kernel for cleaning masks; again curiosu
    if smooth:
        kernel = np.ones((5, 5), np.uint8)
        red_mask = cv2.dilate(red_mask, kernel, iterations=2)
        red_mask = cv2.erode(red_mask, kernel, iterations=2)
        green_mask = cv2.dilate(green_mask, kernel, iterations=2)
        green_mask = cv2.erode(green_mask, kernel, iterations=2)
        blue_mask = cv2.dilate(blue_mask, kernel, iterations=2)
        blue_mask = cv2.erode(blue_mask, kernel, iterations=2)

    # Create white background
    white_background = np.ones_like(image, dtype=np.uint8) * 255

    # Apply masks to isolate color regions
    red_result = np.where(red_mask[:, :, None] == 255, image, white_background)
    green_result = np.where(green_mask[:, :, None] == 255, image, white_background)
    blue_result = np.where(blue_mask[:, :, None] == 255, image, white_background)

    result_images = [red_result, green_result, blue_result]
    masks = [red_mask, green_mask, blue_mask]

    return result_images, masks

if __name__ == "__main__":
    image = cv2.imread("rg.jpeg")

    result_images, masks = split_image_by_rgb(image, tolerance=30, smooth=True)

    cv2.imshow("Original Image", image)
    for i, (result, mask) in enumerate(zip(result_images, masks)):
        color_names = ["Red", "Green", "Blue"]
        cv2.imshow(f"{color_names[i]} Region", result)
        cv2.imshow(f"{color_names[i]} Mask", mask)
    cv2.waitKey(0)
    cv2.destroyAllWindows()