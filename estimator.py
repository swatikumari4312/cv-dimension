import cv2
import numpy as np
from utils import (
    get_aruco_scale, 
    get_perspective_corrected_image, 
    draw_dimensions,
    load_calibration
)
import os

def estimate_dimensions(image_path, marker_size_mm=50.0):
    """
    Main pipeline to estimate object dimensions using a marker.
    """
    image = cv2.imread(image_path)
    if image is None:
        print(f"Could not read image: {image_path}")
        return

    # Load calibration if available for undistortion
    calib = load_calibration()
    if calib:
        mtx = np.array(calib["camera_matrix"])
        dist = np.array(calib["dist_coeff"])
        image = cv2.undistort(image, mtx, dist)

    # 1. Scale Detection
    pixels_per_mm, marker_corners = get_aruco_scale(image, marker_size_mm)
    if pixels_per_mm is None:
        print("Reference ArUco marker not found. Cannot estimate scale.")
        return
    
    # 2. Perspective Correction (Critical for accuracy)
    warped_image, pixel_scale = get_perspective_corrected_image(image, marker_corners, marker_size_mm)
    
    # 3. Preprocessing on warped image
    gray = cv2.cvtColor(warped_image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (7, 7), 0)
    
    # 4. Adaptive Thresholding for robust object detection
    thresh = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
    kernel = np.ones((5, 5), np.uint8)
    closing = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    
    # 5. Find Contours
    cnts, _ = cv2.findContours(closing, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # 6. Filter and Measure
    for c in cnts:
        area = cv2.contourArea(c)
        if area < 3000: # Filter small noise
            continue
            
        # Get bounding box
        rect = cv2.minAreaRect(c)
        box = cv2.boxPoints(rect)
        box = np.array(box, dtype="int")
        
        # Order points: tl, tr, br, bl
        # Sort by x then by y
        box = box[np.argsort(box[:, 0])]
        left = box[:2][np.argsort(box[:2, 1])]
        right = box[2:][np.argsort(box[2:, 1])]
        ordered_box = np.array([left[0], right[0], right[1], left[1]], dtype="float32")
        
        # Avoid drawing on the marker itself (marker is near margin 100)
        center = np.mean(ordered_box, axis=0)
        if center[0] < 250 and center[1] < 250:
            continue

        warped_image = draw_dimensions(warped_image, ordered_box, pixel_scale)
        cv2.drawContours(warped_image, [ordered_box.astype("int")], -1, (0, 255, 0), 2)

    # Save Result
    output_path = "output_" + os.path.basename(image_path)
    cv2.imwrite(output_path, warped_image)

    print(f"Results saved to {output_path}")

import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Estimate object dimensions using an ArUco marker as a reference.")
    parser.add_argument("--image", type=str, required=True, help="Path to the image file.")
    parser.add_argument("--marker_size", type=float, default=50.0, help="Size of the ArUco marker in mm (default: 50.0).")
    
    args = parser.parse_args()
    
    estimate_dimensions(args.image, args.marker_size)
