import cv2
import numpy as np
import json
import os

def get_aruco_scale(image, marker_size_mm=50.0):
    """
    Detects ArUco marker and returns pixels_per_mm and corners.
    Uses DICT_5X5_50 by default.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_50)
    parameters = cv2.aruco.DetectorParameters()
    detector = cv2.aruco.ArucoDetector(aruco_dict, parameters)
    
    corners, ids, rejected = detector.detectMarkers(gray)
    
    if ids is not None:
        # Calculate pixels per mm using the first detected marker
        # Perimeter of marker in pixels / (4 * marker_size_mm)
        peri = cv2.arcLength(corners[0], True)
        pixels_per_mm = peri / (4 * marker_size_mm)
        return pixels_per_mm, corners[0][0]
    
    return None, None

def get_perspective_corrected_image(image, marker_corners, marker_size_mm):
    """
    Corrects image perspective based on the ArUco marker corners.
    """
    # Define destination points for a top-down view of the marker
    # We'll place the marker in the top-left with some margin
    margin = 100 
    side_px = int(marker_size_mm * (cv2.arcLength(marker_corners, True) / (4 * marker_size_mm)))
    
    dst_pts = np.array([
        [margin, margin],
        [margin + side_px, margin],
        [margin + side_px, margin + side_px],
        [margin, margin + side_px]
    ], dtype="float32")
    
    M = cv2.getPerspectiveTransform(marker_corners, dst_pts)
    
    # Estimate output size (could be refined)
    h, w = image.shape[:2]
    warped = cv2.warpPerspective(image, M, (w, h))
    
    # Calculate scale in the warped image
    pixel_scale = side_px / marker_size_mm
    
    return warped, pixel_scale

def draw_dimensions(image, ordered_box, pixel_scale):
    """
    Draws width and height on the image based on the ordered box points.
    ordered_box: [top-left, top-right, bottom-right, bottom-left]
    """
    (tl, tr, br, bl) = ordered_box
    
    # Compute midpoint between top-left and top-right
    (mpx, mpy) = ((tl[0] + tr[0]) * 0.5, (tl[1] + tr[1]) * 0.5)
    # Compute midpoint between top-right and bottom-right
    (mpy_r, mpx_r) = ((tr[0] + br[0]) * 0.5, (tr[1] + br[1]) * 0.5)
    
    # Euclidean distance
    width_mm = np.linalg.norm(tl - tr) / pixel_scale
    height_mm = np.linalg.norm(tr - br) / pixel_scale
    
    cv2.putText(image, f"{width_mm:.1f}mm", (int(mpx), int(mpy - 10)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)
    cv2.putText(image, f"{height_mm:.1f}mm", (int(mpy_r + 10), int(mpx_r)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)
    
    return image

def match_features(img1, img2):
    """
    Finds and matches features between two images using ORB.
    """
    orb = cv2.ORB_create(nfeatures=5000) # Increased features for better accuracy
    
    # Find keypoints and descriptors
    kp1, des1 = orb.detectAndCompute(img1, None)
    kp2, des2 = orb.detectAndCompute(img2, None)
    
    if des1 is None or des2 is None:
        return None, None
        
    # Match descriptors using Brute Force Hamming distance
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)
    
    # Sort matches by distance (best first)
    matches = sorted(matches, key=lambda x: x.distance)
    
    # Extract location of good matches (top 500)
    num_matches = min(500, len(matches))
    pts1 = np.float32([kp1[m.queryIdx].pt for m in matches[:num_matches]])
    pts2 = np.float32([kp2[m.trainIdx].pt for m in matches[:num_matches]])
    
    return pts1, pts2

def get_triangulated_points(pts1, pts2, baseline_mm, focal_length_px, img_shape):
    """
    Triangulates 3D points from 2D matches using a simple stereo model.
    """
    disparity = pts1[:, 0] - pts2[:, 0]
    
    # Filter out small or negative disparity
    mask = disparity > 2.0 # Threshold for noise
    pts1 = pts1[mask]
    disparity = disparity[mask]
    
    if len(disparity) == 0:
        return None
        
    Z = (focal_length_px * baseline_mm) / disparity
    
    h, w = img_shape[:2]
    cx, cy = w / 2, h / 2
    
    X = ((pts1[:, 0] - cx) * Z) / focal_length_px
    Y = ((pts1[:, 1] - cy) * Z) / focal_length_px
    
    return np.vstack((X, Y, Z)).T

def estimate_object_dimensions(pts3d):
    """
    Estimates dimensions based on the 3D points.
    Uses IQR to filter outliers for better accuracy.
    """
    if pts3d is None or len(pts3d) < 10:
        return None
        
    # Stats-based outlier removal (Interquartile Range)
    q1 = np.percentile(pts3d, 25, axis=0)
    q3 = np.percentile(pts3d, 75, axis=0)
    iqr = q3 - q1
    
    mask = np.all((pts3d >= q1 - 1.5 * iqr) & (pts3d <= q3 + 1.5 * iqr), axis=1)
    filtered_pts = pts3d[mask]
    
    if len(filtered_pts) < 5:
        filtered_pts = pts3d # Fallback
        
    dims = np.max(filtered_pts, axis=0) - np.min(filtered_pts, axis=0)
    return dims

def draw_3d_results(image, dims):
    """
    Overlays the calculated dimensions on the result image.
    """
    if dims is None:
        return image
        
    # Rich aesthetic for results
    overlay = image.copy()
    cv2.rectangle(overlay, (20, 20), (350, 180), (40, 40, 40), -1)
    image = cv2.addWeighted(overlay, 0.7, image, 0.3, 0)
    
    text_color = (0, 255, 127) # Spring Green
    cv2.putText(image, "ESTIMATED DIMENSIONS", (40, 50), cv2.FONT_HERSHEY_DUPLEX, 0.8, (255, 255, 255), 1)
    cv2.putText(image, f"Width:  {dims[0]:.1f} mm", (40, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, text_color, 2)
    cv2.putText(image, f"Height: {dims[1]:.1f} mm", (40, 125), cv2.FONT_HERSHEY_SIMPLEX, 0.7, text_color, 2)
    cv2.putText(image, f"Depth:  {dims[2]:.1f} mm", (40, 160), cv2.FONT_HERSHEY_SIMPLEX, 0.7, text_color, 2)
    
    return image

def load_calibration():
    """Reads calibration data if available."""
    if os.path.exists("calibration.json"):
        with open("calibration.json", "r") as f:
            return json.load(f)
    return None
