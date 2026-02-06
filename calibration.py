import cv2
import numpy as np
import os
import json

def calibrate_camera(image_dir, grid_size=(9, 6), square_size=25.0):
    """
    Calibrates the camera using a series of chessboard images.
    :param image_dir: Directory containing calibration images.
    :param grid_size: Number of inner corners (cols, rows).
    :param square_size: Length of a square side in mm.
    """
    # Termination criteria for sub-pixel accuracy
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    # Prepare object points (0,0,0), (1,0,0), (2,0,0) ....,(8,5,0)
    objp = np.zeros((grid_size[0] * grid_size[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:grid_size[0], 0:grid_size[1]].T.reshape(-1, 2)
    objp *= square_size

    objpoints = [] # 3d point in real world space
    imgpoints = [] # 2d points in image plane.

    if not os.path.exists(image_dir):
        print(f"Directory {image_dir} does not exist.")
        return

    images = [f for f in os.listdir(image_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]

    if not images:
        print("No calibration images found.")
        return

    h, w = 0, 0
    for fname in images:
        img = cv2.imread(os.path.join(image_dir, fname))
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape[:2]

        # Find the chess board corners
        ret, corners = cv2.findChessboardCorners(gray, grid_size, None)

        if ret:
            objpoints.append(objp)
            corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            imgpoints.append(corners2)

            # Draw and display the corners (optional)
            cv2.drawChessboardCorners(img, grid_size, corners2, ret)
            # cv2.imshow('img', img)
            # cv2.waitKey(500)

    # cv2.destroyAllWindows()

    # Calibrate
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, (w, h), None, None)

    if ret:
        calibration_data = {
            "camera_matrix": mtx.tolist(),
            "dist_coeff": dist.tolist(),
            "reprojection_error": float(ret)
        }
        with open("calibration.json", "w") as f:
            json.dump(calibration_data, f, indent=4)
        print("Calibration successful. Saved to calibration.json")
        return mtx, dist
    else:
        print("Calibration failed.")
        return None, None

import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calibrate camera using a series of chessboard images.")
    parser.add_argument("--dir", type=str, required=True, help="Directory containing calibration images.")
    parser.add_argument("--grid", type=int, nargs=2, default=[9, 6], help="Number of inner corners (cols rows) (default: 9 6).")
    parser.add_argument("--size", type=float, default=25.0, help="Length of a square side in mm (default: 25.0).")
    
    args = parser.parse_args()
    
    calibrate_camera(args.dir, tuple(args.grid), args.size)
