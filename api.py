from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
import cv2
import numpy as np
import os
import shutil
import uuid
from utils import (
    match_features, 
    get_triangulated_points, 
    estimate_object_dimensions, 
    draw_3d_results,
    load_calibration
)

app = FastAPI()

# Enable CORS for frontend interaction
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_DIR = "uploads"
OUTPUT_DIR = "outputs"
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Load calibration data at startup
CALIB = load_calibration()

@app.post("/estimate")
async def estimate(
    file1: UploadFile = File(...), 
    file2: UploadFile = File(...), 
    baseline: float = Form(100.0)
):
    # Save uploaded files
    req_id = str(uuid.uuid4())
    ext1 = os.path.splitext(file1.filename)[1]
    ext2 = os.path.splitext(file2.filename)[1]
    path1 = os.path.join(UPLOAD_DIR, f"{req_id}_1{ext1}")
    path2 = os.path.join(UPLOAD_DIR, f"{req_id}_2{ext2}")
    
    with open(path1, "wb") as b1, open(path2, "wb") as b2:
        shutil.copyfileobj(file1.file, b1)
        shutil.copyfileobj(file2.file, b2)
    
    # Process images
    img1 = cv2.imread(path1)
    img2 = cv2.imread(path2)
    
    if img1 is None or img2 is None:
        return JSONResponse(content={"error": "Invalid images"}, status_code=400)
    
    # Optional: Undistort if calibration is available
    if CALIB:
        mtx = np.array(CALIB["camera_matrix"])
        dist = np.array(CALIB["dist_coeff"])
        img1 = cv2.undistort(img1, mtx, dist)
        img2 = cv2.undistort(img2, mtx, dist)
        focal_length = (mtx[0, 0] + mtx[1, 1]) / 2.0
    else:
        # Default focal length (approximate for typical smartphone)
        focal_length = 0.8 * img1.shape[1] 
    
    # 1. Feature Matching
    pts1, pts2 = match_features(img1, img2)
    if pts1 is None or len(pts1) < 15:
        return JSONResponse(content={"error": "Not enough matching features found"}, status_code=400)
    
    # 2. Triangulation
    pts3d = get_triangulated_points(pts1, pts2, baseline, focal_length, img1.shape)
    
    if pts3d is None:
        return JSONResponse(content={"error": "Triangulation failed. Ensure horizontal movement and overlap."}, status_code=400)
    
    # 3. Dimension Estimation
    dims = estimate_object_dimensions(pts3d)
    
    if dims is None:
        return JSONResponse(content={"error": "Could not estimate dimensions from 3D data."}, status_code=400)
    
    # 4. Draw results on First Image
    result_img = draw_3d_results(img1.copy(), dims)
    
    output_filename = f"out_{req_id}.jpg" # Convert to jpg for consistent display
    output_path = os.path.join(OUTPUT_DIR, output_filename)
    cv2.imwrite(output_path, result_img)
    
    return {
        "output_url": f"/output/{output_filename}",
        "dimensions": {
            "width": float(dims[0]),
            "height": float(dims[1]),
            "depth": float(dims[2])
        },
        "success": True,
        "calibrated": CALIB is not None
    }

@app.get("/output/{filename}")
async def get_output(filename: str):
    path = os.path.join(OUTPUT_DIR, filename)
    if os.path.exists(path):
        return FileResponse(path)
    return JSONResponse(content={"error": "File not found"}, status_code=404)

@app.get("/status")
async def status():
    return {
        "status": "ready", 
        "mode": "stereo",
        "has_calibration": CALIB is not None
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
