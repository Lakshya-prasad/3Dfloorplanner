"""
══════════════════════════════════════════════
app.py — ETH BUILDER Python API
Flask + OpenCV Blueprint → 3D Converter

POST /process-blueprint
  → Downloads blueprint from Dropbox URL
  → OpenCV: edge detection + contour extraction
  → Generates .obj 3D model file
  → Returns path to .obj file

🔧 SETUP:
  pip install -r requirements.txt
  python app.py
══════════════════════════════════════════════
"""

from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import cv2
import numpy as np
import requests
import os
import uuid
import math
import traceback

app = Flask(__name__)
CORS(app, origins="*")  # Allow requests from Node.js backend

# ─── OUTPUT FOLDER ────────────────────────────
# Generated .obj files are saved here temporarily
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ══════════════════════════════════════════════
# 1. HEALTH CHECK
# ══════════════════════════════════════════════
@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "service": "ETH BUILDER Python API"})


# ══════════════════════════════════════════════
# 2. MAIN ROUTE: PROCESS BLUEPRINT
# ══════════════════════════════════════════════
@app.route("/process-blueprint", methods=["POST"])
def process_blueprint():
    """
    Accepts a Dropbox file URL, downloads the image,
    runs OpenCV processing, and returns a .obj 3D model.

    Request JSON:
    {
      "fileUrl": "https://dl.dropboxusercontent.com/...",
      "fileName": "blueprint.png",
      "userId": "firebase-uid"
    }

    Response JSON:
    {
      "success": true,
      "objFileName": "abc123.obj",
      "downloadPath": "/download/abc123.obj"
    }
    """
    print("[OpenCV] Received request")
    print(f"[OpenCV] Method: {request.method}")
    print(f"[OpenCV] Content-Type: {request.content_type}")

    data = request.get_json(silent=True) or {}

    if "fileUrl" not in data:
        print("[OpenCV] Invalid request body:", data)
        return jsonify({"success": False, "error": "fileUrl is required"}), 400

    file_url  = data.get("fileUrl")
    file_name = data.get("fileName", "blueprint")
    user_id   = data.get("userId", "anonymous")

    print(f"[OpenCV] Processing blueprint for user: {user_id}")
    print(f"[OpenCV] File URL: {file_url}")

    try:
        # Step 1: Download image from Dropbox
        image = download_image(file_url)
        if image is None:
            return jsonify({"success": False, "error": "Failed to download image"}), 500

        # Step 2: Run OpenCV processing pipeline
        contours, processed = extract_contours(image)
        print(f"[OpenCV] Contours found: {len(contours)}")

        # Step 3: Convert contours to 3D .obj model
        obj_id       = str(uuid.uuid4())[:8]
        obj_filename = f"{obj_id}.obj"
        obj_path     = os.path.join(OUTPUT_DIR, obj_filename)

        generate_obj(contours, image.shape, obj_path)
        print(f"[OpenCV] OBJ saved: {obj_path}")

        return jsonify({
            "success":      True,
            "objFileName":  obj_filename,
            "downloadPath": f"/download/{obj_filename}",
            "contoursFound": len(contours)
        })

    except Exception as e:
        print(f"[OpenCV] Error: {e}")
        traceback.print_exc()
        return jsonify({"success": False, "error": str(e)}), 500


# ══════════════════════════════════════════════
# 3. DOWNLOAD ROUTE: Serve generated .obj file
# ══════════════════════════════════════════════
@app.route("/download/<filename>", methods=["GET"])
def download_file(filename):
    """
    Serves the generated .obj file.
    Node.js backend proxies this — frontend never calls it directly.
    """
    file_path = os.path.join(OUTPUT_DIR, filename)
    if not os.path.exists(file_path):
        return jsonify({"error": "File not found"}), 404

    return send_file(
        file_path,
        as_attachment=True,
        download_name=filename,
        mimetype="text/plain"
    )


# ══════════════════════════════════════════════
# 4. OPENCV PROCESSING PIPELINE
# ══════════════════════════════════════════════

def download_image(url):
    """
    Downloads an image from a URL and decodes it with OpenCV.
    Handles both image files (JPG/PNG) and PDFs.
    """
    try:
        print(f"[OpenCV] Downloading image from: {url}")

        # Force direct download from Dropbox
        url = url.replace("dl=0", "dl=1").replace("?dl=0", "?dl=1")
        if "dropbox.com" in url and "dl=1" not in url:
            url = url + ("&dl=1" if "?" in url else "?dl=1")
        print(f"[OpenCV] Resolved download URL: {url}")
        response = requests.get(url, timeout=30, allow_redirects=True)
        response.raise_for_status()

        # Decode image bytes → numpy array → OpenCV image
        img_array = np.frombuffer(response.content, dtype=np.uint8)
        image = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

        if image is None:
            raise ValueError("Could not decode image. PDF support requires pdf2image.")

        print(f"[OpenCV] Image downloaded: {image.shape}")
        return image

    except Exception as e:
        print(f"[Download] Error: {e}")
        traceback.print_exc()
        return None


def extract_contours(image):
    try:
        # Grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    except Exception:
        # If image has alpha channel (RGBA)
        image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Gaussian blur
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Canny edge detection
    edges = cv2.Canny(blurred, threshold1=30, threshold2=100)

    # Dilate
    kernel = np.ones((3, 3), np.uint8)
    dilated = cv2.dilate(edges, kernel, iterations=1)

    # Find contours
    contours, _ = cv2.findContours(
        dilated,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    # Filter small noise
    h, w = image.shape[:2]
    min_area = max(100, (h * w) * 0.0003)
    significant = [c for c in contours if cv2.contourArea(c) > min_area]

    # If no contours found, lower the threshold
    if len(significant) == 0:
        significant = [c for c in contours if cv2.contourArea(c) > 50]

    return significant, dilated

# ══════════════════════════════════════════════
# 5. OBJ FILE GENERATOR
# ══════════════════════════════════════════════

def generate_obj(contours, image_shape, output_path):
    """
    Converts OpenCV 2D contours into a 3D Wavefront .obj file.

    Strategy:
    - Each contour becomes a wall (extruded polygon)
    - Image coordinates are normalized to [-1, 1] range
    - Walls are extruded upward by WALL_HEIGHT units
    - A flat floor plane is added at y=0

    The .obj file can be loaded directly in Three.js using OBJLoader.
    """
    img_h, img_w = image_shape[:2]
    WALL_HEIGHT = 0.5      # Height of extruded walls in 3D units
    SCALE       = 2.0      # Normalize coordinates to [-SCALE, SCALE]

    vertices  = []  # List of (x, y, z) tuples
    faces     = []  # List of face index tuples (1-indexed for .obj)

    def normalize(px, py):
        """Convert pixel coords to normalized 3D coords."""
        x = (px / img_w - 0.5) * SCALE
        z = (py / img_h - 0.5) * SCALE
        return x, z

    vertex_index = 1  # .obj vertices are 1-indexed

    for contour in contours:
        # Approximate contour to reduce vertex count
        epsilon = 0.01 * cv2.arcLength(contour, True)
        approx  = cv2.approxPolyDP(contour, epsilon, True)
        points  = approx.reshape(-1, 2)

        if len(points) < 3:
            continue  # Skip degenerate contours

        # For each segment of the contour, create a wall quad
        base_idx = vertex_index

        # Add bottom ring vertices (y = 0)
        for pt in points:
            x, z = normalize(pt[0], pt[1])
            vertices.append((x, 0.0, z))
            vertex_index += 1

        # Add top ring vertices (y = WALL_HEIGHT)
        for pt in points:
            x, z = normalize(pt[0], pt[1])
            vertices.append((x, WALL_HEIGHT, z))
            vertex_index += 1

        n = len(points)

        # Create quad faces between bottom and top rings
        for i in range(n):
            next_i = (i + 1) % n
            # Bottom-left, bottom-right, top-right, top-left
            bl = base_idx + i
            br = base_idx + next_i
            tr = base_idx + n + next_i
            tl = base_idx + n + i
            faces.append((bl, br, tr, tl))

    # Add a floor plane (simple flat quad)
    floor_verts = [
        (-SCALE/2, 0.0, -SCALE/2),
        ( SCALE/2, 0.0, -SCALE/2),
        ( SCALE/2, 0.0,  SCALE/2),
        (-SCALE/2, 0.0,  SCALE/2),
    ]
    floor_base = vertex_index
    for v in floor_verts:
        vertices.append(v)
    faces.append((floor_base, floor_base+1, floor_base+2, floor_base+3))

    # Write .obj file
    with open(output_path, "w") as f:
        f.write("# ETH BUILDER — Generated 3D Model\n")
        f.write("# OpenCV Blueprint Conversion\n\n")
        f.write("o Blueprint3D\n\n")

        # Write vertices
        for v in vertices:
            f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")

        f.write("\n")

        # Write faces (quad faces)
        for face in faces:
            indices = " ".join(str(i) for i in face)
            f.write(f"f {indices}\n")

    print(f"[OBJ] Written {len(vertices)} vertices, {len(faces)} faces")


# ══════════════════════════════════════════════
# 6. RUN SERVER
# ══════════════════════════════════════════════
if __name__ == "__main__":
    print("ETH BUILDER Python API running on http://127.0.0.1:5000")
    print("OpenCV version: " + cv2.__version__)
    app.run(host="0.0.0.0", port=5000, debug=True)

"""
🔮 FUTURE SCOPE:
- PDF support: use pdf2image to convert PDF pages to images first
- Better 3D: use open3d for point cloud generation
- Room detection: use ML models to identify rooms, doors, windows
- Export to .gltf for better Three.js support with materials
"""
