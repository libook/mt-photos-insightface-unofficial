import cv2
import numpy as np
from PIL import Image
from io import BytesIO
import sys
import os

def initialize_insightface():
    import insightface
    from insightface.utils import storage
    from insightface.app import FaceAnalysis

    recognition_model = os.getenv("RECOGNITION_MODEL", "buffalo_l")
    detection_thresh = float(os.getenv("DETECTION_THRESH", "0.65"))

    storage.BASE_REPO_URL = 'https://github.com/kqstone/mt-photos-insightface-unofficial/releases/download/models'
    face_analysis = FaceAnalysis(providers=['CUDAExecutionProvider', 'CPUExecutionProvider'], allowed_modules=['detection', 'recognition'], name=recognition_model)
    face_analysis.prepare(ctx_id=0, det_thresh=detection_thresh, det_size=(640, 640))
    return face_analysis

def process_image(image_bytes, content_type, face_analysis):
    img = None

    try:
        if content_type in ['image/webp', 'image/gif']:
            with Image.open(BytesIO(image_bytes)) as img:
                if getattr(img, "is_animated", False):  # Check if the image supports animation
                    img.seek(0)
                frame = img.convert('RGB')
                np_arr = np.array(frame)
                img = cv2.cvtColor(np_arr, cv2.COLOR_RGB2BGR)
        if img is None:
            np_arr = np.frombuffer(image_bytes, np.uint8)
            img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        if img is None:
            return {'result': [], 'msg': 'Invalid image format or corrupted image.'}

        height, width, _ = img.shape
        if width > 10000 or height > 10000:
            return {'result': [], 'msg': 'height or width out of range'}

        faces = face_analysis.get(img)
        results = []
        for face in faces:
            resp_obj = {}
            embedding = face.normed_embedding.astype(float)
            resp_obj["embedding"] = embedding.tolist()
            box = face.bbox
            resp_obj["facial_area"] = {"x": int(box[0]), "y": int(box[1]), "w": int(box[2] - box[0]), "h": int(box[3] - box[1])}
            resp_obj["face_confidence"] = face.det_score.astype(float)
            results.append(resp_obj)

        return {'result': results}

    except Exception as e:
        return {'result': [], 'msg': str(e)}

def process_loop(conn):
    # 只在子进程中初始化 face_analysis
    face_analysis = initialize_insightface()

    while True:
        data = conn.recv()
        if data == "STOP":
            break

        image_bytes, content_type = data
        result = process_image(image_bytes, content_type, face_analysis)
        conn.send(result)
