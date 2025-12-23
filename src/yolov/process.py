import os
import cv2
import numpy as np

from .box import Box
from src.setup import get_hardware_mode

HARDWARE_MODE = get_hardware_mode()

if HARDWARE_MODE == "310b":
    import acl

    from acllite.acllite_model import AclLiteModel
    from acllite.acllite_resource import AclLiteResource

    acl_resource = AclLiteResource()
    acl_resource.init()
elif HARDWARE_MODE == "normal":
    import onnxruntime as ort
elif HARDWARE_MODE == "rk3588":
    from rknn.api import RKNN
else:
    raise ValueError(f"不支持的硬件模式: {HARDWARE_MODE}")

# 初始化模型
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if HARDWARE_MODE == "310b":
    MODEL_PATH = os.path.join(BASE_DIR, 'models', 'tennis.om')
    model = AclLiteModel(MODEL_PATH)
elif HARDWARE_MODE == "normal":
    MODEL_PATH = os.path.join(BASE_DIR, 'models', 'tennis.onnx')
    session = ort.InferenceSession(MODEL_PATH, providers=['CPUExecutionProvider'])
    input_name = session.get_inputs()[0].name
elif HARDWARE_MODE == "rk3588":
    MODEL_PATH = os.path.join(BASE_DIR, 'models', 'tennis.rknn')
    rknn = RKNN()
    rknn.load_rknn(MODEL_PATH)
    rknn.init_runtime(target="rk3588")
else:
    raise ValueError(f"不支持的硬件模式: {HARDWARE_MODE}")

img_size = 640

def yolo_infer(frame):
    if frame is None or frame.size == 0:
        print("无效的图像输入")
        return []

    H, W = frame.shape[:2]
    scale = min(img_size / H, img_size / W)
    new_h, new_w = int(H * scale), int(W * scale)
    resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)

    pad_top = (img_size - new_h) // 2
    pad_left = (img_size - new_w) // 2
    input_img = np.full((img_size, img_size, 3), 114, dtype=np.uint8)
    input_img[pad_top:pad_top + new_h, pad_left:pad_left + new_w] = resized

    if HARDWARE_MODE == "310b":
        # 图像预处理：BGR转RGB，归一化，维度调整
        input_img = cv2.cvtColor(input_img, cv2.COLOR_BGR2RGB)
        input_img = input_img.astype(np.float32) / 255.0
        input_img = np.transpose(input_img, (2, 0, 1))  # HWC->CHW
        input_img = np.expand_dims(input_img, axis=0)  # 添加批次维度
        outputs = model.execute([input_img])
    elif HARDWARE_MODE == "normal":
        blob = cv2.dnn.blobFromImage(input_img, scalefactor=1 / 255.0, size=(img_size, img_size), swapRB=True, crop=False)
        outputs = session.run(None, {input_name: blob})
    elif HARDWARE_MODE == "rk3588":
        outputs = rknn.inference(inputs=[input_img],data_format="nchw")

    pred = outputs[0].squeeze().T  # [C, N] -> [N, C]

    if pred.ndim != 2 or pred.shape[0] == 0:
        return []

    scores = pred[:, 4:]
    class_ids = np.argmax(scores, axis=1)
    conf_scores = scores[np.arange(len(scores)), class_ids]
    mask = conf_scores > 0.70

    pred = pred[mask]
    conf_scores = conf_scores[mask]
    class_ids = class_ids[mask]

    boxes = []
    raw_boxes = []

    for p in pred:
        cx, cy, w, h = p[:4]
        x1 = cx - 0.5 * w
        y1 = cy - 0.5 * h
        x2 = cx + 0.5 * w
        y2 = cy + 0.5 * h
        x1 = max(0, (x1 - pad_left) / scale)
        y1 = max(0, (y1 - pad_top) / scale)
        x2 = min(W, (x2 - pad_left) / scale)
        y2 = min(H, (y2 - pad_top) / scale)
        raw_boxes.append([x1, y1, x2, y2])

    raw_boxes = np.array(raw_boxes, dtype=np.float32)
    indices = cv2.dnn.NMSBoxes(raw_boxes.tolist(), conf_scores.tolist(), 0.25, 0.45)

    if indices is not None and len(indices) > 0:
        for idx in indices:
            i = int(idx) if np.isscalar(idx) else int(idx[0])
            x1, y1, x2, y2 = raw_boxes[i]
            box = Box(int(x1), int(y1), int(x2 - x1), int(y2 - y1))
            boxes.append(box)

    return boxes