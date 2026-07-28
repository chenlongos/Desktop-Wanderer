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
elif HARDWARE_MODE == "rk3588" or HARDWARE_MODE == "rk3576":
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

    BLUE_BUCKET_MODEL_PATH = os.path.join(BASE_DIR, 'models', 'yolov8-blue_bucket-fp16-rk3588.rknn')
    rknn_blue_bucket = RKNN()
    rknn_blue_bucket.load_rknn(BLUE_BUCKET_MODEL_PATH)
    rknn_blue_bucket.init_runtime(target="rk3588")
elif HARDWARE_MODE == "rk3576":
    MODEL_PATH = os.path.join(BASE_DIR, 'models', 'yolov8-int8-tennis.rknn')
    rknn = RKNN()
    rknn.load_rknn(MODEL_PATH)
    rknn.init_runtime(target="rk3576")
else:
    raise ValueError(f"不支持的硬件模式: {HARDWARE_MODE}")

img_size = 640
OBJ_THRESH = 0.50
NMS_THRESH = 0.45

def letterbox(img, new_shape=(img_size, img_size), color=(0, 0, 0)):
    """
    YOLOv8 官方预处理函数，保持宽高比 resize + center pad
    """
    shape = img.shape[:2]  # current shape [H, W]
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]
    dw /= 2  # divide padding into 2 sides
    dh /= 2
    if shape[::-1] != new_unpad:
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    return img, r, (dw, dh)

def dfl(position):
    import torch
    x = torch.tensor(position)
    n, c, h, w = x.shape
    p_num = 4
    mc = c // p_num
    y = x.reshape(n, p_num, mc, h, w)
    y = y.softmax(2)
    acc_metrix = torch.tensor(range(mc)).float().reshape(1, 1, mc, 1, 1)
    y = (y * acc_metrix).sum(2)
    return y.numpy()

def box_process(position):
    grid_h, grid_w = position.shape[2:4]
    col, row = np.meshgrid(np.arange(0, grid_w), np.arange(0, grid_h))
    col = col.reshape(1, 1, grid_h, grid_w)
    row = row.reshape(1, 1, grid_h, grid_w)
    grid = np.concatenate((col, row), axis=1)
    stride = np.array([img_size // grid_h, img_size // grid_w]).reshape(1, 2, 1, 1)

    position = dfl(position)
    box_xy = grid + 0.5 - position[:, 0:2, :, :]
    box_xy2 = grid + 0.5 + position[:, 2:4, :, :]
    xyxy = np.concatenate((box_xy * stride, box_xy2 * stride), axis=1)

    return xyxy

def filter_by_shape(box):
    """检查检测框是否接近正方形（网球是圆形，在图像中接近正方形）"""
    aspect_ratio = box.w / box.h
    return 0.8 < aspect_ratio < 1.2  # 接近正方形

def filter_boxes(boxes, box_confidences, box_class_probs):
    box_confidences = box_confidences.reshape(-1)
    candidate, class_num = box_class_probs.shape

    class_max_score = np.max(box_class_probs, axis=-1)
    classes = np.argmax(box_class_probs, axis=-1)

    _class_pos = np.where(class_max_score * box_confidences >= OBJ_THRESH)
    scores = (class_max_score * box_confidences)[_class_pos]

    boxes = boxes[_class_pos]
    classes = classes[_class_pos]

    return boxes, classes, scores

def nms_boxes(boxes, scores):
    x = boxes[:, 0]
    y = boxes[:, 1]
    w = boxes[:, 2] - boxes[:, 0]
    h = boxes[:, 3] - boxes[:, 1]

    areas = w * h
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)

        xx1 = np.maximum(x[i], x[order[1:]])
        yy1 = np.maximum(y[i], y[order[1:]])
        xx2 = np.minimum(x[i] + w[i], x[order[1:]] + w[i] + w[order[1:]])
        yy2 = np.minimum(y[i] + h[i], y[order[1:]] + h[i] + h[order[1:]])

        w1 = np.maximum(0.0, xx2 - xx1 + 0.00001)
        h1 = np.maximum(0.0, yy2 - yy1 + 0.00001)
        inter = w1 * h1

        ovr = inter / (areas[i] + areas[order[1:]] - inter)
        inds = np.where(ovr <= NMS_THRESH)[0]
        order = order[inds + 1]
    keep = np.array(keep)
    return keep

def post_process(input_data):
    boxes, scores, classes_conf = [], [], []
    default_branch = 3
    pair_per_branch = len(input_data) // default_branch
    for i in range(default_branch):
        boxes.append(box_process(input_data[pair_per_branch * i]))
        classes_conf.append(input_data[pair_per_branch * i + 1])
        scores.append(np.ones_like(input_data[pair_per_branch * i + 1][:, :1, :, :], dtype=np.float32))

    def sp_flatten(_in):
        ch = _in.shape[1]
        _in = _in.transpose(0, 2, 3, 1)
        return _in.reshape(-1, ch)

    boxes = [sp_flatten(_v) for _v in boxes]
    classes_conf = [sp_flatten(_v) for _v in classes_conf]
    scores = [sp_flatten(_v) for _v in scores]

    boxes = np.concatenate(boxes)
    classes_conf = np.concatenate(classes_conf)
    scores = np.concatenate(scores)

    boxes, classes, scores = filter_boxes(boxes, scores, classes_conf)

    nboxes, nclasses, nscores = [], [], []
    for c in set(classes):
        inds = np.where(classes == c)
        b = boxes[inds]
        c = classes[inds]
        s = scores[inds]
        keep = nms_boxes(b, s)

        if len(keep) != 0:
            nboxes.append(b[keep])
            nclasses.append(c[keep])
            nscores.append(s[keep])

    if not nclasses and not nscores:
        return None, None, None

    boxes = np.concatenate(nboxes)
    classes = np.concatenate(nclasses)
    scores = np.concatenate(nscores)

    return boxes, classes, scores

def yolo_infer(frame):
    if frame is None or frame.size == 0:
        print("无效的图像输入")
        return []

    H, W = frame.shape[:2]

    # 预处理：letterbox + BGR2RGB
    input_img, r, (dw, dh) = letterbox(frame, new_shape=(img_size, img_size), color=(0, 0, 0))
    input_img = cv2.cvtColor(input_img, cv2.COLOR_BGR2RGB)

    # 推理
    if HARDWARE_MODE == "310b":
        # 图像预处理：归一化，维度调整
        input_img = input_img.astype(np.float32) / 255.0
        input_img = np.transpose(input_img, (2, 0, 1))  # HWC->CHW
        input_img = np.expand_dims(input_img, axis=0)  # 添加批次维度
        outputs = model.execute([input_img])
    elif HARDWARE_MODE == "normal":
        input_data = input_img.transpose((2, 0, 1))
        input_data = input_data.reshape(1, *input_data.shape).astype(np.float32)
        input_data = input_data / 255.
        outputs = session.run(None, {input_name: input_data})
    elif HARDWARE_MODE == "rk3588" or HARDWARE_MODE == "rk3576":
        outputs = rknn.inference(inputs=[input_img], data_format='nhwc')

    # 后处理（根据硬件模式选择不同的后处理方式）
    if HARDWARE_MODE == "310b" or HARDWARE_MODE == "normal":
        # ONNX模型输出已经是检测框格式，直接解析
        pred = outputs[0].squeeze()
        
        if pred.ndim != 2 or pred.shape[0] == 0:
            return []
        
        # yolov8是[C, N], 需要转置为[N, C]
        pred = pred.T
        
        boxes_orig = pred[:, :4]  # [cx, cy, w, h]格式
        conf_scores = pred[:, 4]
        mask = conf_scores > OBJ_THRESH
        
        pred = pred[mask]
        boxes_orig = boxes_orig[mask]
        conf_scores = conf_scores[mask]
        
        raw_boxes = []
        for i in range(len(boxes_orig)):
            cx, cy, w, h = boxes_orig[i]
            x1 = (cx - w / 2 - dw) / r
            y1 = (cy - h / 2 - dh) / r
            x2 = (cx + w / 2 - dw) / r
            y2 = (cy + h / 2 - dh) / r
            
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(W, x2)
            y2 = min(H, y2)
            raw_boxes.append([x1, y1, x2, y2])
        
        # 应用NMS
        raw_boxes = np.array(raw_boxes, dtype=np.float32)
        indices = cv2.dnn.NMSBoxes(raw_boxes.tolist(), conf_scores.tolist(), OBJ_THRESH, NMS_THRESH)
        
        result_boxes = []
        if indices is not None and len(indices) > 0:
            for idx in indices:
                i = int(idx) if np.isscalar(idx) else int(idx[0])
                x1, y1, x2, y2 = raw_boxes[i]
                box = Box(int(x1), int(y1), int(x2 - x1), int(y2 - y1))
                result_boxes.append(box)
    elif HARDWARE_MODE == "rk3588":
        # fp16模型输出已经是检测框格式，直接解析
        pred = outputs[0].squeeze().T  # [C, N] -> [N, C]
        
        if pred.ndim != 2 or pred.shape[0] == 0:
            return []
        
        scores = pred[:, 4:]
        class_ids = np.argmax(scores, axis=1)
        conf_scores = scores[np.arange(len(scores)), class_ids]
        mask = conf_scores > OBJ_THRESH
        
        pred = pred[mask]
        conf_scores = conf_scores[mask]
        
        raw_boxes = []
        for p in pred:
            cx, cy, w, h = p[:4]
            x1 = cx - 0.5 * w
            y1 = cy - 0.5 * h
            x2 = cx + 0.5 * w
            y2 = cy + 0.5 * h
            # 反归一化到原始图像尺寸（使用letterbox的参数）
            x1 = (x1 - dw) / r
            y1 = (y1 - dh) / r
            x2 = (x2 - dw) / r
            y2 = (y2 - dh) / r
            # 裁剪到图像边界
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(W, x2)
            y2 = min(H, y2)
            raw_boxes.append([x1, y1, x2, y2])
        
        raw_boxes = np.array(raw_boxes, dtype=np.float32)
        indices = cv2.dnn.NMSBoxes(raw_boxes.tolist(), conf_scores.tolist(), OBJ_THRESH, NMS_THRESH)
        
        result_boxes = []
        if indices is not None and len(indices) > 0:
            for idx in indices:
                i = int(idx) if np.isscalar(idx) else int(idx[0])
                x1, y1, x2, y2 = raw_boxes[i]
                box = Box(int(x1), int(y1), int(x2 - x1), int(y2 - y1))
                result_boxes.append(box)
    elif HARDWARE_MODE == "rk3576":
        # int8模型输出是特征图格式，需要DFL解码
        boxes, classes, scores = post_process(outputs)
        
        # 将检测框坐标还原到原始图像
        result_boxes = []
        if boxes is not None:
            for i in range(boxes.shape[0]):
                x1, y1, x2, y2 = boxes[i]
                
                # 反归一化到原始图像尺寸
                x1 = (x1 - dw) / r
                y1 = (y1 - dh) / r
                x2 = (x2 - dw) / r
                y2 = (y2 - dh) / r
                
                # 裁剪到图像边界
                x1 = max(0, x1)
                y1 = max(0, y1)
                x2 = min(W, x2)
                y2 = min(H, y2)
                
                box = Box(int(x1), int(y1), int(x2 - x1), int(y2 - y1))
                result_boxes.append(box)

    # 过滤形状
    filtered_boxes = []
    for box in result_boxes:
        if filter_by_shape(box):
            filtered_boxes.append(box)

    return filtered_boxes

def yolo_infer_blue_bucket(frame):
    if frame is None or frame.size == 0:
        print("无效的图像输入")
        return []

    H, W = frame.shape[:2]

    input_img, r, (dw, dh) = letterbox(frame, new_shape=(img_size, img_size), color=(0, 0, 0))
    input_img = cv2.cvtColor(input_img, cv2.COLOR_BGR2RGB)

    if HARDWARE_MODE == "rk3588":
        outputs = rknn_blue_bucket.inference(inputs=[input_img], data_format='nhwc')

        pred = outputs[0].squeeze().T

        if pred.ndim != 2 or pred.shape[0] == 0:
            return []

        scores = pred[:, 4:]
        class_ids = np.argmax(scores, axis=1)
        conf_scores = scores[np.arange(len(scores)), class_ids]
        mask = conf_scores > OBJ_THRESH

        pred = pred[mask]
        conf_scores = conf_scores[mask]

        raw_boxes = []
        for p in pred:
            cx, cy, w, h = p[:4]
            x1 = cx - 0.5 * w
            y1 = cy - 0.5 * h
            x2 = cx + 0.5 * w
            y2 = cy + 0.5 * h
            x1 = (x1 - dw) / r
            y1 = (y1 - dh) / r
            x2 = (x2 - dw) / r
            y2 = (y2 - dh) / r
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(W, x2)
            y2 = min(H, y2)
            raw_boxes.append([x1, y1, x2, y2])

        raw_boxes = np.array(raw_boxes, dtype=np.float32)
        indices = cv2.dnn.NMSBoxes(raw_boxes.tolist(), conf_scores.tolist(), OBJ_THRESH, NMS_THRESH)

        result_boxes = []
        if indices is not None and len(indices) > 0:
            for idx in indices:
                i = int(idx) if np.isscalar(idx) else int(idx[0])
                x1, y1, x2, y2 = raw_boxes[i]
                box = Box(int(x1), int(y1), int(x2 - x1), int(y2 - y1))
                result_boxes.append(box)

        if len(result_boxes) > 1:
            min_x = min(b.x for b in result_boxes)
            min_y = min(b.y for b in result_boxes)
            max_x = max(b.x + b.w for b in result_boxes)
            max_y = max(b.y + b.h for b in result_boxes)
            result_boxes = [Box(min_x, min_y, max_x - min_x, max_y - min_y)]

        return result_boxes
    else:
        return get_bucket_local(frame, color="blue")

def get_bucket_local(frame, color="red"):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    if color == "red":
        lower_red1 = np.array([0, 80, 50])
        upper_red1 = np.array([10, 255, 255])
        lower_red2 = np.array([170, 80, 50])
        upper_red2 = np.array([180, 255, 255])

        mask = (
                cv2.inRange(hsv, lower_red1, upper_red1)
                | cv2.inRange(hsv, lower_red2, upper_red2)
        )
    elif color == "blue":
        lower_blue = np.array([90, 120, 120])
        upper_blue = np.array([130, 255, 255])

        mask = cv2.inRange(hsv, lower_blue, upper_blue)
        
    else:
        raise ValueError(f"Invalid color: {color}")

    num_labels, _labels, stats, _centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)

    boxes = []
    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        if area > 1000:
            x = int(stats[i, cv2.CC_STAT_LEFT])
            y = int(stats[i, cv2.CC_STAT_TOP])
            w = int(stats[i, cv2.CC_STAT_WIDTH])
            h = int(stats[i, cv2.CC_STAT_HEIGHT])
            boxes.append(Box(x, y, w, h))

    return boxes


def get_black_bucket_local(frame):
    # 转换色彩空间
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # 拉高饱和度，消除光的影响
    hsv[:, :, 1] = np.clip(hsv[:, :, 1] * 1.5, 0, 255).astype(np.uint8)

    # 定义黑色的 HSV 范围
    lower_black = np.array([0, 0, 0])
    upper_black = np.array([180, 255, 80])

    # 创建掩膜
    mask = cv2.inRange(hsv, lower_black, upper_black)

    # 形态学操作：开运算 + 腐蚀
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    # 保存开运算后的mask，用于后续补救
    mask_before_erode = mask.copy()

    # 腐蚀：分离相连的两个色块
    mask = cv2.erode(mask, kernel, iterations=4)

    # 连通域分析
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)

    candidates = []
    for i in range(1, num_labels):
        x, y, w, h, area = stats[i]

        # 过滤太小的区域
        if area < 1000:
            continue

        # 过滤长宽比例
        aspect_ratio = w / h if h > 0 else 0
        if aspect_ratio < 1 or aspect_ratio > 4:
            continue

        # 过滤框内黑色比例
        roi = mask[y:y+h, x:x+w]
        black_pixels = cv2.countNonZero(roi)
        total_pixels = w * h
        black_ratio = black_pixels / total_pixels if total_pixels > 0 else 0

        if black_ratio < 0.7:
            # 用原始mask再检查一次
            roi_before = mask_before_erode[y:y+h, x:x+w]
            black_pixels_before = cv2.countNonZero(roi_before)
            black_ratio_before = black_pixels_before / total_pixels if total_pixels > 0 else 0
            if black_ratio_before < 0.7:
                continue

        # 计算矩形度
        rect_area = w * h
        rectangularity = area / rect_area if rect_area > 0 else 0

        candidates.append((area, rectangularity, x, y, w, h))

    if not candidates:
        return []

    # 按矩形度降序，再按面积降序
    candidates.sort(key=lambda x: (x[1], x[0]), reverse=True)

    # 返回最接近矩形且面积最大的那个
    _, _, x, y, w, h = candidates[0]
    return [Box(int(x), int(y), int(w), int(h))]
