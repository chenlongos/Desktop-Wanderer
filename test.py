import cv2
import numpy as np


def extract_black_objects():
    # 打开摄像头
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("无法打开摄像头")
        return

    print("按 ESC 键退出")
    print("窗口说明:")
    print(" - 'Black Mask': 这里显示分离出的黑色区域 (白色部分代表检测到的黑色物体)")
    print(" - 'Result': 原图上叠加了检测框")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # 1. 转换色彩空间：BGR 转 HSV
        # 这是分离颜色的关键步骤
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # 1.5 拉高饱和度，消除光的影响
        hsv[:, :, 1] = np.clip(hsv[:, :, 1] * 1.5, 0, 255).astype(np.uint8)

        # 2. 定义黑色的 HSV 范围
        # 注意：这里只通过 V (亮度) 来限制黑色
        # H (色相) 和 S (饱和度) 的范围可以放宽，因为黑色不挑颜色
        lower_black = np.array([0, 0, 0])  # 最低值
        upper_black = np.array([180, 255, 80])  # 关键是 V <= 90 (亮度很低就是黑色)

        # 3. 根据范围创建掩膜 (Mask)
        # 这个 mask 里，符合黑色特征的区域是白色 (255)，其他区域是黑色 (0)
        mask = cv2.inRange(hsv, lower_black, upper_black)

        # 4. 形态学操作：去噪和填充
        # 目的：去掉小的噪点，并让黑色物体的轮廓更完整
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        # 闭运算：先膨胀后腐蚀，用于连接断裂的黑色区域
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        # 开运算：先腐蚀后膨胀，用于去除小的噪点
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

        # 腐蚀：分离相连的两个色块（只腐蚀一次，不会影响独立色块）
        mask = cv2.erode(mask, kernel, iterations=4)

        # 5. 连通域分析：把黑色区域分成独立的小块
        # 注意：连通域分析要求前景是白色 (255)，背景是黑色 (0)，刚好符合 mask 的定义
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)

        # 6. 遍历每一个找到的黑色块
        for i in range(1, num_labels):  # 0 是背景，所以从 1 开始
            x, y, w, h, area = stats[i]

            # 过滤掉太小的区域 (防止噪点)
            if area < 1000:  # 你可以根据实际情况调整这个值
                continue

            # 过滤掉长宽比例太离谱的框（宽高比在 0.25~4 之间）
            aspect_ratio = w / h if h > 0 else 0
            if aspect_ratio < 1 or aspect_ratio > 4:
                continue

            # 过滤掉框内非黑色区域太多的框
            roi = mask[y:y+h, x:x+w]
            black_pixels = cv2.countNonZero(roi)
            total_pixels = w * h
            black_ratio = black_pixels / total_pixels if total_pixels > 0 else 0
            if black_ratio < 0.7:  # 框内黑色像素少于70%则过滤
                continue

            # 在原图上画出矩形框
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, f"Black Block {i}", (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # 7. 显示结果
        # 注意：mask 窗口里，白色的部分就是被分离出来的黑色物体
        cv2.imshow('Black Mask (Separated Black)', mask)
        cv2.imshow('Result', frame)

        if cv2.waitKey(1) == 27:  # ESC键
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    extract_black_objects()