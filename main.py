import os
import cv2
import numpy as np


# =====================================================================
# 函数 1: 读取并解析 KITTI Calib (标定) 文件
# =====================================================================
def read_kitti_calib(calib_file_path):
    """
    读取并解析 KITTI 数据集中的 calib 文件。
    """
    if not os.path.exists(calib_file_path):
        raise FileNotFoundError(f"找不到标定文件: {calib_file_path}")

    calib_data = {}

    with open(calib_file_path, 'r') as f:
        for line in f.readlines():
            if not line.strip():
                continue

            key, value_str = line.split(':', 1)
            key = key.strip()
            values = np.array([float(x) for x in value_str.split()])

            if key.startswith('P'):  # P0, P1, P2, P3
                calib_data[key] = values.reshape(3, 4)
            elif key == 'R0_rect':
                mat = values.reshape(3, 3)
                mat_4x4 = np.eye(4)
                mat_4x4[:3, :3] = mat
                calib_data[key] = mat_4x4
            elif key in ['Tr_velo_to_cam', 'Tr_imu_to_velo']:
                mat = values.reshape(3, 4)
                mat_4x4 = np.eye(4)
                mat_4x4[:3, :4] = mat
                calib_data[key] = mat_4x4
            else:
                calib_data[key] = values

    return calib_data


# =====================================================================
# 函数 2: 读取并解析 KITTI Label (标签) 文件
# =====================================================================
def read_kitti_label(label_file_path):
    """
    读取并解析 KITTI 数据集的 label 文件。
    """
    if not os.path.exists(label_file_path):
        raise FileNotFoundError(f"找不到标签文件: {label_file_path}")

    objects = []

    with open(label_file_path, 'r') as f:
        for line in f.readlines():
            line = line.strip()
            if not line:
                continue

            parts = line.split()

            if len(parts) < 15:
                continue

            obj = {
                'type': parts[0],
                'truncated': float(parts[1]),
                'occluded': int(parts[2]),
                'alpha': float(parts[3]),
                'bbox_2d': [float(parts[4]), float(parts[5]), float(parts[6]), float(parts[7])],
                'dimensions': [float(parts[8]), float(parts[9]), float(parts[10])],
                'location': [float(parts[11]), float(parts[12]), float(parts[13])],
                'rotation_y': float(parts[14])
            }

            if len(parts) >= 16:
                obj['score'] = float(parts[15])
            else:
                obj['score'] = 1.0

            objects.append(obj)

    return objects


# =====================================================================
# 函数 3: 可视化绘制 (传入图像和标签数据)
# =====================================================================
def draw_kitti_labels(frame, labels, filter_dontcare=True):
    """
    在图像上绘制 KITTI 标签（2D框 + 多行详细信息）。

    参数:
        frame (np.ndarray): OpenCV 读入的图像数据 (BGR 格式)。
        labels (list): read_kitti_label 返回的字典列表。
        filter_dontcare (bool): 是否隐藏 DontCare（默认 True 隐藏，画面更干净）。

    返回:
        np.ndarray: 绘制好信息的图像。
    """
    vis_img = frame.copy()
    h_img, w_img = vis_img.shape[:2]

    # 为不同类别定义不同的颜色 (B, G, R)
    color_map = {
        'Car': (0, 255, 0),  # 绿色
        'Pedestrian': (255, 0, 0),  # 蓝色
        'Cyclist': (0, 0, 255),  # 红色
        'Truck': (0, 165, 255),  # 橙色
        'Van': (0, 255, 255),  # 黄色
        'Tram': (255, 0, 255),  # 品红
        'Person_sitting': (255, 255, 0),  # 青色
        'DontCare': (128, 128, 128)  # 灰色
    }
    default_color = (255, 255, 255)  # 默认白色

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.4
    thickness = 1
    line_height = 15

    for obj in labels:
        obj_type = obj['type']

        if filter_dontcare and obj_type == 'DontCare':
            continue

        color = color_map.get(obj_type, default_color)

        # 提取 2D 框坐标并转为整型
        xmin, ymin, xmax, ymax = map(int, obj['bbox_2d'])

        # 坐标边界检查
        xmin = max(0, xmin)
        ymin = max(0, ymin)
        xmax = min(w_img - 1, xmax)
        ymax = min(h_img - 1, ymax)

        if xmax <= xmin or ymax <= ymin:
            continue

        # 1. 画 2D 边界框
        cv2.rectangle(vis_img, (xmin, ymin), (xmax, ymax), color, 2)

        # 2. 准备多行文本信息
        text_lines = [
            f"Type: {obj_type}",
            f"Trunc: {obj['truncated']:.2f} | Occl: {obj['occluded']}",
            f"Alpha: {obj['alpha']:.2f} | Ry: {obj['rotation_y']:.2f}",
            f"Loc(X,Y,Z): {obj['location'][0]:.1f}, {obj['location'][1]:.1f}, {obj['location'][2]:.1f}",
            f"Dim(H,W,L): {obj['dimensions'][0]:.1f}, {obj['dimensions'][1]:.1f}, {obj['dimensions'][2]:.1f}"
        ]

        # 计算文字背景框的尺寸
        max_text_w = max([cv2.getTextSize(line, font, font_scale, thickness)[0][0] for line in text_lines])
        text_block_h = len(text_lines) * line_height + 5

        # 背景框起始坐标
        text_bg_x1 = xmin
        text_bg_y1 = ymin - text_block_h - 2
        text_bg_x2 = xmin + max_text_w + 5
        text_bg_y2 = ymin

        # 智能避让边界
        if text_bg_y1 < 0:
            text_bg_y1 = ymin
            text_bg_y2 = ymin + text_block_h
        if text_bg_x2 > w_img:
            diff = text_bg_x2 - w_img
            text_bg_x1 -= diff
            text_bg_x2 -= diff

        # 3. 画纯黑文字背景块
        cv2.rectangle(vis_img, (int(text_bg_x1), int(text_bg_y1)), (int(text_bg_x2), int(text_bg_y2)), (0, 0, 0), -1)

        # 4. 逐行写入白色文字
        text_x = text_bg_x1 + 2
        text_y = text_bg_y1 + line_height - 2
        for line in text_lines:
            cv2.putText(vis_img, line, (int(text_x), int(text_y)), font, font_scale, (255, 255, 255), thickness,
                        cv2.LINE_AA)
            text_y += line_height

    return vis_img


# =====================================================================
# Main: 综合测试与可视化演示
# =====================================================================
if __name__ == "__main__":
    # 1. 定义基础目录 (根据你的实际路径调整)
    calib_dir = r"D:\LB\Datasets\KITTI\calib\training\calib"
    label_dir = r"D:\LB\Datasets\KITTI\label\training\label"
    image_dir = r"D:\LB\Datasets\KITTI\left_image\training\image"  # 新增了图像目录

    # 2. 定义需要测试的帧 ID
    frame_ids = ["000000", "000001"]

    for frame_id in frame_ids:
        print(f"\n{'=' * 50}")
        print(f"正在处理帧 ID: {frame_id}")
        print(f"{'=' * 50}")

        # ---------------- 读取 Calib 文件 ----------------
        calib_path = os.path.join(calib_dir, f"{frame_id}.txt")
        try:
            calib = read_kitti_calib(calib_path)
            P2 = calib['P2']
            fx, cx = P2[0, 0], P2[0, 2]
            fy, cy = P2[1, 1], P2[1, 2]
            print(f"[标定文件] 解析成功! 左目内参: fx={fx:.2f}, fy={fy:.2f}, cx={cx:.2f}, cy={cy:.2f}")
        except Exception as e:
            print(f"[标定文件] 解析失败: {e}")
            continue

        # ---------------- 读取 Label 文件 ----------------
        label_path = os.path.join(label_dir, f"{frame_id}.txt")
        try:
            labels = read_kitti_label(label_path)
            valid_objects = [obj for obj in labels if obj['type'] != 'DontCare']
            dontcare_objects = [obj for obj in labels if obj['type'] == 'DontCare']
            print(f"[标签文件] 解析成功! {len(valid_objects)} 个有效目标, {len(dontcare_objects)} 个 DontCare 区域。")
        except Exception as e:
            print(f"[标签文件] 解析失败: {e}")
            continue

        # ---------------- 读取 Image 并绘制 ----------------
        image_path = os.path.join(image_dir, f"{frame_id}.png")
        if not os.path.exists(image_path):
            print(f"[图像文件] 找不到图片: {image_path}，跳过可视化。")
            continue

        # 使用 OpenCV 读取图像
        frame = cv2.imread(image_path)

        # 调用刚才写的绘制函数 (可以选择过滤掉 DontCare)
        result_img = draw_kitti_labels(frame, labels, filter_dontcare=True)

        # 展示结果 (KITTI 图像很宽，使用 WINDOW_NORMAL 允许用户拖拽缩放)
        window_name = f"KITTI Visualization - {frame_id}"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, 1242, 375)  # 设置初始窗口大小
        cv2.imshow(window_name, result_img)

        print(f"已弹出图像窗口，按【任意键】继续处理下一帧...")

        # 保存一张结果图到本地（可选）
        # cv2.imwrite(f"output_{frame_id}.jpg", result_img)

        # 等待按键，按任意键关闭当前窗口并继续循环
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    print(f"\n{'=' * 50}")
    print("所有测试处理完毕！")
    print(f"{'=' * 50}")