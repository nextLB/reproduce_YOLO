
import os
import cv2
import numpy as np



def build_kitti_file_lists(kitti_root_dir):
    """
    遍历 KITTI 数据集目录，提取并排序所有文件的绝对路径。

    参数:
        kitti_root_dir (str): 数据集根目录 (如 r"D:\LB\Datasets\KITTI")

    返回:
        dict: 包含四个列表的字典，键分别为 'calibs', 'labels', 'left_imgs', 'right_imgs'。
              这四个列表的长度严格一致，且索引一一对应。
    """
    # 1. 定义四个子目录的相对路径 (根据你的实际目录结构)
    # 你的结构比较特殊：calib 嵌套了 calib，label 嵌套了 label
    calib_dir = os.path.join(kitti_root_dir, "calib", "training", "calib")
    label_dir = os.path.join(kitti_root_dir, "label", "training", "label")
    left_img_dir = os.path.join(kitti_root_dir, "left_image", "training", "image")
    right_img_dir = os.path.join(kitti_root_dir, "right_image", "training", "image")

    # 检查基础目录是否存在
    for d in [calib_dir, label_dir, left_img_dir, right_img_dir]:
        if not os.path.exists(d):
            raise FileNotFoundError(f"找不到目录: {d}")

    # 2. 定义获取文件并排序的辅助函数
    def get_sorted_files(directory, ext):
        files = []
        for f in os.listdir(directory):
            # 过滤掉隐藏文件，并只保留指定扩展名（.txt 或 .png/.jpg）
            if not f.startswith('.') and f.lower().endswith(ext):
                # 使用 os.path.join 拼接成绝对路径
                full_path = os.path.join(directory, f)
                files.append(full_path)

        # 【核心】：按文件名（字符串）进行升序排序
        # 因为你的文件名是 000000, 000001 这种补零格式，直接对字符串排序即可
        files.sort()
        return files

    # 3. 获取排序后的四个文件路径列表
    calib_files = get_sorted_files(calib_dir, ".txt")
    label_files = get_sorted_files(label_dir, ".txt")
    left_img_files = get_sorted_files(left_img_dir, ".png")  # 如果你的图是 jpg，改成 ".jpg"
    right_img_files = get_sorted_files(right_img_dir, ".png")

    # 4. 严格的一致性校验
    total_num = len(left_img_files)
    if not (len(calib_files) == len(label_files) == total_num == len(right_img_files)):
        raise ValueError(
            f"文件数量不匹配！请检查数据集是否缺失文件。\n"
            f"Calib: {len(calib_files)}, Label: {len(label_files)}, "
            f"Left Img: {len(left_img_files)}, Right Img: {len(right_img_files)}"
        )

    print(f"[索引构建成功] 共发现 {total_num} 帧对齐数据。")

    # 5. 返回字典
    return {
        "calibs": calib_files,
        "labels": label_files,
        "left_imgs": left_img_files,
        "right_imgs": right_img_files,
        "length": total_num
    }


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





