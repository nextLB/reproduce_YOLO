import cv2
import numpy as np


def compute_sgbm_disparity(left_img, right_img):
    """
    仅使用左右图像，计算 SGBM 视差图。

    参数:
        left_img (np.ndarray): 左目图像 (BGR)
        right_img (np.ndarray): 右目图像 (BGR)

    返回:
        disparity (np.ndarray): 浮点型视差图 (float32, 单位：像素)
    """
    # 1. SGBM 必须在灰度图上运行
    left_gray = cv2.cvtColor(left_img, cv2.COLOR_BGR2GRAY)
    right_gray = cv2.cvtColor(right_img, cv2.COLOR_BGR2GRAY)

    # 2. 配置 SGBM 参数 (针对 KITTI 1242x375 调优)
    min_disp = 0
    # 视差搜索范围，必须是 16 的倍数。KITTI 图像较宽，通常 256 足够
    num_disp = 256
    block_size = 5  # 匹配块大小，必须是奇数

    # 平滑惩罚系数
    channels = 1
    P1 = 8 * channels * block_size ** 2
    P2 = 32 * channels * block_size ** 2

    # 创建 SGBM 对象
    stereo = cv2.StereoSGBM_create(
        minDisparity=min_disp,
        numDisparities=num_disp,
        blockSize=block_size,
        P1=P1,
        P2=P2,
        disp12MaxDiff=1,
        uniquenessRatio=10,
        speckleWindowSize=100,
        speckleRange=2,
        preFilterCap=63,
        mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY  # 推荐模式
    )

    # 3. 计算视差 (OpenCV 返回的是 16 倍放大的有符号整型)
    disp_16s = stereo.compute(left_gray, right_gray).astype(np.float32)

    # 4. 转换为真实的浮点视差
    disparity = disp_16s / 16.0

    # 5. 过滤无效视差 (SGBM 中默认用 -16 表示匹配失败，除以 16 后就是 -1)
    disparity[disparity < 0] = 0.0

    return disparity


def compute_depth_map(disparity, calib):
    """
    通过视差图和相机标定参数计算真实的物理深度。

    参数:
        disparity (np.ndarray): 视差图 (float32)
        calib (dict): 标定数据字典

    返回:
        depth_map (np.ndarray): 深度图 (float32, 单位：米)
    """
    # 获取焦距 f
    f = calib['P2'][0, 0]
    # 计算基线 B (单位：米)，即左右相机的物理距离
    baseline = abs(calib['P3'][0, 3] - calib['P2'][0, 3]) / f

    # 初始化深度图，全部设为 0
    depth_map = np.zeros_like(disparity, dtype=np.float32)

    # 过滤掉无效视差（视差为0或者负数的地方没有深度）
    valid_mask = disparity > 0

    # 核心公式：Z = (f * B) / d
    depth_map[valid_mask] = (f * baseline) / disparity[valid_mask]

    return depth_map


def visualize_stereo_results(left_frame, disparity, depth_map, frame_id=""):
    """
    将原图、视差图、深度图处理并拼接成一张画布。

    参数:
        left_frame (np.ndarray): 左目原图 (BGR)
        disparity (np.ndarray): 视差图数据 (float32)
        depth_map (np.ndarray): 深度图数据 (float32)
        frame_id (str): 当前帧的标识，用于显示

    返回:
        combined_canvas (np.ndarray): 拼接好的可视化图像
    """
    # --- 视差图可视化 (范围 0~128) ---
    disp_vis = np.clip(disparity, 0, 128)
    disp_vis = (disp_vis / 128.0 * 255).astype(np.uint8)
    disp_color = cv2.applyColorMap(disp_vis, cv2.COLORMAP_JET)

    # --- 深度图可视化 (范围 0~80米) ---
    depth_vis = np.clip(depth_map, 0, 80)
    depth_vis = (depth_vis / 80.0 * 255).astype(np.uint8)
    # 把无效区域强制设为纯黑
    depth_vis[depth_map == 0] = 0
    depth_color = cv2.applyColorMap(depth_vis, cv2.COLORMAP_JET)

    # --- 加文字标签 ---
    cv2.putText(disp_color, "Disparity (0-128)", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
    cv2.putText(depth_color, "Depth (0-80m)", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)

    # 可以在原图上加个帧号
    if frame_id:
        cv2.putText(left_frame, f"Frame: {frame_id}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)

    # 确保尺寸一致以便水平拼接
    h, w = left_frame.shape[:2]
    disp_color_resized = cv2.resize(disp_color, (w, h))
    depth_color_resized = cv2.resize(depth_color, (w, h))

    # 拼接画布：左原图，中视差图，右深度图
    combined_canvas = np.hstack((left_frame, disp_color_resized, depth_color_resized))

    return combined_canvas