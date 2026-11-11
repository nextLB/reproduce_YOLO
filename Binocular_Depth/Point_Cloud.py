import cv2
import numpy as np
import matplotlib.pyplot as plt


def project_2d_to_3d_manual(depth_map, calib, image=None):
    P2 = calib['P2']
    fx, cx = P2[0, 0], P2[0, 2]
    fy, cy = P2[1, 1], P2[1, 2]

    valid_mask = (depth_map > 0) & (depth_map <= 100.0)
    v, u = np.where(valid_mask)  # v是行(y)，u是列(x)
    Z = depth_map[v, u]

    if len(Z) == 0:
        return (np.empty((0, 3), dtype=np.float32), np.empty((0, 3), dtype=np.uint8), np.empty(0),
                np.empty(0)) if image is not None else (np.empty((0, 3), dtype=np.float32), np.empty(0), np.empty(0))

    X = (u - cx) * Z / fx
    Y = (v - cy) * Z / fy
    points_3d = np.stack((X, Y, Z), axis=-1)

    if image is not None:
        colors_rgb = image[v, u][:, ::-1]
        # 返回：坐标, 颜色, 像素u, 像素v
        return points_3d, colors_rgb, u, v
    return points_3d, u, v


def project_2d_to_3d_opencv(disparity, calib, image=None):
    P2 = calib['P2']
    P3 = calib['P3']
    fx, cx = P2[0, 0], P2[0, 2]
    cy = P2[1, 2]
    tx = (P3[0, 3] - P2[0, 3]) / fx

    Q = np.float32([
        [1, 0, 0, -cx],
        [0, 1, 0, -cy],
        [0, 0, 0, fx],
        [0, 0, -1.0 / tx, 0]
    ])

    points_3d_dense = cv2.reprojectImageTo3D(disparity, Q)
    Z = points_3d_dense[:, :, 2]

    valid_mask = (Z > 0) & (Z <= 100.0) & np.isfinite(Z)
    v, u = np.where(valid_mask)
    points_3d = points_3d_dense[v, u]

    if len(points_3d) == 0:
        return (np.empty((0, 3), dtype=np.float32), np.empty((0, 3), dtype=np.uint8), np.empty(0),
                np.empty(0)) if image is not None else (np.empty((0, 3), dtype=np.float32), np.empty(0), np.empty(0))

    if image is not None:
        colors_rgb = image[v, u][:, ::-1]
        return points_3d, colors_rgb, u, v
    return points_3d, u, v


def _downsample(points, colors=None, max_points=15000, max_depth=80.0):
    """过滤远点并随机降采样，保证 matplotlib 交互流畅"""
    # 先过滤过远的点（防止离群点撑大坐标轴范围）
    valid = points[:, 2] < max_depth
    points = points[valid]
    if colors is not None:
        colors = colors[valid]

    if len(points) > max_points:
        idx = np.random.choice(len(points), max_points, replace=False)
        points = points[idx]
        if colors is not None:
            colors = colors[idx]

    return points, colors


def _setup_axes(ax, points, title):
    """统一设置 3D 坐标轴（把相机 Z 轴放水平方向，Y 轴倒过来作为高度）"""
    x, y, z = points[:, 0], points[:, 1], points[:, 2]
    # 用 99% 分位数防止个别远点把范围撑爆
    z_max = max(np.percentile(z, 99), 5.0)
    x_lim = max(np.percentile(np.abs(x), 99), 5.0)
    y_abs_max = max(np.percentile(np.abs(y), 99), 3.0)

    ax.set_xlim([-x_lim, x_lim])
    ax.set_ylim([0, z_max])
    ax.set_zlim([-y_abs_max, y_abs_max])
    ax.set_xlabel('X (Right)')
    ax.set_ylabel('Z (Forward)')
    ax.set_zlabel('Y (Up)')
    ax.set_title(title, fontsize=11)
    # 让视角更符合"司机视角"
    ax.view_init(elev=15, azim=-70)


# =====================================================================
# 函数 1：可视化两种计算方式的点云对比（无颜色）
# =====================================================================
def visualize_compare_no_color(pts_manual, pts_cv, max_points=15000, max_depth=80.0):
    """
    并排显示手写公式版和 OpenCV 版的 3D 点云（纯色）。

    参数:
        pts_manual (np.ndarray): 手写版点云 (N, 3)
        pts_cv (np.ndarray): OpenCV 版点云 (N, 3)
        max_points (int): 每个图降采样后的点数
        max_depth (float): 最大深度过滤（米）
    """
    p1, _ = _downsample(pts_manual, None, max_points, max_depth)
    p2, _ = _downsample(pts_cv, None, max_points, max_depth)

    fig = plt.figure(figsize=(16, 7))
    fig.suptitle('3D Point Cloud Comparison (No Color)', fontsize=14)

    ax1 = fig.add_subplot(121, projection='3d')
    ax1.scatter(p1[:, 0], p1[:, 2], -p1[:, 1], s=0.3, c='dodgerblue', alpha=0.6)
    _setup_axes(ax1, p1, f'Manual Formula  (N={len(p1)})')

    ax2 = fig.add_subplot(122, projection='3d')
    ax2.scatter(p2[:, 0], p2[:, 2], -p2[:, 1], s=0.3, c='coral', alpha=0.6)
    _setup_axes(ax2, p2, f'OpenCV reprojectImageTo3D  (N={len(p2)})')

    plt.tight_layout()
    plt.show()


# =====================================================================
# 函数 2：可视化两种计算方式的点云对比（带颜色）
# =====================================================================
def visualize_compare_with_color(pts_manual, colors_manual, pts_cv, colors_cv,
                                 max_points=15000, max_depth=80.0):
    """
    并排显示手写公式版和 OpenCV 版的 3D 点云（带 RGB 颜色）。
    注意：颜色必须和坐标数量一致，所以除了 colors 之外，还需要传入对应的 pts。

    参数:
        pts_manual (np.ndarray): 手写版点云 (N, 3)
        colors_manual (np.ndarray): 手写版颜色 (N, 3) RGB
        pts_cv (np.ndarray): OpenCV 版点云 (N, 3)
        colors_cv (np.ndarray): OpenCV 版颜色 (N, 3) RGB
    """
    p1, c1 = _downsample(pts_manual, colors_manual, max_points, max_depth)
    p2, c2 = _downsample(pts_cv, colors_cv, max_points, max_depth)

    # 颜色归一化到 0~1（matplotlib 要求）
    c1_norm = c1.astype(np.float32) / 255.0
    c2_norm = c2.astype(np.float32) / 255.0

    fig = plt.figure(figsize=(16, 7))
    fig.suptitle('3D Point Cloud Comparison (With Color)', fontsize=14)

    ax1 = fig.add_subplot(121, projection='3d')
    ax1.scatter(p1[:, 0], p1[:, 2], -p1[:, 1], s=0.5, c=c1_norm, alpha=0.9)
    _setup_axes(ax1, p1, f'Manual Formula  (N={len(p1)})')

    ax2 = fig.add_subplot(122, projection='3d')
    ax2.scatter(p2[:, 0], p2[:, 2], -p2[:, 1], s=0.5, c=c2_norm, alpha=0.9)
    _setup_axes(ax2, p2, f'OpenCV reprojectImageTo3D  (N={len(p2)})')

    plt.tight_layout()
    plt.show()


def compare_pointcloud_with_labels(points_3d, u, v, labels, method_name="Method"):
    """
    提取点云中属于每个 Label 目标的子点云，并计算与 Label 真值（3D 中心）的误差。
    """
    print(f"\n[{method_name}] 点云与 Label 真值对比:")
    print(f"{'类别':<12} | {'点云三维质心 (X, Y, Z)':<25} | {'标签真值中心 (X, Y, Z)':<25} | {'中心偏差':<8}")
    print(f"{'-' * 85}")

    for obj in labels:
        if obj['type'] == 'DontCare':
            continue

        xmin, ymin, xmax, ymax = obj['bbox_2d']

        # 用 2D 框筛选属于该目标的点云
        mask = (u >= xmin) & (u <= xmax) & (v >= ymin) & (v <= ymax)
        obj_points = points_3d[mask]

        if len(obj_points) == 0:
            continue

        # 计算质心
        pc_center = np.mean(obj_points, axis=0)
        label_center = np.array(obj['location'])
        distance_error = np.linalg.norm(pc_center - label_center)

        pc_str = f"({pc_center[0]:>6.2f}, {pc_center[1]:>6.2f}, {pc_center[2]:>6.2f})"
        lb_str = f"({label_center[0]:>6.2f}, {label_center[1]:>6.2f}, {label_center[2]:>6.2f})"
        print(f"{obj['type']:<12} | {pc_str:<25} | {lb_str:<25} | {distance_error:.3f} m")
    print(f"{'-' * 85}")

