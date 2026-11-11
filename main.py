import os
import sys
import cv2
import time

# 解决 Windows 终端中文乱码问题
if sys.platform == 'win32':
    os.system("chcp 65001 > nul")


from Binocular_Depth.KITTI_handle import build_kitti_file_lists, read_kitti_calib, read_kitti_label
from Binocular_Depth.SGBM import compute_sgbm_disparity, compute_depth_map, visualize_stereo_results
from Binocular_Depth.Point_Cloud import project_2d_to_3d_manual, project_2d_to_3d_opencv, visualize_compare_no_color, visualize_compare_with_color, compare_pointcloud_with_labels


if __name__ == "__main__":
    ROOT_DIR = r"D:\LB\Datasets\KITTI"

    try:
        dataset_index = build_kitti_file_lists(ROOT_DIR)
        total_frames = dataset_index['length']

        print(f"\n总帧数: {total_frames}")
        print("提示：按任意键查看下一帧，按 'q' 键退出程序。")
        print("-" * 50)

        for i in range(total_frames):
            calib_path = dataset_index['calibs'][i]
            label_path = dataset_index['labels'][i]
            left_img_path = dataset_index['left_imgs'][i]
            right_img_path = dataset_index['right_imgs'][i]

            print(f"正在处理第 {i + 1}/{total_frames} 帧...")

            # 读取数据
            calib = read_kitti_calib(calib_path)
            labels = read_kitti_label(label_path)
            left_frame = cv2.imread(left_img_path)
            right_frame = cv2.imread(right_img_path)

            if left_frame is None or right_frame is None:
                continue

            # 核心算法计算
            disparity = compute_sgbm_disparity(left_frame, right_frame)
            depth_map = compute_depth_map(disparity, calib)

            print(f"--- 当前帧点云生成对比 ---")

            # 1. 测试手写公式版 (传入 depth_map)
            start_time = time.time()
            pts_manual, colors_manual, u_m, v_m = project_2d_to_3d_manual(depth_map, calib, image=left_frame)
            time_manual = (time.time() - start_time) * 1000

            # 2. 测试 OpenCV 版 (传入 disparity)
            start_time = time.time()
            pts_cv, colors_cv, u_cv, v_cv = project_2d_to_3d_opencv(disparity, calib, image=left_frame)
            time_cv = (time.time() - start_time) * 1000

            print(f"手写公式版: 生成点数 {pts_manual.shape[0]}, 耗时 {time_manual:.2f} ms")
            print(f"OpenCV 版 : 生成点数 {pts_cv.shape[0]}, 耗时 {time_cv:.2f} ms")

            # 先看无颜色的对比
            visualize_compare_no_color(pts_manual, pts_cv)

            # 再看带颜色的对比
            visualize_compare_with_color(pts_manual, colors_manual, pts_cv, colors_cv)

            compare_pointcloud_with_labels(pts_manual, u_m, v_m, labels, method_name="手写公式版")
            compare_pointcloud_with_labels(pts_cv, u_cv, v_cv, labels, method_name="OpenCV 版")



        print("-" * 50)
        print("所有数据处理完毕！")

    except Exception as e:
        print(f"发生错误: {e}")