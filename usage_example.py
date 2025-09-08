"""
Contour Context Loop Closure Detection - Usage Example
使用示例
"""

import numpy as np
import os
import sys

# 添加当前目录到Python路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from main_loop_closure import LoopClosureDetector
from contour_manager import ContourManager
from contour_types import ContourManagerConfig


def test_single_point_cloud():
    """测试单个点云的轮廓提取"""
    print("Testing single point cloud contour extraction...")

    # 创建配置
    config = ContourManagerConfig()
    config.lv_grads = [1.0, 2.0, 3.0, 4.0]
    config.n_row = 100
    config.n_col = 100
    config.reso_row = 0.5
    config.reso_col = 0.5

    # 创建轮廓管理器
    cm = ContourManager(config, 0)

    # 生成测试点云数据（模拟一个简单场景）
    points = []

    # 添加一些地面点
    for x in np.linspace(-10, 10, 100):
        for y in np.linspace(-10, 10, 50):
            z = 0.1 * np.random.randn()  # 地面噪声
            points.append([x, y, z])

    # 添加一些建筑物/障碍物
    for x in np.linspace(-5, -3, 20):
        for y in np.linspace(2, 4, 20):
            for z in np.linspace(0, 3, 10):
                points.append([x, y, z])

    # 添加另一个建筑物
    for x in np.linspace(3, 5, 20):
        for y in np.linspace(-4, -2, 20):
            for z in np.linspace(0, 2.5, 8):
                points.append([x, y, z])

    point_cloud = np.array(points)
    print(f"Generated test point cloud with {len(point_cloud)} points")

    # 处理点云
    cm.make_bev(point_cloud, "test_scan")
    cm.make_contours_recursive()

    # 显示结果
    print(f"Generated BEV image shape: {cm.get_bev_image().shape}")

    for level in range(len(config.lv_grads)):
        contours = cm.get_lev_contours(level)
        keys = cm.get_lev_retrieval_key(level)
        print(f"Level {level}: {len(contours)} contours, {len(keys)} keys")

        for i, contour in enumerate(contours[:3]):  # 显示前3个轮廓
            print(f"  Contour {i}: {contour.cell_cnt} cells, "
                  f"pos=({contour.pos_mean[0]:.2f}, {contour.pos_mean[1]:.2f}), "
                  f"eig=({contour.eig_vals[0]:.3f}, {contour.eig_vals[1]:.3f})")


def test_loop_closure_with_synthetic_data():
    """使用合成数据测试回环检测"""
    print("\nTesting loop closure detection with synthetic data...")

    # 创建两个相似的轮廓管理器
    config = ContourManagerConfig()
    config.lv_grads = [1.0, 2.0, 3.0]

    cm1 = ContourManager(config, 1)
    cm2 = ContourManager(config, 2)

    # 生成相似的点云（模拟同一位置的两次扫描）
    base_points = []
    for x in np.linspace(-8, 8, 80):
        for y in np.linspace(-8, 8, 80):
            z = 0.05 * np.random.randn()
            base_points.append([x, y, z])

    # 添加一些标志性结构
    for x in np.linspace(-3, -1, 15):
        for y in np.linspace(1, 3, 15):
            for z in np.linspace(0, 2, 8):
                base_points.append([x, y, z])

    base_points = np.array(base_points)

    # 第一个扫描（原始）
    cm1.make_bev(base_points, "scan_1")
    cm1.make_contours_recursive()

    # 第二个扫描（稍有不同，模拟噪声和小变化）
    noise = 0.1 * np.random.randn(*base_points.shape)
    perturbed_points = base_points + noise

    cm2.make_bev(perturbed_points, "scan_2")
    cm2.make_contours_recursive()

    # 比较轮廓
    print(f"CM1 contours per level: {[len(cm1.get_lev_contours(i)) for i in range(len(config.lv_grads))]}")
    print(f"CM2 contours per level: {[len(cm2.get_lev_contours(i)) for i in range(len(config.lv_grads))]}")

    # 测试相似性检查
    from contour_types import ContourSimThresConfig
    sim_config = ContourSimThresConfig()

    # 检查第一层的前两个轮廓
    if (len(cm1.get_lev_contours(0)) > 0 and len(cm2.get_lev_contours(0)) > 0):
        from contour_view import ContourView
        is_similar = ContourView.check_sim(
            cm1.get_lev_contours(0)[0],
            cm2.get_lev_contours(0)[0],
            sim_config)
        print(f"Top contours similarity: {is_similar}")


def create_sample_data_files():
    """创建示例数据文件用于测试"""
    print("\nCreating sample data files...")

    # 创建示例姿态文件
    poses_data = []
    for i in range(100):
        # 生成简单的轨迹
        t = i * 0.1
        x = 10 * np.cos(0.1 * t)
        y = 10 * np.sin(0.1 * t)
        z = 0.0

        # 简单的旋转（面向运动方向）
        yaw = 0.1 * t + np.pi / 2

        # 创建变换矩阵
        R = np.array([[np.cos(yaw), -np.sin(yaw), 0],
                      [np.sin(yaw), np.cos(yaw), 0],
                      [0, 0, 1]])

        pose_line = f"{t:.3f} " + " ".join(
            [f"{R[j, k]:.6f}" for j in range(3) for k in range(3)]) + f" {x:.6f} {y:.6f} {z:.6f}"
        poses_data.append(pose_line)

    # 保存姿态文件
    os.makedirs("sample_data", exist_ok=True)
    with open("sample_data/poses.txt", "w") as f:
        f.write("\n".join(poses_data))

    # 创建示例激光扫描文件列表
    lidar_data = []
    for i in range(100):
        t = i * 0.1
        lidar_line = f"{t:.3f} {i} sample_data/scan_{i:06d}.bin"
        lidar_data.append(lidar_line)

    with open("sample_data/scans.txt", "w") as f:
        f.write("\n".join(lidar_data))

    # 创建示例点云文件（简化版本）
    for i in range(5):  # 只创建前5个作为示例
        points = []
        t = i * 0.1

        # 基本地面
        for x in np.linspace(-15, 15, 50):
            for y in np.linspace(-15, 15, 50):
                if np.random.rand() < 0.3:  # 稀疏化
                    z = 0.05 * np.random.randn()
                    points.append([x, y, z, 0.5])  # KITTI格式包含强度

        # 添加一些结构
        phase = 0.1 * t
        cx = 5 * np.cos(phase)
        cy = 5 * np.sin(phase)

        for x in np.linspace(cx - 2, cx + 2, 10):
            for y in np.linspace(cy - 1, cy + 1, 5):
                for z in np.linspace(0, 3, 8):
                    points.append([x, y, z, 0.8])

        # 保存为二进制文件
        points_array = np.array(points, dtype=np.float32)
        points_array.tofile(f"sample_data/scan_{i:06d}.bin")

    print("Sample data files created in 'sample_data/' directory")

    # 创建示例配置文件
    sample_config = {
        'fpath_sens_gt_pose': 'sample_data/poses.txt',
        'fpath_lidar_bins': 'sample_data/scans.txt',
        'fpath_outcome_sav': 'sample_data/results.txt',
        'correlation_thres': 0.5,
        'ContourDBConfig': {
            'nnk_': 10,
            'max_fine_opt_': 3,
            'q_levels_': [1, 2],
            'TreeBucketConfig': {
                'max_elapse_': 25.0,
                'min_elapse_': 15.0
            },
            'ContourSimThresConfig': {
                'ta_cell_cnt': 6.0,
                'tp_cell_cnt': 0.2,
                'tp_eigval': 0.2,
                'ta_h_bar': 0.3,
                'ta_rcom': 0.4,
                'tp_rcom': 0.25
            }
        },
        'ContourManagerConfig': {
            'lv_grads_': [1.0, 2.0, 3.0],
            'reso_row_': 0.5,
            'reso_col_': 0.5,
            'n_row_': 80,
            'n_col_': 80,
            'lidar_height_': 2.0,
            'blind_sq_': 4.0,
            'min_cont_key_cnt_': 5,
            'min_cont_cell_cnt_': 3,
            'piv_firsts_': 4,
            'dist_firsts_': 6,
            'roi_radius_': 8.0
        },
        'thres_lb_': {
            'i_ovlp_sum': 2,
            'i_ovlp_max_one': 2,
            'i_in_ang_rng': 2,
            'i_indiv_sim': 2,
            'i_orie_sim': 2,
            'correlation': 0.2,
            'area_perc': 0.02,
            'neg_est_dist': -8.0
        },
        'thres_ub_': {
            'i_ovlp_sum': 10,
            'i_ovlp_max_one': 10,
            'i_in_ang_rng': 10,
            'i_indiv_sim': 10,
            'i_orie_sim': 10,
            'correlation': 0.9,
            'area_perc': 0.5,
            'neg_est_dist': -1.0
        }
    }

    import yaml
    with open("sample_data/config.yaml", "w") as f:
        yaml.dump(sample_config, f, default_flow_style=False)

    print("Sample configuration saved to 'sample_data/config.yaml'")


def run_sample_loop_closure():
    """运行示例回环检测"""
    print("\nRunning sample loop closure detection...")

    # 首先创建示例数据
    create_sample_data_files()

    try:
        # 创建检测器
        detector = LoopClosureDetector("sample_data/config.yaml")

        # 运行少量扫描作为演示
        print("Processing first few scans...")
        for i in range(3):  # 只处理前3个扫描
            ret_code = detector.process_single_scan(i)
            if ret_code != 0:
                break

        print("Sample run completed!")

    except Exception as e:
        print(f"Error in sample run: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    print("Contour Context Loop Closure Detection - Python Implementation")
    print("=" * 60)

    # 运行测试
    try:
        test_single_point_cloud()
        test_loop_closure_with_synthetic_data()
        run_sample_loop_closure()

        print("\n" + "=" * 60)
        print("All tests completed successfully!")
        print("\nTo run the full pipeline with your own data:")
        print("1. Prepare your data files (poses.txt and scans.txt)")
        print("2. Update the configuration file")
        print("3. Run: python main_loop_closure.py --config your_config.yaml")

    except Exception as e:
        print(f"Error during testing: {e}")
        import traceback

        traceback.print_exc()