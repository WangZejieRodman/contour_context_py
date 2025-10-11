#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Contour Context Chilean场景识别评估
基于main_loop_closure.py改编，用于Chilean数据集的跨时间段场景识别
评估任务：历史地图(100-100) vs 当前观测(190-190)
"""

import numpy as np
import os
import sys
import pickle
import time
from typing import List, Dict, Tuple, Optional
import logging
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import ListedColormap
import cv2

# 导入Contour Context相关模块
from contour_types import (
    ContourManagerConfig, ContourDBConfig, CandidateScoreEnsemble,
    ContourSimThresConfig, TreeBucketConfig
)
from contour_manager_区间分割_垂直结构复杂度_BCI邻域搜索层数_BCI分bin数 import ContourManager
from contour_database import ContourDB


class ChileanContourEvaluator:
    """Chilean数据集上的Contour Context场景识别评估器"""

    def __init__(self, dataset_folder: str,
                 database_file: str = 'chilean_NoRot_NoScale_5cm_evaluation_database_100_100.pickle',
                 query_file: str = 'chilean_NoRot_NoScale_5cm_evaluation_query_190_190.pickle',
                 log_file: str = 'contour_chilean_不相同时段_log.txt'):
        self.dataset_folder = dataset_folder
        self.database_file = database_file
        self.query_file = query_file

        # 设置随机种子以确保可重复的实验结果
        np.random.seed(42)

        # 设置日志记录
        logging.basicConfig(
            level=logging.INFO,
            format='%(message)s',
            handlers=[
                logging.FileHandler(log_file, mode='w'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)

        self.contour_stats = {
            'total_contours': [],
            'tiny_contour_ratios': [],
            'small_contour_ratios': [],
            'medium_small_contour_ratios': [],
            'medium_contour_ratios': [],
            'large_contour_ratios': [],
            'super_large_contour_ratios': [],
            'avg_eccentricities': [],
            'std_eccentricities': [],
            'significant_ecc_ratios': [],
            'significant_com_ratios': [],
            'avg_sizes': [],
            'std_sizes': [],
            'min_sizes': [],
            'max_sizes': [],
            'avg_eigenvalue_ratios': [],
            'avg_heights': []
        }

        self.key_stats = {
            'sparsity_ratios': [],
            'quality_scores': [],
            'ring_activations': []
        }

        self.bci_stats = {
            'avg_neighbors': [],
            'std_neighbors': [],
            'min_neighbors': [],
            'max_neighbors': [],
            'neighbor_dist_0': [],
            'neighbor_dist_1_3': [],
            'neighbor_dist_4_6': [],
            'neighbor_dist_7_10': [],
            'neighbor_dist_10_plus': [],
            'avg_distances': [],
            'std_distances': [],
            'min_distances': [],
            'max_distances': [],
            'angle_diversities': [],
            'angle_uniformities': [],
            'cross_layer_ratios': [],
            'activation_rates': [],
            'constellation_complexities': [],
            'connection_qualities': []
        }

        self.similarity_stats = {
            'total_searches': 0,
            'total_checks': 0,
            'check1_passed': 0,
            'check2_passed': 0,
            'check3_passed': 0
        }

        # 加载数据集
        self.database_sets = self.load_sets_dict(self.database_file)
        self.query_sets = self.load_sets_dict(self.query_file)

        # 创建配置
        self.cm_config = self.create_contour_manager_config()
        self.db_config = self.create_contour_db_config()
        self.thres_lb, self.thres_ub = self.create_similarity_thresholds()

        # 创建数据库
        self.contour_db = ContourDB(self.db_config)

        # 存储轮廓管理器用于重叠计算
        self.database_cms = []

        self.logger.info(f"初始化完成：{len(self.database_sets)}个数据库时间段，{len(self.query_sets)}个查询时间段")
        self.logger.info(f"评估任务：历史地图(100-100) vs 当前观测(190-190)")
        self.processed_cloud_count = 0

        # 添加可视化器
        self.visualizer = ContourVisualizer()

        # 添加统计收集器
        self.contour_stats = {
            'total_contours': [],
            'tiny_contour_ratios': [],
            'small_contour_ratios': [],
            'medium_small_contour_ratios': [],
            'medium_contour_ratios': [],
            'large_contour_ratios': [],
            'super_large_contour_ratios': [],
            'avg_eccentricities': [],
            'std_eccentricities': [],
            'significant_ecc_ratios': [],
            'significant_com_ratios': [],
            'avg_sizes': [],
            'std_sizes': [],
            'min_sizes': [],
            'max_sizes': [],
            'avg_eigenvalue_ratios': [],
            'avg_heights': []
        }

        self.key_stats = {
            'sparsity_ratios': [],
            'quality_scores': [],
            'ring_activations': []
        }

        self.bci_stats = {
            'avg_neighbors': [],
            'std_neighbors': [],
            'min_neighbors': [],
            'max_neighbors': [],
            'neighbor_dist_0': [],
            'neighbor_dist_1_3': [],
            'neighbor_dist_4_6': [],
            'neighbor_dist_7_10': [],
            'neighbor_dist_10_plus': [],
            'avg_distances': [],
            'std_distances': [],
            'min_distances': [],
            'max_distances': [],
            'angle_diversities': [],
            'angle_uniformities': [],
            'cross_layer_ratios': [],
            'activation_rates': [],
            'constellation_complexities': [],
            'connection_qualities': []
        }

        self.similarity_stats = {
            'total_searches': 0,
            'total_checks': 0,
            'check1_passed': 0,
            'check2_passed': 0,
            'check3_passed': 0
        }

    def create_contour_manager_config(self) -> ContourManagerConfig:
        """为Chilean数据集创建轮廓管理器配置"""
        config = ContourManagerConfig()

        # 根据Chilean地下矿井环境调整参数
        config.lv_grads = [0.0, 0.625, 1.25, 1.875, 2.5, 3.125, 3.75, 4.375, 5.0]  # 高度阈值

        # # 0.1m分辨率配置
        # config.reso_row = 0.1
        # config.reso_col = 0.1
        # config.n_row = 400
        # config.n_col = 400

        # 0.2m分辨率配置（基线）
        config.reso_row = 0.2
        config.reso_col = 0.2
        config.n_row = 200
        config.n_col = 200

        # # 0.4m分辨率配置（基线）
        # config.reso_row = 0.4
        # config.reso_col = 0.4
        # config.n_row = 100
        # config.n_col = 100

        # # 0.5m分辨率配置
        # config.reso_row = 0.5
        # config.reso_col = 0.5
        # config.n_row = 80
        # config.n_col = 80

        config.lidar_height = 0.0  # 地下环境激光雷达高度较低
        config.blind_sq = 0.0  # 减小盲区
        config.min_cont_key_cnt = 1  # 降低最小轮廓键 像素数量
        config.min_cont_cell_cnt = 1  # 降低最小轮廓 像素数量
        config.piv_firsts = 12  # 检索键生成数量，每层搜索前piv_firsts个轮廓生成检索键
        config.dist_firsts = 12  # BCI邻居搜索范围，每层搜索前dist_firsts个轮廓作为潜在邻居
        config.roi_radius = 15.0  # 增大感兴趣区域半径

        return config

    def create_contour_db_config(self) -> ContourDBConfig:
        """创建轮廓数据库配置"""
        config = ContourDBConfig()

        config.nnk = 30  # 降低KNN搜索数量以提高召回率
        config.max_fine_opt = 5  # 减少精细优化候选数
        config.q_levels = [0, 1, 2, 3, 4, 5, 6, 7]


        # 树桶配置
        tb_cfg = TreeBucketConfig()
        tb_cfg.max_elapse = 30.0
        tb_cfg.min_elapse = 10.0
        config.tb_cfg = tb_cfg

        # 轮廓相似性配置 - 放宽阈值以适应地下环境
        sim_cfg = ContourSimThresConfig()

        # 基线（当前）
        sim_cfg.ta_cell_cnt = 15.0  # 放宽绝对面积差异
        sim_cfg.tp_cell_cnt = 0.6  # 放宽相对面积差异
        sim_cfg.tp_eigval = 0.6  # 放宽特征值差异
        # 超宽松（实验组）
        # sim_cfg.ta_cell_cnt = 100.0  # 基线的6.7倍
        # sim_cfg.tp_cell_cnt = 0.9  # 基线的1.5倍
        # sim_cfg.tp_eigval = 0.9  # 基线的1.5倍
        # 超严格（实验组）
        # sim_cfg.ta_cell_cnt = 5.0  # 基线的1/3
        # sim_cfg.tp_cell_cnt = 0.3  # 基线的1/2
        # sim_cfg.tp_eigval = 0.3  # 基线的1/2

        sim_cfg.ta_h_bar = 1.2  # 放宽高度差异
        sim_cfg.ta_rcom = 1.5  # 放宽质心半径差异
        sim_cfg.tp_rcom = 0.7  # 放宽质心半径相对差异
        config.cont_sim_cfg = sim_cfg

        return config

    def create_similarity_thresholds(self) -> Tuple[CandidateScoreEnsemble, CandidateScoreEnsemble]:
        """创建检测阈值"""
        # 下界阈值（较宽松以适应时间跨度大的数据）
        thres_lb = CandidateScoreEnsemble()

        DISABLE_CHECK2 = False  # DISABLE_CHECK2 = True 表示禁用Check2（位重叠和角度重叠）星座检查

        if DISABLE_CHECK2:
            thres_lb.sim_constell.i_ovlp_sum = 0  # 不要求任何位重叠
            thres_lb.sim_constell.i_ovlp_max_one = 0
            thres_lb.sim_constell.i_in_ang_rng = 0  # 不要求角度一致
        else:
            # 基线配置
            thres_lb.sim_constell.i_ovlp_sum = 1
            thres_lb.sim_constell.i_ovlp_max_one = 1
            thres_lb.sim_constell.i_in_ang_rng = 1

        thres_lb.sim_pair.i_indiv_sim = 1
        thres_lb.sim_pair.i_orie_sim = 1  # 最低要求
        #########后处理三道门槛独立贡献
        # ========== 配置1：全禁用 ==========
        thres_lb.sim_post.correlation = 0.0001
        thres_lb.sim_post.area_perc = 0.0001
        thres_lb.sim_post.neg_est_dist = -100.0
        # # ========== 配置2：只面积 ==========
        # thres_lb.sim_post.correlation = 0.0001  # 禁用GMM
        # thres_lb.sim_post.area_perc = 0.01  # 启用面积 ✓
        # thres_lb.sim_post.neg_est_dist = -100.0  # 禁用距离
        # # ========== 配置3：只距离 ==========
        # thres_lb.sim_post.correlation = 0.0001  # 禁用GMM
        # thres_lb.sim_post.area_perc = 0.0001  # 禁用面积
        # thres_lb.sim_post.neg_est_dist = -10.0  # 启用距离 ✓
        # # ========== 配置4：只GMM ==========
        # thres_lb.sim_post.correlation = 0.2  # 启用GMM ✓
        # thres_lb.sim_post.area_perc = 0.0001  # 禁用面积
        # thres_lb.sim_post.neg_est_dist = -100.0  # 禁用距离
        # # ========== 配置5：面积+距离 ==========
        # thres_lb.sim_post.correlation = 0.0001  # 禁用GMM
        # thres_lb.sim_post.area_perc = 0.01  # 启用面积 ✓
        # thres_lb.sim_post.neg_est_dist = -10.0  # 启用距离 ✓
        # # ========== 配置6：面积+GMM ==========
        # thres_lb.sim_post.correlation = 0.2  # 启用GMM ✓
        # thres_lb.sim_post.area_perc = 0.01  # 启用面积 ✓
        # thres_lb.sim_post.neg_est_dist = -100.0  # 禁用距离
        # # ========== 配置7：距离+GMM ==========
        # thres_lb.sim_post.correlation = 0.2  # 启用GMM ✓
        # thres_lb.sim_post.area_perc = 0.0001  # 禁用面积
        # thres_lb.sim_post.neg_est_dist = -10.0  # 启用距离 ✓
        # # ========== 配置8：全启用 ==========
        # thres_lb.sim_post.correlation = 0.2  # 启用GMM ✓
        # thres_lb.sim_post.area_perc = 0.01  # 启用面积 ✓
        # thres_lb.sim_post.neg_est_dist = -10.0  # 启用距离 ✓

        # 上界阈值
        thres_ub = CandidateScoreEnsemble()
        thres_ub.sim_constell.i_ovlp_sum = 10
        thres_ub.sim_constell.i_ovlp_max_one = 10
        thres_ub.sim_constell.i_in_ang_rng = 10
        thres_ub.sim_pair.i_indiv_sim = 10
        thres_ub.sim_pair.i_orie_sim = 10
        thres_ub.sim_post.correlation = 0.9
        thres_ub.sim_post.area_perc = 0.3
        thres_ub.sim_post.neg_est_dist = -0.1

        return thres_lb, thres_ub

    def load_sets_dict(self, filename: str) -> List[Dict]:
        """加载数据集字典"""
        try:
            with open(filename, 'rb') as handle:
                sets = pickle.load(handle)
                self.logger.info(f"加载 {filename}: {len(sets)} 个时间段")
                return sets
        except Exception as e:
            self.logger.error(f"加载 {filename} 失败: {e}")
            return []

    def load_chilean_pointcloud(self, filename: str) -> np.ndarray:
        """加载Chilean点云文件"""
        try:
            full_path = os.path.join(self.dataset_folder, filename)

            # 读取二进制数据
            pc = np.fromfile(full_path, dtype=np.float64)

            # 检查数据长度是否是3的倍数
            if len(pc) % 3 != 0:
                self.logger.warning(f"Chilean点云数据长度不是3的倍数: {len(pc)}")
                return np.array([])

            # reshape为 [N, 3] 格式
            num_points = len(pc) // 3
            pc = pc.reshape(num_points, 3)

            return pc

        except Exception as e:
            self.logger.error(f"加载Chilean点云 {filename} 失败: {e}")
            return np.array([])

    def apply_random_rotation(self, pointcloud: np.ndarray) -> np.ndarray:
        """对点云应用随机旋转（绕x、y、z轴）"""
        if len(pointcloud) == 0:
            return pointcloud

        # 生成随机旋转角度（弧度）
        angle_x = np.random.uniform(-np.pi, np.pi)
        angle_y = np.random.uniform(-np.pi, np.pi)
        angle_z = np.random.uniform(-np.pi, np.pi)

        # 创建旋转矩阵
        R_x = np.array([
            [1, 0, 0],
            [0, np.cos(angle_x), -np.sin(angle_x)],
            [0, np.sin(angle_x), np.cos(angle_x)]
        ])

        R_y = np.array([
            [np.cos(angle_y), 0, np.sin(angle_y)],
            [0, 1, 0],
            [-np.sin(angle_y), 0, np.cos(angle_y)]
        ])

        R_z = np.array([
            [np.cos(angle_z), -np.sin(angle_z), 0],
            [np.sin(angle_z), np.cos(angle_z), 0],
            [0, 0, 1]
        ])

        # 组合旋转矩阵
        R = R_x @ R_y @ R_z

        # 应用旋转到点云的xyz坐标
        rotated_pointcloud = (R @ pointcloud.T).T

        return rotated_pointcloud

    def save_pointcloud_to_txt(self, pointcloud: np.ndarray, filename: str):
        """将点云保存为txt文件"""
        try:
            save_dir = "不旋转点云&随机旋转点云"
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)

            filepath = os.path.join(save_dir, filename)
            np.savetxt(filepath, pointcloud, fmt='%.6f', delimiter=' ',
                       header='x y z', comments='')

            self.logger.info(f"点云已保存到: {filepath}")
            self.logger.info(f"点云信息: {len(pointcloud)} 个点")

        except Exception as e:
            self.logger.error(f"保存点云失败: {e}")

    def build_database(self, set_idx: int) -> bool:
        """为指定时间段构建数据库"""
        session_id = 100 + set_idx
        self.logger.info(f"构建数据库时间段 {set_idx} (session {session_id})...")

        database_set = self.database_sets[set_idx]
        processed_count = 0
        total_count = len(database_set)

        for key in sorted(database_set.keys()):
            item = database_set[key]
            filename = item['query']

            # 加载点云
            pointcloud = self.load_chilean_pointcloud(filename)
            if len(pointcloud) == 0:
                continue

            # 如果是第10个点云，保存原始和旋转后的点云
            if self.processed_cloud_count == 9:
                self.save_pointcloud_to_txt(pointcloud, "tenth_pointcloud_original.txt")
                self.logger.info(f"已保存第10个原始点云")

            # 应用随机旋转
            #pointcloud = self.apply_random_rotation(pointcloud)

            if self.processed_cloud_count == 9:
                self.save_pointcloud_to_txt(pointcloud, "tenth_pointcloud_rotated.txt")
                self.logger.info(f"已保存第10个旋转后点云")

            self.processed_cloud_count += 1

            # 创建轮廓管理器
            start_time = time.time()
            cm = ContourManager(self.cm_config, key)

            # 生成字符串ID
            str_id = f"db_session_{session_id}_frame_{key}"

            # 处理点云
            cm.make_bev(pointcloud, str_id)
            cm.make_contours_recursive()

            try:
                contour_stats = cm._output_detailed_contour_statistics()
                key_stats = cm._output_retrieval_key_statistics()
                bci_stats = cm._output_bci_statistics()

                # 收集轮廓统计
                self.contour_stats['total_contours'].append(contour_stats['total_contours'])
                self.contour_stats['tiny_contour_ratios'].append(contour_stats['tiny_contour_ratio'])
                self.contour_stats['small_contour_ratios'].append(contour_stats['small_contour_ratio'])
                self.contour_stats['medium_small_contour_ratios'].append(contour_stats['medium_small_contour_ratio'])
                self.contour_stats['medium_contour_ratios'].append(contour_stats['medium_contour_ratio'])
                self.contour_stats['large_contour_ratios'].append(contour_stats['large_contour_ratio'])
                self.contour_stats['super_large_contour_ratios'].append(contour_stats['super_large_contour_ratio'])
                self.contour_stats['avg_eccentricities'].append(contour_stats['avg_eccentricity'])
                self.contour_stats['std_eccentricities'].append(contour_stats['std_eccentricity'])
                self.contour_stats['significant_ecc_ratios'].append(contour_stats['significant_ecc_ratio'])
                self.contour_stats['significant_com_ratios'].append(contour_stats['significant_com_ratio'])
                self.contour_stats['avg_sizes'].append(contour_stats['avg_size'])
                self.contour_stats['std_sizes'].append(contour_stats['std_size'])
                self.contour_stats['min_sizes'].append(contour_stats['min_size'])
                self.contour_stats['max_sizes'].append(contour_stats['max_size'])
                self.contour_stats['avg_eigenvalue_ratios'].append(contour_stats['avg_eigenvalue_ratio'])
                self.contour_stats['avg_heights'].append(contour_stats['avg_height'])

                # 收集特征质量统计
                self.key_stats['sparsity_ratios'].append(key_stats['sparsity_ratio'])
                self.key_stats['quality_scores'].append(key_stats['quality_score'])
                self.key_stats['ring_activations'].append(key_stats['ring_activation'])

                # 收集BCI统计
                self.bci_stats['avg_neighbors'].append(bci_stats['avg_neighbors'])
                self.bci_stats['std_neighbors'].append(bci_stats['std_neighbors'])
                self.bci_stats['min_neighbors'].append(bci_stats['min_neighbors'])
                self.bci_stats['max_neighbors'].append(bci_stats['max_neighbors'])
                self.bci_stats['neighbor_dist_0'].append(bci_stats['neighbor_dist_0'])
                self.bci_stats['neighbor_dist_1_3'].append(bci_stats['neighbor_dist_1_3'])
                self.bci_stats['neighbor_dist_4_6'].append(bci_stats['neighbor_dist_4_6'])
                self.bci_stats['neighbor_dist_7_10'].append(bci_stats['neighbor_dist_7_10'])
                self.bci_stats['neighbor_dist_10_plus'].append(bci_stats['neighbor_dist_10_plus'])
                self.bci_stats['avg_distances'].append(bci_stats['avg_distance'])
                self.bci_stats['std_distances'].append(bci_stats['std_distance'])
                self.bci_stats['min_distances'].append(bci_stats['min_distance'])
                self.bci_stats['max_distances'].append(bci_stats['max_distance'])
                self.bci_stats['angle_diversities'].append(bci_stats['angle_diversity'])
                self.bci_stats['angle_uniformities'].append(bci_stats['angle_uniformity'])
                self.bci_stats['cross_layer_ratios'].append(bci_stats['cross_layer_ratio'])
                self.bci_stats['activation_rates'].append(bci_stats['activation_rate'])
                self.bci_stats['constellation_complexities'].append(bci_stats['constellation_complexity'])
                self.bci_stats['connection_qualities'].append(bci_stats['connection_quality'])

            except Exception as e:
                self.logger.warning(f"统计数据收集失败: {e}")


            desc_time = (time.time() - start_time) * 1000

            # 记录处理信息
            total_contours = sum(len(cm.get_lev_contours(i))
                                 for i in range(len(self.cm_config.lv_grads)))

            self.logger.info(f"[DB {set_idx}] Frame {key}: 总轮廓数: {total_contours}, "
                             f"处理时间: {desc_time:.1f}ms")

            # 添加到数据库
            start_time = time.time()
            self.contour_db.add_scan(cm, processed_count)  # 使用processed_count作为时间戳
            update_time = (time.time() - start_time) * 1000

            # 存储轮廓管理器
            self.database_cms.append(cm)

            processed_count += 1

            if processed_count % 50 == 0:
                self.logger.info(f"  已处理: {processed_count}/{total_count}")

        self.logger.info(f"数据库时间段 {set_idx} (session {session_id}) 构建完成: {processed_count}/{total_count}")

        # 强制将所有缓冲区数据移动到KD树
        if processed_count > 0:
            self.logger.info("开始强制平衡，将缓冲区数据移动到KD树...")

            # 使用一个很大的时间戳强制触发数据迁移
            final_timestamp = 999999.0

            # 打印平衡前的状态
            self.logger.info("平衡前的状态：")
            for ll in range(len(self.contour_db.layer_db)):
                layer_db = self.contour_db.layer_db[ll]
                tree_size = sum(bucket.get_tree_size() for bucket in layer_db.buckets)
                buffer_size = sum(len(bucket.buffer) for bucket in layer_db.buckets)
                self.logger.info(f"  LayerDB {ll}: 树大小={tree_size}, 缓冲区大小={buffer_size}")

            # 多次调用平衡操作确保所有数据都被处理
            self.logger.info("执行自动平衡操作...")
            for balance_round in range(10):
                for bucket_pair in range(5):  # 0到4，对应bucket pairs (0,1), (1,2), (2,3), (3,4), (4,5)
                    self.contour_db.push_and_balance(bucket_pair, final_timestamp)

            # 检查自动平衡后的状态
            self.logger.info("自动平衡后的状态：")
            any_data_in_tree = False
            for ll in range(len(self.contour_db.layer_db)):
                layer_db = self.contour_db.layer_db[ll]
                tree_size = sum(bucket.get_tree_size() for bucket in layer_db.buckets)
                buffer_size = sum(len(bucket.buffer) for bucket in layer_db.buckets)
                if tree_size > 0:
                    any_data_in_tree = True
                self.logger.info(f"  LayerDB {ll}: 树大小={tree_size}, 缓冲区大小={buffer_size}")

            # 如果自动平衡失败，手动清空缓冲区到树中
            if not any_data_in_tree:
                self.logger.info("自动平衡失败，开始手动清空缓冲区...")

                for ll in range(len(self.contour_db.layer_db)):
                    layer_db = self.contour_db.layer_db[ll]
                    self.logger.info(f"处理LayerDB {ll}...")

                    for bucket_idx, bucket in enumerate(layer_db.buckets):
                        if len(bucket.buffer) > 0:
                            self.logger.info(f"  手动清空Bucket {bucket_idx}: {len(bucket.buffer)} 个项目")

                            # 将缓冲区数据强制移动到树
                            for item in bucket.buffer:
                                tree_key, ts, iok = item
                                bucket.data_tree.append(tree_key.copy())
                                bucket.gkidx_tree.append(iok)

                            # 清空缓冲区
                            bucket.buffer.clear()

                            # 重建树
                            try:
                                bucket.rebuild_tree()
                                self.logger.info(f"    Bucket {bucket_idx} 树重建成功，新大小: {bucket.get_tree_size()}")
                            except Exception as e:
                                self.logger.error(f"    Bucket {bucket_idx} 树重建失败: {e}")

            # 最终检查和验证
            self.logger.info("最终状态检查：")
            total_tree_size = 0
            total_buffer_size = 0

            for ll in range(len(self.contour_db.layer_db)):
                layer_db = self.contour_db.layer_db[ll]
                tree_size = sum(bucket.get_tree_size() for bucket in layer_db.buckets)
                buffer_size = sum(len(bucket.buffer) for bucket in layer_db.buckets)
                total_tree_size += tree_size
                total_buffer_size += buffer_size

                self.logger.info(f"  LayerDB {ll} (q_level={self.contour_db.cfg.q_levels[ll]}): "
                                 f"树大小={tree_size}, 缓冲区大小={buffer_size}")

                # 打印每个非空桶的详细信息
                for bucket_idx, bucket in enumerate(layer_db.buckets):
                    bucket_tree_size = bucket.get_tree_size()
                    bucket_buffer_size = len(bucket.buffer)
                    if bucket_tree_size > 0 or bucket_buffer_size > 0:
                        self.logger.info(f"    Bucket {bucket_idx}: 树={bucket_tree_size}, 缓冲区={bucket_buffer_size}")

            self.logger.info(f"总计: 树大小={total_tree_size}, 缓冲区大小={total_buffer_size}")

            # 验证结果
            if total_tree_size == 0:
                self.logger.error("严重错误：所有数据仍在缓冲区中，KD树为空！")
                self.logger.error("这将导致查询时无法找到任何匹配。")

                # 尝试最后的手段：直接操作内部数据结构
                self.logger.info("尝试最后的修复手段...")
                try:
                    for ll in range(len(self.contour_db.layer_db)):
                        layer_db = self.contour_db.layer_db[ll]
                        for bucket in layer_db.buckets:
                            if len(bucket.buffer) > 0:
                                # 强制调用pop_buffer_max
                                bucket.pop_buffer_max(final_timestamp)
                                self.logger.info(f"强制调用pop_buffer_max后，树大小: {bucket.get_tree_size()}")

                    # 再次检查
                    final_tree_size = sum(
                        sum(bucket.get_tree_size() for bucket in layer_db.buckets)
                        for layer_db in self.contour_db.layer_db
                    )
                    self.logger.info(f"最终修复后树大小: {final_tree_size}")

                    if final_tree_size == 0:
                        return False

                except Exception as e:
                    self.logger.error(f"最后修复尝试失败: {e}")
                    return False
            else:
                self.logger.info(f"✅ 成功：数据已移动到KD树中")


        return processed_count > 0

    def query_in_database(self, query_set_idx: int, database_set_idx: int,
                          k: int = 25) -> Tuple[List, List, float]:
        """在指定数据库时间段中查询"""
        query_set = self.query_sets[query_set_idx]

        recall = [0] * k
        top1_similarity_score = []
        one_percent_retrieved = 0
        threshold = max(int(round(len(self.database_sets[database_set_idx]) / 100.0)), 1)
        num_evaluated = 0

        # 失败统计
        failure_statistics = {
            'no_ground_truth': 0,
            'pointcloud_load_failed': 0,
            'no_retrieval_results': 0,
            'no_valid_matches': 0,
            'total_queries': 0
        }

        # ✅ 创建失败日志文件
        failure_log_path = f"query_failure_analysis_session_{190 + query_set_idx}_to_{100 + database_set_idx}.txt"
        failure_log = open(failure_log_path, 'w', encoding='utf-8')
        failure_log.write("=== 查询失败详细分析报告 ===\n")
        failure_log.write(f"查询时间段: {190 + query_set_idx}\n")
        failure_log.write(f"数据库时间段: {100 + database_set_idx}\n")
        failure_log.write(f"生成时间: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        failure_log.write("=" * 50 + "\n\n")

        try:
            for query_key in sorted(query_set.keys()):
                query_item = query_set[query_key]
                failure_statistics['total_queries'] += 1

                # 检查ground truth
                if database_set_idx not in query_item:
                    failure_statistics['no_ground_truth'] += 1
                    self._log_failure(failure_log, query_key, "NO_GROUND_TRUTH",
                                      "缺少真值标注", query_item)
                    continue

                true_neighbors = query_item[database_set_idx]
                if len(true_neighbors) == 0:
                    failure_statistics['no_ground_truth'] += 1
                    self._log_failure(failure_log, query_key, "EMPTY_GROUND_TRUTH",
                                      "真值标注为空", query_item)
                    continue

                num_evaluated += 1

                # 加载查询点云
                filename = query_item['query']
                pointcloud = self.load_chilean_pointcloud(filename)
                if len(pointcloud) == 0:
                    failure_statistics['pointcloud_load_failed'] += 1
                    self._log_failure(failure_log, query_key, "POINTCLOUD_LOAD_FAILED",
                                      f"点云加载失败: {filename}", query_item)
                    continue

                # 创建查询轮廓管理器
                query_cm = ContourManager(self.cm_config, query_key + 10000)  # 避免ID冲突
                str_id = f"query_session_{190 + query_set_idx}_frame_{query_key}"
                query_cm.make_bev(pointcloud, str_id)
                query_cm.make_contours_recursive()

                # ✅ 预检查：收集轮廓和特征键信息（在失败时使用）
                contour_info = self._collect_contour_info(query_cm)
                feature_key_info = self._collect_feature_key_info(query_cm)

                # 在数据库中检索
                try:
                    # ✅ 修改这里：接收相似性统计数据
                    candidate_cms, correlations, transforms, query_similarity_stats = self.contour_db.query_ranged_knn(
                        query_cm, self.thres_lb, self.thres_ub)

                    # ✅ 累加相似性统计
                    self.similarity_stats['total_searches'] += query_similarity_stats['total_searches']
                    self.similarity_stats['total_checks'] += query_similarity_stats['total_checks']
                    self.similarity_stats['check1_passed'] += query_similarity_stats['check1_passed']
                    self.similarity_stats['check2_passed'] += query_similarity_stats['check2_passed']
                    self.similarity_stats['check3_passed'] += query_similarity_stats['check3_passed']

                    if len(candidate_cms) == 0:
                        failure_statistics['no_retrieval_results'] += 1
                        self._log_retrieval_failure(failure_log, query_key, query_item,
                                                    contour_info, feature_key_info,
                                                    pointcloud.shape, "KNN搜索返回空结果")
                        continue

                    # 将候选CM ID映射回数据库索引
                    results = []
                    for i, (candidate_cm, correlation) in enumerate(zip(candidate_cms, correlations)):
                        # 假设数据库CM的ID就是其在database_cms中的索引
                        db_idx = candidate_cm.get_int_id()
                        if db_idx < len(self.database_cms):
                            distance = 1.0 - correlation  # 将相关性转换为距离
                            results.append((db_idx, distance))

                    if len(results) == 0:
                        failure_statistics['no_retrieval_results'] += 1
                        self._log_retrieval_failure(failure_log, query_key, query_item,
                                                    contour_info, feature_key_info,
                                                    pointcloud.shape,
                                                    f"候选映射失败: {len(candidate_cms)} 个候选无法映射")
                        continue

                    # 按距离排序，返回前k个
                    results.sort(key=lambda x: x[1])
                    results = results[:k]

                    indices = [result[0] for result in results]

                except Exception as e:
                    self.logger.error(f"查询失败: {e}")
                    failure_statistics['no_retrieval_results'] += 1
                    self._log_retrieval_failure(failure_log, query_key, query_item,
                                                contour_info, feature_key_info,
                                                pointcloud.shape,
                                                f"查询异常: {str(e)}")
                    continue

                # 检查是否有有效匹配
                has_valid_match = False
                for j, idx in enumerate(indices):
                    if idx in true_neighbors:
                        has_valid_match = True
                        if j == 0:
                            similarity = 1.0 - results[0][1]
                            top1_similarity_score.append(similarity)
                        for k_idx in range(j, len(recall)):
                            recall[k_idx] += 1
                        break

                if not has_valid_match:
                    failure_statistics['no_valid_matches'] += 1
                    self._log_matching_failure(failure_log, query_key, query_item,
                                               contour_info, feature_key_info,
                                               pointcloud.shape, indices, true_neighbors)

                # 计算top 1% recall
                top_percent_indices = indices[:threshold]
                if len(set(top_percent_indices).intersection(set(true_neighbors))) > 0:
                    one_percent_retrieved += 1

        finally:
            # ✅ 关闭失败日志文件
            failure_log.write(f"\n=== 失败统计汇总 ===\n")
            failure_log.write(f"总查询数: {failure_statistics['total_queries']}\n")
            failure_log.write(f"成功评估数: {num_evaluated}\n")
            for reason, count in failure_statistics.items():
                if reason != 'total_queries' and count > 0:
                    failure_log.write(f"{reason}: {count}\n")
            failure_log.close()

            self.logger.info(f"失败分析报告已保存到: {failure_log_path}")

        # 记录失败统计到主日志
        if failure_statistics['total_queries'] > 0:
            self.logger.info(f"\n=== 失败查询分析 (DB{database_set_idx} <- Query{query_set_idx}) ===")
            self.logger.info(f"总查询数: {failure_statistics['total_queries']}")
            self.logger.info(f"成功评估数: {num_evaluated}")
            for reason, count in failure_statistics.items():
                if reason != 'total_queries' and count > 0:
                    self.logger.info(f"  {reason}: {count}")

        if num_evaluated > 0:
            recall = [(r / num_evaluated) * 100 for r in recall]
            one_percent_recall = (one_percent_retrieved / num_evaluated) * 100
        else:
            recall = [0] * k
            one_percent_recall = 0

        return recall, top1_similarity_score, one_percent_recall

    def _collect_contour_info(self, query_cm: ContourManager) -> dict:
        """收集轮廓信息用于失败分析"""
        contour_info = {
            'total_contours': 0,
            'contours_per_level': [],
            'bev_stats': {}
        }

        for ll in range(len(self.cm_config.lv_grads)):
            contours = query_cm.get_lev_contours(ll)
            level_info = {
                'level': ll,
                'height_threshold': self.cm_config.lv_grads[ll],
                'contour_count': len(contours),
                'contour_details': []
            }

            for seq, contour in enumerate(contours[:5]):  # 只记录前5个
                level_info['contour_details'].append({
                    'seq': seq,
                    'cell_cnt': contour.cell_cnt,
                    'eig_vals': contour.eig_vals.tolist() if hasattr(contour, 'eig_vals') else [0, 0],
                    'pos_mean': contour.pos_mean.tolist() if hasattr(contour, 'pos_mean') else [0, 0]
                })

            contour_info['contours_per_level'].append(level_info)
            contour_info['total_contours'] += len(contours)

        # BEV统计
        if hasattr(query_cm, 'bev') and query_cm.bev is not None:
            for ll in range(len(self.cm_config.lv_grads)):
                non_empty = np.sum(query_cm.bev > self.cm_config.lv_grads[ll])
                contour_info['bev_stats'][f'level_{ll}_pixels'] = int(non_empty)

        return contour_info

    def _collect_feature_key_info(self, query_cm: ContourManager) -> dict:
        """收集特征键信息用于失败分析"""
        feature_info = {
            'total_keys': 0,
            'zero_keys': 0,
            'valid_keys': 0,
            'key_details': []
        }

        for ll in range(len(self.cm_config.lv_grads)):
            keys = query_cm.get_lev_retrieval_key(ll)
            for seq, key in enumerate(keys):
                key_sum = np.sum(key)
                key_detail = {
                    'level': ll,
                    'seq': seq,
                    'key_sum': float(key_sum),
                    'key_first_3': key[:3].tolist() if len(key) >= 3 else key.tolist(),
                    'is_zero': key_sum == 0
                }

                feature_info['key_details'].append(key_detail)
                feature_info['total_keys'] += 1

                if key_sum == 0:
                    feature_info['zero_keys'] += 1
                else:
                    feature_info['valid_keys'] += 1

        return feature_info

    def _log_failure(self, failure_log, query_key: int, failure_type: str,
                     reason: str, query_item: dict):
        """记录简单失败情况"""
        failure_log.write(f"【失败案例 {failure_type}】\n")
        failure_log.write(f"查询ID: {query_key}\n")
        failure_log.write(f"失败原因: {reason}\n")
        failure_log.write(f"查询文件: {query_item.get('query', 'unknown')}\n")
        if 'northing' in query_item:
            failure_log.write(f"查询坐标: ({query_item['northing']:.2f}, {query_item['easting']:.2f})\n")
        failure_log.write("-" * 40 + "\n\n")

    def _log_retrieval_failure(self, failure_log, query_key: int, query_item: dict,
                               contour_info: dict, feature_key_info: dict,
                               pointcloud_shape: tuple, reason: str):
        """记录检索失败的详细信息"""
        failure_log.write(f"【检索失败详细分析】\n")
        failure_log.write(f"查询ID: {query_key}\n")
        failure_log.write(f"失败原因: {reason}\n")
        failure_log.write(f"查询文件: {query_item.get('query', 'unknown')}\n")
        if 'northing' in query_item:
            failure_log.write(f"查询坐标: ({query_item['northing']:.2f}, {query_item['easting']:.2f})\n")
        failure_log.write(f"点云大小: {pointcloud_shape}\n")

        # 轮廓信息
        failure_log.write(f"\n轮廓提取情况:\n")
        failure_log.write(f"  总轮廓数: {contour_info['total_contours']}\n")
        for level_info in contour_info['contours_per_level']:
            failure_log.write(f"  L{level_info['level']} (h>{level_info['height_threshold']}): "
                              f"{level_info['contour_count']} 个轮廓\n")
            for detail in level_info['contour_details']:
                failure_log.write(f"    S{detail['seq']}: cells={detail['cell_cnt']}, "
                                  f"eig_vals={detail['eig_vals']}\n")

        # BEV像素统计
        if contour_info['bev_stats']:
            failure_log.write(f"\nBEV像素统计:\n")
            for level_key, pixel_count in contour_info['bev_stats'].items():
                failure_log.write(f"  {level_key}: {pixel_count} 像素\n")

        # 特征键信息
        failure_log.write(f"\n特征键生成情况:\n")
        failure_log.write(f"  总特征键数: {feature_key_info['total_keys']}\n")
        failure_log.write(f"  全0键数量: {feature_key_info['zero_keys']}\n")
        failure_log.write(f"  有效键数量: {feature_key_info['valid_keys']}\n")

        # 详细键信息
        failure_log.write(f"  键详情:\n")
        for key_detail in feature_key_info['key_details']:
            status = "❌全0" if key_detail['is_zero'] else "✅有效"
            failure_log.write(f"    L{key_detail['level']}S{key_detail['seq']}: "
                              f"{status}, sum={key_detail['key_sum']:.6f}, "
                              f"first3={key_detail['key_first_3']}\n")

        failure_log.write("=" * 50 + "\n\n")

    def _log_matching_failure(self, failure_log, query_key: int, query_item: dict,
                              contour_info: dict, feature_key_info: dict,
                              pointcloud_shape: tuple, retrieved_indices: list,
                              true_neighbors: list):
        """记录匹配失败的详细信息"""
        failure_log.write(f"【匹配失败详细分析】\n")
        failure_log.write(f"查询ID: {query_key}\n")
        failure_log.write(f"失败原因: 检索成功但无有效匹配\n")
        failure_log.write(f"查询文件: {query_item.get('query', 'unknown')}\n")
        if 'northing' in query_item:
            failure_log.write(f"查询坐标: ({query_item['northing']:.2f}, {query_item['easting']:.2f})\n")
        failure_log.write(f"点云大小: {pointcloud_shape}\n")

        failure_log.write(f"\n检索结果分析:\n")
        failure_log.write(f"  检索到的索引: {retrieved_indices[:10]}{'...' if len(retrieved_indices) > 10 else ''}\n")
        failure_log.write(f"  真值索引: {true_neighbors[:10]}{'...' if len(true_neighbors) > 10 else ''}\n")
        failure_log.write(f"  交集: {list(set(retrieved_indices) & set(true_neighbors))}\n")

        # 简化的轮廓和特征信息
        failure_log.write(f"\n简要特征信息:\n")
        failure_log.write(f"  总轮廓数: {contour_info['total_contours']}\n")
        failure_log.write(f"  有效特征键: {feature_key_info['valid_keys']}/{feature_key_info['total_keys']}\n")

        failure_log.write("=" * 50 + "\n\n")

    def evaluate(self) -> float:
        """执行完整评估"""
        self.logger.info("开始Chilean数据集Contour Context评估...")

        # 构建所有数据库
        for i in range(len(self.database_sets)):
            if not self.build_database(i):
                self.logger.error(f"数据库时间段 {i} 构建失败")
                return 0.0

        self.logger.info("\n开始跨时间段评估...")

        recall = np.zeros(25)
        count = 0
        similarity = []
        one_percent_recall = []

        # 跨时间段评估
        for m in range(len(self.database_sets)):  # 数据库时间段 (100-100)
            for n in range(len(self.query_sets)):  # 查询时间段 (190-190)
                db_session_id = 100 + m
                query_session_id = 190 + n
                self.logger.info(f"评估：查询时间段{query_session_id} -> 数据库时间段{db_session_id}")

                pair_recall, pair_similarity, pair_opr = self.query_in_database(n, m)
                recall += np.array(pair_recall)
                count += 1
                one_percent_recall.append(pair_opr)

                for x in pair_similarity:
                    similarity.append(x)

                self.logger.info(f"  Recall@1: {pair_recall[0]:.2f}%, Top1%: {pair_opr:.2f}%")

        # 计算平均结果
        if count > 0:
            ave_recall = recall / count
            average_similarity = np.mean(similarity) if similarity else 0
            ave_one_percent_recall = np.mean(one_percent_recall)
        else:
            ave_recall = np.zeros(25)
            average_similarity = 0
            ave_one_percent_recall = 0

        # 输出结果
        self.logger.info(f"\n=== Chilean数据集 Contour Context 评估结果 ===")
        self.logger.info(f"数据库时间段: 100-100 (历史地图)")
        self.logger.info(f"查询时间段: 190-190 (当前观测)")
        self.logger.info(f"Average Recall @1: {ave_recall[0]:.2f}%")
        self.logger.info(f"Average Recall @5: {ave_recall[4]:.2f}%")
        self.logger.info(f"Average Recall @10: {ave_recall[9]:.2f}%")
        self.logger.info(f"Average Recall @25: {ave_recall[24]:.2f}%")
        self.logger.info(f"Average Similarity: {average_similarity:.4f}")
        self.logger.info(f"Average Top 1% Recall: {ave_one_percent_recall:.2f}%")

        # 保存详细结果
        self.save_results(ave_recall, average_similarity, ave_one_percent_recall)

        return ave_recall[0]

    def save_results(self, ave_recall: np.ndarray, average_similarity: float,
                     ave_one_percent_recall: float):
        """保存评估结果"""
        results_dir = "results"
        if not os.path.exists(results_dir):
            os.makedirs(results_dir)

        output_file = os.path.join(results_dir, "contour_results_chilean_不相同时段_session.txt")

        with open(output_file, "w") as f:
            f.write("Chilean Dataset Contour Context Evaluation Results\n")
            f.write("=" * 50 + "\n\n")
            f.write("Dataset Info:\n")
            f.write("Database: Sessions 100-100 (Historical Map)\n")
            f.write("Query: Sessions 190-190 (Current Observations)\n\n")
            f.write("Average Recall @N:\n")
            f.write(str(ave_recall) + "\n\n")
            f.write("Average Similarity:\n")
            f.write(str(average_similarity) + "\n\n")
            f.write("Average Top 1% Recall:\n")
            f.write(str(ave_one_percent_recall) + "\n")

        self.logger.info(f"结果已保存到: {output_file}")

    def visualize_sample_cases(self):
        """可视化示例案例"""

        print(f"\n开始可视化指定的点云案例...")

        database_set = self.database_sets[0]

        # 指定要可视化的索引（第13个点云对应索引12）
        target_indices = [12]  # 可以添加更多索引，如 [12, 15, 20]
        sample_keys = [list(database_set.keys())[i] for i in target_indices if i < len(database_set)]

        for i, key in enumerate(sample_keys):
            item = database_set[key]
            filename = item['query']

            print(f"正在可视化指定案例 {i + 1}: 索引{key}, 文件{filename}")

            # 加载点云
            pointcloud = self.load_chilean_pointcloud(filename)
            if len(pointcloud) == 0:
                print(f"跳过: 无法加载点云 {filename}")
                continue

            # 创建轮廓管理器
            cm = ContourManager(self.cm_config, key)
            str_id = f"specified_sample_{key}"
            cm.make_bev(pointcloud, str_id)
            cm.make_contours_recursive()

            # 可视化
            prefix = f"specified_sample_index_{key}"
            self.visualizer.visualize_pointcloud_pipeline(cm, pointcloud, prefix)

    def output_ablation_summary(self, experiment_name: str, param_value, top1_recall):
        """输出消融实验汇总"""

        # 创建专门的消融实验报告文件
        report_file = f"ablation_report_{experiment_name}_param_{param_value}.txt"

        # 准备报告内容
        report_lines = []

        # report_lines.append(f"\n实验{experiment_name}: min_cont_cell_cnt消融实验")  # B1
        # report_lines.append(f"\n实验{experiment_name}: min_cont_key_cnt消融实验")   # B2
        # report_lines.append(f"\n实验{experiment_name}: piv_firsts消融实验")        # D2
        # report_lines.append(f"\n实验{experiment_name}: dist_firsts消融实验")       # D3
        # report_lines.append(f"\n实验{experiment_name}: nnk消融实验")  # G1
        # report_lines.append(f"\n实验{experiment_name}: 精细优化候选数量max_fine_opt消融实验")  # H2
        # report_lines.append(f"\n实验{experiment_name}: 空间分辨率消融实验")  # I1
        # report_lines.append(f"\n实验{experiment_name}: ROI半径roi_radius消融实验")  # C3
        # report_lines.append(f"\n实验{experiment_name}: 检索键维度影响（3维 vs 5维 vs 7维 vs 10维 ）消融实验")  # C1
        # report_lines.append(f"\n实验{experiment_name}: 垂直结构复杂度消融实验")  # C4
        # report_lines.append(f"\n实验{experiment_name}: BCI邻域搜索层级范围消融实验")  # D1
        # report_lines.append(f"\n实验{experiment_name}: BCI分bin数量消融实验")  # D4
        # report_lines.append(f"\n实验{experiment_name}: BCI角度一致性检查的角度阈值消融实验")  # D5
        # report_lines.append(f"\n实验{experiment_name}: GMM相关性阈值消融实验")  # F1
        # report_lines.append(f"\n实验{experiment_name}: 轮廓相似性（Check1）消融实验") #E0
        # report_lines.append(f"\n实验{experiment_name}: Check2星座检查消融实验") #E4
        # report_lines.append(f"\n实验{experiment_name}: Check3成对相似性检查消融实验")  # E2
        report_lines.append(f"\n实验{experiment_name}: 后处理三道门槛（面积、距离、GMM）独立贡献消融实验")  # H0


        report_lines.append("=" * 40)
        report_lines.append(f"参数设置: {param_value}")

        # ✅ 新增：D5实验显示角度信息
        if experiment_name == "D5":
            angle_deg = np.degrees(param_value)
            report_lines.append(f"角度阈值: {param_value:.4f} rad ({angle_deg:.2f}°)")

        report_lines.append("基线配置: 8层, 高度0-5m, 层间距0.625m")

        # ✅ 新增：星座相似性检查详细统计（在轮廓统计之前）
        if experiment_name == "D5":
            from contour_types import CONSTELL_CHECK_STATS

            report_lines.append("\n" + "=" * 60)
            report_lines.append("星座相似性检查详细统计（验证原因1和原因2）")
            report_lines.append("=" * 60)

            stats = CONSTELL_CHECK_STATS

            if stats.total_calls > 0:
                # 原因1验证：各阶段过滤统计
                report_lines.append("\n【原因1验证】各阶段过滤统计:")
                report_lines.append(f"  • 总调用次数: {stats.total_calls}")
                overlap_pct = stats.filtered_by_overlap / stats.total_calls * 100
                angle_pct = stats.filtered_by_angle / stats.total_calls * 100
                passed_pct = stats.passed / stats.total_calls * 100
                report_lines.append(f"  • 位重叠过滤: {stats.filtered_by_overlap} ({overlap_pct:.2f}%)")
                report_lines.append(f"  • 角度一致性过滤: {stats.filtered_by_angle} ({angle_pct:.2f}%)")
                report_lines.append(f"  • 通过检查: {stats.passed} ({passed_pct:.2f}%)")

                # 验证一致性
                total_accounted = stats.filtered_by_overlap + stats.filtered_by_angle + stats.passed
                report_lines.append(f"  • 统计一致性检查: {total_accounted} / {stats.total_calls}")

                # 原因2验证：角度差异分布
                angle_dist = stats.get_angle_distribution()

                if angle_dist and angle_dist['total_pairs'] > 0:
                    report_lines.append("\n【原因2验证】通过位重叠后的角度差异分布:")
                    report_lines.append(f"  • 总配对数: {angle_dist['total_pairs']}")
                    report_lines.append(f"  • 平均角度差异: {angle_dist['mean']:.2f}° ± {angle_dist['std']:.2f}°")
                    report_lines.append(f"  • 中位数: {angle_dist['median']:.2f}°")
                    report_lines.append(f"\n  角度差异范围分布:")
                    report_lines.append(f"    - <1°:  {angle_dist['less_1deg']:.1f}%")
                    report_lines.append(f"    - <3°:  {angle_dist['less_3deg']:.1f}%")
                    report_lines.append(f"    - <5°:  {angle_dist['less_5deg']:.1f}%")
                    report_lines.append(f"    - <10°: {angle_dist['less_10deg']:.1f}%")
                    report_lines.append(f"    - <20°: {angle_dist['less_20deg']:.1f}%")
                    report_lines.append(f"    - ≥20°: {angle_dist['greater_20deg']:.1f}%")

                    # 关键判断
                    report_lines.append(f"\n  【关键判断】")
                    if angle_dist['less_3deg'] > 80:
                        report_lines.append(f"    ✓ 原因2成立：{angle_dist['less_3deg']:.1f}% 的配对角度差异<3°")
                        report_lines.append(f"      说明Chilean环境结构高度规整，真实匹配天然角度一致")
                    else:
                        report_lines.append(f"    ✗ 原因2不成立：只有{angle_dist['less_3deg']:.1f}% 的配对<3°")

                    if overlap_pct > 70:
                        report_lines.append(f"    ✓ 原因1成立：{overlap_pct:.1f}% 在位重叠阶段被过滤")
                        report_lines.append(f"      说明位重叠是主要过滤机制")
                    else:
                        report_lines.append(f"    ✗ 原因1不完全成立：只有{overlap_pct:.1f}% 在位重叠过滤")
                else:
                    report_lines.append("\n【原因2验证】无角度差异数据")
            else:
                report_lines.append("\n星座相似性检查统计：无数据")

            report_lines.append("\n" + "=" * 60)

        # 以下是原有内容，保持不变
        report_lines.append("\n轮廓统计:")
        if self.contour_stats['total_contours']:
            avg_total_contours = np.mean(self.contour_stats['total_contours'])
            avg_tiny_ratio = np.mean(self.contour_stats['tiny_contour_ratios']) * 100
            avg_small_ratio = np.mean(self.contour_stats['small_contour_ratios']) * 100
            avg_medium_small_ratio = np.mean(self.contour_stats['medium_small_contour_ratios']) * 100
            avg_medium_ratio = np.mean(self.contour_stats['medium_contour_ratios']) * 100
            avg_large_ratio = np.mean(self.contour_stats['large_contour_ratios']) * 100
            avg_super_large_ratio = np.mean(self.contour_stats['super_large_contour_ratios']) * 100
            avg_eccentricity = np.mean(self.contour_stats['avg_eccentricities'])
            std_eccentricity = np.mean(self.contour_stats['std_eccentricities'])
            avg_significant_ecc_ratio = np.mean(self.contour_stats['significant_ecc_ratios']) * 100
            avg_significant_com_ratio = np.mean(self.contour_stats['significant_com_ratios']) * 100
            avg_size = np.mean(self.contour_stats['avg_sizes'])
            std_size = np.mean(self.contour_stats['std_sizes'])
            avg_min_size = np.mean(self.contour_stats['min_sizes'])
            avg_max_size = np.mean(self.contour_stats['max_sizes'])
            avg_eigenvalue_ratio = np.mean(self.contour_stats['avg_eigenvalue_ratios'])
            avg_height = np.mean(self.contour_stats['avg_heights'])

            report_lines.append(f"- 平均总轮廓数: {avg_total_contours:.1f}")
            report_lines.append(f"- 轮廓尺寸分布:")
            report_lines.append(f"  * 极小轮廓(1-5): {avg_tiny_ratio:.1f}%")
            report_lines.append(f"  * 小轮廓(6-15): {avg_small_ratio:.1f}%")
            report_lines.append(f"  * 中小轮廓(16-50): {avg_medium_small_ratio:.1f}%")
            report_lines.append(f"  * 中等轮廓(51-150): {avg_medium_ratio:.1f}%")
            report_lines.append(f"  * 大轮廓(151-500): {avg_large_ratio:.1f}%")
            report_lines.append(f"  * 超大轮廓(500+): {avg_super_large_ratio:.1f}%")
            report_lines.append(f"- 平均偏心率: {avg_eccentricity:.3f} ± {std_eccentricity:.3f}")
            report_lines.append(f"- 显著特征比例:")
            report_lines.append(f"  * 显著偏心率特征: {avg_significant_ecc_ratio:.1f}%")
            report_lines.append(f"  * 显著质心特征: {avg_significant_com_ratio:.1f}%")
            report_lines.append(
                f"- 轮廓尺寸统计: 平均{avg_size:.1f} ± {std_size:.1f} (范围: {avg_min_size:.1f}-{avg_max_size:.1f})")
            report_lines.append(f"- 平均特征值比例: {avg_eigenvalue_ratio:.3f}")
            report_lines.append(f"- 平均轮廓高度: {avg_height:.2f}m")
        else:
            report_lines.append("- 轮廓统计数据为空")

        report_lines.append("\n特征质量:")
        if self.key_stats['sparsity_ratios']:
            avg_sparsity = np.mean(self.key_stats['sparsity_ratios'])
            avg_quality = np.mean(self.key_stats['quality_scores'])
            avg_ring = np.mean(self.key_stats['ring_activations'])

            report_lines.append(f"- 平均稀疏度: {avg_sparsity:.3f}")
            report_lines.append(f"- 平均质量得分: {avg_quality:.3f}")
            report_lines.append(f"- 平均环形激活: {avg_ring:.1f}")
        else:
            report_lines.append("- 特征质量统计数据为空")

        report_lines.append("\nBCI连接:")

        if experiment_name == "D4":
            min_dist = 1.0
            max_dist = 20.0  # 假设使用固定范围方案
            bin_width = (max_dist - min_dist) / param_value
            report_lines.append(f"Bin宽度: {bin_width:.3f}米")
            report_lines.append(f"Bin数量: {param_value:.1f}")
            report_lines.append(f"距离范围: {min_dist:.1f}-{max_dist:.1f}米")

        if self.bci_stats['avg_neighbors']:
            avg_neighbors = np.mean(self.bci_stats['avg_neighbors'])
            std_neighbors = np.mean(self.bci_stats['std_neighbors'])
            avg_min_neighbors = np.mean(self.bci_stats['min_neighbors'])
            avg_max_neighbors = np.mean(self.bci_stats['max_neighbors'])
            avg_neighbor_dist_0 = np.mean(self.bci_stats['neighbor_dist_0']) * 100
            avg_neighbor_dist_1_3 = np.mean(self.bci_stats['neighbor_dist_1_3']) * 100
            avg_neighbor_dist_4_6 = np.mean(self.bci_stats['neighbor_dist_4_6']) * 100
            avg_neighbor_dist_7_10 = np.mean(self.bci_stats['neighbor_dist_7_10']) * 100
            avg_neighbor_dist_10_plus = np.mean(self.bci_stats['neighbor_dist_10_plus']) * 100
            avg_distance = np.mean(self.bci_stats['avg_distances'])
            std_distance = np.mean(self.bci_stats['std_distances'])
            avg_min_distance = np.mean(self.bci_stats['min_distances'])
            avg_max_distance = np.mean(self.bci_stats['max_distances'])
            avg_angle_diversity = np.mean(self.bci_stats['angle_diversities'])
            avg_angle_uniformity = np.mean(self.bci_stats['angle_uniformities'])
            avg_cross_layer = np.mean(self.bci_stats['cross_layer_ratios']) * 100
            avg_activation_rate = np.mean(self.bci_stats['activation_rates'])
            avg_complexity = np.mean(self.bci_stats['constellation_complexities'])
            avg_connection_quality = np.mean(self.bci_stats['connection_qualities'])

            report_lines.append(
                f"- 平均邻居数: {avg_neighbors:.1f} ± {std_neighbors:.1f} (范围: {avg_min_neighbors:.1f}-{avg_max_neighbors:.1f})")
            report_lines.append(f"- BCI邻居分布:")
            report_lines.append(f"  * 0个邻居: {avg_neighbor_dist_0:.1f}%")
            report_lines.append(f"  * 1-3个邻居: {avg_neighbor_dist_1_3:.1f}%")
            report_lines.append(f"  * 4-6个邻居: {avg_neighbor_dist_4_6:.1f}%")
            report_lines.append(f"  * 7-10个邻居: {avg_neighbor_dist_7_10:.1f}%")
            report_lines.append(f"  * 10+个邻居: {avg_neighbor_dist_10_plus:.1f}%")
            report_lines.append(
                f"- 邻居距离统计: {avg_distance:.2f} ± {std_distance:.2f} (范围: {avg_min_distance:.2f}-{avg_max_distance:.2f})")
            report_lines.append(f"- 角度特征: 多样性{avg_angle_diversity:.3f}, 均匀性{avg_angle_uniformity:.3f}")
            report_lines.append(f"- 平均跨层连接比例: {avg_cross_layer:.1f}%")
            report_lines.append(f"- 平均距离位激活率: {avg_activation_rate:.3f}")
            report_lines.append(f"- 平均星座复杂度: {avg_complexity:.3f}")
            report_lines.append(f"- 平均连接质量: {avg_connection_quality:.3f}")
        else:
            report_lines.append("- BCI连接统计数据为空")

        report_lines.append(f"\n相似性检查统计:")
        if self.similarity_stats['total_checks'] > 0:
            check1_rate = self.similarity_stats['check1_passed'] / self.similarity_stats['total_checks'] * 100
            report_lines.append(f"- 总搜索次数: {self.similarity_stats['total_searches']}")
            report_lines.append(f"- 总检查次数: {self.similarity_stats['total_checks']}")
            report_lines.append(f"- 轮廓相似性通过率: {check1_rate:.1f}%")
            if self.similarity_stats['check1_passed'] > 0:
                check2_rate = self.similarity_stats['check2_passed'] / self.similarity_stats['check1_passed'] * 100
                report_lines.append(f"- 星座相似性通过率: {check2_rate:.1f}%")
            if self.similarity_stats['check2_passed'] > 0:
                check3_rate = self.similarity_stats['check3_passed'] / self.similarity_stats['check2_passed'] * 100
                report_lines.append(f"- 成对相似性通过率: {check3_rate:.1f}%")
        else:
            report_lines.append("- 相似性检查统计数据为空")

        if top1_recall is not None:
            report_lines.append(f"Top 1 Recall: {top1_recall:.2f}%")

        # 同时输出到控制台和文件
        for line in report_lines:
            print(line)  # 输出到控制台

        # 保存到文件
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write('\n'.join(report_lines))

        print(f"\n消融实验报告已保存到: {report_file}")

    def clear_statistics(self):
        """清空统计数据，为下一轮实验准备"""
        self.contour_stats = {
            'total_contours': [],
            'tiny_contour_ratios': [],
            'avg_eccentricities': []
        }

        self.key_stats = {
            'sparsity_ratios': [],
            'quality_scores': [],
            'ring_activations': []
        }

        self.bci_stats = {
            'avg_neighbors': [],
            'cross_layer_ratios': [],
            'constellation_complexities': []
        }

        self.similarity_stats = {
            'total_searches': 0,
            'total_checks': 0,
            'check1_passed': 0,
            'check2_passed': 0,
            'check3_passed': 0
        }
    # ===== 统计汇总方法添加结束 =====

# ========== 添加可视化函数类 ==========
class ContourVisualizer:
    """轮廓上下文可视化器 - 改进版本"""

    def __init__(self, output_dir: str = "visualization_output"):
        self.output_dir = output_dir
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

    def visualize_pointcloud_pipeline(self, cm: ContourManager, pointcloud: np.ndarray,
                                      filename_prefix: str):
        """可视化单个点云的完整处理流程 - 改进版本"""

        # 1. 原始点云 3D散点图
        fig1 = plt.figure(figsize=(12, 8))
        ax1 = fig1.add_subplot(111, projection='3d')
        self._plot_3d_pointcloud(ax1, pointcloud, title="Original Point Cloud 3D")
        plt.tight_layout()
        output_path_3d = os.path.join(self.output_dir, f"{filename_prefix}_01_3d_pointcloud.png")
        plt.savefig(output_path_3d, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"3D point cloud saved to: {output_path_3d}")

        # 2. 每层BEV和该层椭圆图 - 分别保存
        lv_grads = cm.get_config().lv_grads
        num_levels = len(lv_grads) - 1

        for level in range(num_levels):
            self._plot_layered_bev_with_ellipses(cm, level, filename_prefix)

        # 3. 3D椭圆总览
        self._plot_3d_ellipses_overview(cm, filename_prefix)

        # 4. 3D BCI星座总览
        self._plot_3d_bci_constellation_overview(cm, filename_prefix)

        # 5. 单个BCI的详细星座图
        self._plot_single_bci_constellation(cm, filename_prefix)

        # 6. 特征键热力图
        fig_heatmap = plt.figure(figsize=(14, 8))
        ax_heatmap = fig_heatmap.add_subplot(111)
        self._plot_feature_keys_heatmap(ax_heatmap, cm, title="Feature Key Heatmap")
        plt.tight_layout()
        output_path_heatmap = os.path.join(self.output_dir, f"{filename_prefix}_06_feature_heatmap.png")
        plt.savefig(output_path_heatmap, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Feature heatmap saved to: {output_path_heatmap}")

        print(f"All visualizations for {filename_prefix} completed!")

    def _plot_layered_bev_with_ellipses(self, cm: ContourManager, level: int, filename_prefix: str):
        """在每层BEV图中显示该层的椭圆"""
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111)

        # 获取BEV图像和轮廓
        bev_image = cm.get_contour_image(level)
        contours_level = cm.get_lev_contours(level)

        # ✅ 新增验证代码1：验证区间定义一致性
        print(f"\n=== VERIFICATION L{level} ===")
        lv_grads = cm.get_config().lv_grads
        h_min = lv_grads[level]
        h_max = lv_grads[level + 1]
        print(f"Expected interval: [{h_min:.1f}, {h_max:.1f})")

        # 检查BEV原始数据在这个区间的像素数
        original_bev = cm.get_bev_image()
        original_mask = ((original_bev >= h_min) & (original_bev < h_max))
        original_pixels = np.sum(original_mask)
        display_pixels = np.sum(bev_image > 0)
        print(f"Original BEV pixels in interval: {original_pixels}")
        print(f"Display BEV pixels: {display_pixels}")
        print(f"Pixel count match: {original_pixels == display_pixels}")

        # 显示BEV二值图
        ax.imshow(bev_image, cmap='gray', origin='lower', alpha=0.8)

        colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown', 'pink', 'cyan']

        # ✅ 新增验证代码2：逐个验证轮廓和椭圆的对应关系
        ellipse_count = 0
        for seq, contour in enumerate(contours_level[:12]):  # 显示前12个
            color = colors[seq % len(colors)]
            pos_mean = contour.pos_mean
            cell_cnt = contour.cell_cnt

            print(f"\n--- Contour L{level}S{seq} ---")
            print(f"pos_mean (row,col): ({pos_mean[0]:.2f}, {pos_mean[1]:.2f})")
            print(f"cell_cnt: {cell_cnt}")
            print(f"eig_vals: [{contour.eig_vals[0]:.4f}, {contour.eig_vals[1]:.4f}]")

            # ✅ 验证代码2a：检查椭圆中心是否在显示的白色区域内
            center_row, center_col = int(round(pos_mean[0])), int(round(pos_mean[1]))
            if (0 <= center_row < bev_image.shape[0] and 0 <= center_col < bev_image.shape[1]):
                is_center_white = bev_image[center_row, center_col] > 0
                print(
                    f"Ellipse center at display pixel ({center_row}, {center_col}): {'WHITE' if is_center_white else 'BLACK'}")

                # 检查椭圆中心周围3x3区域
                white_neighbors = 0
                for dr in [-1, 0, 1]:
                    for dc in [-1, 0, 1]:
                        nr, nc = center_row + dr, center_col + dc
                        if (0 <= nr < bev_image.shape[0] and 0 <= nc < bev_image.shape[1]):
                            if bev_image[nr, nc] > 0:
                                white_neighbors += 1
                print(f"White pixels in 3x3 around center: {white_neighbors}/9")
            else:
                print(f"Ellipse center OUTSIDE image bounds!")

            # 绘制质心
            ax.plot(pos_mean[1], pos_mean[0], 'o', color=color,
                    markersize=6, markeredgecolor='black', markeredgewidth=1)

            # 绘制协方差椭圆
            if hasattr(contour, 'eig_vals') and hasattr(contour, 'eig_vecs'):
                # ✅ 验证代码2b：计算椭圆理论覆盖范围
                major_radius = 2 * np.sqrt(contour.eig_vals[1]) * 2 # 与可视化代码一致
                minor_radius = 2 * np.sqrt(contour.eig_vals[0]) * 2
                print(f"Ellipse radii: major={major_radius:.2f}, minor={minor_radius:.2f} pixels")

                # 椭圆主轴方向
                main_direction = np.arctan2(contour.eig_vecs[1, 1], contour.eig_vecs[0, 1]) * 180 / np.pi
                print(f"Ellipse main axis angle: {main_direction:.1f} degrees")

                self._draw_covariance_ellipse(ax, pos_mean, contour.eig_vals,
                                              contour.eig_vecs, color, alpha=0.6)

                # 绘制主轴方向
                main_axis = contour.eig_vecs[:, 1] * np.sqrt(contour.eig_vals[1]) * 2
                ax.arrow(pos_mean[1], pos_mean[0], main_axis[1], main_axis[0],
                         head_width=2, head_length=2, fc=color, ec='black', alpha=0.8, linewidth=1.5)

                ellipse_count += 1

            # ✅ 验证代码2c：检查轮廓实际分布范围
            # 找到属于这个轮廓的所有像素（通过连通组件分析）
            binary_mask = (bev_image > 0).astype(np.uint8)
            num_labels, labels = cv2.connectedComponents(binary_mask)

            # 找到包含椭圆中心的连通组件
            if (0 <= center_row < labels.shape[0] and 0 <= center_col < labels.shape[1]):
                center_label = labels[center_row, center_col]
                if center_label > 0:  # 0是背景
                    component_pixels = np.where(labels == center_label)
                    if len(component_pixels[0]) > 0:
                        actual_rows = component_pixels[0]
                        actual_cols = component_pixels[1]
                        actual_row_range = (actual_rows.min(), actual_rows.max())
                        actual_col_range = (actual_cols.min(), actual_cols.max())
                        actual_span_row = actual_row_range[1] - actual_row_range[0] + 1
                        actual_span_col = actual_col_range[1] - actual_col_range[0] + 1

                        print(f"Actual contour spans: row={actual_span_row} pixels, col={actual_span_col} pixels")
                        print(
                            f"Actual contour bounds: rows=[{actual_row_range[0]}, {actual_row_range[1]}], cols=[{actual_col_range[0]}, {actual_col_range[1]}]")

                        # 比较椭圆覆盖范围与实际轮廓范围
                        if hasattr(contour, 'eig_vals'):
                            ellipse_span_estimate = max(major_radius * 2, minor_radius * 2)
                            print(f"Ellipse span estimate: {ellipse_span_estimate:.1f} pixels")
                            if ellipse_span_estimate > 0:
                                print(
                                    f"Actual vs Ellipse ratio: row={actual_span_row / ellipse_span_estimate:.2f}, col={actual_span_col / ellipse_span_estimate:.2f}")

            # 标注
            ax.text(pos_mean[1] + 8, pos_mean[0] + 8, f'S{seq}\n({cell_cnt})',
                    color='black', fontsize=10, weight='bold',
                    bbox=dict(boxstyle="round,pad=0.3", facecolor=color, alpha=0.8))

        # ✅ 验证代码3：添加实际轮廓边界叠加（可选）
        print(f"\n=== Adding contour boundaries for verification ===")
        binary_mask = (bev_image > 0).astype(np.uint8)
        try:
            contours_cv, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            print(f"Found {len(contours_cv)} contour boundaries via cv2.findContours")

            for i, contour_cv in enumerate(contours_cv[:len(contours_level)]):  # 限制数量匹配
                # 绘制轮廓边界
                contour_points = contour_cv.squeeze()
                if len(contour_points.shape) == 2 and contour_points.shape[0] > 2:
                    # 注意：cv2.findContours返回的是(x,y)坐标，对应(col,row)
                    ax.plot(contour_points[:, 0], contour_points[:, 1], '--',
                            color='white', linewidth=2, alpha=0.8,
                            label=f'Actual boundary' if i == 0 else "")
                    print(f"Drew boundary {i} with {len(contour_points)} points")
        except Exception as e:
            print(f"Error drawing contour boundaries: {e}")

        # 设置标题和标签
        title = f'L{level}: Height [{h_min:.1f}m, {h_max:.1f}m) - {len(contours_level)} contours, {ellipse_count} ellipses'
        ax.set_title(title, fontsize=14, weight='bold')

        # 坐标轴设置
        config = cm.get_config()
        x_center = config.n_col // 2
        y_center = config.n_row // 2
        meter_range = 12
        grid_range = int(meter_range / config.reso_col)

        ax.set_xlim(x_center - grid_range, x_center + grid_range)
        ax.set_ylim(y_center - grid_range, y_center + grid_range)

        # 米制标签
        x_ticks = np.linspace(x_center - grid_range, x_center + grid_range, 5)
        y_ticks = np.linspace(y_center - grid_range, y_center + grid_range, 5)
        x_labels = [f'{(x - x_center) * config.reso_col:.1f}' for x in x_ticks]
        y_labels = [f'{(y - y_center) * config.reso_row:.1f}' for y in y_ticks]

        ax.set_xticks(x_ticks)
        ax.set_yticks(y_ticks)
        ax.set_xticklabels(x_labels)
        ax.set_yticklabels(y_labels)
        ax.set_xlabel('X (m)', fontsize=12)
        ax.set_ylabel('Y (m)', fontsize=12)
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal')

        # 添加图例（如果有轮廓边界的话）
        if len(plt.gca().get_lines()) > len(contours_level):  # 有额外的边界线
            ax.legend(loc='upper right', fontsize=8)

        plt.tight_layout()
        output_path = os.path.join(self.output_dir, f"{filename_prefix}_02_L{level}_bev_ellipses.png")
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Level {level} BEV with ellipses saved to: {output_path}")
        print(f"=== END VERIFICATION L{level} ===\n")

    def _plot_3d_ellipses_overview(self, cm: ContourManager, filename_prefix: str):
        """3D椭圆总览"""
        fig = plt.figure(figsize=(14, 10))
        ax = fig.add_subplot(111, projection='3d')

        lv_grads = cm.get_config().lv_grads
        num_levels = len(lv_grads) - 1
        config = cm.get_config()

        # 层级颜色
        colors = plt.cm.Set1(np.linspace(0, 1, num_levels))

        total_ellipses = 0

        for level in range(num_levels):
            contours_level = cm.get_lev_contours(level)
            color = colors[level]

            # 计算层级的Z坐标（使用区间中点）
            z_coord = (lv_grads[level] + lv_grads[level + 1]) / 2

            for seq, contour in enumerate(contours_level):  # 每层最多显示5个
                pos_mean = contour.pos_mean

                # 转换到米制坐标
                x_meter = (pos_mean[1] - config.n_col // 2) * config.reso_col
                y_meter = (pos_mean[0] - config.n_row // 2) * config.reso_row

                # 绘制质心点
                ax.scatter(x_meter, y_meter, z_coord,
                           c=[color], s=15, alpha=1.0, edgecolors='black')

                # 绘制3D椭圆投影（在对应高度平面上）
                if hasattr(contour, 'eig_vals') and hasattr(contour, 'eig_vecs'):
                    self._draw_3d_ellipse_projection(ax, x_meter, y_meter, z_coord,
                                                     contour.eig_vals, contour.eig_vecs,
                                                     color, config.reso_col)

                # 添加标签
                ax.text(x_meter + 0.5, y_meter + 0.5, z_coord + 0.1,
                        f'L{level}S{seq}', fontsize=6)

                total_ellipses += 1

        ax.set_xlabel('X (m)', fontsize=12)
        ax.set_ylabel('Y (m)', fontsize=12)
        ax.set_zlabel('Height (m)', fontsize=12)
        ax.set_title(f'3D Contour Ellipses Overview ({total_ellipses} ellipses)', fontsize=14, weight='bold')

        # 设置视角
        ax.view_init(elev=20, azim=45)

        # 设置坐标范围
        ax.set_xlim(-10, 10)
        ax.set_ylim(-10, 10)
        ax.set_zlim(lv_grads[0], lv_grads[-1])

        plt.tight_layout()
        output_path = os.path.join(self.output_dir, f"{filename_prefix}_03_3d_ellipses_overview.png")
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"3D ellipses overview saved to: {output_path}")

    def _plot_3d_bci_constellation_overview(self, cm: ContourManager, filename_prefix: str):
        """3D BCI星座总览"""
        fig = plt.figure(figsize=(14, 10))
        ax = fig.add_subplot(111, projection='3d')

        lv_grads = cm.get_config().lv_grads
        num_levels = len(lv_grads) - 1
        config = cm.get_config()

        # 层级颜色
        colors = plt.cm.tab10(np.linspace(0, 1, num_levels))

        total_connections = 0

        for level in range(num_levels):
            bcis = cm.get_lev_bci(level)
            contours_level = cm.get_lev_contours(level)

            for seq, bci in enumerate(bcis[:1]):  # 每层最多显示1个BCI
                if seq >= len(contours_level):
                    continue

                pivot_contour = contours_level[bci.piv_seq] if bci.piv_seq < len(contours_level) else None
                if pivot_contour is None:
                    continue

                # 中心点坐标
                pivot_pos = pivot_contour.pos_mean
                pivot_x = (pivot_pos[1] - config.n_col // 2) * config.reso_col
                pivot_y = (pivot_pos[0] - config.n_row // 2) * config.reso_row
                pivot_z = (lv_grads[level] + lv_grads[level + 1]) / 2

                # 绘制中心点
                ax.scatter(pivot_x, pivot_y, pivot_z,
                           c=[colors[level]], s=150, alpha=0.9,
                           edgecolors='black', linewidth=2, marker='o')

                # 绘制邻居连接
                for nei_pt in bci.nei_pts:  # 最多显示10个邻居
                    nei_level = nei_pt.level
                    nei_seq = nei_pt.seq

                    if nei_level >= num_levels:
                        continue

                    nei_contours = cm.get_lev_contours(nei_level)
                    if nei_seq >= len(nei_contours):
                        continue

                    nei_contour = nei_contours[nei_seq]
                    nei_pos = nei_contour.pos_mean

                    nei_x = (nei_pos[1] - config.n_col // 2) * config.reso_col
                    nei_y = (nei_pos[0] - config.n_row // 2) * config.reso_row
                    nei_z = (lv_grads[nei_level] + lv_grads[nei_level + 1]) / 2

                    # 绘制邻居点
                    ax.scatter(nei_x, nei_y, nei_z,
                               c=[colors[nei_level]], s=50, alpha=0.7, marker='s')

                    # 绘制连接线
                    ax.plot([pivot_x, nei_x], [pivot_y, nei_y], [pivot_z, nei_z],
                            color=colors[level], alpha=0.5, linewidth=1)

                    total_connections += 1

                # 添加标签
                ax.text(pivot_x + 0.5, pivot_y + 0.5, pivot_z + 0.2,
                        f'L{level}S{seq}', fontsize=9, weight='bold')

        ax.set_xlabel('X (m)', fontsize=12)
        ax.set_ylabel('Y (m)', fontsize=12)
        ax.set_zlabel('Height (m)', fontsize=12)
        ax.set_title(f'3D BCI Constellation Overview ({total_connections} connections)',
                     fontsize=14, weight='bold')

        # 设置视角
        ax.view_init(elev=25, azim=60)

        # 设置坐标范围
        ax.set_xlim(-10, 10)
        ax.set_ylim(-10, 10)
        ax.set_zlim(lv_grads[0], lv_grads[-1])

        plt.tight_layout()
        output_path = os.path.join(self.output_dir, f"{filename_prefix}_04_3d_bci_overview.png")
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"3D BCI constellation overview saved to: {output_path}")

    def _plot_single_bci_constellation(self, cm: ContourManager, filename_prefix: str):
        """单个BCI的详细星座图"""
        # 选择一个有较多邻居的BCI进行详细可视化
        selected_bci = None
        selected_level = -1
        selected_seq = -1
        max_neighbors = 0

        lv_grads = cm.get_config().lv_grads
        num_levels = len(lv_grads) - 1

        # 找到邻居最多的BCI
        for level in range(num_levels):
            bcis = cm.get_lev_bci(level)
            for seq, bci in enumerate(bcis):
                if len(bci.nei_pts) > max_neighbors:
                    max_neighbors = len(bci.nei_pts)
                    selected_bci = bci
                    selected_level = level
                    selected_seq = seq

        if selected_bci is None or max_neighbors == 0:
            print(f"No BCI with neighbors found for detailed visualization")
            return

        fig = plt.figure(figsize=(14, 10))
        ax = fig.add_subplot(111, projection='3d')

        config = cm.get_config()
        contours_level = cm.get_lev_contours(selected_level)

        if selected_bci.piv_seq >= len(contours_level):
            print(f"Invalid pivot sequence {selected_bci.piv_seq} for level {selected_level}")
            return

        pivot_contour = contours_level[selected_bci.piv_seq]

        # 中心点坐标
        pivot_pos = pivot_contour.pos_mean
        pivot_x = (pivot_pos[1] - config.n_col // 2) * config.reso_col
        pivot_y = (pivot_pos[0] - config.n_row // 2) * config.reso_row
        pivot_z = (lv_grads[selected_level] + lv_grads[selected_level + 1]) / 2

        # 绘制中心椭圆
        if hasattr(pivot_contour, 'eig_vals') and hasattr(pivot_contour, 'eig_vecs'):
            self._draw_3d_ellipse_projection(ax, pivot_x, pivot_y, pivot_z,
                                             pivot_contour.eig_vals, pivot_contour.eig_vecs,
                                             'red', config.reso_col, alpha=0.8)

        # 绘制中心点
        ax.scatter(pivot_x, pivot_y, pivot_z,
                   c='red', s=100, alpha=1.0,
                   edgecolors='black', linewidth=2, marker='o')

        # 层级颜色
        colors = plt.cm.tab10(np.linspace(0, 1, num_levels))

        # 绘制所有邻居
        for i, nei_pt in enumerate(selected_bci.nei_pts):
            nei_level = nei_pt.level
            nei_seq = nei_pt.seq

            if nei_level >= num_levels:
                continue

            nei_contours = cm.get_lev_contours(nei_level)
            if nei_seq >= len(nei_contours):
                continue

            nei_contour = nei_contours[nei_seq]
            nei_pos = nei_contour.pos_mean

            nei_x = (nei_pos[1] - config.n_col // 2) * config.reso_col
            nei_y = (nei_pos[0] - config.n_row // 2) * config.reso_row
            nei_z = (lv_grads[nei_level] + lv_grads[nei_level + 1]) / 2

            # 绘制邻居椭圆
            if hasattr(nei_contour, 'eig_vals') and hasattr(nei_contour, 'eig_vecs'):
                self._draw_3d_ellipse_projection(ax, nei_x, nei_y, nei_z,
                                                 nei_contour.eig_vals, nei_contour.eig_vecs,
                                                 colors[nei_level], config.reso_col, alpha=0.6)

            # 绘制邻居点
            ax.scatter(nei_x, nei_y, nei_z,
                       c=[colors[nei_level]], s=20, alpha=1.0, marker='s')

            # 绘制连接线，线宽根据距离调整
            line_width = max(1, 3 - nei_pt.r / 5)  # 距离越近线越粗
            ax.plot([pivot_x, nei_x], [pivot_y, nei_y], [pivot_z, nei_z],
                    color=colors[nei_level], alpha=0.7, linewidth=line_width)

            # 添加距离和角度标签
            mid_x = (pivot_x + nei_x) / 2
            mid_y = (pivot_y + nei_y) / 2
            mid_z = (pivot_z + nei_z) / 2

            if i < 10:  # 只标注前10个邻居，避免过于拥挤
                ax.text(mid_x, mid_y, mid_z,
                        f'r={nei_pt.r:.1f}\nθ={nei_pt.theta * 180 / np.pi:.0f}°',
                        fontsize=7, alpha=0.8)

            # 邻居标签
            ax.text(nei_x + 0.3, nei_y + 0.3, nei_z + 0.1,
                    f'L{nei_level}S{nei_seq}', fontsize=8)

        # 中心点标签
        ax.text(pivot_x + 0.5, pivot_y + 0.5, pivot_z + 0.3,
                f'CENTER\nL{selected_level}S{selected_seq}',
                fontsize=10, weight='bold', color='red')

        ax.set_xlabel('X (m)', fontsize=12)
        ax.set_ylabel('Y (m)', fontsize=12)
        ax.set_zlabel('Height (m)', fontsize=12)
        ax.set_title(f'Detailed BCI Constellation: L{selected_level}S{selected_seq} '
                     f'({len(selected_bci.nei_pts)} neighbors)',
                     fontsize=14, weight='bold')

        # 设置视角
        ax.view_init(elev=20, azim=45)

        # 动态设置坐标范围
        all_x = [pivot_x] + [nei_x for nei_x in [pivot_x]]  # 简化版本，实际应收集所有点
        all_y = [pivot_y] + [nei_y for nei_y in [pivot_y]]

        range_x = max(15, (max(all_x) - min(all_x)) * 1.2) if len(all_x) > 1 else 15
        range_y = max(15, (max(all_y) - min(all_y)) * 1.2) if len(all_y) > 1 else 15

        ax.set_xlim(pivot_x - range_x / 2, pivot_x + range_x / 2)
        ax.set_ylim(pivot_y - range_y / 2, pivot_y + range_y / 2)
        ax.set_zlim(lv_grads[0], lv_grads[-1])

        plt.tight_layout()
        output_path = os.path.join(self.output_dir, f"{filename_prefix}_05_detailed_bci_constellation.png")
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Detailed BCI constellation saved to: {output_path}")

    def _draw_3d_ellipse_projection(self, ax, center_x, center_y, center_z,
                                    eig_vals, eig_vecs, color, resolution, alpha=0.6):
        """在3D空间中绘制椭圆投影"""
        # 创建椭圆参数
        theta = np.linspace(0, 2 * np.pi, 50)

        # 椭圆在标准坐标系中的点
        ellipse_x = np.sqrt(eig_vals[1]) * np.cos(theta) * resolution
        ellipse_y = np.sqrt(eig_vals[0]) * np.sin(theta) * resolution

        # 旋转椭圆
        cos_angle = eig_vecs[1, 1]  # 主轴方向
        sin_angle = eig_vecs[0, 1]

        rotated_x = cos_angle * ellipse_x - sin_angle * ellipse_y
        rotated_y = sin_angle * ellipse_x + cos_angle * ellipse_y

        # 平移到中心位置
        final_x = rotated_x + center_x
        final_y = rotated_y + center_y
        final_z = np.full_like(final_x, center_z)

        # 绘制椭圆
        ax.plot(final_x, final_y, final_z, color=color, alpha=alpha, linewidth=2)

    def _draw_covariance_ellipse(self, ax, center, eig_vals, eig_vecs, color, alpha=0.3):
        """绘制协方差椭圆 (2D版本)"""
        from matplotlib.patches import Ellipse

        # 计算椭圆参数
        angle = np.degrees(np.arctan2(eig_vecs[0, 1], eig_vecs[1, 1]))
        #                              ^row分量(Y)    ^col分量(X)
        # 相当于 atan2(Y, X)
        width = 2 * np.sqrt(eig_vals[1])  # 2倍标准差，覆盖95%的数据
        height = 2 * np.sqrt(eig_vals[0])  # 2倍标准差

        # 创建椭圆
        ellipse = Ellipse((center[1], center[0]), width, height,
                          angle=angle, facecolor=color, alpha=alpha,
                          edgecolor='black', linewidth=1)
        ax.add_patch(ellipse)

    def _plot_3d_pointcloud(self, ax, pointcloud: np.ndarray, title: str):
        """绘制3D点云"""
        # 采样以提高显示性能
        if len(pointcloud) > 15000:
            indices = np.random.choice(len(pointcloud), 15000, replace=False)
            sample_cloud = pointcloud[indices]
        else:
            sample_cloud = pointcloud

        # 按z值着色
        colors = sample_cloud[:, 2]
        scatter = ax.scatter(sample_cloud[:, 0], sample_cloud[:, 1], sample_cloud[:, 2],
                             c=colors, s=1.0, alpha=0.7, cmap='viridis')

        ax.set_xlabel('X (m)', fontsize=12)
        ax.set_ylabel('Y (m)', fontsize=12)
        ax.set_zlabel('Z (m)', fontsize=12)
        ax.set_title(title, fontsize=14, weight='bold')

        # 设置合理的轴范围
        ax.set_xlim(-10, 10)
        ax.set_ylim(-10, 10)
        ax.set_zlim(-5, 5)

        # 添加颜色条
        plt.colorbar(scatter, ax=ax, shrink=0.6, aspect=30, label='Height (m)')

        # 添加统计信息
        ax.text2D(0.02, 0.98, f'Points: {len(sample_cloud):,}\nZ range: [{colors.min():.2f}, {colors.max():.2f}]m',
                  transform=ax.transAxes, verticalalignment='top', fontsize=10,
                  bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))

    def _plot_feature_keys_heatmap(self, ax, cm: ContourManager, title: str):
        """绘制特征键热力图"""
        lv_grads = cm.get_config().lv_grads
        n_levels = len(lv_grads) - 1
        n_seqs = 8

        # 收集所有特征键
        feature_matrix = np.zeros((n_levels, n_seqs))

        for level in range(n_levels):
            keys = cm.get_lev_retrieval_key(level)
            for seq in range(min(len(keys), n_seqs)):
                key_sum = np.sum(keys[seq])
                feature_matrix[level, seq] = key_sum

        # 绘制热力图
        im = ax.imshow(feature_matrix, cmap='RdYlBu_r', aspect='auto', origin='lower')

        # 添加数值标注
        for level in range(n_levels):
            for seq in range(n_seqs):
                value = feature_matrix[level, seq]
                color = 'white' if value > np.max(feature_matrix) * 0.5 else 'black'
                ax.text(seq, level, f'{value:.0f}' if value > 0 else '0',
                        ha='center', va='center', color=color, fontsize=12, weight='bold')

        # 设置标签
        ax.set_xlabel('Sequence Index (S0-S7)', fontsize=12)
        ax.set_ylabel('Level Index (L0-L7)', fontsize=12)

        valid_keys = np.sum(feature_matrix > 0)
        zero_keys = np.sum(feature_matrix == 0)
        ax.set_title(f'{title}\nValid Keys: {valid_keys}, Zero Keys: {zero_keys}',
                     fontsize=14, weight='bold')

        # 设置刻度
        ax.set_xticks(range(n_seqs))
        ax.set_xticklabels([f'S{i}' for i in range(n_seqs)])
        ax.set_yticks(range(n_levels))

        # 正确的区间标签
        interval_labels = []
        for i in range(n_levels):
            h_min = lv_grads[i]
            h_max = lv_grads[i + 1]
            interval_labels.append(f'L{i}\n[{h_min:.1f},{h_max:.1f})')

        ax.set_yticklabels(interval_labels)

        # 添加颜色条
        cbar = plt.colorbar(im, ax=ax, label='Feature Key Value')
        cbar.ax.tick_params(labelsize=10)

        # 添加统计信息
        max_value = np.max(feature_matrix)
        mean_nonzero = np.mean(feature_matrix[feature_matrix > 0]) if valid_keys > 0 else 0

        stats_text = f'Max Value: {max_value:.0f}\nMean (non-zero): {mean_nonzero:.1f}\nSparsity: {zero_keys}/{n_levels * n_seqs} ({100 * zero_keys / (n_levels * n_seqs):.1f}%)'
        ax.text(1.02, 0.5, stats_text, transform=ax.transAxes, verticalalignment='center',
                bbox=dict(boxstyle="round,pad=0.4", facecolor="lightyellow", alpha=0.9),
                fontsize=10)


def main():
    """主函数"""
    # ========== 配置参数 - 在PyCharm中直接修改这里 ==========
    DATASET_FOLDER = '/home/wzj/pan2/Chilean_Underground_Mine_Dataset_Many_Times'  # 修改为你的Chilean数据集路径
    DATABASE_FILE = 'chilean_NoRot_NoScale_5cm_evaluation_database_100_100.pickle'  # 数据库pickle文件
    QUERY_FILE = 'chilean_NoRot_NoScale_5cm_evaluation_query_190_190.pickle'  # 查询pickle文件
    LOG_FILE = 'contour_chilean_不相同时段_log.txt'  # 日志文件
    # =====================================================

    # 如果仍然希望支持命令行参数，可以取消注释以下代码
    import argparse
    parser = argparse.ArgumentParser(description='Chilean数据集上的Contour Context评估')
    parser.add_argument('--dataset_folder', type=str, default=DATASET_FOLDER,
                        help='Chilean数据集文件夹路径')
    parser.add_argument('--database_file', type=str, default=DATABASE_FILE,
                        help='数据库pickle文件名')
    parser.add_argument('--query_file', type=str, default=QUERY_FILE,
                        help='查询pickle文件名')
    parser.add_argument('--log_file', type=str, default=LOG_FILE,
                        help='日志文件名')

    args = parser.parse_args()

    # 使用配置参数
    dataset_folder = args.dataset_folder
    database_file = args.database_file
    query_file = args.query_file
    log_file = args.log_file

    # 检查必要文件
    if not os.path.exists(database_file):
        print(f"错误：找不到数据库文件 {database_file}")
        print("请先运行 generate_test_sets_chilean_NoRot_period.py 生成评估数据")
        print(f"当前工作目录: {os.getcwd()}")
        print(f"寻找文件: {os.path.abspath(database_file)}")
        return

    if not os.path.exists(query_file):
        print(f"错误：找不到查询文件 {query_file}")
        print("请先运行 generate_test_sets_chilean_NoRot_period.py 生成评估数据")
        print(f"当前工作目录: {os.getcwd()}")
        print(f"寻找文件: {os.path.abspath(query_file)}")
        return

    if not os.path.exists(dataset_folder):
        print(f"错误：找不到数据集文件夹 {dataset_folder}")
        print(f"当前工作目录: {os.getcwd()}")
        print(f"寻找路径: {os.path.abspath(dataset_folder)}")
        return

    print(f"数据集路径: {dataset_folder}")
    print(f"数据库文件: {database_file}")
    print(f"查询文件: {query_file}")
    print(f"日志文件: {log_file}")
    print(f"当前工作目录: {os.getcwd()}")
    print("-" * 60)

    # 重置星座相似性检查统计
    from contour_types import CONSTELL_CHECK_STATS
    CONSTELL_CHECK_STATS.reset()
    print("已重置星座相似性检查统计")
    print("-" * 60)

    # 创建评估器
    evaluator = ChileanContourEvaluator(
        dataset_folder=dataset_folder,
        database_file=database_file,
        query_file=query_file,
        log_file=log_file
    )
    evaluator.cm_config.use_vertical_complexity = True #是否启用垂直结构复杂度
    evaluator.cm_config.neighbor_layer_range = 7 #BCI邻域搜索：neighbor_layer_range = 0（仅本层）， 1（本层±1）...，7（本层±7）； 对于8层配置，neighbor_layer_range=7是上限， 在contour_types.py里修改
    evaluator.cm_config.angular_consistency_threshold = np.pi / 16

    # 执行可视化（在评估之前）
    print("=" * 60)
    print("开始可视化分析...")
    print("=" * 60)
    # 可视化（现在会显示区间信息）
    evaluator.visualize_sample_cases()  # 先测试一个
    print("=" * 60)
    print("可视化完成，开始正常评估...")
    print("=" * 60)

    # 执行评估
    start_time = time.time()
    top1_recall = evaluator.evaluate()
    end_time = time.time()

    # 输出消融实验汇总
    evaluator.output_ablation_summary("H0", param_value="后处理三道门槛（面积、距离、GMM）独立贡献：全禁用", top1_recall=top1_recall)

    print(f"\n评估完成!")
    print(f"总耗时: {end_time - start_time:.2f} 秒")
    print(f"最终Top 1 Recall: {top1_recall:.2f}%")
    print(f"详细日志已保存到: {log_file}")


if __name__ == "__main__":
    main()
