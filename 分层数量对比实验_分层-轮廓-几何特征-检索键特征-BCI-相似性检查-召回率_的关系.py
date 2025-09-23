#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
综合分层数量分析实验
深入分析层数变化对特征提取和召回率的影响
扩展到1-20层，包含详细的轮廓、BCI、特征统计分析
"""

import os
import sys
import time
import re
import subprocess
from typing import Dict, List, Tuple, Any
import tempfile
import shutil
import statistics
import numpy as np
import json


class ComprehensiveLayersAnalysisExperiment:
    """综合分层数量分析实验类"""

    def __init__(self):
        self.base_script = "contour_chilean_场景识别_不相同时段100-190_不加旋转.py"
        self.base_types = "contour_types.py"
        self.results = []

        # 扩展到1-20层的完整测试
        self.experiments = self._generate_layer_experiments(1, 2)

        # 特征分析的详细配置
        self.contour_size_bins = [
            (1, 5, "极小轮廓"),
            (6, 10, "小轮廓"),
            (11, 20, "中小轮廓"),
            (21, 50, "中等轮廓"),
            (51, 100, "大轮廓"),
            (101, float('inf'), "极大轮廓")
        ]

        self.eccentricity_bins = [
            (0.0, 0.3, "近圆形"),
            (0.3, 0.6, "椭圆形"),
            (0.6, 0.8, "长椭圆"),
            (0.8, 1.0, "极长椭圆")
        ]

    def _generate_layer_experiments(self, min_layers: int, max_layers: int) -> List[Dict]:
        """生成1-20层的实验配置"""
        experiments = []

        for num_layers in range(min_layers, max_layers + 1):
            if num_layers == 1:
                # 1层特殊情况
                lv_grads = [0.0, 5.0]
                q_levels = [0]
                dist_layers = [0]
                weights = [1.0]
            else:
                # 多层情况：均匀分割0-5m
                lv_grads = []
                for i in range(num_layers + 1):
                    height = 5.0 * i / num_layers
                    lv_grads.append(round(height, 3))

                q_levels = list(range(num_layers))
                dist_layers = list(range(num_layers))

                # 生成权重（中间层级权重更高）
                weights = self._generate_layer_weights(num_layers)

            experiment = {
                "name": f"{num_layers}layers",
                "layers_count": num_layers,
                "layer_interval": 5.0 / num_layers,
                "lv_grads": lv_grads,
                "q_levels": q_levels,
                "dist_layers": dist_layers,
                "weights": weights
            }

            experiments.append(experiment)

        return experiments

    def _generate_layer_weights(self, num_layers: int) -> List[float]:
        """生成层级权重"""
        if num_layers <= 3:
            return [1.0 / num_layers] * num_layers

        weights = []
        for i in range(num_layers):
            # 使用高斯分布，中间层级权重更高
            center = (num_layers - 1) / 2
            distance = abs(i - center) / (num_layers / 2)
            weight = np.exp(-distance * distance)
            weights.append(weight)

        # 归一化
        total = sum(weights)
        return [w / total for w in weights]

    def backup_files(self):
        """备份原始文件"""
        print("备份原始文件...")
        if os.path.exists(self.base_script):
            shutil.copy2(self.base_script, f"{self.base_script}.backup")
        if os.path.exists(self.base_types):
            shutil.copy2(self.base_types, f"{self.base_types}.backup")

    def restore_files(self):
        """恢复原始文件"""
        print("恢复原始文件...")
        if os.path.exists(f"{self.base_script}.backup"):
            shutil.copy2(f"{self.base_script}.backup", self.base_script)
            os.remove(f"{self.base_script}.backup")
        if os.path.exists(f"{self.base_types}.backup"):
            shutil.copy2(f"{self.base_types}.backup", self.base_types)
            os.remove(f"{self.base_types}.backup")

    def modify_config_files(self, experiment: Dict):
        """修改配置文件"""
        print(f"修改配置文件为: {experiment['name']} ({experiment['layers_count']}层)")

        # 修改 contour_types.py
        self._modify_contour_types(experiment)

        # 修改主程序文件
        self._modify_main_script(experiment)

    def _modify_contour_types(self, experiment: Dict):
        """修改 contour_types.py 文件"""
        with open(self.base_types, 'r', encoding='utf-8') as f:
            content = f.read()

        # 修改 DIST_BIN_LAYERS
        dist_layers_str = str(experiment['dist_layers'])
        content = re.sub(
            r'DIST_BIN_LAYERS = \[.*?\]',
            f'DIST_BIN_LAYERS = {dist_layers_str}',
            content
        )

        # 修改 LAYER_AREA_WEIGHTS
        weights_str = '[' + ', '.join([f'{w:.4f}' for w in experiment['weights']]) + ']'
        content = re.sub(
            r'LAYER_AREA_WEIGHTS = \[.*?\]',
            f'LAYER_AREA_WEIGHTS = {weights_str}',
            content
        )

        with open(self.base_types, 'w', encoding='utf-8') as f:
            f.write(content)

    def _modify_main_script(self, experiment: Dict):
        """修改主程序文件"""
        with open(self.base_script, 'r', encoding='utf-8') as f:
            content = f.read()

        # 修改 lv_grads
        lv_grads_str = '[' + ', '.join([f'{g:.3f}' for g in experiment['lv_grads']]) + ']'
        content = re.sub(
            r'config\.lv_grads = \[.*?\]',
            f'config.lv_grads = {lv_grads_str}',
            content
        )

        # 修改 q_levels
        q_levels_str = str(experiment['q_levels'])
        content = re.sub(
            r'config\.q_levels = \[.*?\]',
            f'config.q_levels = {q_levels_str}',
            content
        )

        with open(self.base_script, 'w', encoding='utf-8') as f:
            f.write(content)

    def run_single_experiment(self, experiment: Dict) -> Dict:
        """运行单个实验"""
        experiment_name = experiment['name']
        layers_count = experiment['layers_count']
        layer_interval = experiment['layer_interval']

        print(f"\n{'=' * 80}")
        print(f"开始实验: {experiment_name}")
        print(f"层数: {layers_count}, 层间隔: {layer_interval:.3f}m")
        print(f"高度分割: {experiment['lv_grads']}")
        print(f"{'=' * 80}")

        # 修改配置文件
        self.modify_config_files(experiment)

        # === 新增：强制重新加载模块 ===
        import importlib
        import sys

        # 如果模块已经导入，重新加载它
        if 'contour_manager_区间分割_垂直结构复杂度' in sys.modules:
            importlib.reload(sys.modules['contour_manager_区间分割_垂直结构复杂度'])
            print("[INFO] 重新加载了 contour_manager 模块")

        if 'contour_database' in sys.modules:
            importlib.reload(sys.modules['contour_database'])
            print("[INFO] 重新加载了 contour_database 模块")
        # === 重新加载结束 ===

        # 设置日志文件名
        log_file = f"comprehensive_{experiment_name}_log.txt"

        # 运行实验
        start_time = time.time()

        try:
            # 运行主程序
            result = subprocess.run([
                sys.executable, self.base_script,
                "--log_file", log_file
            ], capture_output=True, text=True, timeout=3600)  # 1小时超时

            if result.returncode != 0:
                print(f"实验 {experiment_name} 运行失败:")
                print(f"STDOUT: {result.stdout}")
                print(f"STDERR: {result.stderr}")
                return None

        except subprocess.TimeoutExpired:
            print(f"实验 {experiment_name} 运行超时")
            return None
        except Exception as e:
            print(f"实验 {experiment_name} 运行异常: {e}")
            return None

        end_time = time.time()
        duration = end_time - start_time

        # 解析基本结果
        result_data = self._parse_basic_results(log_file, experiment, duration)

        if result_data:
            # 进行详细的特征分析
            feature_analysis = self._perform_detailed_feature_analysis(log_file, experiment)
            result_data.update(feature_analysis)

            # 在日志文件末尾添加分析信息
            self._append_analysis_info_to_log(log_file, experiment, duration, feature_analysis)

        print(f"实验 {experiment_name} 完成")
        print(f"运行时长: {duration:.2f}秒")
        if result_data:
            print(f"Recall@1: {result_data['recall_1']:.2f}%")
            print(f"平均轮廓数: {result_data.get('avg_contours_per_frame', 0):.1f}")
            print(f"特征质量指数: {result_data.get('feature_quality_index', 0):.3f}")

        return result_data

    def _parse_basic_results(self, log_file: str, experiment: Dict, duration: float) -> Dict:
        """解析基本实验结果"""
        if not os.path.exists(log_file):
            print(f"警告: 日志文件 {log_file} 不存在")
            return None

        try:
            with open(log_file, 'r', encoding='utf-8') as f:
                content = f.read()

            result = {
                'name': experiment['name'],
                'layers_count': experiment['layers_count'],
                'layer_interval': experiment['layer_interval'],
                'duration': duration,
                'log_file': log_file,
                'recall_1': 0.0,
                'recall_5': 0.0,
                'recall_10': 0.0,
                'recall_25': 0.0,
                'similarity': 0.0,
                'top1_recall': 0.0
            }

            # 提取召回率
            recall_1_match = re.search(r'Average Recall @1: ([\d.]+)%', content)
            if recall_1_match:
                result['recall_1'] = float(recall_1_match.group(1))

            recall_5_match = re.search(r'Average Recall @5: ([\d.]+)%', content)
            if recall_5_match:
                result['recall_5'] = float(recall_5_match.group(1))

            recall_10_match = re.search(r'Average Recall @10: ([\d.]+)%', content)
            if recall_10_match:
                result['recall_10'] = float(recall_10_match.group(1))

            recall_25_match = re.search(r'Average Recall @25: ([\d.]+)%', content)
            if recall_25_match:
                result['recall_25'] = float(recall_25_match.group(1))

            # 提取相似性
            similarity_match = re.search(r'Average Similarity: ([\d.]+)', content)
            if similarity_match:
                result['similarity'] = float(similarity_match.group(1))

            # 提取Top 1% Recall
            top1_match = re.search(r'Average Top 1% Recall: ([\d.]+)%', content)
            if top1_match:
                result['top1_recall'] = float(top1_match.group(1))

            return result

        except Exception as e:
            print(f"解析基本结果失败: {e}")
            return None

    def _perform_detailed_feature_analysis(self, log_file: str, experiment: Dict) -> Dict:
        """执行详细的特征分析"""
        try:
            with open(log_file, 'r', encoding='utf-8') as f:
                content = f.read()

            analysis = {}

            # 1. 轮廓统计分析
            analysis.update(self._analyze_contour_statistics(content))

            # 2. 轮廓尺寸分布分析
            analysis.update(self._analyze_contour_size_distribution(content))

            # 3. 轮廓几何特征分析
            analysis.update(self._analyze_contour_geometry_features(content))

            # 4. 检索键特征分析
            analysis.update(self._analyze_retrieval_key_features(content))

            # 5. BCI特征分析
            analysis.update(self._analyze_bci_features(content))

            # 6. 相似性检查分析
            analysis.update(self._analyze_similarity_checks(content))

            # 7. 计算综合特征质量指数
            analysis['feature_quality_index'] = self._calculate_feature_quality_index(analysis)

            return analysis

        except Exception as e:
            print(f"详细特征分析失败: {e}")
            return {}

    def _analyze_contour_statistics(self, content: str) -> Dict:
        """分析轮廓统计信息"""
        stats = {
            'total_frames': 0,
            'total_contours': 0,
            'avg_contours_per_frame': 0.0,
            'std_contours_per_frame': 0.0,
            'min_contours_per_frame': 0,
            'max_contours_per_frame': 0
        }

        try:
            # 提取每帧轮廓数量
            frame_contours = re.findall(r'\[DB \d+\] Frame \d+: 总轮廓数: (\d+)', content)

            if frame_contours:
                contour_counts = [int(x) for x in frame_contours]

                stats['total_frames'] = len(contour_counts)
                stats['total_contours'] = sum(contour_counts)
                stats['avg_contours_per_frame'] = statistics.mean(contour_counts)
                stats['std_contours_per_frame'] = statistics.stdev(contour_counts) if len(contour_counts) > 1 else 0
                stats['min_contours_per_frame'] = min(contour_counts)
                stats['max_contours_per_frame'] = max(contour_counts)

        except Exception as e:
            print(f"轮廓统计分析失败: {e}")

        return stats

    def _analyze_contour_size_distribution(self, content: str) -> Dict:
        """分析轮廓尺寸分布"""
        size_dist = {}

        try:
            # 初始化各尺寸区间的计数
            for min_size, max_size, label in self.contour_size_bins:
                size_dist[f'contours_{label}_count'] = 0
                size_dist[f'contours_{label}_ratio'] = 0.0

            # 提取轮廓尺寸信息
            contour_sizes_match = re.search(r'Contour sizes: ([\d,]+)', content)
            if contour_sizes_match:
                sizes_str = contour_sizes_match.group(1)
                sizes = [int(x) for x in sizes_str.split(',') if x.strip()]

                total_contours = len(sizes)
                if total_contours > 0:
                    for min_size, max_size, label in self.contour_size_bins:
                        if max_size == float('inf'):
                            count = sum(1 for s in sizes if s >= min_size)
                        else:
                            count = sum(1 for s in sizes if min_size <= s <= max_size)

                        size_dist[f'contours_{label}_count'] = count
                        size_dist[f'contours_{label}_ratio'] = count / total_contours

        except Exception as e:
            print(f"轮廓尺寸分布分析失败: {e}")

        return size_dist

    def _analyze_contour_geometry_features(self, content: str) -> Dict:
        """分析轮廓几何特征"""
        geometry = {
            'avg_eccentricity': 0.0,
            'std_eccentricity': 0.0,
            'significant_ecc_features_ratio': 0.0,
            'significant_com_features_ratio': 0.0,
            'avg_eigenvalue_ratio': 0.0,
            'avg_contour_height': 0.0
        }

        try:
            # 提取偏心率信息
            ecc_match = re.search(r'Eccentricities: ([\d.,]+)', content)
            if ecc_match:
                ecc_str = ecc_match.group(1)
                eccentricities = [float(x) for x in ecc_str.split(',') if x.strip()]

                if eccentricities:
                    geometry['avg_eccentricity'] = statistics.mean(eccentricities)
                    geometry['std_eccentricity'] = statistics.stdev(eccentricities) if len(eccentricities) > 1 else 0

                    # 计算各偏心率区间的分布
                    for min_ecc, max_ecc, label in self.eccentricity_bins:
                        count = sum(1 for e in eccentricities if min_ecc <= e < max_ecc)
                        geometry[f'{label}_ratio'] = count / len(eccentricities)

            # 提取特征值比例信息
            eigenratio_match = re.search(r'Eigenvalue ratios: ([\d.,]+)', content)
            if eigenratio_match:
                ratios_str = eigenratio_match.group(1)
                ratios = [float(x) for x in ratios_str.split(',') if x.strip()]

                if ratios:
                    geometry['avg_eigenvalue_ratio'] = statistics.mean(ratios)

            # 提取显著特征比例
            significant_features = re.search(r'Significant features: ecc=(\d+), com=(\d+), total=(\d+)', content)
            if significant_features:
                ecc_count, com_count, total_count = map(int, significant_features.groups())
                if total_count > 0:
                    geometry['significant_ecc_features_ratio'] = ecc_count / total_count
                    geometry['significant_com_features_ratio'] = com_count / total_count

            # 提取高度信息
            heights_match = re.search(r'Contour heights: ([\d.,]+)', content)
            if heights_match:
                heights_str = heights_match.group(1)
                heights = [float(x) for x in heights_str.split(',') if x.strip()]

                if heights:
                    geometry['avg_contour_height'] = statistics.mean(heights)

        except Exception as e:
            print(f"轮廓几何特征分析失败: {e}")

        return geometry

    def _analyze_retrieval_key_features(self, content: str) -> Dict:
        """分析检索键特征"""
        key_features = {
            'avg_key_dimension_0': 0.0,
            'avg_key_dimension_1': 0.0,
            'avg_key_dimension_2': 0.0,
            'key_sparsity_ratio': 0.0,
            'key_distinctiveness': 0.0,
            'ring_feature_activation': 0.0
        }

        try:
            # 提取键的各维度统计
            key_dim0_match = re.search(r'Key dimension 0: avg=([\d.]+)', content)
            if key_dim0_match:
                key_features['avg_key_dimension_0'] = float(key_dim0_match.group(1))

            key_dim1_match = re.search(r'Key dimension 1: avg=([\d.]+)', content)
            if key_dim1_match:
                key_features['avg_key_dimension_1'] = float(key_dim1_match.group(1))

            key_dim2_match = re.search(r'Key dimension 2: avg=([\d.]+)', content)
            if key_dim2_match:
                key_features['avg_key_dimension_2'] = float(key_dim2_match.group(1))

            # 提取稀疏性信息
            sparsity_match = re.search(r'Key sparsity: ([\d.]+)', content)
            if sparsity_match:
                key_features['key_sparsity_ratio'] = float(sparsity_match.group(1))

            # 提取环形特征激活率
            ring_activation_match = re.search(r'Ring feature activation: ([\d.]+)', content)
            if ring_activation_match:
                key_features['ring_feature_activation'] = float(ring_activation_match.group(1))

        except Exception as e:
            print(f"检索键特征分析失败: {e}")

        return key_features

    def _analyze_bci_features(self, content: str) -> Dict:
        """分析BCI特征"""
        bci_features = {
            'avg_neighbors_per_bci': 0.0,
            'std_neighbors_per_bci': 0.0,
            'max_neighbors_per_bci': 0,
            'avg_neighbor_distance': 0.0,
            'std_neighbor_distance': 0.0,
            'avg_neighbor_angle_diversity': 0.0,
            'cross_layer_connections_ratio': 0.0,
            'distance_bit_activation_rate': 0.0,
            'constellation_complexity': 0.0
        }

        try:
            # 提取BCI邻居数量信息
            neighbors_match = re.search(r'BCI neighbors: ([\d,]+)', content)
            if neighbors_match:
                neighbors_str = neighbors_match.group(1)
                neighbor_counts = [int(x) for x in neighbors_str.split(',') if x.strip()]

                if neighbor_counts:
                    bci_features['avg_neighbors_per_bci'] = statistics.mean(neighbor_counts)
                    bci_features['std_neighbors_per_bci'] = statistics.stdev(neighbor_counts) if len(
                        neighbor_counts) > 1 else 0
                    bci_features['max_neighbors_per_bci'] = max(neighbor_counts)

            # 提取邻居距离信息
            distances_match = re.search(r'Neighbor distances: ([\d.,]+)', content)
            if distances_match:
                distances_str = distances_match.group(1)
                distances = [float(x) for x in distances_str.split(',') if x.strip()]

                if distances:
                    bci_features['avg_neighbor_distance'] = statistics.mean(distances)
                    bci_features['std_neighbor_distance'] = statistics.stdev(distances) if len(distances) > 1 else 0

            # 提取角度信息
            angles_match = re.search(r'Neighbor angles: ([\d.,\-]+)', content)
            if angles_match:
                angles_str = angles_match.group(1)
                angles = [float(x) for x in angles_str.split(',') if x.strip()]

                if angles:
                    # 计算角度多样性（标准差）
                    bci_features['avg_neighbor_angle_diversity'] = statistics.stdev(angles) if len(angles) > 1 else 0

            # 提取跨层连接信息
            cross_layer_match = re.search(r'Cross layer connections: (\d+)/(\d+)', content)
            if cross_layer_match:
                cross_layer_count = int(cross_layer_match.group(1))
                total_connections = int(cross_layer_match.group(2))
                if total_connections > 0:
                    bci_features['cross_layer_connections_ratio'] = cross_layer_count / total_connections

            # 提取距离位激活率
            bit_activation_match = re.search(r'Distance bit activation: ([\d.]+)', content)
            if bit_activation_match:
                bci_features['distance_bit_activation_rate'] = float(bit_activation_match.group(1))

            # 计算星座复杂度（基于平均邻居数和角度多样性）
            avg_neighbors = bci_features['avg_neighbors_per_bci']
            angle_diversity = bci_features['avg_neighbor_angle_diversity']
            if avg_neighbors > 0 and angle_diversity > 0:
                bci_features['constellation_complexity'] = avg_neighbors * angle_diversity / 10.0  # 归一化

        except Exception as e:
            print(f"BCI特征分析失败: {e}")

        return bci_features

    def _analyze_similarity_checks(self, content: str) -> Dict:
        """分析相似性检查统计"""
        similarity_stats = {
            'contour_similarity_pass_rate': 0.0,
            'constellation_similarity_pass_rate': 0.0,
            'pairwise_similarity_pass_rate': 0.0,
            'area_check_fail_rate': 0.0,
            'eigenvalue_check_fail_rate': 0.0,
            'height_check_fail_rate': 0.0,
            'centroid_check_fail_rate': 0.0,
            'geometric_consistency_pass_rate': 0.0
        }

        try:
            # 提取各检查步骤的通过率
            check1_match = re.search(r'After check 1: (\d+)', content)
            check2_match = re.search(r'After check 2: (\d+)', content)
            check3_match = re.search(r'After check 3: (\d+)', content)

            if check1_match and check2_match and check3_match:
                check1_count = int(check1_match.group(1))
                check2_count = int(check2_match.group(1))
                check3_count = int(check3_match.group(1))

                # 假设总查询数为1000（可以根据实际调整）
                total_queries = 1000
                if total_queries > 0:
                    similarity_stats['contour_similarity_pass_rate'] = check1_count / total_queries

                if check1_count > 0:
                    similarity_stats['constellation_similarity_pass_rate'] = check2_count / check1_count

                if check2_count > 0:
                    similarity_stats['pairwise_similarity_pass_rate'] = check3_count / check2_count

            # 提取详细的相似性检查通过率（如果有的话）
            sim_pass_rate_match = re.search(r'Similarity check pass rates: check1=([\d.]+)', content)
            if sim_pass_rate_match:
                similarity_stats['contour_similarity_pass_rate'] = float(sim_pass_rate_match.group(1))

            const_pass_rate_match = re.search(r'Constellation similarity pass rate: ([\d.]+)', content)
            if const_pass_rate_match:
                similarity_stats['constellation_similarity_pass_rate'] = float(const_pass_rate_match.group(1))

            pair_pass_rate_match = re.search(r'Pairwise similarity pass rate: ([\d.]+)', content)
            if pair_pass_rate_match:
                similarity_stats['pairwise_similarity_pass_rate'] = float(pair_pass_rate_match.group(1))

        except Exception as e:
            print(f"相似性检查分析失败: {e}")

        return similarity_stats

    def _calculate_feature_quality_index(self, analysis: Dict) -> float:
        """计算综合特征质量指数"""
        try:
            # 综合多个指标计算特征质量
            quality_factors = []

            # 轮廓质量因子（轮廓数量适中，不过度碎片化）
            avg_contours = analysis.get('avg_contours_per_frame', 0)
            if avg_contours > 0:
                contour_quality = min(1.0, avg_contours / 200.0)  # 200为理想轮廓数
                # 惩罚过多的小轮廓
                small_contour_penalty = analysis.get('contours_小轮廓_ratio', 0) * 0.5
                contour_quality = max(0.1, contour_quality - small_contour_penalty)
                quality_factors.append(contour_quality)

            # 几何特征质量因子（偏心率和特征值比例的合理性）
            avg_ecc = analysis.get('avg_eccentricity', 0)
            if avg_ecc > 0:
                # 适度的偏心率表示良好的结构特征
                ecc_quality = 1.0 - abs(avg_ecc - 0.5) * 2  # 0.5为理想偏心率
                ecc_quality = max(0.1, ecc_quality)
                quality_factors.append(ecc_quality)

            # 检索键质量因子（非稀疏且有区分度）
            key_sparsity = analysis.get('key_sparsity_ratio', 1.0)
            if key_sparsity < 1.0:
                key_quality = 1.0 - key_sparsity
                quality_factors.append(key_quality)

            # BCI连接质量因子（适度的邻居数量）
            avg_neighbors = analysis.get('avg_neighbors_per_bci', 0)
            if avg_neighbors > 0:
                # 3-8个邻居为理想范围
                if 3 <= avg_neighbors <= 8:
                    bci_quality = 1.0
                elif avg_neighbors < 3:
                    bci_quality = avg_neighbors / 3.0
                else:
                    bci_quality = max(0.1, 1.0 - (avg_neighbors - 8) * 0.1)
                quality_factors.append(bci_quality)

            # 相似性检查质量因子（合理的通过率）
            check_pass_rate = analysis.get('constellation_similarity_pass_rate', 0)
            if check_pass_rate > 0:
                # 10-50%的通过率为合理范围
                if 0.1 <= check_pass_rate <= 0.5:
                    check_quality = 1.0
                elif check_pass_rate < 0.1:
                    check_quality = check_pass_rate / 0.1
                else:
                    check_quality = max(0.1, 1.0 - (check_pass_rate - 0.5) * 2)
                quality_factors.append(check_quality)

            # 计算综合质量指数
            if quality_factors:
                feature_quality_index = statistics.mean(quality_factors)
            else:
                feature_quality_index = 0.0

            return round(feature_quality_index, 4)

        except Exception as e:
            print(f"计算特征质量指数失败: {e}")
            return 0.0

    def _append_analysis_info_to_log(self, log_file: str, experiment: Dict,
                                     duration: float, analysis: Dict):
        """在日志文件末尾添加分析信息"""
        try:
            with open(log_file, 'a', encoding='utf-8') as f:
                f.write(f"\n{'=' * 80}\n")
                f.write(f"综合特征分析报告\n")
                f.write(f"{'=' * 80}\n")
                f.write(f"实验配置: {experiment['name']} ({experiment['layers_count']}层)\n")
                f.write(f"层间隔: {experiment['layer_interval']:.3f}m\n")
                f.write(f"运行时长: {duration:.2f}秒\n")
                f.write(f"特征质量指数: {analysis.get('feature_quality_index', 0):.4f}\n")
                f.write(f"完成时间: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")

                # 轮廓统计
                f.write("轮廓统计分析:\n")
                f.write("-" * 40 + "\n")
                f.write(f"总帧数: {analysis.get('total_frames', 0)}\n")
                f.write(f"总轮廓数: {analysis.get('total_contours', 0)}\n")
                f.write(f"平均每帧轮廓数: {analysis.get('avg_contours_per_frame', 0):.1f}\n")
                f.write(f"轮廓数标准差: {analysis.get('std_contours_per_frame', 0):.1f}\n")
                f.write(f"最少轮廓数: {analysis.get('min_contours_per_frame', 0)}\n")
                f.write(f"最多轮廓数: {analysis.get('max_contours_per_frame', 0)}\n\n")

                # 轮廓尺寸分布
                f.write("轮廓尺寸分布:\n")
                f.write("-" * 40 + "\n")
                for min_size, max_size, label in self.contour_size_bins:
                    count = analysis.get(f'contours_{label}_count', 0)
                    ratio = analysis.get(f'contours_{label}_ratio', 0)
                    if max_size == float('inf'):
                        size_range = f"≥{min_size}像素"
                    else:
                        size_range = f"{min_size}-{max_size}像素"
                    f.write(f"{label} ({size_range}): {count}个 ({ratio:.1%})\n")
                f.write("\n")

                # 几何特征
                f.write("轮廓几何特征:\n")
                f.write("-" * 40 + "\n")
                f.write(f"平均偏心率: {analysis.get('avg_eccentricity', 0):.3f}\n")
                f.write(f"偏心率标准差: {analysis.get('std_eccentricity', 0):.3f}\n")
                f.write(f"显著偏心率特征比例: {analysis.get('significant_ecc_features_ratio', 0):.1%}\n")
                f.write(f"显著质心特征比例: {analysis.get('significant_com_features_ratio', 0):.1%}\n")
                f.write(f"平均特征值比例: {analysis.get('avg_eigenvalue_ratio', 0):.3f}\n")
                for min_ecc, max_ecc, label in self.eccentricity_bins:
                    ratio = analysis.get(f'{label}_ratio', 0)
                    f.write(f"{label} ({min_ecc:.1f}-{max_ecc:.1f}): {ratio:.1%}\n")
                f.write("\n")

                # 检索键特征
                f.write("检索键特征:\n")
                f.write("-" * 40 + "\n")
                f.write(f"主特征值*面积 平均值: {analysis.get('avg_key_dimension_0', 0):.2f}\n")
                f.write(f"次特征值*面积 平均值: {analysis.get('avg_key_dimension_1', 0):.2f}\n")
                f.write(f"累积面积 平均值: {analysis.get('avg_key_dimension_2', 0):.2f}\n")
                f.write(f"键稀疏性比例: {analysis.get('key_sparsity_ratio', 0):.1%}\n")
                f.write(f"键区分度: {analysis.get('key_distinctiveness', 0):.3f}\n")
                f.write(f"环形特征激活率: {analysis.get('ring_feature_activation', 0):.1%}\n\n")

                # BCI特征
                f.write("BCI特征:\n")
                f.write("-" * 40 + "\n")
                f.write(f"平均邻居数: {analysis.get('avg_neighbors_per_bci', 0):.1f}\n")
                f.write(f"邻居数标准差: {analysis.get('std_neighbors_per_bci', 0):.1f}\n")
                f.write(f"最大邻居数: {analysis.get('max_neighbors_per_bci', 0)}\n")
                f.write(f"平均邻居距离: {analysis.get('avg_neighbor_distance', 0):.2f}\n")
                f.write(f"邻居距离标准差: {analysis.get('std_neighbor_distance', 0):.2f}\n")
                f.write(f"跨层连接比例: {analysis.get('cross_layer_connections_ratio', 0):.1%}\n")
                f.write(f"距离位激活率: {analysis.get('distance_bit_activation_rate', 0):.1%}\n")
                f.write(f"星座复杂度: {analysis.get('constellation_complexity', 0):.3f}\n\n")

                # 相似性检查
                f.write("相似性检查统计:\n")
                f.write("-" * 40 + "\n")
                f.write(f"轮廓相似性通过率: {analysis.get('contour_similarity_pass_rate', 0):.1%}\n")
                f.write(f"星座相似性通过率: {analysis.get('constellation_similarity_pass_rate', 0):.1%}\n")
                f.write(f"成对相似性通过率: {analysis.get('pairwise_similarity_pass_rate', 0):.1%}\n")
                f.write(f"面积检查失败率: {analysis.get('area_check_fail_rate', 0):.1%}\n")
                f.write(f"特征值检查失败率: {analysis.get('eigenvalue_check_fail_rate', 0):.1%}\n")
                f.write(f"高度检查失败率: {analysis.get('height_check_fail_rate', 0):.1%}\n")
                f.write(f"质心检查失败率: {analysis.get('centroid_check_fail_rate', 0):.1%}\n")

                f.write(f"{'=' * 80}\n")

        except Exception as e:
            print(f"添加分析信息失败: {e}")

    def run_all_experiments(self):
        """运行所有实验"""
        print("开始综合分层数量分析实验")
        print(f"实验范围: {len(self.experiments)} 个配置 (1-20层)")
        print(f"预计总时长: {len(self.experiments) * 15} 分钟")

        # 备份原始文件
        self.backup_files()

        try:
            # 运行所有实验
            for i, experiment in enumerate(self.experiments, 1):
                print(f"\n总进度: {i}/{len(self.experiments)}")

                result = self.run_single_experiment(experiment)
                if result:
                    self.results.append(result)
                else:
                    print(f"实验 {experiment['name']} 失败，跳过")

            # 生成综合分析报告
            self.generate_comprehensive_report()

        finally:
            # 恢复原始文件
            self.restore_files()

    def generate_comprehensive_report(self):
        """生成综合分析报告"""
        if not self.results:
            print("没有成功的实验结果")
            return

        # 创建综合报告文件
        report_file = "comprehensive_layers_analysis_report.txt"

        with open(report_file, 'w', encoding='utf-8') as f:
            f.write("综合分层数量分析实验报告\n")
            f.write("=" * 100 + "\n")
            f.write(f"生成时间: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"实验数量: {len(self.results)}\n")
            f.write(f"层数范围: 1-20层\n")
            f.write(f"分析目的: 深入理解层数变化对特征提取和召回率的影响\n\n")

            # 基本性能表格
            f.write("基本性能结果:\n")
            f.write("-" * 120 + "\n")
            header = (f"{'层数':<4} {'层间隔':<8} {'Recall@1':<10} {'相似性':<8} {'运行时长':<8} "
                      f"{'平均轮廓':<8} {'特征质量':<8} {'小轮廓比例':<10} {'BCI邻居':<8}\n")
            f.write(header)
            f.write("-" * 120 + "\n")

            # 按层数排序
            sorted_results = sorted(self.results, key=lambda x: x['layers_count'])

            for result in sorted_results:
                line = (f"{result['layers_count']:<4d} {result['layer_interval']:<8.3f} "
                        f"{result['recall_1']:<10.2f} {result['similarity']:<8.4f} "
                        f"{result['duration']:<8.0f} {result.get('avg_contours_per_frame', 0):<8.1f} "
                        f"{result.get('feature_quality_index', 0):<8.3f} "
                        f"{result.get('contours_小轮廓_ratio', 0):<10.2f} "
                        f"{result.get('avg_neighbors_per_bci', 0):<8.1f}\n")
                f.write(line)

            f.write("-" * 120 + "\n\n")

            # 性能趋势分析
            f.write("性能趋势分析:\n")
            f.write("-" * 50 + "\n")

            recall_values = [r['recall_1'] for r in sorted_results]
            if recall_values:
                max_recall = max(recall_values)
                max_idx = recall_values.index(max_recall)
                optimal_layers = sorted_results[max_idx]['layers_count']

                f.write(f"最高召回率: {max_recall:.2f}% (在{optimal_layers}层时达到)\n")
                f.write(f"最优层间隔: {sorted_results[max_idx]['layer_interval']:.3f}m\n")

                # 寻找性能下降点
                if len(recall_values) >= optimal_layers:
                    decline_threshold = max_recall * 0.95  # 95%阈值
                    decline_start = -1

                    for i in range(max_idx, len(recall_values)):
                        if recall_values[i] < decline_threshold:
                            decline_start = i
                            break

                    if decline_start != -1:
                        f.write(f"性能开始下降: 从{sorted_results[decline_start]['layers_count']}层开始\n")
                        f.write(f"过分割阈值: 约{sorted_results[decline_start]['layer_interval']:.3f}m层间隔\n")
                    else:
                        f.write("在测试范围内未观察到明显性能下降\n")
                        f.write("建议扩展到更多层数以找到过分割阈值\n")

            f.write("\n")

            # 特征质量分析
            f.write("特征质量分析:\n")
            f.write("-" * 50 + "\n")

            quality_values = [r.get('feature_quality_index', 0) for r in sorted_results]
            if quality_values:
                avg_quality = statistics.mean(quality_values)
                f.write(f"平均特征质量指数: {avg_quality:.3f}\n")

                # 找出特征质量最高的配置
                max_quality = max(quality_values)
                max_quality_idx = quality_values.index(max_quality)
                f.write(f"最高特征质量: {max_quality:.3f} (在{sorted_results[max_quality_idx]['layers_count']}层时)\n")

            # 轮廓碎片化分析
            f.write("\n轮廓碎片化分析:\n")
            f.write("-" * 50 + "\n")

            small_contour_ratios = [r.get('contours_小轮廓_ratio', 0) for r in sorted_results]
            avg_contour_counts = [r.get('avg_contours_per_frame', 0) for r in sorted_results]

            if small_contour_ratios and avg_contour_counts:
                # 分析轮廓碎片化趋势
                f.write("层数 -> 小轮廓比例 -> 平均轮廓数 的关系:\n")
                for i, result in enumerate(sorted_results):
                    if i % 3 == 0:  # 每3层显示一次
                        f.write(f"{result['layers_count']:2d}层: "
                                f"小轮廓比例={small_contour_ratios[i]:.1%}, "
                                f"平均轮廓数={avg_contour_counts[i]:.1f}\n")

                # 计算碎片化趋势
                if len(small_contour_ratios) > 5:
                    early_avg = statistics.mean(small_contour_ratios[:5])
                    late_avg = statistics.mean(small_contour_ratios[-5:])
                    f.write(f"\n碎片化趋势: 前5层平均{early_avg:.1%} -> 后5层平均{late_avg:.1%}\n")
                    if late_avg > early_avg * 1.2:
                        f.write("结论: 存在明显的轮廓碎片化趋势\n")
                    else:
                        f.write("结论: 轮廓碎片化不明显\n")

            # BCI复杂度分析
            f.write("\nBCI复杂度分析:\n")
            f.write("-" * 50 + "\n")

            neighbor_counts = [r.get('avg_neighbors_per_bci', 0) for r in sorted_results]
            if neighbor_counts:
                f.write("BCI邻居数随层数变化:\n")
                for i, result in enumerate(sorted_results):
                    if i % 4 == 0:  # 每4层显示一次
                        f.write(f"{result['layers_count']:2d}层: 平均邻居数={neighbor_counts[i]:.1f}\n")

                # 分析复杂度趋势
                if len(neighbor_counts) > 5:
                    early_neighbors = statistics.mean(neighbor_counts[:5])
                    late_neighbors = statistics.mean(neighbor_counts[-5:])
                    f.write(f"\n复杂度变化: 前5层平均{early_neighbors:.1f}个邻居 -> "
                            f"后5层平均{late_neighbors:.1f}个邻居\n")

            # 因果关系分析
            f.write("\n因果关系分析:\n")
            f.write("=" * 60 + "\n")

            # 建立特征指标与性能的相关性
            if len(sorted_results) >= 10:
                f.write("特征指标与召回率的关系分析:\n\n")

                # 层间隔 vs 召回率
                intervals = [r['layer_interval'] for r in sorted_results]
                recalls = [r['recall_1'] for r in sorted_results]

                f.write("1. 层间隔与召回率关系:\n")
                f.write("   最优层间隔区间: ")

                # 找出召回率前25%的层间隔范围
                top_25_percent_count = max(1, len(recalls) // 4)
                top_indices = sorted(range(len(recalls)), key=lambda i: recalls[i], reverse=True)[:top_25_percent_count]
                top_intervals = [intervals[i] for i in top_indices]

                if top_intervals:
                    f.write(f"{min(top_intervals):.3f}m - {max(top_intervals):.3f}m\n")
                    f.write(f"   对应层数范围: {min([sorted_results[i]['layers_count'] for i in top_indices])}-"
                            f"{max([sorted_results[i]['layers_count'] for i in top_indices])}层\n")

                # 轮廓质量 vs 召回率
                f.write("\n2. 轮廓碎片化对性能的影响:\n")
                avg_contours = [r.get('avg_contours_per_frame', 0) for r in sorted_results]

                # 找出轮廓数适中且召回率高的配置
                moderate_contour_configs = []
                for i, result in enumerate(sorted_results):
                    contour_count = avg_contours[i]
                    if 50 <= contour_count <= 200:  # 适中的轮廓数量
                        moderate_contour_configs.append((result['layers_count'], recalls[i], contour_count))

                if moderate_contour_configs:
                    best_moderate = max(moderate_contour_configs, key=lambda x: x[1])
                    f.write(f"   轮廓数适中时的最佳性能: {best_moderate[0]}层 "
                            f"({best_moderate[2]:.1f}个轮廓, {best_moderate[1]:.1f}%召回率)\n")

                # BCI复杂度 vs 召回率
                f.write("\n3. BCI复杂度对性能的影响:\n")
                bci_neighbors = [r.get('avg_neighbors_per_bci', 0) for r in sorted_results]

                # 分析理想的邻居数量范围
                ideal_neighbor_configs = []
                for i, result in enumerate(sorted_results):
                    neighbor_count = bci_neighbors[i]
                    if 3 <= neighbor_count <= 8:  # 理想的邻居数量
                        ideal_neighbor_configs.append((result['layers_count'], recalls[i], neighbor_count))

                if ideal_neighbor_configs:
                    best_ideal = max(ideal_neighbor_configs, key=lambda x: x[1])
                    f.write(f"   BCI邻居数适中时的最佳性能: {best_ideal[0]}层 "
                            f"({best_ideal[2]:.1f}个邻居, {best_ideal[1]:.1f}%召回率)\n")

            # 推荐建议
            f.write("\n推荐建议:\n")
            f.write("=" * 40 + "\n")

            if sorted_results:
                # 基于综合分析的推荐
                best_overall = max(sorted_results, key=lambda x: x['recall_1'])
                best_quality = max(sorted_results, key=lambda x: x.get('feature_quality_index', 0))
                best_efficiency = max(sorted_results, key=lambda x: x['recall_1'] / x['duration'])

                f.write(f"1. 最佳性能配置: {best_overall['layers_count']}层 ")
                f.write(f"(召回率: {best_overall['recall_1']:.1f}%)\n")

                f.write(f"2. 最佳特征质量配置: {best_quality['layers_count']}层 ")
                f.write(f"(质量指数: {best_quality.get('feature_quality_index', 0):.3f})\n")

                f.write(f"3. 最高效率配置: {best_efficiency['layers_count']}层 ")
                f.write(f"(效率比: {best_efficiency['recall_1'] / best_efficiency['duration']:.4f})\n")

                # 实际应用建议
                f.write("\n实际应用建议:\n")

                if len(sorted_results) >= 15:
                    # 如果测试了足够多的层数
                    high_perf_configs = [r for r in sorted_results if r['recall_1'] >= max(recall_values) * 0.95]
                    if high_perf_configs:
                        recommended = min(high_perf_configs, key=lambda x: x['duration'])
                        f.write(f"- 推荐配置: {recommended['layers_count']}层 ")
                        f.write(f"(平衡性能{recommended['recall_1']:.1f}%和效率{recommended['duration']:.0f}s)\n")
                else:
                    f.write("- 建议扩展测试范围到更多层数以获得更全面的分析\n")

        print(f"\n综合分析报告已保存到: {report_file}")

        # 控制台输出关键发现
        self._print_key_findings()

    def _print_key_findings(self):
        """输出关键发现"""
        print("\n" + "=" * 80)
        print("关键发现总结")
        print("=" * 80)

        if not self.results:
            return

        sorted_results = sorted(self.results, key=lambda x: x['layers_count'])
        recall_values = [r['recall_1'] for r in sorted_results]

        # 找出最佳配置
        if recall_values:
            max_recall = max(recall_values)
            max_idx = recall_values.index(max_recall)
            optimal_config = sorted_results[max_idx]

            print(f"🎯 最佳性能: {optimal_config['layers_count']}层")
            print(f"   召回率: {max_recall:.2f}%")
            print(f"   层间隔: {optimal_config['layer_interval']:.3f}m")
            print(f"   特征质量: {optimal_config.get('feature_quality_index', 0):.3f}")

        # 分析性能趋势
        if len(recall_values) >= 10:
            # 寻找性能峰值和下降点
            peak_found = False
            decline_found = False

            for i in range(1, len(recall_values) - 1):
                # 寻找局部峰值
                if not peak_found and recall_values[i] > recall_values[i - 1] and recall_values[i] > recall_values[
                    i + 1]:
                    peak_layers = sorted_results[i]['layers_count']
                    peak_recall = recall_values[i]
                    print(f"📈 性能峰值: {peak_layers}层 ({peak_recall:.2f}%)")
                    peak_found = True

            # 寻找明显下降点
            threshold = max_recall * 0.9  # 90%阈值
            for i in range(max_idx + 1, len(recall_values)):
                if recall_values[i] < threshold:
                    decline_layers = sorted_results[i]['layers_count']
                    decline_recall = recall_values[i]
                    print(f"📉 性能下降点: {decline_layers}层 ({decline_recall:.2f}%)")
                    print(f"   过分割阈值: 约{sorted_results[i]['layer_interval']:.3f}m")
                    decline_found = True
                    break

            if not decline_found:
                print("⚠️  在测试范围内未发现明显的过分割现象")
                print("   建议扩展到更多层数 (如25-30层) 来找到性能下降点")

        # 轮廓碎片化分析
        small_ratios = [r.get('contours_小轮廓_ratio', 0) for r in sorted_results]
        if small_ratios:
            if len(small_ratios) >= 5:
                early_frag = statistics.mean(small_ratios[:5])
                late_frag = statistics.mean(small_ratios[-5:])

                if late_frag > early_frag * 1.5:
                    print(f"🔍 轮廓碎片化趋势: {early_frag:.1%} -> {late_frag:.1%}")
                    print("   证实了过分割导致轮廓碎片化的假设")

        print(f"\n详细报告: comprehensive_layers_analysis_report.txt")


def main():
    """主函数"""
    print("开始综合分层数量分析实验")
    print("分析目标: 深入理解层数变化对特征提取和召回率的因果影响")
    print("实验范围: 1-20层完整测试")

    # 检查必要文件
    experiment = ComprehensiveLayersAnalysisExperiment()

    if not os.path.exists(experiment.base_script):
        print(f"错误: 找不到主程序文件 {experiment.base_script}")
        return

    if not os.path.exists(experiment.base_types):
        print(f"错误: 找不到类型定义文件 {experiment.base_types}")
        return

    print(f"\n实验设计:")
    print(f"  层数范围: 1-20层")
    print(f"  实验数量: {len(experiment.experiments)}")
    print(f"  分析维度: 轮廓统计、几何特征、检索键、BCI特征、相似性检查")
    print(f"  预计时长: {len(experiment.experiments) * 15} 分钟")

    # 询问用户确认
    confirm = input(f"\n继续运行综合分析实验? (y/n): ")
    if confirm.lower() != 'y':
        print("实验已取消")
        return

    # 运行所有实验
    experiment.run_all_experiments()

    print("\n所有综合分析实验完成!")
    print("请查看生成的详细报告了解层数对特征提取的具体影响")


if __name__ == "__main__":
    main()
