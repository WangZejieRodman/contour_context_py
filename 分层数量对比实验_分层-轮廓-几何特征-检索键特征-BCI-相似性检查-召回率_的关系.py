#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
分层数量消融实验 - subprocess分离式
每层实验都启动独立Python进程，彻底解决模块缓存和配置一致性问题
"""

import os
import sys
import time
import re
import subprocess
import shutil
import statistics
import json
from typing import List, Dict, Tuple, Any, Optional
import math


class SubprocessLayersExperiment:
    """分层数量消融实验 - 子进程分离式"""

    def __init__(self):
        # 检查必要文件
        self.main_script = "contour_chilean_场景识别_不相同时段100-190_不加旋转.py"
        self.types_file = "contour_types.py"
        self.manager_file = "contour_manager_区间分割_垂直结构复杂度.py"

        self.results = []

        # 轮廓大小分桶策略
        self.contour_size_bins = [
            (1, 5, "极小轮廓"),
            (6, 15, "小轮廓"),
            (16, 50, "中小轮廓"),
            (51, 150, "中等轮廓"),
            (151, 500, "大轮廓"),
            (501, float('inf'), "超大轮廓")
        ]

        # 偏心率分桶
        self.eccentricity_bins = [
            (0.0, 0.3, "近圆形"),
            (0.3, 0.6, "椭圆形"),
            (0.6, 0.8, "长椭圆"),
            (0.8, 1.0, "极长椭圆")
        ]

    def check_required_files(self) -> bool:
        """检查必要文件是否存在"""
        files_to_check = [
            self.main_script,
            self.types_file,
            self.manager_file,
            'chilean_NoRot_NoScale_5cm_evaluation_database_100_100.pickle',
            'chilean_NoRot_NoScale_5cm_evaluation_query_190_190.pickle'
        ]

        missing_files = []
        for file in files_to_check:
            if not os.path.exists(file):
                missing_files.append(file)

        if missing_files:
            print("错误：缺少以下必要文件：")
            for file in missing_files:
                print(f"  - {file}")
            return False

        return True

    def generate_layer_configs(self, num_layers: int) -> Dict:
        """生成指定层数的配置"""
        if num_layers == 1:
            lv_grads = [0.0, 5.0]
            q_levels = [0]
            dist_layers = [0]
            weights = [1.0]
        else:
            # 均匀分割0-5m
            lv_grads = []
            for i in range(num_layers + 1):
                height = 5.0 * i / num_layers
                lv_grads.append(round(height, 3))

            q_levels = list(range(num_layers))
            dist_layers = list(range(num_layers))

            # 生成权重： 简单高斯分布，中间层级权重更高
            center = (num_layers - 1) / 2
            weights = []
            for i in range(num_layers):
                # 简单高斯权重
                weight = math.exp(-0.5 * ((i - center) / (num_layers / 4)) ** 2)
                weights.append(weight)

            # 归一化权重
            total_weight = sum(weights)
            weights = [w / total_weight for w in weights]

        return {
            'num_layers': num_layers,
            'lv_grads': lv_grads,
            'q_levels': q_levels,
            'dist_layers': dist_layers,
            'weights': weights,
            'layer_interval': 5.0 / num_layers
        }

    def backup_files(self):
        """备份原始文件"""
        print("备份原始文件...")
        backup_files = [self.main_script, self.types_file, self.manager_file]

        for file in backup_files:
            if os.path.exists(file):
                backup_name = f"{file}.backup"
                if not os.path.exists(backup_name):  # 避免重复备份
                    shutil.copy2(file, backup_name)
                    print(f"  已备份: {file} -> {backup_name}")

    def restore_files(self):
        """恢复原始文件"""
        print("恢复原始文件...")
        backup_files = [self.main_script, self.types_file, self.manager_file]

        for file in backup_files:
            backup_name = f"{file}.backup"
            if os.path.exists(backup_name):
                shutil.copy2(backup_name, file)
                os.remove(backup_name)
                print(f"  已恢复: {backup_name} -> {file}")

    def modify_config_files(self, layer_config: Dict):
        """修改配置文件"""
        num_layers = layer_config['num_layers']
        print(f"修改配置文件为: {num_layers}层配置")

        # 1. 修改contour_types.py
        self._modify_contour_types(layer_config)

        # 2. 修改主程序文件
        self._modify_main_script(layer_config)

        # 3. 修改轮廓管理器文件（如果需要）
        self._modify_manager_script(layer_config)

    def _modify_contour_types(self, layer_config: Dict):
        """修改contour_types.py"""
        with open(self.types_file, 'r', encoding='utf-8') as f:
            content = f.read()

        # 修改DIST_BIN_LAYERS
        dist_layers_str = str(layer_config['dist_layers'])
        content = re.sub(
            r'DIST_BIN_LAYERS = \[.*?\]',
            f'DIST_BIN_LAYERS = {dist_layers_str}',
            content
        )

        # 修改LAYER_AREA_WEIGHTS
        weights_str = '[' + ', '.join([f'{w:.4f}' for w in layer_config['weights']]) + ']'
        content = re.sub(
            r'LAYER_AREA_WEIGHTS = \[.*?\]',
            f'LAYER_AREA_WEIGHTS = {weights_str}',
            content
        )

        # 修改NUM_BIN_KEY_LAYER
        num_bin_layers = len(layer_config['dist_layers'])
        content = re.sub(
            r'NUM_BIN_KEY_LAYER = \d+',
            f'NUM_BIN_KEY_LAYER = {num_bin_layers}',
            content
        )

        with open(self.types_file, 'w', encoding='utf-8') as f:
            f.write(content)

    def _modify_main_script(self, layer_config: Dict):
        """修改主程序文件"""
        with open(self.main_script, 'r', encoding='utf-8') as f:
            content = f.read()

        # 修改lv_grads
        lv_grads_str = '[' + ', '.join([f'{g:.3f}' for g in layer_config['lv_grads']]) + ']'
        content = re.sub(
            r'config\.lv_grads = \[.*?\]',
            f'config.lv_grads = {lv_grads_str}',
            content
        )

        # 修改q_levels
        q_levels_str = str(layer_config['q_levels'])
        content = re.sub(
            r'config\.q_levels = \[.*?\]',
            f'config.q_levels = {q_levels_str}',
            content
        )

        with open(self.main_script, 'w', encoding='utf-8') as f:
            f.write(content)

    def _modify_manager_script(self, layer_config: Dict):
        """修改轮廓管理器文件（如果需要特殊处理）"""
        # 大部分情况下不需要修改manager文件
        # 如果有特殊需求可以在这里添加
        pass

    def run_single_layer_experiment(self, num_layers: int) -> Optional[Dict]:
        """运行单层实验"""
        print(f"\n{'=' * 80}")
        print(f"开始实验: {num_layers}层配置")
        print(f"层间距: {5.0 / num_layers:.3f}m")
        print(f"{'=' * 80}")

        start_time = time.time()

        try:
            # 1. 生成层级配置
            layer_config = self.generate_layer_configs(num_layers)

            # 2. 修改配置文件
            self.modify_config_files(layer_config)

            # 3. 设置日志文件
            log_file = f"layers_{num_layers}_experiment.log"

            # 4. 启动独立Python进程运行实验
            print(f"启动独立进程运行{num_layers}层实验...")

            # 构建命令行参数
            cmd = [
                sys.executable, self.main_script,
                "--log_file", log_file
            ]

            # 运行实验
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                encoding='utf-8',
                errors='replace'
            )

            end_time = time.time()
            duration = end_time - start_time

            # 5. 检查执行结果
            if result.returncode != 0:
                print(f"实验{num_layers}层执行失败 (返回码: {result.returncode})")
                print(f"STDERR: {result.stderr[:1000]}")  # 只显示前1000字符
                return None

            # 6. 解析结果
            experiment_result = self._parse_experiment_results(
                log_file, layer_config, duration
            )

            if experiment_result:
                print(f"实验{num_layers}层成功完成")
                print(f"Recall@1: {experiment_result.get('recall_1', 0):.2f}%")
                print(f"运行时长: {duration:.1f}秒")
                print(f"平均轮廓数: {experiment_result.get('avg_contours_per_frame', 0):.1f}")
                return experiment_result
            else:
                print(f"实验{num_layers}层结果解析失败")
                return None

        except subprocess.TimeoutExpired:
            print(f"实验{num_layers}层超时")
            return None
        except Exception as e:
            print(f"实验{num_layers}层异常: {e}")
            return None

    def _parse_experiment_results(self, log_file: str, layer_config: Dict,
                                  duration: float) -> Optional[Dict]:
        """解析实验结果"""
        if not os.path.exists(log_file):
            print(f"警告: 日志文件不存在: {log_file}")
            return None

        try:
            with open(log_file, 'r', encoding='utf-8') as f:
                content = f.read()

            result = {
                'num_layers': layer_config['num_layers'],
                'layer_interval': layer_config['layer_interval'],
                'duration': duration,
                'log_file': log_file
            }

            # 解析基本性能指标
            recall_1_match = re.search(r'Average Recall @1: ([\d.]+)%', content)
            if recall_1_match:
                result['recall_1'] = float(recall_1_match.group(1))
            else:
                result['recall_1'] = 0.0

            recall_5_match = re.search(r'Average Recall @5: ([\d.]+)%', content)
            result['recall_5'] = float(recall_5_match.group(1)) if recall_5_match else 0.0

            recall_10_match = re.search(r'Average Recall @10: ([\d.]+)%', content)
            result['recall_10'] = float(recall_10_match.group(1)) if recall_10_match else 0.0

            recall_25_match = re.search(r'Average Recall @25: ([\d.]+)%', content)
            result['recall_25'] = float(recall_25_match.group(1)) if recall_25_match else 0.0

            similarity_match = re.search(r'Average Similarity: ([\d.]+)', content)
            result['average_similarity'] = float(similarity_match.group(1)) if similarity_match else 0.0

            top1_match = re.search(r'Average Top 1% Recall: ([\d.]+)%', content)
            result['top1_percent_recall'] = float(top1_match.group(1)) if top1_match else 0.0

            # 解析详细特征统计
            detailed_stats = self._parse_detailed_statistics(content)
            result.update(detailed_stats)

            return result

        except Exception as e:
            print(f"解析实验结果失败: {e}")
            return None

    def _parse_detailed_statistics(self, content: str) -> Dict:
        """解析详细统计信息 - 增强版"""
        stats = {}

        try:
            # 1. 解析轮廓统计 - 从新的logging格式
            basic_match = re.search(
                r'CONTOUR_STATS_BASIC: total=(\d+), min=(\d+), max=(\d+), avg=([\d.]+), std=([\d.]+)', content)
            if basic_match:
                stats['total_contours'] = int(basic_match.group(1))
                stats['min_contours_per_frame'] = int(basic_match.group(2))
                stats['max_contours_per_frame'] = int(basic_match.group(3))
                stats['avg_contours_per_frame'] = float(basic_match.group(4))
                stats['std_contours_per_frame'] = float(basic_match.group(5))

            # 轮廓尺寸分布
            size_bins = ["极小轮廓", "小轮廓", "中小轮廓", "中等轮廓", "大轮廓", "超大轮廓"]
            for bin_name in size_bins:
                pattern = f'CONTOUR_SIZE_DIST: {bin_name}=(\\d+)\\(([\\d.]+)\\)'
                match = re.search(pattern, content)
                if match:
                    stats[f'contours_{bin_name}_count'] = int(match.group(1))
                    stats[f'contours_{bin_name}_ratio'] = float(match.group(2))

            # 几何特征
            geom_match = re.search(r'CONTOUR_GEOMETRY: avg_eccentricity=([\d.]+), std_eccentricity=([\d.]+)', content)
            if geom_match:
                stats['avg_eccentricity'] = float(geom_match.group(1))
                stats['std_eccentricity'] = float(geom_match.group(2))

            # 偏心率分布
            ecc_bins = ["近圆形", "椭圆形", "长椭圆", "极长椭圆"]
            for bin_name in ecc_bins:
                pattern = f'CONTOUR_ECC_DIST: {bin_name}=(\\d+)\\(([\\d.]+)\\)'
                match = re.search(pattern, content)
                if match:
                    stats[f'{bin_name}_count'] = int(match.group(1))
                    stats[f'{bin_name}_ratio'] = float(match.group(2))

            # 特征值比例
            eigval_match = re.search(r'CONTOUR_EIGENVALUE: avg_ratio=([\d.]+)', content)
            if eigval_match:
                stats['avg_eigenvalue_ratio'] = float(eigval_match.group(1))

            # 显著特征
            feat_match = re.search(
                r'CONTOUR_SIGNIFICANT_FEATURES: ecc_count=(\d+)\(([^\)]+)\), com_count=(\d+)\(([^\)]+)\)', content)
            if feat_match:
                stats['significant_ecc_features_count'] = int(feat_match.group(1))
                stats['significant_ecc_features_ratio'] = float(feat_match.group(2))
                stats['significant_com_features_count'] = int(feat_match.group(3))
                stats['significant_com_features_ratio'] = float(feat_match.group(4))

            # 高度信息
            height_match = re.search(r'CONTOUR_HEIGHT: avg_height=([\d.]+)', content)
            if height_match:
                stats['avg_contour_height'] = float(height_match.group(1))

            # 2. 解析检索键统计
            dim_match = re.search(r'KEY_DIMENSIONS: dim0_avg=([\d.]+), dim1_avg=([\d.]+), dim2_avg=([\d.]+)', content)
            if dim_match:
                stats['avg_key_dimension_0'] = float(dim_match.group(1))
                stats['avg_key_dimension_1'] = float(dim_match.group(2))
                stats['avg_key_dimension_2'] = float(dim_match.group(3))

            # 键维度分布
            for dim in [0, 1, 2]:
                pattern = f'KEY_DIM{dim}_DIST: min=([\\d.]+), max=([\\d.]+), std=([\\d.]+)'
                match = re.search(pattern, content)
                if match:
                    stats[f'key_dim{dim}_min'] = float(match.group(1))
                    stats[f'key_dim{dim}_max'] = float(match.group(2))
                    stats[f'key_dim{dim}_std'] = float(match.group(3))

            # 稀疏性统计
            sparse_match = re.search(
                r'KEY_SPARSITY: total_keys=(\d+), zero_keys=(\d+), sparsity=([\d.]+), valid_keys=(\d+)', content)
            if sparse_match:
                stats['total_retrieval_keys'] = int(sparse_match.group(1))
                stats['zero_keys_count'] = int(sparse_match.group(2))
                stats['key_sparsity_ratio'] = float(sparse_match.group(3))
                stats['valid_keys_count'] = int(sparse_match.group(4))

            # 环形特征
            ring_match = re.search(
                r'KEY_RING_FEATURES: avg_activation=([\d.]+), std_activation=([\d.]+), max_activation=([\d.]+), active_count=(\d+)',
                content)
            if ring_match:
                stats['ring_feature_avg_activation'] = float(ring_match.group(1))
                stats['ring_feature_std_activation'] = float(ring_match.group(2))
                stats['ring_feature_max_activation'] = float(ring_match.group(3))
                stats['ring_feature_active_count'] = int(ring_match.group(4))

            # 键质量
            key_quality_match = re.search(r'KEY_QUALITY: quality_score=([\d.]+)', content)
            if key_quality_match:
                stats['key_quality_score'] = float(key_quality_match.group(1))

            # 3. 解析BCI统计
            bci_basic_match = re.search(
                r'BCI_BASIC_STATS: total_bcis=(\d+), avg_neighbors=([\d.]+), std_neighbors=([\d.]+), min_neighbors=(\d+), max_neighbors=(\d+)',
                content)
            if bci_basic_match:
                stats['total_bcis'] = int(bci_basic_match.group(1))
                stats['avg_neighbors_per_bci'] = float(bci_basic_match.group(2))
                stats['std_neighbors_per_bci'] = float(bci_basic_match.group(3))
                stats['min_neighbors_per_bci'] = int(bci_basic_match.group(4))
                stats['max_neighbors_per_bci'] = int(bci_basic_match.group(5))

            # BCI邻居分布
            neighbor_ranges = ['0_neighbors', '1-3_neighbors', '4-6_neighbors', '7-10_neighbors', '10+_neighbors']
            for range_name in neighbor_ranges:
                safe_range_name = range_name.replace('+', '\\+').replace('-', '\\-')
                pattern = f'BCI_NEIGHBOR_DIST: {safe_range_name}=(\\d+)\\(([\\d.]+)\\)'
                match = re.search(pattern, content)
                if match:
                    stats[f'bci_{range_name}_count'] = int(match.group(1))
                    stats[f'bci_{range_name}_ratio'] = float(match.group(2))

            # 距离统计
            dist_match = re.search(
                r'BCI_DISTANCES: avg_distance=([\d.]+), std_distance=([\d.]+), min_distance=([\d.]+), max_distance=([\d.]+)',
                content)
            if dist_match:
                stats['avg_neighbor_distance'] = float(dist_match.group(1))
                stats['std_neighbor_distance'] = float(dist_match.group(2))
                stats['min_neighbor_distance'] = float(dist_match.group(3))
                stats['max_neighbor_distance'] = float(dist_match.group(4))

            # 角度统计
            angle_match = re.search(r'BCI_ANGLES: angle_diversity=([\d.]+), angle_uniformity=([\d.]+)', content)
            if angle_match:
                stats['angle_diversity'] = float(angle_match.group(1))
                stats['angle_uniformity'] = float(angle_match.group(2))

            # 跨层连接
            cross_layer_match = re.search(
                r'BCI_CROSS_LAYER: cross_layer_connections=(\d+), intra_layer_connections=(\d+), cross_layer_ratio=([\d.]+)',
                content)
            if cross_layer_match:
                stats['cross_layer_connections'] = int(cross_layer_match.group(1))
                stats['intra_layer_connections'] = int(cross_layer_match.group(2))
                stats['cross_layer_connections_ratio'] = float(cross_layer_match.group(3))

            # 距离位激活
            bit_match = re.search(
                r'BCI_DISTANCE_BITS: total_bits=(\d+), activated_bits=(\d+), activation_rate=([\d.]+)', content)
            if bit_match:
                stats['total_distance_bits'] = int(bit_match.group(1))
                stats['activated_distance_bits'] = int(bit_match.group(2))
                stats['distance_bit_activation_rate'] = float(bit_match.group(3))

            # 星座复杂度
            complexity_match = re.search(r'BCI_CONSTELLATION_COMPLEXITY: complexity_score=([\d.]+)', content)
            if complexity_match:
                stats['constellation_complexity'] = float(complexity_match.group(1))

            # 连接质量
            conn_quality_match = re.search(r'BCI_CONNECTION_QUALITY: quality_score=([\d.]+), ideal_bcis_ratio=([\d.]+)',
                                           content)
            if conn_quality_match:
                stats['bci_connection_quality'] = float(conn_quality_match.group(1))
                stats['ideal_bcis_ratio'] = float(conn_quality_match.group(2))

            # 每层连接统计
            layer_pattern = r'BCI_LAYER_(\d+): bcis=(\d+), avg_connections=([\d.]+), cross_layer_ratio=([\d.]+)'
            layer_matches = re.findall(layer_pattern, content)
            layer_stats = {}
            for layer_id, bcis_count, avg_conn, cross_ratio in layer_matches:
                layer_stats[f'layer_{layer_id}'] = {
                    'bcis_count': int(bcis_count),
                    'avg_connections': float(avg_conn),
                    'cross_layer_ratio': float(cross_ratio)
                }
            if layer_stats:
                stats['layer_connectivity'] = layer_stats

            # 4. 解析相似性检查统计 - 从现有格式
            similarity_matches = re.findall(
                r'SIMILARITY_CHECKS: check1_passed=(\d+), check2_passed=(\d+), check3_passed=(\d+)', content)
            context_matches = re.findall(
                r'SIMILARITY_CHECKS_CONTEXT: query_id=([^,]+), total_searches=(\d+), total_found=(\d+), total_checks=(\d+), final_candidates=(\d+)',
                content)

            if similarity_matches and context_matches:
                import statistics

                # 解析检查通过数量
                check1_values = [int(match[0]) for match in similarity_matches]
                check2_values = [int(match[1]) for match in similarity_matches]
                check3_values = [int(match[2]) for match in similarity_matches]

                stats['avg_check1_passed'] = statistics.mean(check1_values) if check1_values else 0
                stats['avg_check2_passed'] = statistics.mean(check2_values) if check2_values else 0
                stats['avg_check3_passed'] = statistics.mean(check3_values) if check3_values else 0

                # 解析搜索上下文信息
                total_searches_values = [int(match[1]) for match in context_matches]
                total_found_values = [int(match[2]) for match in context_matches]
                total_checks_values = [int(match[3]) for match in context_matches]
                final_candidates_values = [int(match[4]) for match in context_matches]

                stats['avg_total_searches'] = statistics.mean(total_searches_values) if total_searches_values else 0
                stats['avg_total_found'] = statistics.mean(total_found_values) if total_found_values else 0
                stats['avg_total_checks'] = statistics.mean(total_checks_values) if total_checks_values else 0
                stats['avg_final_candidates'] = statistics.mean(
                    final_candidates_values) if final_candidates_values else 0

                # 计算总查询数
                total_queries = len(similarity_matches)
                stats['estimated_total_queries'] = total_queries

                # 计算通过率
                if total_queries > 0 and stats['avg_total_checks'] > 0:
                    # 轮廓相似性通过率 = 平均check1通过数 / 平均总检查数
                    stats['contour_similarity_pass_rate'] = stats['avg_check1_passed'] / stats['avg_total_checks']

                    # 星座相似性通过率 = 平均check2通过数 / 平均check1通过数
                    if stats['avg_check1_passed'] > 0:
                        stats['constellation_similarity_pass_rate'] = stats['avg_check2_passed'] / stats[
                            'avg_check1_passed']
                    else:
                        stats['constellation_similarity_pass_rate'] = 0

                    # 对偶相似性通过率 = 平均check3通过数 / 平均check2通过数
                    if stats['avg_check2_passed'] > 0:
                        stats['pairwise_similarity_pass_rate'] = stats['avg_check3_passed'] / stats['avg_check2_passed']
                    else:
                        stats['pairwise_similarity_pass_rate'] = 0
                else:
                    stats['contour_similarity_pass_rate'] = 0
                    stats['constellation_similarity_pass_rate'] = 0
                    stats['pairwise_similarity_pass_rate'] = 0

            # 如果没有找到新格式，尝试解析旧格式（向后兼容）
            elif 'avg_check1_passed' not in stats:
                check1_matches = re.findall(r'After check 1: (\d+)', content)
                check2_matches = re.findall(r'After check 2: (\d+)', content)
                check3_matches = re.findall(r'After check 3: (\d+)', content)

                if check1_matches and check2_matches and check3_matches:
                    import statistics
                    stats['avg_check1_passed'] = statistics.mean([int(x) for x in check1_matches])
                    stats['avg_check2_passed'] = statistics.mean([int(x) for x in check2_matches])
                    stats['avg_check3_passed'] = statistics.mean([int(x) for x in check3_matches])

                    # 尝试从日志中解析实际的查询数量（保持原有逻辑）
                    total_queries = None
                    total_queries_match = re.search(r'总查询数[：:]\s*(\d+)', content)
                    if total_queries_match:
                        total_queries = int(total_queries_match.group(1))
                    else:
                        total_queries = len(check1_matches) * 10  # 估算

                    stats['estimated_total_queries'] = total_queries

                    # 计算通过率（使用估算的总查询数）
                    if total_queries > 0:
                        stats['contour_similarity_pass_rate'] = stats['avg_check1_passed'] / total_queries
                    if stats['avg_check1_passed'] > 0:
                        stats['constellation_similarity_pass_rate'] = stats['avg_check2_passed'] / stats[
                            'avg_check1_passed']
                    if stats['avg_check2_passed'] > 0:
                        stats['pairwise_similarity_pass_rate'] = stats['avg_check3_passed'] / stats['avg_check2_passed']

            # 5. 计算综合特征质量指数
            stats['feature_quality_index'] = self._calculate_enhanced_feature_quality_index(stats)

            # 6. 处理缺失值 - 为所有可能的统计项设置默认值
            default_values = {
                'total_contours': 0, 'avg_contours_per_frame': 0.0, 'std_contours_per_frame': 0.0,
                'avg_eccentricity': 0.0, 'std_eccentricity': 0.0,
                'avg_eigenvalue_ratio': 0.0, 'avg_contour_height': 0.0,
                'avg_key_dimension_0': 0.0, 'avg_key_dimension_1': 0.0, 'avg_key_dimension_2': 0.0,
                'key_sparsity_ratio': 1.0, 'ring_feature_avg_activation': 0.0,
                'total_retrieval_keys': 0, 'key_quality_score': 0.0,
                'total_bcis': 0, 'avg_neighbors_per_bci': 0.0, 'std_neighbors_per_bci': 0.0,
                'avg_neighbor_distance': 0.0, 'angle_diversity': 0.0, 'angle_uniformity': 0.0,
                'cross_layer_connections_ratio': 0.0, 'distance_bit_activation_rate': 0.0,
                'constellation_complexity': 0.0, 'bci_connection_quality': 0.0,
                'contour_similarity_pass_rate': 0.0, 'constellation_similarity_pass_rate': 0.0,
                'pairwise_similarity_pass_rate': 0.0
            }

            # 尺寸分布默认值
            for bin_name in size_bins:
                default_values[f'contours_{bin_name}_count'] = 0
                default_values[f'contours_{bin_name}_ratio'] = 0.0

            # 偏心率分布默认值
            for bin_name in ecc_bins:
                default_values[f'{bin_name}_count'] = 0
                default_values[f'{bin_name}_ratio'] = 0.0

            # BCI邻居分布默认值
            for range_name in neighbor_ranges:
                default_values[f'bci_{range_name}_count'] = 0
                default_values[f'bci_{range_name}_ratio'] = 0.0

            # 填充缺失值
            for key, default_val in default_values.items():
                if key not in stats:
                    stats[key] = default_val

        except Exception as e:
            print(f"解析详细统计失败: {e}")
            import traceback
            print(f"详细错误信息: {traceback.format_exc()}")

        return stats

    def _calculate_enhanced_feature_quality_index(self, stats: Dict) -> float:
        """计算增强的特征质量指数"""
        try:
            quality_factors = []
            weights = []

            # 1. 轮廓质量因子 (权重: 0.3)
            avg_contours = stats.get('avg_contours_per_frame', 0)
            if avg_contours > 0:
                # 理想轮廓数范围: 50-200
                if 50 <= avg_contours <= 200:
                    contour_quality = 1.0
                elif avg_contours < 50:
                    contour_quality = avg_contours / 50.0
                else:
                    contour_quality = max(0.1, 1.0 - (avg_contours - 200) * 0.002)

                # 考虑轮廓尺寸分布的影响
                small_ratio = stats.get('contours_极小轮廓_ratio', 0)
                large_ratio = stats.get('contours_大轮廓_ratio', 0) + stats.get('contours_超大轮廓_ratio', 0)

                # 理想情况下，小轮廓占比不应过高，大轮廓应有一定比例
                size_balance = 1.0 - small_ratio * 0.5 + large_ratio * 0.3
                contour_quality = contour_quality * max(0.3, min(1.0, size_balance))

                quality_factors.append(contour_quality)
                weights.append(0.3)

            # 2. 几何特征质量因子 (权重: 0.2)
            avg_ecc = stats.get('avg_eccentricity', 0)
            if avg_ecc > 0:
                # 理想偏心率应该有一定变化但不过于极端
                if 0.2 <= avg_ecc <= 0.6:
                    ecc_quality = 1.0
                elif avg_ecc < 0.2:
                    ecc_quality = avg_ecc / 0.2 * 0.8 + 0.2  # 太圆形不好
                else:
                    ecc_quality = max(0.2, 1.0 - (avg_ecc - 0.6) * 1.25)

                # 考虑偏心率分布的平衡性
                near_circle = stats.get('近圆形_ratio', 0)
                ellipse = stats.get('椭圆形_ratio', 0)
                long_ellipse = stats.get('长椭圆_ratio', 0)

                # 理想分布应该是椭圆形占主导，其他形状有一定比例
                distribution_balance = ellipse * 0.4 + near_circle * 0.3 + long_ellipse * 0.3
                ecc_quality = ecc_quality * max(0.3, min(1.0, distribution_balance * 2))

                quality_factors.append(ecc_quality)
                weights.append(0.2)

            # 3. 检索键质量因子 (权重: 0.25)
            key_sparsity = stats.get('key_sparsity_ratio', 1.0)
            key_quality_score = stats.get('key_quality_score', 0)

            if key_sparsity < 1.0 or key_quality_score > 0:
                # 基础质量: 1 - 稀疏性
                base_quality = 1.0 - key_sparsity

                # 考虑环形特征激活
                ring_activation = stats.get('ring_feature_avg_activation', 0)
                ring_quality = min(1.0, ring_activation * 10)  # 归一化

                # 考虑键维度的有效性
                dim0 = stats.get('avg_key_dimension_0', 0)
                dim1 = stats.get('avg_key_dimension_1', 0)
                dim2 = stats.get('avg_key_dimension_2', 0)

                dim_quality = 0.0
                if dim0 > 0 or dim1 > 0 or dim2 > 0:
                    # 维度值应该有合理的范围，不应过小或过大
                    dim_scores = []
                    for dim_val in [dim0, dim1, dim2]:
                        if dim_val > 0:
                            if 0.1 <= dim_val <= 10:
                                dim_scores.append(1.0)
                            elif dim_val < 0.1:
                                dim_scores.append(dim_val / 0.1)
                            else:
                                dim_scores.append(max(0.1, 1.0 - (dim_val - 10) * 0.05))

                    dim_quality = sum(dim_scores) / len(dim_scores) if dim_scores else 0.0

                # 综合键质量
                key_quality = (base_quality * 0.4 + ring_quality * 0.3 + dim_quality * 0.3)
                quality_factors.append(key_quality)
                weights.append(0.25)

            # 4. BCI连接质量因子 (权重: 0.25)
            avg_neighbors = stats.get('avg_neighbors_per_bci', 0)
            bci_connection_quality = stats.get('bci_connection_quality', 0)

            if avg_neighbors > 0 or bci_connection_quality > 0:
                # 理想邻居数: 3-8个
                if 3 <= avg_neighbors <= 8:
                    neighbor_quality = 1.0
                elif avg_neighbors < 3:
                    neighbor_quality = avg_neighbors / 3.0
                else:
                    neighbor_quality = max(0.1, 1.0 - (avg_neighbors - 8) * 0.05)

                # 角度多样性 - 邻居角度应该分布均匀
                angle_diversity = stats.get('angle_diversity', 0)
                angle_uniformity = stats.get('angle_uniformity', 0)

                angle_quality = 0.0
                if angle_diversity > 0 and angle_uniformity > 0:
                    # 理想情况下应该有适中的角度多样性和高的均匀性
                    diversity_score = min(1.0, angle_diversity / 2.0)  # 假设理想多样性为2.0
                    angle_quality = diversity_score * 0.4 + angle_uniformity * 0.6

                # 跨层连接质量
                cross_layer_ratio = stats.get('cross_layer_connections_ratio', 0)
                # 理想的跨层连接比例应该在20%-60%之间
                if 0.2 <= cross_layer_ratio <= 0.6:
                    cross_layer_quality = 1.0
                elif cross_layer_ratio < 0.2:
                    cross_layer_quality = cross_layer_ratio / 0.2
                else:
                    cross_layer_quality = max(0.2, 1.0 - (cross_layer_ratio - 0.6) * 1.25)

                # 星座复杂度
                constellation_complexity = stats.get('constellation_complexity', 0)
                complexity_quality = min(1.0, constellation_complexity * 2)  # 归一化

                # 综合BCI质量
                bci_quality = (neighbor_quality * 0.3 + angle_quality * 0.25 +
                               cross_layer_quality * 0.25 + complexity_quality * 0.2)

                quality_factors.append(bci_quality)
                weights.append(0.25)

            # 计算加权平均
            if quality_factors and weights:
                weighted_sum = sum(q * w for q, w in zip(quality_factors, weights))
                total_weight = sum(weights)
                return weighted_sum / total_weight
            else:
                return 0.0

        except Exception as e:
            print(f"计算特征质量指数失败: {e}")
            return 0.0

    def _format_experiment_report(self, result: Dict) -> str:
        """格式化单层实验报告"""
        try:
            num_layers = result['num_layers']
            layer_interval = result['layer_interval']

            # 生成高度分割列表
            height_splits = []
            for i in range(num_layers + 1):
                height = layer_interval * i
                height_splits.append(f"{height:.1f}")

            # 生成查询层级列表
            query_levels = list(range(num_layers))

            report = []
            report.append(f"分层数量消融实验 - {num_layers}层配置")
            report.append("=" * 80)
            report.append("实验配置:")
            report.append(f"  层数: {num_layers}")
            report.append(f"  层间距: {layer_interval:.3f}m")
            report.append(f"  高度分割: [{', '.join(height_splits)}]")
            report.append(f"  查询层级: {query_levels}")
            report.append(f"  运行时长: {result['duration']:.2f}秒")

            report.append("")
            report.append("基本性能结果:")
            report.append("-" * 40)
            report.append(f"Average Recall @1: {result.get('recall_1', 0):.2f}%")
            report.append(f"Average Recall @5: {result.get('recall_5', 0):.2f}%")
            report.append(f"Average Recall @10: {result.get('recall_10', 0):.2f}%")
            report.append(f"Average Recall @25: {result.get('recall_25', 0):.2f}%")
            report.append(f"Average Similarity: {result.get('average_similarity', 0):.4f}")
            report.append(f"Average Top 1% Recall: {result.get('top1_percent_recall', 0):.2f}%")

            report.append("")
            report.append("轮廓统计分析:")
            report.append("-" * 40)
            report.append(f"总轮廓数: {result.get('total_contours', 0)}")
            report.append(f"平均每帧轮廓数: {result.get('avg_contours_per_frame', 0):.1f}")
            report.append(f"轮廓数标准差: {result.get('std_contours_per_frame', 0):.1f}")
            report.append(f"最少轮廓数: {result.get('min_contours_per_frame', 0)}")
            report.append(f"最多轮廓数: {result.get('max_contours_per_frame', 0)}")

            report.append("")
            report.append("轮廓尺寸分布:")
            report.append("-" * 40)

            size_bins_info = [
                ("极小轮廓", "(1-5像素)"),
                ("小轮廓", "(6-15像素)"),
                ("中小轮廓", "(16-50像素)"),
                ("中等轮廓", "(51-150像素)"),
                ("大轮廓", "(151-500像素)"),
                ("超大轮廓", "(≥501像素)")
            ]

            for bin_name, size_range in size_bins_info:
                count = result.get(f'contours_{bin_name}_count', 0)
                ratio = result.get(f'contours_{bin_name}_ratio', 0) * 100
                report.append(f"{bin_name} {size_range}: {count}个 ({ratio:.1f}%)")

            report.append("")
            report.append("轮廓几何特征:")
            report.append("-" * 40)
            report.append(f"平均偏心率: {result.get('avg_eccentricity', 0):.3f}")
            report.append(f"偏心率标准差: {result.get('std_eccentricity', 0):.3f}")
            report.append(f"显著偏心率特征比例: {result.get('significant_ecc_features_ratio', 0) * 100:.1f}%")
            report.append(f"显著质心特征比例: {result.get('significant_com_features_ratio', 0) * 100:.1f}%")
            report.append(f"平均特征值比例: {result.get('avg_eigenvalue_ratio', 0):.3f}")
            report.append(f"平均轮廓高度: {result.get('avg_contour_height', 0):.2f}")

            # 偏心率分布
            ecc_bins_info = [
                ("近圆形", "(0.0-0.3)"),
                ("椭圆形", "(0.3-0.6)"),
                ("长椭圆", "(0.6-0.8)"),
                ("极长椭圆", "(0.8-1.0)")
            ]

            for bin_name, ecc_range in ecc_bins_info:
                ratio = result.get(f'{bin_name}_ratio', 0) * 100
                report.append(f"{bin_name} {ecc_range}: {ratio:.1f}%")

            report.append("")
            report.append("检索键特征:")
            report.append("-" * 40)
            report.append(f"Key dimension 0: avg={result.get('avg_key_dimension_0', 0):.4f}")
            report.append(f"Key dimension 1: avg={result.get('avg_key_dimension_1', 0):.4f}")
            report.append(f"Key dimension 2: avg={result.get('avg_key_dimension_2', 0):.4f}")

            # 键维度分布细节
            for dim in [0, 1, 2]:
                dim_min = result.get(f'key_dim{dim}_min', 0)
                dim_max = result.get(f'key_dim{dim}_max', 0)
                dim_std = result.get(f'key_dim{dim}_std', 0)
                if dim_min > 0 or dim_max > 0 or dim_std > 0:
                    report.append(f"  Dimension {dim} 分布: min={dim_min:.4f}, max={dim_max:.4f}, std={dim_std:.4f}")

            report.append(f"Key sparsity: {result.get('key_sparsity_ratio', 1):.4f}")
            report.append(f"Ring feature activation: {result.get('ring_feature_avg_activation', 0):.4f}")

            # 环形特征详细信息
            ring_std = result.get('ring_feature_std_activation', 0)
            ring_max = result.get('ring_feature_max_activation', 0)
            ring_active_count = result.get('ring_feature_active_count', 0)
            if ring_std > 0 or ring_max > 0 or ring_active_count > 0:
                report.append(
                    f"  Ring feature std: {ring_std:.4f}, max: {ring_max:.4f}, active count: {ring_active_count}")

            report.append(f"Total retrieval keys: {result.get('total_retrieval_keys', 0)}")
            report.append(f"Valid keys: {result.get('valid_keys_count', 0)}")
            report.append(f"Key quality score: {result.get('key_quality_score', 0):.4f}")

            report.append("")
            report.append("BCI特征:")
            report.append("-" * 40)
            report.append(f"BCI neighbors: 平均{result.get('avg_neighbors_per_bci', 0):.1f}个")
            report.append(f"邻居数标准差: {result.get('std_neighbors_per_bci', 0):.1f}")
            report.append(f"最大邻居数: {result.get('max_neighbors_per_bci', 0)}")
            report.append(f"Neighbor distances: 平均{result.get('avg_neighbor_distance', 0):.2f}")
            report.append(f"邻居距离标准差: {result.get('std_neighbor_distance', 0):.2f}")

            # 距离范围
            min_dist = result.get('min_neighbor_distance', 0)
            max_dist = result.get('max_neighbor_distance', 0)
            if min_dist > 0 or max_dist > 0:
                report.append(f"距离范围: {min_dist:.2f} - {max_dist:.2f}")

            report.append(f"Neighbor angles: 多样性{result.get('angle_diversity', 0):.3f}")
            report.append(f"角度均匀性: {result.get('angle_uniformity', 0):.3f}")
            report.append(f"Cross layer connections: {result.get('cross_layer_connections_ratio', 0) * 100:.1f}%")

            # 跨层连接详细信息
            cross_connections = result.get('cross_layer_connections', 0)
            intra_connections = result.get('intra_layer_connections', 0)
            if cross_connections > 0 or intra_connections > 0:
                report.append(f"  跨层连接数: {cross_connections}, 层内连接数: {intra_connections}")

            report.append(f"Distance bit activation: {result.get('distance_bit_activation_rate', 0):.4f}")

            # 距离位详细信息
            total_bits = result.get('total_distance_bits', 0)
            activated_bits = result.get('activated_distance_bits', 0)
            if total_bits > 0:
                report.append(f"  总距离位: {total_bits}, 激活位: {activated_bits}")

            report.append(f"星座复杂度: {result.get('constellation_complexity', 0):.3f}")
            report.append(f"Total BCIs: {result.get('total_bcis', 0)}")
            report.append(f"BCI连接质量: {result.get('bci_connection_quality', 0):.4f}")
            report.append(f"理想BCI比例: {result.get('ideal_bcis_ratio', 0) * 100:.1f}%")

            # BCI邻居分布
            neighbor_ranges = [
                ('0_neighbors', '0邻居'),
                ('1-3_neighbors', '1-3邻居'),
                ('4-6_neighbors', '4-6邻居'),
                ('7-10_neighbors', '7-10邻居'),
                ('10+_neighbors', '10+邻居')
            ]

            neighbor_dist_found = False
            for range_key, range_label in neighbor_ranges:
                count = result.get(f'bci_{range_key}_count', 0)
                ratio = result.get(f'bci_{range_key}_ratio', 0)
                if count > 0:
                    if not neighbor_dist_found:
                        report.append("BCI邻居分布:")
                        neighbor_dist_found = True
                    report.append(f"  {range_label}: {count}个 ({ratio * 100:.1f}%)")

            # 每层连接统计
            layer_connectivity = result.get('layer_connectivity', {})
            if layer_connectivity:
                report.append("每层BCI连接统计:")
                for layer_key, layer_stats in layer_connectivity.items():
                    layer_id = layer_key.replace('layer_', '')
                    bcis_count = layer_stats.get('bcis_count', 0)
                    avg_conn = layer_stats.get('avg_connections', 0)
                    cross_ratio = layer_stats.get('cross_layer_ratio', 0)
                    report.append(
                        f"  第{layer_id}层: {bcis_count}个BCI, 平均连接{avg_conn:.1f}个, 跨层比例{cross_ratio * 100:.1f}%")

            report.append("")
            report.append("相似性检查统计:")
            report.append("-" * 40)

            # 相似性检查通过率
            contour_pass_rate = result.get('contour_similarity_pass_rate', 0) * 100
            constellation_pass_rate = result.get('constellation_similarity_pass_rate', 0) * 100
            pairwise_pass_rate = result.get('pairwise_similarity_pass_rate', 0) * 100

            report.append(f"轮廓相似性检查通过率: {contour_pass_rate:.2f}%")
            report.append(f"星座相似性检查通过率: {constellation_pass_rate:.2f}%")
            report.append(f"对偶相似性检查通过率: {pairwise_pass_rate:.2f}%")

            # 检查数量统计
            check1_avg = result.get('avg_check1_passed', 0)
            check2_avg = result.get('avg_check2_passed', 0)
            check3_avg = result.get('avg_check3_passed', 0)
            total_queries = result.get('estimated_total_queries', 0)

            if check1_avg > 0 or check2_avg > 0 or check3_avg > 0:
                report.append(f"平均通过数量: 检查1={check1_avg:.1f}, 检查2={check2_avg:.1f}, 检查3={check3_avg:.1f}")
                if total_queries > 0:
                    report.append(f"估算总查询数: {total_queries}")

            report.append("")
            report.append("综合质量评估:")
            report.append("-" * 40)
            feature_quality_index = result.get('feature_quality_index', 0)
            report.append(f"特征质量指数: {feature_quality_index:.4f}")

            # 质量等级评估
            if feature_quality_index >= 0.8:
                quality_grade = "优秀"
            elif feature_quality_index >= 0.6:
                quality_grade = "良好"
            elif feature_quality_index >= 0.4:
                quality_grade = "中等"
            elif feature_quality_index >= 0.2:
                quality_grade = "较差"
            else:
                quality_grade = "很差"

            report.append(f"质量等级: {quality_grade}")

            # 添加质量建议
            suggestions = []
            if result.get('avg_contours_per_frame', 0) < 50:
                suggestions.append("建议增加轮廓检测敏感度或降低过滤阈值")
            if result.get('key_sparsity_ratio', 1) > 0.8:
                suggestions.append("建议优化检索键生成算法，减少空键比例")
            if result.get('avg_neighbors_per_bci', 0) < 3:
                suggestions.append("建议调整BCI邻域搜索参数，增加连接密度")
            if result.get('cross_layer_connections_ratio', 0) < 0.2:
                suggestions.append("建议优化层间连接策略，提高跨层特征关联")

            if suggestions:
                report.append("")
                report.append("优化建议:")
                for i, suggestion in enumerate(suggestions, 1):
                    report.append(f"  {i}. {suggestion}")

            # 添加实验完成信息
            completion_time = result.get('completion_time', 'Unknown')
            report.append("")
            report.append(f"完成时间: {completion_time}")
            report.append("=" * 80)

            return "\n".join(report)

        except Exception as e:
            print(f"格式化实验报告失败: {e}")
            return f"报告生成失败: {str(e)}"

    def run_all_experiments(self, start_layers: int = 1, end_layers: int = 20):
        """运行所有实验"""
        print("开始分层数量消融实验 - subprocess分离式")
        print(f"实验范围: {start_layers}-{end_layers}层")
        print(f"预计总时长: {(end_layers - start_layers + 1) * 20} 分钟")

        # 检查必要文件
        if not self.check_required_files():
            return

        # 备份原始文件
        self.backup_files()

        try:
            # 运行所有实验
            for num_layers in range(start_layers, end_layers + 1):
                print(f"\n总进度: {num_layers - start_layers + 1}/{end_layers - start_layers + 1}")

                result = self.run_single_layer_experiment(num_layers)
                if result:
                    self.results.append(result)
                    print(f"✅ 实验{num_layers}层成功")
                else:
                    print(f"❌ 实验{num_layers}层失败，跳过")

                # 短暂休息避免系统负载过高
                time.sleep(2)

            # 生成综合报告
            if self.results:
                self.generate_comprehensive_report()
            else:
                print("警告：没有成功的实验结果")

        finally:
            # 恢复原始文件
            self.restore_files()

    def generate_comprehensive_report(self):
        """生成综合分析报告"""
        if not self.results:
            print("没有成功的实验结果")
            return

        import datetime
        import os

        # 创建报告目录
        report_dir = "分层数量-多种特征-召回率关系-实验报告"
        os.makedirs(report_dir, exist_ok=True)

        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        comprehensive_report_file = os.path.join(report_dir, f"comprehensive_report_{timestamp}.txt")

        # 按层数排序结果
        sorted_results = sorted(self.results, key=lambda x: x['num_layers'])

        with open(comprehensive_report_file, 'w', encoding='utf-8') as f:
            # =============== 综合报告头部 ===============
            f.write("分层数量消融实验 - 综合分析报告\n")
            f.write("=" * 100 + "\n")
            f.write(f"报告生成时间: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"实验数量: {len(self.results)}\n")
            f.write(f"层数范围: {sorted_results[0]['num_layers']}-{sorted_results[-1]['num_layers']}层\n")
            f.write(f"分析目的: 深入理解层数变化对特征提取和召回率的影响\n")
            f.write(f"数据集: Chilean地下矿井场景识别 (历史地图 vs 当前观测)\n")
            f.write("\n")

            # =============== 实验摘要 ===============
            f.write("实验摘要\n")
            f.write("-" * 50 + "\n")

            if sorted_results:
                # 找到各项最佳性能
                best_recall_1 = max(sorted_results, key=lambda x: x.get('recall_1', 0))
                best_recall_5 = max(sorted_results, key=lambda x: x.get('recall_5', 0))
                best_similarity = max(sorted_results, key=lambda x: x.get('average_similarity', 0))
                best_feature_quality = max(sorted_results, key=lambda x: x.get('feature_quality_index', 0))
                fastest_experiment = min(sorted_results, key=lambda x: x.get('duration', float('inf')))

                f.write(
                    f"最佳 Recall@1 性能: {best_recall_1['num_layers']}层 ({best_recall_1.get('recall_1', 0):.2f}%)\n")
                f.write(
                    f"最佳 Recall@5 性能: {best_recall_5['num_layers']}层 ({best_recall_5.get('recall_5', 0):.2f}%)\n")
                f.write(
                    f"最佳平均相似度: {best_similarity['num_layers']}层 ({best_similarity.get('average_similarity', 0):.4f})\n")
                f.write(
                    f"最佳特征质量: {best_feature_quality['num_layers']}层 ({best_feature_quality.get('feature_quality_index', 0):.4f})\n")
                f.write(
                    f"最快实验: {fastest_experiment['num_layers']}层 ({fastest_experiment.get('duration', 0):.1f}秒)\n")

                # 计算平均值
                avg_recall_1 = sum(r.get('recall_1', 0) for r in sorted_results) / len(sorted_results)
                avg_duration = sum(r.get('duration', 0) for r in sorted_results) / len(sorted_results)
                avg_contours = sum(r.get('avg_contours_per_frame', 0) for r in sorted_results) / len(sorted_results)

                f.write(f"\n平均性能指标:\n")
                f.write(f"  平均 Recall@1: {avg_recall_1:.2f}%\n")
                f.write(f"  平均运行时长: {avg_duration:.1f}秒\n")
                f.write(f"  平均轮廓数: {avg_contours:.1f}个/帧\n")

            f.write("\n")

            # =============== 性能对比表格 ===============
            f.write("性能对比表格\n")
            f.write("-" * 120 + "\n")
            header = (f"{'层数':<4} {'层间距':<8} {'Recall@1':<9} {'Recall@5':<9} {'相似度':<8} "
                      f"{'运行时长':<8} {'轮廓数':<8} {'特征质量':<8} {'小轮廓%':<8} {'BCI邻居':<8}\n")
            f.write(header)
            f.write("-" * 120 + "\n")

            for result in sorted_results:
                line = (f"{result['num_layers']:<4d} {result['layer_interval']:<8.3f} "
                        f"{result.get('recall_1', 0):<9.2f} {result.get('recall_5', 0):<9.2f} "
                        f"{result.get('average_similarity', 0):<8.4f} {result.get('duration', 0):<8.0f} "
                        f"{result.get('avg_contours_per_frame', 0):<8.1f} "
                        f"{result.get('feature_quality_index', 0):<8.4f} "
                        f"{result.get('contours_小轮廓_ratio', 0) * 100:<8.1f} "
                        f"{result.get('avg_neighbors_per_bci', 0):<8.1f}\n")
                f.write(line)

            f.write("-" * 120 + "\n\n")

            # =============== 趋势分析 ===============
            f.write("性能趋势分析\n")
            f.write("-" * 50 + "\n")

            recall_values = [r.get('recall_1', 0) for r in sorted_results]
            layer_counts = [r['num_layers'] for r in sorted_results]

            if len(recall_values) >= 5:
                # 找出性能峰值
                max_recall = max(recall_values)
                max_idx = recall_values.index(max_recall)
                optimal_layers = layer_counts[max_idx]

                f.write(f"1. 性能峰值分析:\n")
                f.write(f"   最高召回率: {max_recall:.2f}% (在{optimal_layers}层时达到)\n")
                f.write(f"   最优层间距: {sorted_results[max_idx]['layer_interval']:.3f}m\n")

                # 寻找性能下降点
                decline_threshold = max_recall * 0.9
                decline_start = -1
                for i in range(max_idx, len(recall_values)):
                    if recall_values[i] < decline_threshold:
                        decline_start = i
                        break

                if decline_start != -1:
                    f.write(f"   性能开始下降: 从{layer_counts[decline_start]}层开始\n")
                    f.write(f"   过分割阈值: 约{sorted_results[decline_start]['layer_interval']:.3f}m层间距\n")
                else:
                    f.write(f"   在测试范围内未观察到明显性能下降\n")

                # 分析性能稳定区间
                stable_configs = []
                for i, recall in enumerate(recall_values):
                    if recall >= max_recall * 0.95:  # 95%的性能区间
                        stable_configs.append((layer_counts[i], recall))

                if len(stable_configs) > 1:
                    f.write(f"   高性能稳定区间: {stable_configs[0][0]}-{stable_configs[-1][0]}层\n")

                f.write(f"\n2. 特征变化趋势:\n")

                # 轮廓碎片化趋势
                small_ratios = [r.get('contours_小轮廓_ratio', 0) for r in sorted_results]
                if len(small_ratios) >= 5:
                    early_frag = sum(small_ratios[:3]) / 3
                    late_frag = sum(small_ratios[-3:]) / 3
                    f.write(f"   轮廓碎片化趋势: {early_frag:.1%} -> {late_frag:.1%}")
                    if late_frag > early_frag * 1.2:
                        f.write(" (明显上升)\n")
                    else:
                        f.write(" (相对稳定)\n")

                # BCI复杂度变化
                neighbor_counts = [r.get('avg_neighbors_per_bci', 0) for r in sorted_results]
                if len(neighbor_counts) >= 5:
                    early_neighbors = sum(neighbor_counts[:3]) / 3
                    late_neighbors = sum(neighbor_counts[-3:]) / 3
                    f.write(f"   BCI复杂度变化: {early_neighbors:.1f} -> {late_neighbors:.1f} 平均邻居数")
                    if abs(late_neighbors - early_neighbors) > 1.0:
                        trend = "上升" if late_neighbors > early_neighbors else "下降"
                        f.write(f" ({trend}趋势)\n")
                    else:
                        f.write(" (相对稳定)\n")

                # 检索键稀疏性变化
                sparsity_values = [r.get('key_sparsity_ratio', 0) for r in sorted_results]
                if len(sparsity_values) >= 5:
                    early_sparsity = sum(sparsity_values[:3]) / 3
                    late_sparsity = sum(sparsity_values[-3:]) / 3
                    f.write(f"   检索键稀疏性变化: {early_sparsity:.1%} -> {late_sparsity:.1%}")
                    if abs(late_sparsity - early_sparsity) > 0.1:
                        trend = "上升" if late_sparsity > early_sparsity else "下降"
                        f.write(f" (稀疏性{trend})\n")
                    else:
                        f.write(" (相对稳定)\n")

            f.write("\n")

            # =============== 因果关系分析 ===============
            f.write("因果关系分析\n")
            f.write("=" * 60 + "\n")

            if len(sorted_results) >= 10:
                f.write("特征指标与召回率的关系分析:\n\n")

                # 1. 层间距与召回率关系
                intervals = [r['layer_interval'] for r in sorted_results]
                recalls = [r.get('recall_1', 0) for r in sorted_results]

                f.write("1. 层间距与召回率关系:\n")
                top_25_percent_count = max(1, len(recalls) // 4)
                top_indices = sorted(range(len(recalls)), key=lambda i: recalls[i], reverse=True)[:top_25_percent_count]
                top_intervals = [intervals[i] for i in top_indices]

                if top_intervals:
                    f.write(f"   最优层间距区间: {min(top_intervals):.3f}m - {max(top_intervals):.3f}m\n")
                    f.write(f"   对应层数范围: {min([sorted_results[i]['num_layers'] for i in top_indices])}-"
                            f"{max([sorted_results[i]['num_layers'] for i in top_indices])}层\n")

                # 2. 轮廓质量与性能关系
                f.write("\n2. 轮廓特征对性能的影响:\n")
                avg_contours = [r.get('avg_contours_per_frame', 0) for r in sorted_results]

                # 找出轮廓数适中且召回率高的配置
                moderate_contour_configs = []
                for i, result in enumerate(sorted_results):
                    contour_count = avg_contours[i]
                    if 50 <= contour_count <= 200:  # 适中的轮廓数量
                        moderate_contour_configs.append((result['num_layers'], recalls[i], contour_count))

                if moderate_contour_configs:
                    best_moderate = max(moderate_contour_configs, key=lambda x: x[1])
                    f.write(f"   轮廓数适中时的最佳性能: {best_moderate[0]}层 "
                            f"({best_moderate[2]:.1f}个轮廓, {best_moderate[1]:.1f}%召回率)\n")

                # 3. BCI复杂度与性能关系
                f.write("\n3. BCI复杂度对性能的影响:\n")
                bci_neighbors = [r.get('avg_neighbors_per_bci', 0) for r in sorted_results]

                ideal_neighbor_configs = []
                for i, result in enumerate(sorted_results):
                    neighbor_count = bci_neighbors[i]
                    if 3 <= neighbor_count <= 8:  # 理想的邻居数量
                        ideal_neighbor_configs.append((result['num_layers'], recalls[i], neighbor_count))

                if ideal_neighbor_configs:
                    best_ideal = max(ideal_neighbor_configs, key=lambda x: x[1])
                    f.write(f"   BCI邻居数适中时的最佳性能: {best_ideal[0]}层 "
                            f"({best_ideal[2]:.1f}个邻居, {best_ideal[1]:.1f}%召回率)\n")

                # 4. 检索键质量与性能关系
                f.write("\n4. 检索键质量对性能的影响:\n")
                key_qualities = [1 - r.get('key_sparsity_ratio', 1) for r in sorted_results]

                high_quality_key_configs = []
                for i, result in enumerate(sorted_results):
                    key_quality = key_qualities[i]
                    if key_quality > 0.5:  # 高质量检索键
                        high_quality_key_configs.append((result['num_layers'], recalls[i], key_quality))

                if high_quality_key_configs:
                    best_key_quality = max(high_quality_key_configs, key=lambda x: x[1])
                    f.write(f"   检索键质量高时的最佳性能: {best_key_quality[0]}层 "
                            f"(质量={best_key_quality[2]:.1%}, {best_key_quality[1]:.1f}%召回率)\n")

            f.write("\n")

            # =============== 推荐配置 ===============
            f.write("推荐配置\n")
            f.write("=" * 40 + "\n")

            if sorted_results:
                best_overall = max(sorted_results, key=lambda x: x.get('recall_1', 0))
                best_quality = max(sorted_results, key=lambda x: x.get('feature_quality_index', 0))
                best_efficiency = max(sorted_results, key=lambda x: x.get('recall_1', 0) / max(x.get('duration', 1), 1))

                f.write(f"1. 最佳性能配置: {best_overall['num_layers']}层\n")
                f.write(f"   召回率: {best_overall.get('recall_1', 0):.2f}%\n")
                f.write(f"   层间距: {best_overall['layer_interval']:.3f}m\n")
                f.write(f"   运行时长: {best_overall.get('duration', 0):.1f}秒\n")

                f.write(f"\n2. 最佳特征质量配置: {best_quality['num_layers']}层\n")
                f.write(f"   特征质量指数: {best_quality.get('feature_quality_index', 0):.4f}\n")
                f.write(f"   召回率: {best_quality.get('recall_1', 0):.2f}%\n")

                f.write(f"\n3. 最高效率配置: {best_efficiency['num_layers']}层\n")
                f.write(
                    f"   效率比: {best_efficiency.get('recall_1', 0) / max(best_efficiency.get('duration', 1), 1):.4f}\n")
                f.write(f"   召回率: {best_efficiency.get('recall_1', 0):.2f}%\n")

                # 实际应用建议
                f.write(f"\n4. 实际应用建议:\n")
                recall_values = [r.get('recall_1', 0) for r in sorted_results]
                max_recall = max(recall_values)
                high_perf_configs = [r for r in sorted_results if r.get('recall_1', 0) >= max_recall * 0.95]

                if high_perf_configs:
                    recommended = min(high_perf_configs, key=lambda x: x.get('duration', float('inf')))
                    f.write(f"   推荐配置: {recommended['num_layers']}层\n")
                    f.write(
                        f"   平衡性能({recommended.get('recall_1', 0):.1f}%)和效率({recommended.get('duration', 0):.0f}s)\n")
                    f.write(f"   层间距: {recommended['layer_interval']:.3f}m\n")
                    f.write(f"   特征质量指数: {recommended.get('feature_quality_index', 0):.4f}\n")

            f.write("\n")

            # =============== 生成单独的格式化报告 ===============
            f.write("单层实验详细报告\n")
            f.write("=" * 50 + "\n")
            f.write("以下为每层实验的详细分析报告，包含完整的特征统计信息:\n\n")

            # 为每层实验生成单独的格式化报告并保存
            individual_reports_dir = os.path.join(report_dir, "individual_reports")
            os.makedirs(individual_reports_dir, exist_ok=True)

            for result in sorted_results:
                num_layers = result['num_layers']

                # 添加完成时间戳
                result['completion_time'] = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')

                # 生成单独的格式化报告
                individual_report_content = self._format_experiment_report(result)

                # 保存单独的报告文件
                individual_report_file = os.path.join(individual_reports_dir, f"layer_{num_layers}_detailed_report.txt")
                with open(individual_report_file, 'w', encoding='utf-8') as individual_f:
                    individual_f.write(individual_report_content)

                # 在综合报告中添加引用
                f.write(f"{num_layers}层实验: 详见 individual_reports/layer_{num_layers}_detailed_report.txt\n")

                # 在综合报告中添加简要摘要
                f.write(f"  性能摘要: Recall@1={result.get('recall_1', 0):.2f}%, ")
                f.write(f"轮廓数={result.get('avg_contours_per_frame', 0):.1f}, ")
                f.write(f"特征质量={result.get('feature_quality_index', 0):.4f}\n")

            f.write(f"\n所有单层详细报告已保存到: {individual_reports_dir}/\n")

            # =============== 报告尾部 ===============
            f.write("\n")
            f.write("实验结论\n")
            f.write("=" * 40 + "\n")

            if sorted_results:
                best_config = max(sorted_results, key=lambda x: x.get('recall_1', 0))
                f.write(f"1. 在本次实验中，{best_config['num_layers']}层配置获得了最佳性能\n")
                f.write(
                    f"2. 最优层间距为{best_config['layer_interval']:.3f}m，对应的召回率为{best_config.get('recall_1', 0):.2f}%\n")

                # 分析过分割现象
                decline_found = False
                max_recall = best_config.get('recall_1', 0)
                for result in sorted_results:
                    if result['num_layers'] > best_config['num_layers']:
                        if result.get('recall_1', 0) < max_recall * 0.9:
                            f.write(f"3. 从{result['num_layers']}层开始出现过分割现象，性能开始下降\n")
                            decline_found = True
                            break

                if not decline_found:
                    f.write(f"3. 在测试范围内未发现明显的过分割现象，建议扩展到更多层数进行测试\n")

                # 特征质量分析
                avg_quality = sum(r.get('feature_quality_index', 0) for r in sorted_results) / len(sorted_results)
                f.write(f"4. 整体特征质量平均为{avg_quality:.4f}，")
                if avg_quality >= 0.6:
                    f.write("特征提取效果良好\n")
                elif avg_quality >= 0.4:
                    f.write("特征提取效果中等，有优化空间\n")
                else:
                    f.write("特征提取效果较差，需要算法改进\n")

            f.write(f"\n报告生成完成时间: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("=" * 100 + "\n")

        print(f"\n综合分析报告已保存到: {comprehensive_report_file}")
        print(f"单层详细报告已保存到: {report_dir}/individual_reports/")

        # 输出关键发现到控制台
        self._print_key_findings()

    def _print_key_findings(self):
        """输出关键发现到控制台"""
        print("\n" + "=" * 80)
        print("关键发现总结")
        print("=" * 80)

        if not self.results:
            return

        sorted_results = sorted(self.results, key=lambda x: x['num_layers'])
        recall_values = [r.get('recall_1', 0) for r in sorted_results]

        if recall_values:
            max_recall = max(recall_values)
            max_idx = recall_values.index(max_recall)
            optimal_config = sorted_results[max_idx]

            print(f"🎯 最佳性能配置: {optimal_config['num_layers']}层")
            print(f"   召回率: {max_recall:.2f}%")
            print(f"   层间距: {optimal_config['layer_interval']:.3f}m")
            print(f"   运行时长: {optimal_config.get('duration', 0):.1f}秒")
            print(f"   特征质量: {optimal_config.get('feature_quality_index', 0):.4f}")

            # 分析性能趋势
            if len(recall_values) >= 10:
                # 寻找性能峰值和下降点
                threshold = max_recall * 0.9
                for i in range(max_idx + 1, len(recall_values)):
                    if recall_values[i] < threshold:
                        decline_layers = sorted_results[i]['num_layers']
                        print(f"📉 性能下降点: {decline_layers}层")
                        print(f"   过分割阈值: 约{sorted_results[i]['layer_interval']:.3f}m")
                        break
                else:
                    print("📈 在测试范围内未发现明显的过分割现象")
                    print("   建议扩展到更多层数 (如25-30层) 来找到性能下降点")

            # 效率分析
            efficiency_scores = [(r.get('recall_1', 0) / max(r.get('duration', 1), 1)) for r in sorted_results]
            best_efficiency_idx = efficiency_scores.index(max(efficiency_scores))
            best_efficiency_config = sorted_results[best_efficiency_idx]

            print(f"⚡ 最高效率配置: {best_efficiency_config['num_layers']}层")
            print(f"   效率比: {max(efficiency_scores):.4f} (召回率/秒)")

            # 质量分析
            quality_scores = [r.get('feature_quality_index', 0) for r in sorted_results]
            best_quality_idx = quality_scores.index(max(quality_scores))
            best_quality_config = sorted_results[best_quality_idx]

            print(f"🔍 最佳特征质量: {best_quality_config['num_layers']}层")
            print(f"   质量指数: {max(quality_scores):.4f}")

            # 实用建议
            print(f"\n💡 实用建议:")
            if max_recall > 50:
                print(f"   ✅ 算法在该数据集上表现良好")
            elif max_recall > 20:
                print(f"   ⚠️  算法性能中等，可考虑参数优化")
            else:
                print(f"   ❌ 算法性能较差，需要算法改进")

            print(f"\n📊 详细分析报告请查看生成的文件")
            print(f"📁 单层实验报告: 分层数量-多种特征-召回率关系-实验报告/individual_reports/")


def main():
    """主函数"""
    print("开始分层数量消融实验 - subprocess分离式")
    print("分析目标: 深入理解层数变化对特征提取和召回率的因果影响")
    print("技术方案: 每层实验独立进程 + 文件配置修改，确保配置一致性")

    # 创建实验器
    experiment = SubprocessLayersExperiment()

    print(f"\n实验设计:")
    print(f"  层数范围: 1-20层")
    print(f"  数据集: Chilean地下矿井")
    print(f"  数据库: Session 100 (历史地图)")
    print(f"  查询: Session 190 (当前观测)")
    print(f"  分析维度: 轮廓统计、几何特征、检索键、BCI特征、相似性检查")
    print(f"  隔离机制: 每层独立subprocess + 配置文件修改")

    # 询问用户确认
    confirm = input(f"\n继续运行消融实验? (y/n): ")
    if confirm.lower() != 'y':
        print("实验已取消")
        return

    # 可选择实验范围
    try:
        start_layers = int(input("起始层数 (默认1): ") or "1")
        end_layers = int(input("结束层数 (默认20): ") or "20")

        if start_layers < 1 or end_layers < start_layers or end_layers > 30:
            print("层数范围无效，使用默认值 1-20")
            start_layers, end_layers = 1, 20
    except:
        print("输入无效，使用默认值 1-20")
        start_layers, end_layers = 1, 20

    # 运行实验
    start_time = time.time()
    experiment.run_all_experiments(start_layers, end_layers)
    end_time = time.time()

    print(f"\n所有消融实验完成!")
    print(f"总耗时: {(end_time - start_time) / 60:.1f} 分钟")
    print("请查看生成的详细报告了解层数对特征提取的具体影响")


if __name__ == "__main__":
    main()
