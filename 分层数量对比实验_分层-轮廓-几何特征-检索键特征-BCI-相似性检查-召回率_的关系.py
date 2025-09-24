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
        self.main_script = "contour_chilean_场景识别_不相同时段_不加旋转.py"
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
            'chilean_NoRot_NoScale_5cm_evaluation_query_180_180.pickle'
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

            # 将stdout追加到日志文件
            if result.stdout:
                with open(log_file, 'a', encoding='utf-8') as f:
                    f.write("\n=== SUBPROCESS STDOUT ===\n")
                    f.write(result.stdout)
                    f.write("\n=== END SUBPROCESS STDOUT ===\n")

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
        """解析详细统计信息"""
        stats = {}

        try:
            # 1. 轮廓统计
            frame_contours = re.findall(r'\[DB \d+\] Frame \d+: 总轮廓数: (\d+)', content)
            if frame_contours:
                contour_counts = [int(x) for x in frame_contours]
                stats['total_frames'] = len(contour_counts)
                stats['total_contours'] = sum(contour_counts)
                stats['avg_contours_per_frame'] = statistics.mean(contour_counts)
                stats['std_contours_per_frame'] = statistics.stdev(contour_counts) if len(contour_counts) > 1 else 0
                stats['min_contours_per_frame'] = min(contour_counts)
                stats['max_contours_per_frame'] = max(contour_counts)

            # 2. 轮廓尺寸分布
            contour_sizes_match = re.search(r'Contour sizes: ([\d,]+)', content)
            if contour_sizes_match:
                sizes_str = contour_sizes_match.group(1)
                sizes = [int(x) for x in sizes_str.split(',') if x.strip()]

                if sizes:
                    total_contours = len(sizes)
                    for min_size, max_size, label in self.contour_size_bins:
                        if max_size == float('inf'):
                            count = sum(1 for s in sizes if s >= min_size)
                        else:
                            count = sum(1 for s in sizes if min_size <= s <= max_size)

                        stats[f'contours_{label}_count'] = count
                        stats[f'contours_{label}_ratio'] = count / total_contours if total_contours > 0 else 0

            # 3. 几何特征
            ecc_match = re.search(r'Eccentricities: ([\d.,]+)', content)
            if ecc_match:
                ecc_str = ecc_match.group(1)
                eccentricities = [float(x) for x in ecc_str.split(',') if x.strip()]

                if eccentricities:
                    stats['avg_eccentricity'] = statistics.mean(eccentricities)
                    stats['std_eccentricity'] = statistics.stdev(eccentricities) if len(eccentricities) > 1 else 0

                    # 偏心率分布
                    for min_ecc, max_ecc, label in self.eccentricity_bins:
                        count = sum(1 for e in eccentricities if min_ecc <= e < max_ecc)
                        stats[f'{label}_ratio'] = count / len(eccentricities)

            eigenratio_match = re.search(r'Eigenvalue ratios: ([\d.,]+)', content)
            if eigenratio_match:
                ratios_str = eigenratio_match.group(1)
                ratios = [float(x) for x in ratios_str.split(',') if x.strip()]
                if ratios:
                    stats['avg_eigenvalue_ratio'] = statistics.mean(ratios)

            # 显著特征
            significant_features = re.search(r'Significant features: ecc=(\d+), com=(\d+), total=(\d+)', content)
            if significant_features:
                ecc_count, com_count, total_count = map(int, significant_features.groups())
                if total_count > 0:
                    stats['significant_ecc_features_ratio'] = ecc_count / total_count
                    stats['significant_com_features_ratio'] = com_count / total_count

            # 轮廓高度
            heights_match = re.search(r'Contour heights: ([\d.,]+)', content)
            if heights_match:
                heights_str = heights_match.group(1)
                heights = [float(x) for x in heights_str.split(',') if x.strip()]
                if heights:
                    stats['avg_contour_height'] = statistics.mean(heights)

            # 4. 检索键特征
            key_dim0_match = re.search(r'Key dimension 0: avg=([\d.]+)', content)
            if key_dim0_match:
                stats['avg_key_dimension_0'] = float(key_dim0_match.group(1))

            key_dim1_match = re.search(r'Key dimension 1: avg=([\d.]+)', content)
            if key_dim1_match:
                stats['avg_key_dimension_1'] = float(key_dim1_match.group(1))

            key_dim2_match = re.search(r'Key dimension 2: avg=([\d.]+)', content)
            if key_dim2_match:
                stats['avg_key_dimension_2'] = float(key_dim2_match.group(1))

            sparsity_match = re.search(r'Key sparsity: ([\d.]+)', content)
            if sparsity_match:
                stats['key_sparsity_ratio'] = float(sparsity_match.group(1))

            ring_activation_match = re.search(r'Ring feature activation: ([\d.]+)', content)
            if ring_activation_match:
                stats['ring_feature_activation'] = float(ring_activation_match.group(1))

            # 5. BCI特征
            neighbors_match = re.search(r'BCI neighbors: ([\d,]+)', content)
            if neighbors_match:
                neighbors_str = neighbors_match.group(1)
                neighbor_counts = [int(x) for x in neighbors_str.split(',') if x.strip()]

                if neighbor_counts:
                    stats['avg_neighbors_per_bci'] = statistics.mean(neighbor_counts)
                    stats['std_neighbors_per_bci'] = statistics.stdev(neighbor_counts) if len(
                        neighbor_counts) > 1 else 0
                    stats['max_neighbors_per_bci'] = max(neighbor_counts)
                    stats['total_bcis'] = len(neighbor_counts)

            distances_match = re.search(r'Neighbor distances: ([\d.,\-]+)', content)
            if distances_match:
                distances_str = distances_match.group(1)
                distances = [float(x) for x in distances_str.split(',') if x.strip()]

                if distances:
                    stats['avg_neighbor_distance'] = statistics.mean(distances)
                    stats['std_neighbor_distance'] = statistics.stdev(distances) if len(distances) > 1 else 0

            angles_match = re.search(r'Neighbor angles: ([\d.,\-]+)', content)
            if angles_match:
                angles_str = angles_match.group(1)
                angles = [float(x) for x in angles_str.split(',') if x.strip()]

                if angles:
                    stats['avg_neighbor_angle_diversity'] = statistics.stdev(angles) if len(angles) > 1 else 0

            cross_layer_match = re.search(r'Cross layer connections: (\d+)/(\d+)', content)
            if cross_layer_match:
                cross_layer_count = int(cross_layer_match.group(1))
                total_connections = int(cross_layer_match.group(2))
                if total_connections > 0:
                    stats['cross_layer_connections_ratio'] = cross_layer_count / total_connections

            bit_activation_match = re.search(r'Distance bit activation: ([\d.]+)', content)
            if bit_activation_match:
                stats['distance_bit_activation_rate'] = float(bit_activation_match.group(1))

            # 计算星座复杂度
            avg_neighbors = stats.get('avg_neighbors_per_bci', 0)
            angle_diversity = stats.get('avg_neighbor_angle_diversity', 0)
            if avg_neighbors > 0 and angle_diversity > 0:
                stats['constellation_complexity'] = avg_neighbors * angle_diversity / 10.0

            # 6. 相似性检查统计
            check1_matches = re.findall(r'After check 1: (\d+)', content)
            check2_matches = re.findall(r'After check 2: (\d+)', content)
            check3_matches = re.findall(r'After check 3: (\d+)', content)

            if check1_matches and check2_matches and check3_matches:
                check1_avg = statistics.mean([int(x) for x in check1_matches])
                check2_avg = statistics.mean([int(x) for x in check2_matches])
                check3_avg = statistics.mean([int(x) for x in check3_matches])

                # 尝试从日志中解析实际的查询数量
                total_queries = None

                # 方法1：直接查找总查询数标记
                total_queries_match = re.search(r'总查询数[：:]\s*(\d+)', content)
                if total_queries_match:
                    total_queries = int(total_queries_match.group(1))

                # 方法2：查找成功评估数
                if not total_queries:
                    eval_match = re.search(r'成功评估数[：:]\s*(\d+)', content)
                    if eval_match:
                        total_queries = int(eval_match.group(1))

                # 方法3：从查询匹配数量推算
                if not total_queries:
                    total_queries = len(check1_matches) * 10  # 假设每个匹配代表10个查询

                # 方法4：最保守的估算
                if not total_queries or total_queries == 0:
                    total_queries = max(100, int(check1_avg))  # 至少100个查询

                # 计算通过率
                if total_queries > 0:
                    stats['contour_similarity_pass_rate'] = check1_avg / total_queries
                if check1_avg > 0:
                    stats['constellation_similarity_pass_rate'] = check2_avg / check1_avg
                if check2_avg > 0:
                    stats['pairwise_similarity_pass_rate'] = check3_avg / check2_avg

                # 记录用于调试的信息
                stats['total_queries_estimated'] = total_queries
                stats['check_matches_count'] = len(check1_matches)

            # 7. 计算特征质量指数
            stats['feature_quality_index'] = self._calculate_feature_quality_index(stats)

        except Exception as e:
            print(f"解析详细统计失败: {e}")

        return stats

    def _calculate_feature_quality_index(self, stats: Dict) -> float:
        """计算特征质量指数"""
        try:
            quality_factors = []

            # 轮廓质量因子
            avg_contours = stats.get('avg_contours_per_frame', 0)
            if avg_contours > 0:
                contour_quality = min(1.0, avg_contours / 200.0)
                small_contour_penalty = stats.get('contours_小轮廓_ratio', 0) * 0.5
                contour_quality = max(0.1, contour_quality - small_contour_penalty)
                quality_factors.append(contour_quality)

            # 几何特征质量因子
            avg_ecc = stats.get('avg_eccentricity', 0)
            if avg_ecc > 0:
                ecc_quality = 1.0 - abs(avg_ecc - 0.5) * 2
                ecc_quality = max(0.1, ecc_quality)
                quality_factors.append(ecc_quality)

            # 检索键质量因子
            key_sparsity = stats.get('key_sparsity_ratio', 1.0)
            if key_sparsity < 1.0:
                key_quality = 1.0 - key_sparsity
                quality_factors.append(key_quality)

            # BCI连接质量因子
            avg_neighbors = stats.get('avg_neighbors_per_bci', 0)
            if avg_neighbors > 0:
                if 3 <= avg_neighbors <= 8:
                    bci_quality = 1.0
                elif avg_neighbors < 3:
                    bci_quality = avg_neighbors / 3.0
                else:
                    bci_quality = max(0.1, 1.0 - (avg_neighbors - 8) * 0.1)
                quality_factors.append(bci_quality)

            if quality_factors:
                return statistics.mean(quality_factors)
            else:
                return 0.0

        except Exception as e:
            print(f"计算特征质量指数失败: {e}")
            return 0.0

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

        report_file = "subprocess_layers_comprehensive_report.txt"

        with open(report_file, 'w', encoding='utf-8') as f:
            f.write("分层数量消融实验综合报告 - subprocess分离式\n")
            f.write("=" * 100 + "\n")
            f.write(f"生成时间: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"实验数量: {len(self.results)}\n")
            f.write(f"分析目的: 深入理解层数变化对特征提取和召回率的影响\n\n")

            # 基本性能表格
            f.write("基本性能结果:\n")
            f.write("-" * 120 + "\n")
            header = (f"{'层数':<4} {'层间距':<8} {'Recall@1':<10} {'相似性':<8} {'运行时长':<8} "
                      f"{'平均轮廓':<8} {'特征质量':<8} {'小轮廓比例':<10} {'BCI邻居':<8}\n")
            f.write(header)
            f.write("-" * 120 + "\n")

            # 按层数排序
            sorted_results = sorted(self.results, key=lambda x: x['num_layers'])

            for result in sorted_results:
                line = (f"{result['num_layers']:<4d} {result['layer_interval']:<8.3f} "
                        f"{result['recall_1']:<10.2f} {result['average_similarity']:<8.4f} "
                        f"{result['duration']:<8.0f} {result.get('avg_contours_per_frame', 0):<8.1f} "
                        f"{result.get('feature_quality_index', 0):<8.3f} "
                        f"{result.get('contours_小轮廓_ratio', 0):<10.2f} "
                        f"{result.get('avg_neighbors_per_bci', 0):<8.1f}\n")
                f.write(line)

            f.write("-" * 120 + "\n\n")

            # 性能趋势分析
            recall_values = [r['recall_1'] for r in sorted_results]
            if recall_values:
                max_recall = max(recall_values)
                max_idx = recall_values.index(max_recall)
                optimal_layers = sorted_results[max_idx]['num_layers']

                f.write("性能趋势分析:\n")
                f.write("-" * 50 + "\n")
                f.write(f"最高召回率: {max_recall:.2f}% (在{optimal_layers}层时达到)\n")
                f.write(f"最优层间距: {sorted_results[max_idx]['layer_interval']:.3f}m\n")

                # 寻找性能下降点
                decline_threshold = max_recall * 0.95
                decline_start = -1

                for i in range(max_idx, len(recall_values)):
                    if recall_values[i] < decline_threshold:
                        decline_start = i
                        break

                if decline_start != -1:
                    f.write(f"性能开始下降: 从{sorted_results[decline_start]['num_layers']}层开始\n")
                    f.write(f"过分割阈值: 约{sorted_results[decline_start]['layer_interval']:.3f}m层间距\n")
                else:
                    f.write("在测试范围内未观察到明显性能下降\n")
                    f.write("建议扩展到更多层数以找到性能下降点\n")

            f.write("\n")

            # 特征变化分析
            f.write("特征变化分析:\n")
            f.write("-" * 50 + "\n")

            # 轮廓碎片化趋势
            small_ratios = [r.get('contours_小轮廓_ratio', 0) for r in sorted_results]
            if len(small_ratios) >= 5:
                early_frag = statistics.mean(small_ratios[:5])
                late_frag = statistics.mean(small_ratios[-5:])

                f.write(f"轮廓碎片化趋势: {early_frag:.1%} -> {late_frag:.1%}\n")
                if late_frag > early_frag * 1.2:
                    f.write("结论: 存在明显的轮廓碎片化趋势\n")
                else:
                    f.write("结论: 轮廓碎片化不明显\n")

            # BCI复杂度变化
            neighbor_counts = [r.get('avg_neighbors_per_bci', 0) for r in sorted_results]
            if len(neighbor_counts) >= 5:
                early_neighbors = statistics.mean(neighbor_counts[:5])
                late_neighbors = statistics.mean(neighbor_counts[-5:])
                f.write(f"BCI复杂度变化: {early_neighbors:.1f} -> {late_neighbors:.1f} 平均邻居数\n")

            # 检索键稀疏性变化
            sparsity_values = [r.get('key_sparsity_ratio', 0) for r in sorted_results]
            if len(sparsity_values) >= 5:
                early_sparsity = statistics.mean(sparsity_values[:5])
                late_sparsity = statistics.mean(sparsity_values[-5:])
                f.write(f"检索键稀疏性变化: {early_sparsity:.1%} -> {late_sparsity:.1%}\n")

            f.write("\n")

            # 因果关系分析
            f.write("因果关系分析:\n")
            f.write("=" * 60 + "\n")

            if len(sorted_results) >= 10:
                f.write("特征指标与召回率的关系分析:\n\n")

                # 层间距与召回率关系
                intervals = [r['layer_interval'] for r in sorted_results]
                recalls = [r['recall_1'] for r in sorted_results]

                f.write("1. 层间距与召回率关系:\n")

                # 找出召回率前25%的层间距范围
                top_25_percent_count = max(1, len(recalls) // 4)
                top_indices = sorted(range(len(recalls)), key=lambda i: recalls[i], reverse=True)[:top_25_percent_count]
                top_intervals = [intervals[i] for i in top_indices]

                if top_intervals:
                    f.write(f"   最优层间距区间: {min(top_intervals):.3f}m - {max(top_intervals):.3f}m\n")
                    f.write(f"   对应层数范围: {min([sorted_results[i]['num_layers'] for i in top_indices])}-"
                            f"{max([sorted_results[i]['num_layers'] for i in top_indices])}层\n")

                # 轮廓质量与性能关系
                f.write("\n2. 轮廓碎片化对性能的影响:\n")
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

                # BCI复杂度与性能关系
                f.write("\n3. BCI复杂度对性能的影响:\n")
                bci_neighbors = [r.get('avg_neighbors_per_bci', 0) for r in sorted_results]

                # 分析理想的邻居数量范围
                ideal_neighbor_configs = []
                for i, result in enumerate(sorted_results):
                    neighbor_count = bci_neighbors[i]
                    if 3 <= neighbor_count <= 8:  # 理想的邻居数量
                        ideal_neighbor_configs.append((result['num_layers'], recalls[i], neighbor_count))

                if ideal_neighbor_configs:
                    best_ideal = max(ideal_neighbor_configs, key=lambda x: x[1])
                    f.write(f"   BCI邻居数适中时的最佳性能: {best_ideal[0]}层 "
                            f"({best_ideal[2]:.1f}个邻居, {best_ideal[1]:.1f}%召回率)\n")

                # 检索键质量与性能关系
                f.write("\n4. 检索键质量对性能的影响:\n")
                key_qualities = [1 - r.get('key_sparsity_ratio', 1) for r in sorted_results]  # 质量 = 1 - 稀疏性

                # 找出检索键质量高且性能好的配置
                high_quality_key_configs = []
                for i, result in enumerate(sorted_results):
                    key_quality = key_qualities[i]
                    if key_quality > 0.5:  # 高质量检索键
                        high_quality_key_configs.append((result['num_layers'], recalls[i], key_quality))

                if high_quality_key_configs:
                    best_key_quality = max(high_quality_key_configs, key=lambda x: x[1])
                    f.write(f"   检索键质量高时的最佳性能: {best_key_quality[0]}层 "
                            f"(质量={best_key_quality[2]:.1%}, {best_key_quality[1]:.1f}%召回率)\n")

            # 推荐建议
            f.write("\n推荐建议:\n")
            f.write("=" * 40 + "\n")

            if sorted_results:
                best_overall = max(sorted_results, key=lambda x: x['recall_1'])
                best_quality = max(sorted_results, key=lambda x: x.get('feature_quality_index', 0))
                best_efficiency = max(sorted_results, key=lambda x: x['recall_1'] / max(x['duration'], 1))

                f.write(f"1. 最佳性能配置: {best_overall['num_layers']}层 ")
                f.write(f"(召回率: {best_overall['recall_1']:.1f}%)\n")

                f.write(f"2. 最佳特征质量配置: {best_quality['num_layers']}层 ")
                f.write(f"(质量指数: {best_quality.get('feature_quality_index', 0):.3f})\n")

                f.write(f"3. 最高效率配置: {best_efficiency['num_layers']}层 ")
                f.write(f"(效率比: {best_efficiency['recall_1'] / max(best_efficiency['duration'], 1):.4f})\n")

                # 实际应用建议
                f.write("\n实际应用建议:\n")
                high_perf_configs = [r for r in sorted_results if r['recall_1'] >= max(recall_values) * 0.95]
                if high_perf_configs:
                    recommended = min(high_perf_configs, key=lambda x: x['duration'])
                    f.write(f"- 推荐配置: {recommended['num_layers']}层 ")
                    f.write(f"(平衡性能{recommended['recall_1']:.1f}%和效率{recommended['duration']:.0f}s)\n")
                    f.write(f"- 层间距: {recommended['layer_interval']:.3f}m\n")
                    f.write(f"- 特征质量指数: {recommended.get('feature_quality_index', 0):.3f}\n")

            # 详细层级分析表格
            f.write("\n\n详细特征分析表格:\n")
            f.write("=" * 150 + "\n")
            detail_header = (f"{'层数':<4} {'轮廓数':<6} {'极小轮廓%':<8} {'偏心率':<8} {'检索键维度0':<10} "
                             f"{'BCI邻居':<8} {'距离位激活':<10} {'特征质量':<8} {'召回率':<8}\n")
            f.write(detail_header)
            f.write("-" * 150 + "\n")

            for result in sorted_results:
                detail_line = (f"{result['num_layers']:<4d} "
                               f"{result.get('avg_contours_per_frame', 0):<6.1f} "
                               f"{result.get('contours_极小轮廓_ratio', 0) * 100:<8.1f} "
                               f"{result.get('avg_eccentricity', 0):<8.3f} "
                               f"{result.get('avg_key_dimension_0', 0):<10.2f} "
                               f"{result.get('avg_neighbors_per_bci', 0):<8.1f} "
                               f"{result.get('distance_bit_activation_rate', 0) * 100:<10.1f} "
                               f"{result.get('feature_quality_index', 0):<8.3f} "
                               f"{result['recall_1']:<8.2f}\n")
                f.write(detail_line)

            f.write("-" * 150 + "\n")

        print(f"\n综合分析报告已保存到: {report_file}")

        # 输出关键发现
        self._print_key_findings()

    def _print_key_findings(self):
        """输出关键发现"""
        print("\n" + "=" * 80)
        print("关键发现总结")
        print("=" * 80)

        if not self.results:
            return

        sorted_results = sorted(self.results, key=lambda x: x['num_layers'])
        recall_values = [r['recall_1'] for r in sorted_results]

        if recall_values:
            max_recall = max(recall_values)
            max_idx = recall_values.index(max_recall)
            optimal_config = sorted_results[max_idx]

            print(f"最佳性能: {optimal_config['num_layers']}层")
            print(f"   召回率: {max_recall:.2f}%")
            print(f"   层间距: {optimal_config['layer_interval']:.3f}m")
            print(f"   特征质量: {optimal_config.get('feature_quality_index', 0):.3f}")

            # 分析性能趋势
            if len(recall_values) >= 10:
                # 寻找性能峰值和下降点
                threshold = max_recall * 0.9
                for i in range(max_idx + 1, len(recall_values)):
                    if recall_values[i] < threshold:
                        decline_layers = sorted_results[i]['num_layers']
                        print(f"性能下降点: {decline_layers}层")
                        print(f"   过分割阈值: 约{sorted_results[i]['layer_interval']:.3f}m")
                        break
                else:
                    print("在测试范围内未发现明显的过分割现象")
                    print("   建议扩展到更多层数 (如25-30层) 来找到性能下降点")

            # 轮廓碎片化分析
            small_ratios = [r.get('contours_小轮廓_ratio', 0) for r in sorted_results]
            if small_ratios and len(small_ratios) >= 5:
                early_frag = statistics.mean(small_ratios[:5])
                late_frag = statistics.mean(small_ratios[-5:])

                if late_frag > early_frag * 1.5:
                    print(f"轮廓碎片化趋势: {early_frag:.1%} -> {late_frag:.1%}")
                    print("   证实了过分割导致轮廓碎片化的假设")

        print(f"\n详细报告: subprocess_layers_comprehensive_report.txt")
        print(f"单层实验日志: layers_N_experiment.log")


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
    print(f"  查询: Session 180 (当前观测)")
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
