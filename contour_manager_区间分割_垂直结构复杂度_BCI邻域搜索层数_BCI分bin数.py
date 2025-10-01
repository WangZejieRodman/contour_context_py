"""
Contour Context Loop Closure Detection - Contour Manager
轮廓管理器实现 - 修复版本
"""

import numpy as np
import cv2
from typing import List, Tuple, Dict, Optional, Any
from dataclasses import dataclass
import os

from contour_types import (
    ContourManagerConfig, ContourViewStatConfig, ContourSimThresConfig,
    ConstellationPair, BCI, RelativePoint, RunningStatRecorder,
    RET_KEY_DIM, DIST_BIN_LAYERS, NUM_BIN_KEY_LAYER, BITS_PER_LAYER,
    gauss_pdf, clamp_angle
)
from contour_view import ContourView


class ContourManager:
    """轮廓管理器类"""

    def __init__(self, config: ContourManagerConfig, int_id: int):
        """
        初始化轮廓管理器

        Args:
            config: 配置
            int_id: 整数ID
        """
        self.cfg = config
        self.view_stat_cfg = ContourViewStatConfig()
        self.int_id = int_id
        self.str_id = ""

        # 验证配置
        assert config.n_col % 2 == 0
        assert config.n_row % 2 == 0
        assert len(config.lv_grads) > 0

        # 坐标范围
        self.x_min = -(config.n_row // 2) * config.reso_row
        self.x_max = -self.x_min
        self.y_min = -(config.n_col // 2) * config.reso_col
        self.y_max = -self.y_min

        # 数据存储
        self.bev = None
        self.cont_views: List[List[ContourView]] = [[] for _ in range(len(config.lv_grads))]
        self.cont_perc: List[List[float]] = [[] for _ in range(len(config.lv_grads))]
        self.layer_cell_cnt: List[int] = [0] * len(config.lv_grads)
        self.layer_keys: List[List[np.ndarray]] = [[] for _ in range(len(config.lv_grads))]
        self.layer_key_bcis: List[List[BCI]] = [[] for _ in range(len(config.lv_grads))]

        # BEV像素信息
        self.bev_pixfs: List[Tuple[int, Tuple[float, float, float]]] = []
        self.max_bin_val = -float('inf')
        self.min_bin_val = float('inf')

        # 初始化BEV
        self._init_bev()

    def _init_bev(self):
        """初始化BEV图像"""
        self.bev = np.full((self.cfg.n_row, self.cfg.n_col), -1000.0, dtype=np.float32)

    def hash_point_to_image(self, pt: np.ndarray) -> Tuple[int, int]:
        """
        将点映射到图像坐标

        Args:
            pt: 点坐标 [x, y, z]

        Returns:
            (row, col) 或 (-1, -1) 如果点在范围外
        """
        padding = 1e-2
        x, y = pt[0], pt[1]

        # 检查范围
        if (x < self.x_min + padding or x > self.x_max - padding or
                y < self.y_min + padding or y > self.y_max - padding or
                (y * y + x * x) < self.cfg.blind_sq):
            return -1, -1

        row = int(np.floor(x / self.cfg.reso_row)) + self.cfg.n_row // 2
        col = int(np.floor(y / self.cfg.reso_col)) + self.cfg.n_col // 2

        # 验证范围
        if not (0 <= row < self.cfg.n_row and 0 <= col < self.cfg.n_col):
            return -1, -1

        return row, col

    def point_to_cont_row_col(self, p_in_l: np.ndarray) -> np.ndarray:
        """
        将激光雷达坐标系中的点转换到连续图像坐标系

        Args:
            p_in_l: 激光雷达坐标系中的点 [x, y]

        Returns:
            连续的行列坐标
        """
        continuous_rc = np.array([
            p_in_l[0] / self.cfg.reso_row + self.cfg.n_row / 2 - 0.5,
            p_in_l[1] / self.cfg.reso_col + self.cfg.n_col / 2 - 0.5
        ], dtype=np.float32)
        return continuous_rc

    def make_bev(self, point_cloud: np.ndarray, str_id: str = ""):
        """
        从点云生成BEV图像

        Args:
            point_cloud: 点云数组，形状为 [N, 3] 或 [N, 4]
            str_id: 字符串ID
        """
        print(f"DEBUG: make_bev() 开始执行 for {str_id}")
        print(f"[BEV_DEBUG] {str_id}: pointcloud shape={point_cloud.shape}")
        print(f"[BEV_DEBUG] {str_id}: pointcloud hash={hash(point_cloud.tobytes())}")
        print(f"[BEV_DEBUG] {str_id}: first 3 points=\n{point_cloud[:3]}")

        assert point_cloud.shape[0] > 10, "点云数量太少"

        self.str_id = str_id if str_id else f"scan_{self.int_id}"

        # 清空之前的数据
        self.bev_pixfs.clear()
        self._init_bev()

        # 为每个栅格保存层级掩码,用作后面环形特征生成时候，计算某像素所在位置有多少个层级存在结构，反映该像素位置的"垂直结构丰富度"
        lv_grads = self.cfg.lv_grads
        num_levels = len(lv_grads) - 1  # 10个层级
        self.layer_masks = np.zeros((self.cfg.n_row, self.cfg.n_col, num_levels), dtype=bool)

        tmp_pillars = {}

        # 处理每个点
        for pt in point_cloud:
            row, col = self.hash_point_to_image(pt)
            if row >= 0:
                height = self.cfg.lidar_height + pt[2]

                # 更新最大高度（保持原有逻辑）
                if self.bev[row, col] < height:
                    self.bev[row, col] = height
                    # 计算连续坐标
                    coor_f = self.point_to_cont_row_col(pt[:2])
                    hash_key = row * self.cfg.n_col + col
                    tmp_pillars[hash_key] = (coor_f[0], coor_f[1], height)

                # 判断该点属于哪个层级并记录
                for level in range(num_levels):
                    h_min = lv_grads[level]
                    h_max = lv_grads[level + 1]
                    if h_min <= height < h_max:
                        self.layer_masks[row, col, level] = True
                        # 不break，因为一个像素位置可能有多个层级的点

                # 更新范围
                self.max_bin_val = max(self.max_bin_val, height)
                self.min_bin_val = min(self.min_bin_val, height)

        # 转换为列表格式
        self.bev_pixfs = [(k, v) for k, v in tmp_pillars.items()]
        self.bev_pixfs.sort(key=lambda x: x[0])  # 按hash键排序

        print(f"Max/Min bin height: {self.max_bin_val:.3f} {self.min_bin_val:.3f}")
        print(f"Continuous Pos size: {len(self.bev_pixfs)}")

        # 新增：输出层级掩码统计信息
        total_pixels_with_data = np.sum(np.any(self.layer_masks, axis=2))
        layer_counts = np.sum(self.layer_masks, axis=(0, 1))
        print(f"Pixels with data: {total_pixels_with_data}")
        print(f"Points per level: {layer_counts}")
        print(f"DEBUG: make_bev() 执行完成 for {str_id}")

    def make_contours_recursive(self):
        """递归生成轮廓 - 修改为区间分割模式"""
        print("DEBUG: make_contours_recursive() 开始执行")
        full_roi = (0, 0, self.cfg.n_col, self.cfg.n_row)
        mask = np.ones((1, 1), dtype=np.uint8)

        # 修改：直接处理每个高度区间，不再递归
        self._make_contours_interval_based(full_roi)
        print("DEBUG: _make_contours_interval_based() 执行完成")

        # 对每层的轮廓按面积排序并计算百分比
        for ll in range(len(self.cont_views)):
            self.cont_views[ll].sort(key=lambda x: x.cell_cnt, reverse=True)

            # 计算层级总像素数
            self.layer_cell_cnt[ll] = sum(cont.cell_cnt for cont in self.cont_views[ll])

            # 计算每个轮廓的面积百分比
            self.cont_perc[ll] = []
            for cont in self.cont_views[ll]:
                if self.layer_cell_cnt[ll] > 0:
                    perc = cont.cell_cnt / self.layer_cell_cnt[ll]
                else:
                    perc = 0.0
                self.cont_perc[ll].append(perc)

        print("DEBUG: 轮廓排序和百分比计算完成")

        # 确保这些函数在方法末尾被调用
        contour_stats = self._output_detailed_contour_statistics()  # 修改这行：接收返回值
        self._make_retrieval_keys()

    def _make_contours_interval_based(self, cc_roi: Tuple[int, int, int, int]):
        """
        基于区间的轮廓提取 - 修复版本

        正确的区间分割：
        - L0: [lv_grads[0], lv_grads[1])  = [0.5, 1.0)
        - L1: [lv_grads[1], lv_grads[2])  = [1.0, 1.5)
        - L2: [lv_grads[2], lv_grads[3])  = [1.5, 2.0)
        - ...
        - L7: [lv_grads[7], lv_grads[8])  = [4.0, 4.5)
        """
        x, y, w, h = cc_roi
        bev_roi = self.bev[y:y + h, x:x + w]

        lv_grads = self.cfg.lv_grads

        # ✅ 修复：应该是 len(lv_grads) - 1 个层级
        num_levels = len(lv_grads) - 1  # 11个阈值点定义10个区间

        for level in range(num_levels):
            print(f"[INTERVAL_DEBUG] Processing level {level}")

            # ✅ 修复：正确的区间定义
            h_min = lv_grads[level]  # 当前区间的下界
            h_max = lv_grads[level + 1]  # 当前区间的上界

            print(f"[INTERVAL_DEBUG] Level {level}: height range [{h_min:.2f}, {h_max:.2f})")

            # 创建区间掩码：[h_min, h_max)
            interval_mask = ((bev_roi >= h_min) & (bev_roi < h_max)).astype(np.uint8) * 255

            print(f"[INTERVAL_DEBUG] Level {level}: {np.sum(interval_mask > 0)} pixels in interval")

            # 连通组件分析
            # 输入: interval_mask - 当前层级的二值图像（0或255）
            # 算法: OpenCV使用8连通性扫描所有白色像素区域
            # 输出: num_labels = 连通区域数 + 1（背景labels=0）
            num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
                interval_mask, connectivity=8)

            # 处理每个连通的像素区域
            for n in range(1, num_labels):  # 跳过背景（labels=0）
                if stats[n, cv2.CC_STAT_AREA] < self.cfg.min_cont_cell_cnt:
                    continue

                # 获取组件的边界框
                comp_x, comp_y, comp_w, comp_h = stats[n, :4]

                # 转换为全局坐标
                global_roi = (comp_x + x, comp_y + y, comp_w, comp_h)

                # 创建组件掩码
                mask_n = (labels[comp_y:comp_y + comp_h, comp_x:comp_x + comp_w] == n).astype(np.uint8)

                # 初始化统计记录器
                rec = RunningStatRecorder()
                poi_r, poi_c = -1, -1

                # 遍历组件内的每个像素
                for i in range(comp_h):
                    for j in range(comp_w):
                        if mask_n[i, j]:
                            global_r = i + global_roi[1]
                            global_c = j + global_roi[0]
                            poi_r, poi_c = global_r, global_c

                            # 查找连续坐标
                            q_hash = global_r * self.cfg.n_col + global_c
                            pixf = self._search_pixf(q_hash)
                            if pixf:
                                rec.running_stats(pixf[0], pixf[1], self.bev[global_r, global_c])

                if poi_r >= 0:
                    # 创建轮廓视图
                    contour = ContourView(level, poi_r, poi_c)
                    contour.calc_stat_vals(rec, self.view_stat_cfg)
                    self.cont_views[level].append(contour)#轮廓被添加到 self.cont_views[level] 列表中

            print(f"[INTERVAL_DEBUG] Level {level}: extracted {len(self.cont_views[level])} contours")

    def _search_pixf(self, q_hash: int) -> Optional[Tuple[float, float, float]]:
        """搜索像素浮点数据"""
        # 二分搜索
        left, right = 0, len(self.bev_pixfs) - 1
        while left <= right:
            mid = (left + right) // 2
            if self.bev_pixfs[mid][0] == q_hash:
                return self.bev_pixfs[mid][1]
            elif self.bev_pixfs[mid][0] < q_hash:
                left = mid + 1
            else:
                right = mid - 1
        return None

    def _make_retrieval_keys(self):
        """生成检索键"""
        roi_radius_padded = int(np.ceil(self.cfg.roi_radius + 1))

        for ll in range(len(self.cfg.lv_grads)-1):
            accumulate_cell_cnt = 0

            for seq in range(self.cfg.piv_firsts):
                key = np.zeros(RET_KEY_DIM, dtype=np.float32)
                bci = BCI(seq, ll)

                if seq < len(self.cont_views[ll]):
                    accumulate_cell_cnt += self.cont_views[ll][seq].cell_cnt

                if (seq < len(self.cont_views[ll]) and
                        self.cont_views[ll][seq].cell_cnt >= self.cfg.min_cont_key_cnt):
                    v_cen = self.cont_views[ll][seq].pos_mean
                    r_cen, c_cen = int(v_cen[0]), int(v_cen[1])

                    # 定义搜索区域
                    r_min = max(0, r_cen - roi_radius_padded)
                    r_max = min(self.cfg.n_row - 1, r_cen + roi_radius_padded)
                    c_min = max(0, c_cen - roi_radius_padded)
                    c_max = min(self.cfg.n_col - 1, c_cen + roi_radius_padded)

                    # 生成环形特征
                    key = self._generate_ring_features(v_cen, r_min, r_max, c_min, c_max,
                                                       accumulate_cell_cnt, ll, seq)  # ✅ 传递层级和序列

                    # 生成二进制星座标识
                    self._generate_bci(bci, ll, seq, v_cen)

                self.layer_key_bcis[ll].append(bci) #bci：邻居（相关）轮廓信息，包含dist_bin(二进制位串记录邻居距离分布)、nei_pts(邻居轮廓的详细信息列表)、nei_idx_segs(按距离分组的索引段)、piv_seq(中心轮廓序号)、level(所属层级)。
                self.layer_keys[ll].append(key) #key：当前轮廓信息，10维特征：[最大特征值×像素数, 最小特征值×像素数, 累积像素数平方根, 10-3个环形分布bins]
            print(f"第{ll}层-全部轮廓的key和bci已生成")

        print(f"全部层-全部轮廓的key和bci已生成")
        # ===== 输出检索键统计信息 =====
        print("DEBUG: 准备输出检索键统计")
        key_stats = self._output_retrieval_key_statistics()
        print("DEBUG: _output_retrieval_key_statistics() 执行完成")

        # ===== BCI统计输出 =====
        print("DEBUG: 准备输出BCI统计")
        bci_stats = self._output_bci_statistics()
        print("DEBUG: _output_bci_statistics() 执行完成")

    def _generate_ring_features(self, v_cen: np.ndarray, r_min: int, r_max: int,
                                c_min: int, c_max: int, accumulate_cell_cnt: int,
                                current_level: int, current_seq: int) -> np.ndarray:
        """生成环形特征"""
        key = np.zeros(RET_KEY_DIM, dtype=np.float32)

        # 前3维：基础特征（特征值+面积）
        if (current_level < len(self.cont_views) and
                current_seq < len(self.cont_views[current_level])):
            cont = self.cont_views[current_level][current_seq]

            # ✅ 添加详细调试输出
            print(f"[KEY_DEBUG] {self.str_id} L{current_level}S{current_seq}: "
                  f"eig_vals=[{cont.eig_vals[0]:.6f}, {cont.eig_vals[1]:.6f}], "
                  f"cell_cnt={cont.cell_cnt}")

            key[0] = np.sqrt(cont.eig_vals[1] * cont.cell_cnt)  # 最大特征值 * 计数
            key[1] = np.sqrt(cont.eig_vals[0] * cont.cell_cnt)  # 最小特征值 * 计数

            print(f"[KEY_DEBUG] {self.str_id} L{current_level}S{current_seq}: "
                  f"key[0]={key[0]:.6f}, key[1]={key[1]:.6f}")

        key[2] = np.sqrt(accumulate_cell_cnt)

        # 第4维及以上：环形分布特征（仅当RET_KEY_DIM > 3时计算）
        if RET_KEY_DIM > 3:
            # 计算实际需要的环形特征数量
            max_ring_bins = 7  # 原始设计最多7个环形bins
            available_dims = RET_KEY_DIM - 3  # 可用于环形特征的维度数
            num_bins = min(available_dims, max_ring_bins)

            if num_bins > 0:
                bin_len = self.cfg.roi_radius / num_bins
                ring_bins = np.zeros(num_bins)

                div_per_bin = 5
                discrete_divs = np.zeros(num_bins * div_per_bin)
                div_len = self.cfg.roi_radius / (num_bins * div_per_bin)
                cnt_point = 0

                # 遍历ROI区域所有像素
                for rr in range(r_min, r_max + 1):
                    for cc in range(c_min, c_max + 1):
                        # 检查是否在搜索半径内
                        q_hash = rr * self.cfg.n_col + cc
                        pixf = self._search_pixf(q_hash)
                        if not pixf:
                            continue

                        pos = np.array([pixf[0], pixf[1]])
                        dist = np.linalg.norm(pos - v_cen)

                        if dist < self.cfg.roi_radius - 1e-2:
                            # 根据垂直结构复杂度开关选择计算方式
                            if self.cfg.use_vertical_complexity:
                                # 使用垂直结构复杂度
                                higher_cnt = np.sum(self.layer_masks[rr, cc, :])
                            else:
                                # 简单计数：只要该位置有数据就计为1
                                higher_cnt = 1 if np.any(self.layer_masks[rr, cc, :]) else 0

                            if higher_cnt > 0:
                                cnt_point += 1

                                # 使用高斯分布分配到bins
                                for div_idx in range(num_bins * div_per_bin):
                                    center = div_idx * div_len + 0.5 * div_len
                                    discrete_divs[div_idx] += higher_cnt * gauss_pdf(center, dist, 1.0)

                # 合并bins
                for b in range(num_bins):
                    for d in range(div_per_bin):
                        ring_bins[b] += discrete_divs[b * div_per_bin + d]
                    if cnt_point > 0:
                        ring_bins[b] *= bin_len / np.sqrt(cnt_point)

                # 填充键的环形部分（第4-10维或更少）
                key[3:3 + num_bins] = ring_bins

        # 扩展维度特征（用于12维、15维、20维实验）
        if RET_KEY_DIM > 10:
            extra_dims_needed = RET_KEY_DIM - 10

            if (current_level < len(self.cont_views) and
                    current_seq < len(self.cont_views[current_level])):
                cont = self.cont_views[current_level][current_seq]

                # 第11维：偏心率
                if extra_dims_needed >= 1:
                    key[10] = cont.eccen if hasattr(cont, 'eccen') else 0.0

                # 第12维：平均高度
                if extra_dims_needed >= 2:
                    key[11] = cont.vol3_mean if hasattr(cont, 'vol3_mean') else 0.0

                # 第13维：质心到几何中心距离
                if extra_dims_needed >= 3:
                    if hasattr(cont, 'com') and hasattr(cont, 'pos_mean'):
                        key[12] = np.linalg.norm(cont.com - cont.pos_mean)
                    else:
                        key[12] = 0.0

                # 第14维：轮廓复杂度（特征值比例）
                if extra_dims_needed >= 4:
                    if hasattr(cont, 'eig_vals') and cont.eig_vals[1] > 0:
                        key[13] = cont.eig_vals[0] / cont.eig_vals[1]
                    else:
                        key[13] = 0.0

                # 第15维：轮廓在层级中的相对大小
                if extra_dims_needed >= 5:
                    layer_total = sum(c.cell_cnt for c in self.cont_views[current_level])
                    if layer_total > 0:
                        key[14] = cont.cell_cnt / layer_total
                    else:
                        key[14] = 0.0

                # 第16-20维：可以添加更多几何特征
                remaining_dims = extra_dims_needed - 5
                for i in range(remaining_dims):
                    if i < 5:  # 最多再添加5个维度
                        # 可以添加其他特征，这里先填0
                        key[15 + i] = 0.0

        print(f"[KEY_DEBUG] {self.str_id} L{current_level}S{current_seq}: "
              f"final_key[0]={key[0]:.6f}, RET_KEY_DIM={RET_KEY_DIM}")

        return key

    def _generate_bci(self, bci: BCI, ll: int, seq: int, v_cen: np.ndarray):
        """生成二进制星座标识"""

        # 固定距离范围
        min_dist = 1.0
        max_dist = 20.0  # 固定为20米

        # 计算允许搜索的层级范围
        min_layer = max(0, ll - self.cfg.neighbor_layer_range)
        max_layer = min(len(self.cfg.lv_grads) - 1, ll + self.cfg.neighbor_layer_range)

        for bl in range(NUM_BIN_KEY_LAYER):
            bit_offset = bl * BITS_PER_LAYER
            layer_idx = DIST_BIN_LAYERS[bl]

            # ✅ 检查该层级是否在允许的搜索范围内
            if layer_idx < min_layer or layer_idx > max_layer:
                continue  # 跳过超出范围的层级

            # 添加边界检查
            if layer_idx >= len(self.cont_views):
                continue

            for j in range(min(self.cfg.dist_firsts, len(self.cont_views[layer_idx]))):
                if ll != layer_idx or j != seq:  # 排除自身
                    vec_cc = self.cont_views[layer_idx][j].pos_mean - v_cen
                    tmp_dist = np.linalg.norm(vec_cc)

                    # 距离范围检查（固定范围）
                    if tmp_dist <= min_dist or tmp_dist > max_dist - 1e-3:
                        continue

                    tmp_orie = np.arctan2(vec_cc[1], vec_cc[0])

                    # 使用动态bin宽度计算索引
                    bin_width = (max_dist - min_dist) / BITS_PER_LAYER
                    dist_idx = min(int(np.floor((tmp_dist - min_dist) / bin_width)), BITS_PER_LAYER - 1)
                    dist_idx += bit_offset

                    if dist_idx < BITS_PER_LAYER * NUM_BIN_KEY_LAYER:
                        bci.dist_bin[dist_idx] = True
                        bci.nei_pts.append(RelativePoint(layer_idx, j, dist_idx, tmp_dist, tmp_orie))

        # 排序并建立索引段（保持不变）
        if bci.nei_pts:
            bci.nei_pts.sort(key=lambda p: p.bit_pos)
            bci.nei_idx_segs = [0]
            for p1 in range(len(bci.nei_pts)):
                if bci.nei_pts[bci.nei_idx_segs[-1]].bit_pos != bci.nei_pts[p1].bit_pos:
                    bci.nei_idx_segs.append(p1)
            bci.nei_idx_segs.append(len(bci.nei_pts))

        print(f"第{ll}层-第{seq}个轮廓的bci已生成")

    # Getter方法
    def get_lev_retrieval_key(self, level: int) -> List[np.ndarray]:
        """获取指定层级的检索键"""
        return self.layer_keys[level]

    def get_retrieval_key(self, level: int, seq: int) -> np.ndarray:
        """获取指定层级和序列的检索键"""
        return self.layer_keys[level][seq]

    def get_lev_contours(self, level: int) -> List[ContourView]:
        """获取指定层级的轮廓"""
        return self.cont_views[level]

    def get_lev_total_pix(self, level: int) -> int:
        """获取指定层级的总像素数"""
        return self.layer_cell_cnt[level]

    def get_lev_bci(self, level: int) -> List[BCI]:
        """获取指定层级的BCI"""
        return self.layer_key_bcis[level]

    def get_bci(self, level: int, seq: int) -> BCI:
        """获取指定层级和序列的BCI"""
        return self.layer_key_bcis[level][seq]

    def get_str_id(self) -> str:
        """获取字符串ID"""
        return self.str_id

    def get_int_id(self) -> int:
        """获取整数ID"""
        return self.int_id

    def get_config(self) -> ContourManagerConfig:
        """获取配置"""
        return self.cfg

    def get_area_perc(self, level: int, seq: int) -> float:
        """获取面积百分比"""
        return self.cont_perc[level][seq]

    def get_bev_image(self) -> np.ndarray:
        """获取BEV图像"""
        return self.bev.copy()

    def get_contour_image(self, level: int) -> np.ndarray:
        """获取指定层级的轮廓图像 - 修复区间模式"""
        if self.bev is None:
            return np.zeros((self.cfg.n_row, self.cfg.n_col), dtype=np.uint8)

        lv_grads = self.cfg.lv_grads

        # ✅ 修复：使用与轮廓提取完全相同的区间定义
        if level < len(lv_grads) - 1:
            h_min = lv_grads[level]  # L3: 1.5
            h_max = lv_grads[level + 1]  # L3: 2.0
            # 创建区间掩码：[h_min, h_max)
            mask = ((self.bev >= h_min) & (self.bev < h_max)).astype(np.uint8) * 255
        else:
            # 最后一层：[lv_grads[level], +∞)
            h_min = lv_grads[level]
            mask = (self.bev >= h_min).astype(np.uint8) * 255

        return mask

    def clear_image(self):
        """清理图像以节省内存"""
        self.bev = None

    def resume_image(self):
        """从像素数据恢复图像"""
        self._init_bev()
        for hash_key, (row_f, col_f, elev) in self.bev_pixfs:
            rr = hash_key // self.cfg.n_col
            cc = hash_key % self.cfg.n_col
            if rr < self.cfg.n_row and cc < self.cfg.n_col:
                self.bev[rr, cc] = elev

    @staticmethod
    def check_constell_corresp_sim(src: 'ContourManager', tgt: 'ContourManager',
                                   cstl_in: List[ConstellationPair],
                                   lb: 'ScorePairwiseSim',
                                   cont_sim: ContourSimThresConfig) -> Tuple[
        'ScorePairwiseSim', List[ConstellationPair], List[float]]:
        """
        检查星座对应相似性

        Args:
            src: 源轮廓管理器
            tgt: 目标轮廓管理器
            cstl_in: 输入星座对列表
            lb: 下界阈值
            cont_sim: 轮廓相似性配置

        Returns:
            (分数, 过滤后的星座对, 面积百分比)
        """
        from contour_types import ScorePairwiseSim

        ret = ScorePairwiseSim()
        cstl_out = []
        area_perc = []

        # 检查个体相似性
        for pr in cstl_in:
            if ContourView.check_sim(src.cont_views[pr.level][pr.seq_src],
                                     tgt.cont_views[pr.level][pr.seq_tgt], cont_sim):
                cstl_out.append(pr)

        ret.i_indiv_sim = len(cstl_out)
        if ret.i_indiv_sim < lb.i_indiv_sim:
            return ret, cstl_out, area_perc

        # 检查方向一致性
        if len(cstl_out) > 1:
            # 计算主轴方向
            shaft_src = np.array([0.0, 0.0])
            shaft_tgt = np.array([0.0, 0.0])
            max_norm = 0.0

            for i in range(1, min(len(cstl_out), 10)):
                for j in range(i):
                    curr_shaft_src = (src.cont_views[cstl_out[i].level][cstl_out[i].seq_src].pos_mean -
                                      src.cont_views[cstl_out[j].level][cstl_out[j].seq_src].pos_mean)
                    curr_norm = np.linalg.norm(curr_shaft_src)

                    if curr_norm > max_norm:
                        max_norm = curr_norm
                        shaft_src = curr_shaft_src / curr_norm
                        shaft_tgt = ((tgt.cont_views[cstl_out[i].level][cstl_out[i].seq_tgt].pos_mean -
                                      tgt.cont_views[cstl_out[j].level][cstl_out[j].seq_tgt].pos_mean) / curr_norm)

            # 过滤方向不一致的对
            num_sim = len(cstl_out)
            i = 0
            while i < num_sim:
                sc1 = src.cont_views[cstl_out[i].level][cstl_out[i].seq_src]
                tc1 = tgt.cont_views[cstl_out[i].level][cstl_out[i].seq_tgt]

                if sc1.ecc_feat and tc1.ecc_feat:
                    theta_s = np.arccos(np.clip(np.dot(shaft_src, sc1.eig_vecs[:, 1]), -1, 1))
                    theta_t = np.arccos(np.clip(np.dot(shaft_tgt, tc1.eig_vecs[:, 1]), -1, 1))

                    from contour_types import diff_delt
                    if (diff_delt(theta_s, theta_t, np.pi / 6) and
                            diff_delt(np.pi - theta_s, theta_t, np.pi / 6)):
                        # 移除此对
                        cstl_out[i], cstl_out[num_sim - 1] = cstl_out[num_sim - 1], cstl_out[i]
                        num_sim -= 1
                        continue
                i += 1

            cstl_out = cstl_out[:num_sim]

        ret.i_orie_sim = len(cstl_out)
        if ret.i_orie_sim < lb.i_orie_sim:
            return ret, cstl_out, area_perc

        # 计算面积百分比
        for pair in cstl_out:
            perc = 0.5 * (src.cont_perc[pair.level][pair.seq_src] +
                          tgt.cont_perc[pair.level][pair.seq_tgt])
            area_perc.append(perc)

        return ret, cstl_out, area_perc

    @staticmethod
    def get_tf_from_constell(src: 'ContourManager', tgt: 'ContourManager',
                             cstl_pairs: List[ConstellationPair]) -> np.ndarray:
        """
        从星座计算变换矩阵

        Args:
            src: 源轮廓管理器
            tgt: 目标轮廓管理器
            cstl_pairs: 星座对列表

        Returns:
            2D同构变换矩阵 (3x3)
        """
        num_elem = len(cstl_pairs)
        if num_elem < 3:
            print(f"警告：对应点不足({num_elem}个)，返回单位变换")
            return np.eye(3)

        # 收集对应点
        pointset1 = np.zeros((2, num_elem))  # src
        pointset2 = np.zeros((2, num_elem))  # tgt

        for i, pair in enumerate(cstl_pairs):
            pointset1[:, i] = src.cont_views[pair.level][pair.seq_src].pos_mean
            pointset2[:, i] = tgt.cont_views[pair.level][pair.seq_tgt].pos_mean

        # 使用Umeyama算法计算变换
        T_delta = umeyama_2d(pointset1, pointset2)

        return T_delta

    def _output_detailed_contour_statistics(self):
        """输出详细的轮廓统计信息到日志"""
        print("DEBUG: _output_detailed_contour_statistics() 函数开始")
        try:
            contour_sizes = []
            eccentricities = []
            eigenvalue_ratios = []
            significant_ecc_count = 0
            significant_com_count = 0
            heights = []

            # 收集所有轮廓的统计信息
            for level in range(len(self.cont_views)):
                for contour in self.cont_views[level]:
                    # 轮廓尺寸
                    contour_sizes.append(contour.cell_cnt)

                    # 偏心率
                    eccentricities.append(contour.eccen)

                    # 特征值比例
                    if len(contour.eig_vals) == 2 and contour.eig_vals[1] > 0:
                        ratio = contour.eig_vals[0] / contour.eig_vals[1]
                        eigenvalue_ratios.append(ratio)

                    # 显著特征计数
                    if contour.ecc_feat:
                        significant_ecc_count += 1
                    if contour.com_feat:
                        significant_com_count += 1

                    # 高度信息
                    heights.append(contour.vol3_mean)

            # 输出到日志
            if contour_sizes:
                # 基本统计
                total_contours = len(contour_sizes)
                min_size = min(contour_sizes)
                max_size = max(contour_sizes)
                avg_size = sum(contour_sizes) / total_contours
                import statistics
                std_size = statistics.stdev(contour_sizes) if total_contours > 1 else 0

                # 同时输出到print和logging
                stats_msg = f"CONTOUR_STATS_BASIC: total={total_contours}, min={min_size}, max={max_size}, avg={avg_size:.1f}, std={std_size:.1f}"
                print(stats_msg)
                import logging
                logging.info(stats_msg)

                # 尺寸分布统计
                size_bins = [
                    (1, 5, "极小轮廓"),
                    (6, 15, "小轮廓"),
                    (16, 50, "中小轮廓"),
                    (51, 150, "中等轮廓"),
                    (151, 500, "大轮廓"),
                    (501, float('inf'), "超大轮廓")
                ]

                size_distribution = {}
                for min_size_bin, max_size_bin, label in size_bins:
                    if max_size_bin == float('inf'):
                        count = sum(1 for s in contour_sizes if s >= min_size_bin)
                    else:
                        count = sum(1 for s in contour_sizes if min_size_bin <= s <= max_size_bin)
                    ratio = count / total_contours if total_contours > 0 else 0
                    size_distribution[label] = ratio

                    size_dist_msg = f"CONTOUR_SIZE_DIST: {label}={count}({ratio:.3f})"
                    print(size_dist_msg)
                    logging.info(size_dist_msg)

                # 几何特征统计
                avg_ecc = 0
                std_ecc = 0
                if eccentricities:
                    avg_ecc = sum(eccentricities) / len(eccentricities)
                    std_ecc = statistics.stdev(eccentricities) if len(eccentricities) > 1 else 0

                    geom_msg = f"CONTOUR_GEOMETRY: avg_eccentricity={avg_ecc:.3f}, std_eccentricity={std_ecc:.3f}"
                    print(geom_msg)
                    logging.info(geom_msg)

                    # 偏心率分布
                    ecc_bins = [
                        (0.0, 0.3, "近圆形"),
                        (0.3, 0.6, "椭圆形"),
                        (0.6, 0.8, "长椭圆"),
                        (0.8, 1.0, "极长椭圆")
                    ]

                    for min_ecc, max_ecc, label in ecc_bins:
                        count = sum(1 for e in eccentricities if min_ecc <= e < max_ecc)
                        ratio = count / len(eccentricities)

                        ecc_dist_msg = f"CONTOUR_ECC_DIST: {label}={count}({ratio:.3f})"
                        print(ecc_dist_msg)
                        logging.info(ecc_dist_msg)

                # 特征值比例
                avg_eigenvalue_ratio = 0
                if eigenvalue_ratios:
                    avg_eigenvalue_ratio = sum(eigenvalue_ratios) / len(eigenvalue_ratios)
                    eigval_msg = f"CONTOUR_EIGENVALUE: avg_ratio={avg_eigenvalue_ratio:.3f}"
                    print(eigval_msg)
                    logging.info(eigval_msg)

                # 显著特征统计
                if total_contours > 0:
                    ecc_feat_ratio = significant_ecc_count / total_contours
                    com_feat_ratio = significant_com_count / total_contours

                    feat_msg = f"CONTOUR_SIGNIFICANT_FEATURES: ecc_count={significant_ecc_count}({ecc_feat_ratio:.3f}), com_count={significant_com_count}({com_feat_ratio:.3f})"
                    print(feat_msg)
                    logging.info(feat_msg)

                # 高度统计
                avg_height = 0
                if heights:
                    avg_height = sum(heights) / len(heights)
                    height_msg = f"CONTOUR_HEIGHT: avg_height={avg_height:.2f}"
                    print(height_msg)
                    logging.info(height_msg)

            print("DEBUG: _output_detailed_contour_statistics() 函数正常结束")

            # 返回详细统计数据
            return {
                'total_contours': total_contours if contour_sizes else 0,
                'tiny_contour_ratio': size_distribution.get("极小轮廓", 0),
                'small_contour_ratio': size_distribution.get("小轮廓", 0),
                'medium_small_contour_ratio': size_distribution.get("中小轮廓", 0),
                'medium_contour_ratio': size_distribution.get("中等轮廓", 0),
                'large_contour_ratio': size_distribution.get("大轮廓", 0),
                'super_large_contour_ratio': size_distribution.get("超大轮廓", 0),
                'avg_eccentricity': avg_ecc,
                'std_eccentricity': std_ecc,
                'significant_ecc_ratio': significant_ecc_count / total_contours if total_contours > 0 else 0,
                'significant_com_ratio': significant_com_count / total_contours if total_contours > 0 else 0,
                'avg_size': avg_size if contour_sizes else 0,
                'std_size': std_size if contour_sizes else 0,
                'min_size': min_size if contour_sizes else 0,
                'max_size': max_size if contour_sizes else 0,
                'avg_eigenvalue_ratio': avg_eigenvalue_ratio,
                'avg_height': avg_height
            }

        except Exception as e:
            error_msg = f"轮廓统计输出失败: {e}"
            print(error_msg)
            import logging
            logging.error(error_msg)
            # 异常情况返回默认值
            return {
                'total_contours': 0,
                'tiny_contour_ratio': 0,
                'small_contour_ratio': 0,
                'medium_small_contour_ratio': 0,
                'medium_contour_ratio': 0,
                'large_contour_ratio': 0,
                'super_large_contour_ratio': 0,
                'avg_eccentricity': 0,
                'std_eccentricity': 0,
                'significant_ecc_ratio': 0,
                'significant_com_ratio': 0,
                'avg_size': 0,
                'std_size': 0,
                'min_size': 0,
                'max_size': 0,
                'avg_eigenvalue_ratio': 0,
                'avg_height': 0
            }

    def _output_retrieval_key_statistics(self):
        """输出检索键特征统计信息到日志"""
        print("DEBUG: _output_retrieval_key_statistics() 函数开始")
        try:
            key_stats = {'dim0': [], 'dim1': [], 'dim2': [], 'zero_keys': 0}
            ring_activations = []
            total_keys = 0

            # 收集所有层级的检索键信息
            for ll in range(len(self.layer_keys)):
                for key in self.layer_keys[ll]:
                    total_keys += 1

                    if len(key) >= 3:
                        key_stats['dim0'].append(float(key[0]))
                        key_stats['dim1'].append(float(key[1]))
                        key_stats['dim2'].append(float(key[2]))

                        # 检查是否为零向量
                        if np.sum(key) == 0:
                            key_stats['zero_keys'] += 1

                        # 收集环形特征
                        if len(key) > 3:
                            ring_features = key[3:]
                            ring_activations.extend([float(x) for x in ring_features if x > 0])

            # 输出统计信息
            import logging
            import statistics

            # 基本维度统计
            if key_stats['dim0']:
                import statistics

                avg_dim0 = statistics.mean(key_stats['dim0'])
                avg_dim1 = statistics.mean(key_stats['dim1'])
                avg_dim2 = statistics.mean(key_stats['dim2'])

                dim_msg = f"KEY_DIMENSIONS: dim0_avg={avg_dim0:.4f}, dim1_avg={avg_dim1:.4f}, dim2_avg={avg_dim2:.4f}"
                print(dim_msg)
                logging.info(dim_msg)

                # 维度分布统计
                if key_stats['dim0']:
                    min_dim0, max_dim0 = min(key_stats['dim0']), max(key_stats['dim0'])
                    std_dim0 = statistics.stdev(key_stats['dim0']) if len(key_stats['dim0']) > 1 else 0

                    dim0_dist_msg = f"KEY_DIM0_DIST: min={min_dim0:.4f}, max={max_dim0:.4f}, std={std_dim0:.4f}"
                    print(dim0_dist_msg)
                    logging.info(dim0_dist_msg)

                if key_stats['dim1']:
                    min_dim1, max_dim1 = min(key_stats['dim1']), max(key_stats['dim1'])
                    std_dim1 = statistics.stdev(key_stats['dim1']) if len(key_stats['dim1']) > 1 else 0

                    dim1_dist_msg = f"KEY_DIM1_DIST: min={min_dim1:.4f}, max={max_dim1:.4f}, std={std_dim1:.4f}"
                    print(dim1_dist_msg)
                    logging.info(dim1_dist_msg)

                if key_stats['dim2']:
                    min_dim2, max_dim2 = min(key_stats['dim2']), max(key_stats['dim2'])
                    std_dim2 = statistics.stdev(key_stats['dim2']) if len(key_stats['dim2']) > 1 else 0

                    dim2_dist_msg = f"KEY_DIM2_DIST: min={min_dim2:.4f}, max={max_dim2:.4f}, std={std_dim2:.4f}"
                    print(dim2_dist_msg)
                    logging.info(dim2_dist_msg)

            # 稀疏性统计
            if total_keys > 0:
                sparsity = key_stats['zero_keys'] / total_keys
                valid_keys = total_keys - key_stats['zero_keys']
                # 解释: 统计所有生成的检索键中，有多少个是零向量（即np.sum(key) == 0
                # 的键）。稀疏度 = 零向量数量 ÷ 总键数量。稀疏度越高说明有效特征越少。
                sparse_msg = f"KEY_SPARSITY: total_keys={total_keys}, zero_keys={key_stats['zero_keys']}, sparsity={sparsity:.4f}, valid_keys={valid_keys}"
                print(sparse_msg)
                logging.info(sparse_msg)

            # 环形特征统计
            if ring_activations:
                avg_activation = statistics.mean(ring_activations)# 所有激活环形特征的平均值，环形特征提取位置:_generate_ring_features() 函数
                # 解释：检索键的第4-10维是环形特征，表示不同半径环上的结构密度。环形激活是所有非零环形特征值的平均值，反映局部空间结构的丰富度。
                std_activation = statistics.stdev(ring_activations) if len(ring_activations) > 1 else 0
                max_activation = max(ring_activations)

                ring_msg = f"KEY_RING_FEATURES: avg_activation={avg_activation:.4f}, std_activation={std_activation:.4f}, max_activation={max_activation:.4f}, active_count={len(ring_activations)}"
                print(ring_msg)
                logging.info(ring_msg)
            else:
                avg_activation = 0.0
                ring_msg = f"KEY_RING_FEATURES: avg_activation=0.0000, std_activation=0.0000, max_activation=0.0000, active_count=0"
                print(ring_msg)
                logging.info(ring_msg)

            # 总体质量评估
            if total_keys > 0:
                quality_score = (1.0 - sparsity) * 0.5
                if ring_activations:
                    quality_score += min(0.5, len(ring_activations) / (total_keys * 7) * 0.5)  # 假设每个key有7个ring features
                # 解释: 质量得分由两部分组成：(1 - 稀疏度) × 0.5 + 环形特征激活度贡献。满分1.0，越高表示特征表达能力越强。
                quality_msg = f"KEY_QUALITY: quality_score={quality_score:.4f}"
                print(quality_msg)
                logging.info(quality_msg)
            else:
                quality_score = 0.0

            print("DEBUG: _output_retrieval_key_statistics() 函数正常结束")

            return {
                'sparsity_ratio': sparsity if total_keys > 0 else 0,
                'quality_score': quality_score,
                'ring_activation': avg_activation
            }

        except Exception as e:
            error_msg = f"检索键统计输出失败: {e}"
            print(error_msg)
            import logging
            logging.error(error_msg)
            return {
                'sparsity_ratio': 0,
                'quality_score': 0,
                'ring_activation': 0
            }

    def _output_bci_statistics(self):
        """输出BCI特征统计信息到日志"""
        print("DEBUG: _output_bci_statistics() 函数开始")
        try:
            bci_neighbors = []
            neighbor_distances = []
            neighbor_angles = []
            cross_layer_connections = 0
            total_connections = 0
            distance_bits_activated = 0
            total_distance_bits = 0
            layer_connectivity = {}

            # 收集所有BCI的信息
            for ll in range(len(self.layer_key_bcis)):
                layer_connections = 0
                layer_cross_connections = 0

                for bci in self.layer_key_bcis[ll]:
                    # 邻居数量
                    neighbor_count = len(bci.nei_pts) # 每个BCI的邻居轮廓数量，邻居生成位置: _generate_bci() 函数
                    bci_neighbors.append(neighbor_count)
                    layer_connections += neighbor_count

                    # 距离位统计
                    total_distance_bits += len(bci.dist_bin)
                    distance_bits_activated += np.sum(bci.dist_bin)

                    # 邻居点详细信息
                    for nei_pt in bci.nei_pts:
                        neighbor_distances.append(float(nei_pt.r))
                        neighbor_angles.append(float(nei_pt.theta))
                        total_connections += 1

                        # 跨层连接统计
                        if nei_pt.level != ll:
                            cross_layer_connections += 1
                            layer_cross_connections += 1

                # 记录每层的连接信息
                layer_connectivity[ll] = {
                    'total_connections': layer_connections,
                    'cross_layer_connections': layer_cross_connections,
                    'bcis_count': len(self.layer_key_bcis[ll])
                }

            # 输出统计信息
            import logging
            import statistics

            # 基本BCI统计
            total_bcis = len([bci for bcis in self.layer_key_bcis for bci in bcis])

            avg_neighbors = 0
            std_neighbors = 0
            min_neighbors = 0
            max_neighbors = 0
            if bci_neighbors:
                avg_neighbors = statistics.mean(bci_neighbors)# 所有BCI邻居数的平均值
                std_neighbors = statistics.stdev(bci_neighbors) if len(bci_neighbors) > 1 else 0
                min_neighbors = min(bci_neighbors)
                max_neighbors = max(bci_neighbors)

                bci_basic_msg = f"BCI_BASIC_STATS: total_bcis={total_bcis}, avg_neighbors={avg_neighbors:.1f}, std_neighbors={std_neighbors:.1f}, min_neighbors={min_neighbors}, max_neighbors={max_neighbors}"
                print(bci_basic_msg)
                logging.info(bci_basic_msg)

                # 邻居数分布
                neighbor_distribution = {
                    '0_neighbors': sum(1 for n in bci_neighbors if n == 0),
                    '1-3_neighbors': sum(1 for n in bci_neighbors if 1 <= n <= 3),
                    '4-6_neighbors': sum(1 for n in bci_neighbors if 4 <= n <= 6),
                    '7-10_neighbors': sum(1 for n in bci_neighbors if 7 <= n <= 10),
                    '10+_neighbors': sum(1 for n in bci_neighbors if n > 10)
                }

                for range_name, count in neighbor_distribution.items():
                    ratio = count / total_bcis if total_bcis > 0 else 0
                    neighbor_dist_msg = f"BCI_NEIGHBOR_DIST: {range_name}={count}({ratio:.3f})"
                    print(neighbor_dist_msg)
                    logging.info(neighbor_dist_msg)

            # 距离统计
            avg_distance = 0
            std_distance = 0
            min_distance = 0
            max_distance = 0
            if neighbor_distances:
                avg_distance = statistics.mean(neighbor_distances)
                std_distance = statistics.stdev(neighbor_distances) if len(neighbor_distances) > 1 else 0
                min_distance = min(neighbor_distances)
                max_distance = max(neighbor_distances)

                distance_msg = f"BCI_DISTANCES: avg_distance={avg_distance:.2f}, std_distance={std_distance:.2f}, min_distance={min_distance:.2f}, max_distance={max_distance:.2f}"
                print(distance_msg)
                logging.info(distance_msg)

            # 角度多样性统计
            angle_diversity = 0
            angle_uniformity = 0
            if neighbor_angles:
                # 角度分布统计 (将角度转换到0-2π范围)
                normalized_angles = [(angle + 2 * np.pi) % (2 * np.pi) for angle in neighbor_angles]
                angle_diversity = statistics.stdev(normalized_angles) if len(normalized_angles) > 1 else 0

                # 角度分布均匀性 (理想情况下应该均匀分布在0-2π)
                angle_bins = [0] * 8  # 8个45度的扇区
                for angle in normalized_angles:
                    bin_idx = int(angle / (np.pi / 4)) % 8
                    angle_bins[bin_idx] += 1

                angle_uniformity = 1.0 - (max(angle_bins) - min(angle_bins)) / len(
                    normalized_angles) if normalized_angles else 0

                angle_msg = f"BCI_ANGLES: angle_diversity={angle_diversity:.3f}, angle_uniformity={angle_uniformity:.3f}"
                print(angle_msg)
                logging.info(angle_msg)

            # 跨层连接统计
            cross_layer_ratio = 0
            if total_connections > 0:
                cross_layer_ratio = cross_layer_connections / total_connections
                intra_layer_connections = total_connections - cross_layer_connections

                cross_layer_msg = f"BCI_CROSS_LAYER: cross_layer_connections={cross_layer_connections}, intra_layer_connections={intra_layer_connections}, cross_layer_ratio={cross_layer_ratio:.3f}"
                print(cross_layer_msg)
                logging.info(cross_layer_msg)

            # 距离位激活统计
            activation_rate = 0
            if total_distance_bits > 0:
                activation_rate = distance_bits_activated / total_distance_bits

                bit_msg = f"BCI_DISTANCE_BITS: total_bits={total_distance_bits}, activated_bits={distance_bits_activated}, activation_rate={activation_rate:.4f}"
                print(bit_msg)
                logging.info(bit_msg)

            # 星座复杂度计算
            constellation_complexity = 0.0
            if bci_neighbors and neighbor_angles:
                avg_neighbors_calc = statistics.mean(bci_neighbors)# 平均邻居数
                angle_diversity_calc = statistics.stdev(normalized_angles) if len(normalized_angles) > 1 else 0# 角度多样性
                constellation_complexity = avg_neighbors_calc * angle_diversity_calc / 10.0# 复杂度 = 邻居数 × 角度多样性 / 10

            complexity_msg = f"BCI_CONSTELLATION_COMPLEXITY: complexity_score={constellation_complexity:.3f}"
            print(complexity_msg)
            logging.info(complexity_msg)

            # 连接质量评估
            connection_quality = 0.0
            if total_bcis > 0 and bci_neighbors:
                # 理想的邻居数是3-8个
                ideal_neighbor_count = sum(1 for n in bci_neighbors if 3 <= n <= 8)
                connection_quality = ideal_neighbor_count / total_bcis

            quality_msg = f"BCI_CONNECTION_QUALITY: quality_score={connection_quality:.3f}, ideal_bcis_ratio={connection_quality:.3f}"
            print(quality_msg)
            logging.info(quality_msg)

            print("DEBUG: _output_bci_statistics() 函数正常结束")

            # 返回详细统计数据
            return {
                'avg_neighbors': avg_neighbors,
                'std_neighbors': std_neighbors,
                'min_neighbors': min_neighbors,
                'max_neighbors': max_neighbors,
                'neighbor_dist_0': neighbor_distribution.get('0_neighbors', 0) / total_bcis if total_bcis > 0 else 0,
                'neighbor_dist_1_3': neighbor_distribution.get('1-3_neighbors',
                                                               0) / total_bcis if total_bcis > 0 else 0,
                'neighbor_dist_4_6': neighbor_distribution.get('4-6_neighbors',
                                                               0) / total_bcis if total_bcis > 0 else 0,
                'neighbor_dist_7_10': neighbor_distribution.get('7-10_neighbors',
                                                                0) / total_bcis if total_bcis > 0 else 0,
                'neighbor_dist_10_plus': neighbor_distribution.get('10+_neighbors',
                                                                   0) / total_bcis if total_bcis > 0 else 0,
                'avg_distance': avg_distance,
                'std_distance': std_distance,
                'min_distance': min_distance,
                'max_distance': max_distance,
                'angle_diversity': angle_diversity,
                'angle_uniformity': angle_uniformity,
                'cross_layer_ratio': cross_layer_ratio,
                'activation_rate': activation_rate,
                'constellation_complexity': constellation_complexity,
                'connection_quality': connection_quality
            }

        except Exception as e:
            error_msg = f"BCI统计输出失败: {e}"
            print(error_msg)
            import logging
            logging.error(error_msg)
            # 异常情况返回默认值
            return {
                'avg_neighbors': 0,
                'std_neighbors': 0,
                'min_neighbors': 0,
                'max_neighbors': 0,
                'neighbor_dist_0': 0,
                'neighbor_dist_1_3': 0,
                'neighbor_dist_4_6': 0,
                'neighbor_dist_7_10': 0,
                'neighbor_dist_10_plus': 0,
                'avg_distance': 0,
                'std_distance': 0,
                'min_distance': 0,
                'max_distance': 0,
                'angle_diversity': 0,
                'angle_uniformity': 0,
                'cross_layer_ratio': 0,
                'activation_rate': 0,
                'constellation_complexity': 0,
                'connection_quality': 0
            }

def umeyama_2d(src_points: np.ndarray, dst_points: np.ndarray) -> np.ndarray:
    """
    2D Umeyama算法计算相似变换
    """
    assert src_points.shape == dst_points.shape
    assert src_points.shape[0] == 2

    n = src_points.shape[1]
    if n < 2:
        return np.eye(3)  # 🔧 处理边界情况

    # 计算质心
    mu_src = np.mean(src_points, axis=1, keepdims=True)
    mu_dst = np.mean(dst_points, axis=1, keepdims=True)

    # 中心化
    src_centered = src_points - mu_src
    dst_centered = dst_points - mu_dst

    # 计算协方差矩阵
    C = src_centered @ dst_centered.T / n

    # SVD分解
    U, S, Vt = np.linalg.svd(C)

    # 计算旋转矩阵
    R = Vt.T @ U.T

    # 确保是旋转矩阵（行列式为正）
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T

    # 🔧 添加数值稳定性检查
    if np.abs(np.linalg.det(R) - 1.0) > 1e-6:
        print(f"Warning: Rotation matrix determinant = {np.linalg.det(R)}")

    # 计算平移
    t = mu_dst - R @ mu_src

    # 构造齐次变换矩阵
    T = np.eye(3)
    T[:2, :2] = R
    T[:2, 2:3] = t

    return T
