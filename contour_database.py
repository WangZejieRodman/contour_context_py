"""
Contour Context Loop Closure Detection - Contour Database
轮廓数据库实现，包含KD树和候选匹配
"""

import numpy as np
from typing import List, Tuple, Dict, Optional, Any, Set
from dataclasses import dataclass
from sklearn.neighbors import NearestNeighbors
import time

from contour_types import (
    ContourDBConfig, CandidateScoreEnsemble, ConstellationPair,
    ScoreConstellSim, ScorePairwiseSim, ScorePostProc,
    BCI, DistSimPair, RelativePoint, clamp_angle,
    BITS_PER_LAYER, NUM_BIN_KEY_LAYER, DIST_BIN_LAYERS, LAYER_AREA_WEIGHTS
)
from contour_manager import ContourManager
from correlation import ConstellCorrelation, GMMOptConfig


@dataclass
class IndexOfKey:
    """键的索引"""
    gidx: int  # 全局索引
    level: int  # 层级
    seq: int   # 序列


class TreeBucket:
    """树桶类 - 管理单个KD树和缓冲区"""

    def __init__(self, config, beg: float, end: float):
        """
        初始化树桶

        Args:
            config: 树桶配置
            beg: 桶开始值
            end: 桶结束值
        """
        self.cfg = config
        self.buc_beg = beg
        self.buc_end = end
        self.data_tree: List[np.ndarray] = []
        self.tree_ptr: Optional[NearestNeighbors] = None
        self.buffer: List[Tuple[np.ndarray, float, IndexOfKey]] = []  # (key, timestamp, index)
        self.gkidx_tree: List[IndexOfKey] = []

        # 最大距离常量
        self.MAX_DIST_SQ = 1e6

    def get_tree_size(self) -> int:
        """获取树大小"""
        assert len(self.data_tree) == len(self.gkidx_tree)
        return len(self.data_tree)

    def push_buffer(self, tree_key: np.ndarray, ts: float, iok: IndexOfKey):
        """推送到缓冲区"""
        self.buffer.append((tree_key.copy(), ts, iok))

    def need_pop_buffer(self, curr_ts: float) -> bool:
        """检查是否需要弹出缓冲区"""
        ts_overflow = curr_ts - self.cfg.max_elapse
        if not self.buffer or self.buffer[0][1] > ts_overflow:
            return False
        return True

    def rebuild_tree(self):
        """重建KD树"""
        if len(self.data_tree) > 0:
            data_matrix = np.array(self.data_tree)
            self.tree_ptr = NearestNeighbors(
                n_neighbors=min(50, len(self.data_tree)),
                algorithm='kd_tree',
                leaf_size=10
            )
            self.tree_ptr.fit(data_matrix)
        else:
            self.tree_ptr = None

    def pop_buffer_max(self, curr_ts: float):
        """从缓冲区弹出到树中并重建树"""
        ts_cutoff = curr_ts - self.cfg.min_elapse
        gap = 0

        # 找到需要移动的元素数量
        for i, (_, ts, _) in enumerate(self.buffer):
            if ts >= ts_cutoff:
                break
            gap += 1

        if gap > 0:
            # 移动数据到树中
            sz0 = len(self.data_tree)
            self.data_tree.extend([self.buffer[i][0] for i in range(gap)])
            self.gkidx_tree.extend([self.buffer[i][2] for i in range(gap)])

            # 移除已处理的缓冲区项
            self.buffer = self.buffer[gap:]

            # 重建树
            self.rebuild_tree()

    def knn_search(self, num_res: int, q_key: np.ndarray, max_dist_sq: float) -> Tuple[List[IndexOfKey], List[float]]:
        """KNN搜索"""
        ret_idx = []
        out_dist_sq = [self.MAX_DIST_SQ] * num_res

        if self.tree_ptr is None or len(self.data_tree) == 0:
            return ret_idx, out_dist_sq[:0]  # 返回空列表

        # 执行搜索
        k = min(num_res, len(self.data_tree))
        try:
            distances, indices = self.tree_ptr.kneighbors([q_key], n_neighbors=k)

            # 过滤距离并构建结果
            for i in range(k):
                dist_sq = distances[0][i] ** 2
                if dist_sq < max_dist_sq:
                    ret_idx.append(self.gkidx_tree[indices[0][i]])
                    if i < num_res:
                        out_dist_sq[i] = dist_sq
                else:
                    break
        except Exception as e:
            print(f"KNN search error: {e}")
            return [], []

        return ret_idx, out_dist_sq[:len(ret_idx)]

    def range_search(self, max_dist_sq: float, q_key: np.ndarray) -> Tuple[List[IndexOfKey], List[float]]:
        """范围搜索"""
        ret_idx = []
        out_dist_sq = []

        if self.tree_ptr is None or len(self.data_tree) == 0:
            return ret_idx, out_dist_sq

        try:
            # sklearn的radius_neighbors返回的是距离，不是距离的平方
            max_dist = np.sqrt(max_dist_sq)
            indices, distances = self.tree_ptr.radius_neighbors([q_key], radius=max_dist)

            if len(indices[0]) > 0:
                for i, idx in enumerate(indices[0]):
                    ret_idx.append(self.gkidx_tree[idx])
                    out_dist_sq.append(distances[0][i] ** 2)
        except Exception as e:
            print(f"Range search error: {e}")

        return ret_idx, out_dist_sq


class LayerDB:
    """层数据库 - 管理一层的多个树桶"""

    MIN_ELEM_SPLIT = 100
    IMBA_DIFF_RATIO = 0.2
    MAX_NUM_BUCKETS = 6
    BUCKET_CHANN = 0  # 用作桶的检索键的第0维
    MAX_BUCKET_VAL = 1000.0

    def __init__(self, tb_cfg):
        """
        初始化层数据库

        Args:
            tb_cfg: 树桶配置
        """
        self.buckets: List[TreeBucket] = []
        self.bucket_ranges: List[float] = []

        # 初始化桶范围
        self.bucket_ranges = [0.0] * (self.MAX_NUM_BUCKETS + 1)
        self.bucket_ranges[0] = -self.MAX_BUCKET_VAL
        self.bucket_ranges[-1] = self.MAX_BUCKET_VAL

        # 创建第一个桶（覆盖整个范围）
        self.buckets.append(TreeBucket(tb_cfg, -self.MAX_BUCKET_VAL, self.MAX_BUCKET_VAL))

        # 创建其他空桶
        for i in range(1, self.MAX_NUM_BUCKETS):
            self.bucket_ranges[i] = self.MAX_BUCKET_VAL
            self.buckets.append(TreeBucket(tb_cfg, self.MAX_BUCKET_VAL, self.MAX_BUCKET_VAL))

    def push_buffer(self, layer_key: np.ndarray, ts: float, scan_key_gidx: IndexOfKey):
        """推送到合适的桶缓冲区"""
        key_val = layer_key[self.BUCKET_CHANN]

        # 找到合适的桶
        for i in range(self.MAX_NUM_BUCKETS):
            if (self.bucket_ranges[i] <= key_val < self.bucket_ranges[i + 1]):
                if np.sum(layer_key) != 0:  # 非零键才添加
                    self.buckets[i].push_buffer(layer_key, ts, scan_key_gidx)
                return

    def rebuild(self, idx_t1: int, curr_ts: float):
        """重建指定的相邻桶对"""
        if idx_t1 >= len(self.buckets) - 1:
            return

        tr1, tr2 = self.buckets[idx_t1], self.buckets[idx_t1 + 1]

        # 检查是否需要弹出缓冲区
        pb1 = tr1.need_pop_buffer(curr_ts)
        pb2 = tr2.need_pop_buffer(curr_ts)

        if not pb1 and not pb2:
            return  # 当我们弹出缓冲区时才重建

        # 获取树大小
        sz1 = tr1.get_tree_size()
        sz2 = tr2.get_tree_size()
        diff_ratio = abs(sz1 - sz2) / max(sz1, sz2) if max(sz1, sz2) > 0 else 0

        # 决定是否需要平衡
        if pb1 and not pb2 and (diff_ratio < self.IMBA_DIFF_RATIO or max(sz1, sz2) < self.MIN_ELEM_SPLIT):
            tr1.pop_buffer_max(curr_ts)
            return

        if not pb1 and pb2 and (diff_ratio < self.IMBA_DIFF_RATIO or max(sz1, sz2) < self.MIN_ELEM_SPLIT):
            tr2.pop_buffer_max(curr_ts)
            return

        # 简化的平衡策略：直接弹出缓冲区
        if pb1:
            tr1.pop_buffer_max(curr_ts)
        if pb2:
            tr2.pop_buffer_max(curr_ts)

    def layer_knn_search(self, q_key: np.ndarray, k_top: int, max_dist_sq: float) -> List[Tuple[IndexOfKey, float]]:
        """层KNN搜索"""
        # 找到中间桶
        key_val = q_key[self.BUCKET_CHANN]
        mid_bucket = 0

        for i in range(self.MAX_NUM_BUCKETS):
            if (self.bucket_ranges[i] <= key_val < self.bucket_ranges[i + 1]):
                mid_bucket = i
                break

        res_pairs = []
        max_dist_sq_run = max_dist_sq

        # 按距离顺序搜索桶
        for i in range(self.MAX_NUM_BUCKETS):
            bucket_idx = -1

            if i == 0:
                bucket_idx = mid_bucket
            elif mid_bucket - i >= 0:
                # 检查距离约束
                dist_to_bucket = abs(key_val - self.bucket_ranges[mid_bucket - i + 1])
                if dist_to_bucket * dist_to_bucket > max_dist_sq_run:
                    continue
                bucket_idx = mid_bucket - i
            elif mid_bucket + i < self.MAX_NUM_BUCKETS:
                # 检查距离约束
                dist_to_bucket = abs(key_val - self.bucket_ranges[mid_bucket + i])
                if dist_to_bucket * dist_to_bucket > max_dist_sq_run:
                    continue
                bucket_idx = mid_bucket + i

            if bucket_idx >= 0:
                tmp_gidx, tmp_dists_sq = self.buckets[bucket_idx].knn_search(k_top, q_key, max_dist_sq_run)

                for gidx, dist_sq in zip(tmp_gidx, tmp_dists_sq):
                    if dist_sq < max_dist_sq_run:
                        res_pairs.append((gidx, dist_sq))

                # 排序并限制数量
                res_pairs.sort(key=lambda x: x[1])
                if len(res_pairs) >= k_top:
                    res_pairs = res_pairs[:k_top]
                    max_dist_sq_run = res_pairs[-1][1]

        return res_pairs

    def layer_range_search(self, q_key: np.ndarray, max_dist_sq: float) -> List[Tuple[IndexOfKey, float]]:
        """层范围搜索"""
        res_pairs = []

        for i in range(self.MAX_NUM_BUCKETS):
            tmp_gidx, tmp_dists_sq = self.buckets[i].range_search(max_dist_sq, q_key)

            for gidx, dist_sq in zip(tmp_gidx, tmp_dists_sq):
                res_pairs.append((gidx, dist_sq))

        return res_pairs


class CandidateAnchorProp:
    """候选锚点提议"""

    def __init__(self):
        self.constell: Dict[ConstellationPair, float] = {}  # 星座匹配：百分比分数
        self.T_delta = np.eye(3)  # 区分不同提议的关键特征
        self.correlation = 0.0
        self.vote_cnt = 0  # 投票给此TF的匹配轮廓数量
        self.area_perc = 0.0  # 所有层级中使用轮廓的面积百分比加权和


class CandidatePoseData:
    """候选姿态数据"""

    def __init__(self, cm_cand: ContourManager):
        self.cm_cand = cm_cand
        self.corr_est: Optional[ConstellCorrelation] = None
        self.anch_props: List[CandidateAnchorProp] = []

    def add_proposal(self, T_prop: np.ndarray, sim_pairs: List[ConstellationPair],
                    sim_area_perc: List[float]):
        """添加锚点提议，合并相似的提议"""
        assert len(sim_pairs) > 3, "底线要求至少4个对"
        assert len(sim_pairs) == len(sim_area_perc)

        # 检查是否与现有提议相似（硬编码阈值：2.0m, 0.3rad）
        for i, prop in enumerate(self.anch_props):
            delta_T = np.linalg.inv(T_prop) @ prop.T_delta
            trans_diff = np.linalg.norm(delta_T[:2, 2])
            rot_diff = abs(np.arctan2(delta_T[1, 0], delta_T[0, 0]))

            if trans_diff < 2.0 and rot_diff < 0.3:
                # 合并到现有提议
                for j, pair in enumerate(sim_pairs):
                    prop.constell[pair] = sim_area_perc[j]  # 覆盖或添加

                old_vote_cnt = prop.vote_cnt
                prop.vote_cnt += len(sim_pairs)

                # 混合变换参数（简单加权平均）
                w1, w2 = old_vote_cnt, len(sim_pairs)
                if w1 + w2 > 0:
                    trans_bl = (prop.T_delta[:2, 2] * w1 + T_prop[:2, 2] * w2) / (w1 + w2)

                    ang1 = np.arctan2(prop.T_delta[1, 0], prop.T_delta[0, 0])
                    ang2 = np.arctan2(T_prop[1, 0], T_prop[0, 0])

                    # 处理角度差
                    diff = ang2 - ang1
                    if diff < 0:
                        diff += 2 * np.pi
                    if diff > np.pi:
                        diff -= 2 * np.pi
                    ang_bl = diff * w2 / (w1 + w2) + ang1

                    # 更新变换矩阵
                    prop.T_delta = np.eye(3)
                    prop.T_delta[:2, :2] = np.array([[np.cos(ang_bl), -np.sin(ang_bl)],
                                                    [np.sin(ang_bl), np.cos(ang_bl)]])
                    prop.T_delta[:2, 2] = trans_bl

                return  # 贪心策略，找到第一个就返回

        # 限制提议数量
        if len(self.anch_props) > 3:
            return

        # 创建新提议
        new_prop = CandidateAnchorProp()
        new_prop.T_delta = T_prop.copy()
        for j, pair in enumerate(sim_pairs):
            new_prop.constell[pair] = sim_area_perc[j]
        new_prop.vote_cnt = len(sim_pairs)

        self.anch_props.append(new_prop)


class CandidateManager:
    """候选管理器 - 处理候选的检查、过滤和优化"""

    def __init__(self, cm_tgt: ContourManager, sim_lb: CandidateScoreEnsemble,
                 sim_ub: CandidateScoreEnsemble):
        """
        初始化候选管理器

        Args:
            cm_tgt: 目标轮廓管理器
            sim_lb: 相似性下界
            sim_ub: 相似性上界
        """
        self.cm_tgt = cm_tgt
        self.sim_var = sim_lb  # 动态下界（会随着检测过程调整）
        self.sim_ub = sim_ub   # 上界

        # 数据结构
        self.cand_id_pos_pair: Dict[int, int] = {}  # 候选ID到位置的映射
        self.candidates: List[CandidatePoseData] = []

        # 统计记录
        self.flow_valve = 0  # 避免反向工作流
        self.cand_aft_check1 = 0  # 第一轮检查后的候选数
        self.cand_aft_check2 = 0  # 第二轮检查后的候选数
        self.cand_aft_check3 = 0  # 第三轮检查后的候选数

        # 验证阈值配置
        assert sim_lb.sim_constell.strict_smaller(sim_ub.sim_constell)
        assert sim_lb.sim_pair.strict_smaller(sim_ub.sim_pair)
        assert sim_lb.sim_post.strict_smaller(sim_ub.sim_post)

    def check_cand_with_hint(self, cm_cand: ContourManager, anchor_pair: ConstellationPair,
                           cont_sim) -> CandidateScoreEnsemble:
        """
        使用提示检查候选

        这是候选管理器的主要功能之一，通过多层级检查验证候选的有效性

        Args:
            cm_cand: 候选轮廓管理器
            anchor_pair: 锚点对提示
            cont_sim: 轮廓相似性配置

        Returns:
            候选分数集合
        """
        assert self.flow_valve == 0, "工作流状态错误"

        cand_id = cm_cand.get_int_id()
        ret_score = CandidateScoreEnsemble()

        # 检查1/4: 锚点相似性
        anchor_sim = self._check_cont_pair_sim(cm_cand, self.cm_tgt, anchor_pair, cont_sim)
        if not anchor_sim:
            return ret_score

        self.cand_aft_check1 += 1

        print("Before check, curr bar:")
        self._print_threshold_status()

        # 检查2/4: 纯星座检查
        tmp_pairs1 = []
        ret_constell_sim = self._check_constell_sim(
            cm_cand.get_bci(anchor_pair.level, anchor_pair.seq_src),
            self.cm_tgt.get_bci(anchor_pair.level, anchor_pair.seq_tgt),
            self.sim_var.sim_constell, tmp_pairs1)

        ret_score.sim_constell = ret_constell_sim
        if ret_constell_sim.overall() < self.sim_var.sim_constell.overall():
            return ret_score

        self.cand_aft_check2 += 1

        # 检查3/4: 个体相似性检查
        tmp_pairs2 = []
        tmp_area_perc = []
        ret_pairwise_sim, tmp_pairs2, tmp_area_perc = self._check_constell_corresp_sim(
            cm_cand, self.cm_tgt, tmp_pairs1, self.sim_var.sim_pair, cont_sim)

        ret_score.sim_pair = ret_pairwise_sim
        if ret_pairwise_sim.overall() < self.sim_var.sim_pair.overall():
            return ret_score

        self.cand_aft_check3 += 1

        # 获取变换矩阵
        T_pass = self._get_tf_from_constell(cm_cand, self.cm_tgt, tmp_pairs2)

        # 动态阈值更新
        self._update_dynamic_thresholds(ret_pairwise_sim.cnt())

        print("Cand passed. New dynamic bar:")
        self._print_threshold_status()

        # 添加到候选列表或更新现有候选
        self._add_or_update_candidate(cand_id, cm_cand, T_pass, tmp_pairs2, tmp_area_perc)

        return ret_score

    def _check_cont_pair_sim(self, src: ContourManager, tgt: ContourManager,
                           cstl: ConstellationPair, cont_sim) -> bool:
        """检查轮廓对相似性"""
        from contour_view import ContourView
        return ContourView.check_sim(
            src.cont_views[cstl.level][cstl.seq_src],
            tgt.cont_views[cstl.level][cstl.seq_tgt],
            cont_sim)

    def _check_constell_sim(self, src: BCI, tgt: BCI, lb: ScoreConstellSim,
                          constell_res: List[ConstellationPair]) -> ScoreConstellSim:
        """检查星座相似性"""
        return BCI.check_constell_sim(src, tgt, lb, constell_res)

    def _check_constell_corresp_sim(self, src: ContourManager, tgt: ContourManager,
                                  cstl_in: List[ConstellationPair],
                                  lb: ScorePairwiseSim, cont_sim) -> Tuple[ScorePairwiseSim, List[ConstellationPair], List[float]]:
        """检查星座对应相似性"""
        return ContourManager.check_constell_corresp_sim(src, tgt, cstl_in, lb, cont_sim)

    def _get_tf_from_constell(self, src: ContourManager, tgt: ContourManager,
                            cstl_pairs: List[ConstellationPair]) -> np.ndarray:
        """从星座计算变换矩阵"""
        return ContourManager.get_tf_from_constell(src, tgt, cstl_pairs)

    def _update_dynamic_thresholds(self, cnt_curr_valid: int):
        """更新动态阈值"""
        # 更新星座相似性阈值
        new_const_lb = ScoreConstellSim()
        new_const_lb.i_ovlp_sum = cnt_curr_valid
        new_const_lb.i_ovlp_max_one = cnt_curr_valid
        new_const_lb.i_in_ang_rng = cnt_curr_valid
        self._align_lb_constell(new_const_lb)
        self._align_ub_constell()

        # 更新成对相似性阈值
        new_pair_lb = ScorePairwiseSim()
        new_pair_lb.i_indiv_sim = cnt_curr_valid
        new_pair_lb.i_orie_sim = cnt_curr_valid
        self._align_lb_pair(new_pair_lb)
        self._align_ub_pair()

    def _align_lb_constell(self, bar: ScoreConstellSim):
        """对齐星座下界"""
        self.sim_var.sim_constell.i_ovlp_sum = max(self.sim_var.sim_constell.i_ovlp_sum, bar.i_ovlp_sum)
        self.sim_var.sim_constell.i_ovlp_max_one = max(self.sim_var.sim_constell.i_ovlp_max_one, bar.i_ovlp_max_one)
        self.sim_var.sim_constell.i_in_ang_rng = max(self.sim_var.sim_constell.i_in_ang_rng, bar.i_in_ang_rng)

    def _align_ub_constell(self):
        """对齐星座上界"""
        self.sim_var.sim_constell.i_ovlp_sum = min(self.sim_var.sim_constell.i_ovlp_sum, self.sim_ub.sim_constell.i_ovlp_sum)
        self.sim_var.sim_constell.i_ovlp_max_one = min(self.sim_var.sim_constell.i_ovlp_max_one, self.sim_ub.sim_constell.i_ovlp_max_one)
        self.sim_var.sim_constell.i_in_ang_rng = min(self.sim_var.sim_constell.i_in_ang_rng, self.sim_ub.sim_constell.i_in_ang_rng)

    def _align_lb_pair(self, bar: ScorePairwiseSim):
        """对齐成对下界"""
        self.sim_var.sim_pair.i_indiv_sim = max(self.sim_var.sim_pair.i_indiv_sim, bar.i_indiv_sim)
        self.sim_var.sim_pair.i_orie_sim = max(self.sim_var.sim_pair.i_orie_sim, bar.i_orie_sim)

    def _align_ub_pair(self):
        """对齐成对上界"""
        self.sim_var.sim_pair.i_indiv_sim = min(self.sim_var.sim_pair.i_indiv_sim, self.sim_ub.sim_pair.i_indiv_sim)
        self.sim_var.sim_pair.i_orie_sim = min(self.sim_var.sim_pair.i_orie_sim, self.sim_ub.sim_pair.i_orie_sim)

    def _print_threshold_status(self):
        """打印当前阈值状态"""
        print(f"Constell: {self.sim_var.sim_constell.i_ovlp_sum}, {self.sim_var.sim_constell.i_ovlp_max_one}, {self.sim_var.sim_constell.i_in_ang_rng}")
        print(f"Pair: {self.sim_var.sim_pair.i_indiv_sim}, {self.sim_var.sim_pair.i_orie_sim}")

    def _add_or_update_candidate(self, cand_id: int, cm_cand: ContourManager,
                               T_pass: np.ndarray, tmp_pairs2: List[ConstellationPair],
                               tmp_area_perc: List[float]):
        """添加或更新候选"""
        cand_it = self.cand_id_pos_pair.get(cand_id)
        if cand_it is not None:
            # 候选姿态已存在，添加提议
            self.candidates[cand_it].add_proposal(T_pass, tmp_pairs2, tmp_area_perc)
        else:
            # 添加新候选
            new_cand = CandidatePoseData(cm_cand)
            new_cand.add_proposal(T_pass, tmp_pairs2, tmp_area_perc)
            self.cand_id_pos_pair[cand_id] = len(self.candidates)
            self.candidates.append(new_cand)

    def tidy_up_candidates(self):
        """整理候选 - 预先计算相关性并过滤候选"""
        assert self.flow_valve < 1
        self.flow_valve += 1

        gmm_config = GMMOptConfig()
        print(f"Tidy up pose {len(self.candidates)} candidates.")

        cnt_to_rm = 0
        valid_candidates = []

        # 分析每个姿态的锚点对