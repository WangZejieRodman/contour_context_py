"""
Contour Context Loop Closure Detection - Basic Data Structures
基础数据结构和配置类
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict, Any
from enum import Enum
import yaml

# 常量定义
BITS_PER_LAYER = 64
DIST_BIN_LAYERS = [1, 2, 3, 4]  # 用于生成距离键和形成星座的层
LAYER_AREA_WEIGHTS = [0.3, 0.3, 0.3, 0.1]  # 计算归一化"使用区域百分比"时每层的权重
NUM_BIN_KEY_LAYER = len(DIST_BIN_LAYERS)
RET_KEY_DIM = 10  # 检索键维度

@dataclass
class ContourViewStatConfig:
    """轮廓视图统计配置"""
    min_cell_cov: int = 4
    point_sigma: float = 1.0
    com_bias_thres: float = 0.5

@dataclass
class ContourSimThresConfig:
    """轮廓相似性阈值配置"""
    ta_cell_cnt: float = 6.0
    tp_cell_cnt: float = 0.2
    tp_eigval: float = 0.2
    ta_h_bar: float = 0.3  # KITTI用，MulRan用0.75
    ta_rcom: float = 0.4
    tp_rcom: float = 0.25

@dataclass
class TreeBucketConfig:
    """树桶配置"""
    max_elapse: float = 25.0
    min_elapse: float = 15.0

@dataclass
class ContourManagerConfig:
    """轮廓管理器配置"""
    lv_grads: List[float] = field(default_factory=lambda: [1.5, 2, 2.5, 3, 3.5, 4])
    reso_row: float = 1.0
    reso_col: float = 1.0
    n_row: int = 150
    n_col: int = 150
    lidar_height: float = 2.0
    blind_sq: float = 9.0
    min_cont_key_cnt: int = 9
    min_cont_cell_cnt: int = 3
    piv_firsts: int = 6
    dist_firsts: int = 10
    roi_radius: float = 10.0

@dataclass
class ContourDBConfig:
    """轮廓数据库配置"""
    nnk: int = 50
    max_fine_opt: int = 10
    q_levels: List[int] = field(default_factory=lambda: [1, 2, 3])
    cont_sim_cfg: ContourSimThresConfig = field(default_factory=ContourSimThresConfig)
    tb_cfg: TreeBucketConfig = field(default_factory=TreeBucketConfig)

@dataclass
class GMMOptConfig:
    """GMM优化配置"""
    min_area_perc: float = 0.95
    levels: List[int] = field(default_factory=lambda: [1, 2, 3, 4])
    cov_dilate_scale: float = 2.0

class PredictionOutcome(Enum):
    """预测结果类型"""
    TP = 0  # True Positive
    FP = 1  # False Positive
    TN = 2  # True Negative
    FN = 3  # False negative

@dataclass
class ScoreConstellSim:
    """星座相似性分数"""
    i_ovlp_sum: int = 0
    i_ovlp_max_one: int = 0
    i_in_ang_rng: int = 0

    def overall(self) -> int:
        return self.i_in_ang_rng

    def cnt(self) -> int:
        return self.i_in_ang_rng

    def strict_smaller(self, other: 'ScoreConstellSim') -> bool:
        return (self.i_ovlp_sum < other.i_ovlp_sum and
                self.i_ovlp_max_one < other.i_ovlp_max_one and
                self.i_in_ang_rng < other.i_in_ang_rng)

@dataclass
class ScorePairwiseSim:
    """成对相似性分数"""
    i_indiv_sim: int = 0
    i_orie_sim: int = 0

    def overall(self) -> int:
        return self.i_orie_sim

    def cnt(self) -> int:
        return self.i_orie_sim

    def strict_smaller(self, other: 'ScorePairwiseSim') -> bool:
        return (self.i_indiv_sim < other.i_indiv_sim and
                self.i_orie_sim < other.i_orie_sim)

@dataclass
class ScorePostProc:
    """后处理分数"""
    correlation: float = 0.0
    area_perc: float = 0.0
    neg_est_dist: float = 0.0  # 负距离（因为越大越好）

    def overall(self) -> float:
        return self.correlation

    def strict_smaller(self, other: 'ScorePostProc') -> bool:
        return (self.correlation < other.correlation and
                self.area_perc < other.area_perc and
                self.neg_est_dist < other.neg_est_dist)

@dataclass
class CandidateScoreEnsemble:
    """候选分数集合"""
    sim_constell: ScoreConstellSim = field(default_factory=ScoreConstellSim)
    sim_pair: ScorePairwiseSim = field(default_factory=ScorePairwiseSim)
    sim_post: ScorePostProc = field(default_factory=ScorePostProc)

@dataclass
class ConstellationPair:
    """星座对"""
    level: int
    seq_src: int
    seq_tgt: int

    def __lt__(self, other):
        return (self.level, self.seq_src, self.seq_tgt) < (other.level, other.seq_src, other.seq_tgt)

    def __eq__(self, other):
        return (self.level == other.level and
                self.seq_src == other.seq_src and
                self.seq_tgt == other.seq_tgt)

@dataclass
class RelativePoint:
    """BCI中的相对点"""
    level: int
    seq: int
    bit_pos: int
    r: float
    theta: float

@dataclass
class DistSimPair:
    """距离相似性对"""
    level: int
    seq_src: int
    seq_tgt: int
    orie_diff: float

class BCI:
    """二进制星座标识"""

    def __init__(self, seq: int, level: int):
        self.piv_seq = seq
        self.level = level
        self.dist_bin = np.zeros(BITS_PER_LAYER * NUM_BIN_KEY_LAYER, dtype=bool)
        self.nei_pts: List[RelativePoint] = []
        self.nei_idx_segs: List[int] = []

@dataclass
class RunningStatRecorder:
    """运行统计记录器"""
    cell_cnt: int = 0
    cell_pos_sum: np.ndarray = field(default_factory=lambda: np.zeros(2))
    cell_pos_tss: np.ndarray = field(default_factory=lambda: np.zeros((2, 2)))
    cell_vol3: float = 0.0
    cell_vol3_torq: np.ndarray = field(default_factory=lambda: np.zeros(2))

    def running_stats(self, curr_row: int, curr_col: int, height: float):
        """添加统计数据"""
        self.cell_cnt += 1
        v_rc = np.array([curr_row, curr_col], dtype=float)
        self.cell_pos_sum += v_rc
        self.cell_pos_tss += np.outer(v_rc, v_rc)
        self.cell_vol3 += height
        self.cell_vol3_torq += height * v_rc

def load_config_from_yaml(yaml_path: str) -> Dict[str, Any]:
    """从YAML文件加载配置"""
    with open(yaml_path, 'r') as f:
        return yaml.safe_load(f)

def clamp_angle(ang: float) -> float:
    """将角度限制在[-π, π)范围内"""
    return ang - np.floor((ang + np.pi) / (2 * np.pi)) * 2 * np.pi

def diff_perc(num1: float, num2: float, perc: float) -> bool:
    """检查两个数的百分比差异是否超过阈值"""
    return abs((num1 - num2) / max(num1, num2)) > perc

def diff_delt(num1: float, num2: float, delta: float) -> bool:
    """检查两个数的绝对差异是否超过阈值"""
    return abs(num1 - num2) > delta

def gauss_pdf(x: float, mean: float, sd: float) -> float:
    """高斯概率密度函数"""
    return np.exp(-0.5 * ((x - mean) / sd) ** 2) / np.sqrt(2 * np.pi * sd * sd)