# Contour Context Loop Closure Detection - Python Implementation

基于轮廓上下文的点云回环检测Python实现版本。本项目将原始的C++版本contour-context改编为Python，便于研究和实验。

## 概述

该系统通过以下步骤实现点云回环检测：

1. **BEV生成**: 将3D点云转换为鸟瞰图(Bird's Eye View)
2. **多层轮廓提取**: 在不同高度层级提取连通组件轮廓
3. **特征计算**: 计算轮廓的几何特征和检索键
4. **数据库查询**: 使用KD树进行高效的相似轮廓检索
5. **候选验证**: 通过多层检查验证回环候选
6. **GMM优化**: 使用高斯混合模型进行精确的变换估计

## 文件结构

```
contour-context-python/
├── contour_types.py          # 基础数据结构和配置类
├── contour_view.py           # 轮廓视图实现
├── contour_manager.py        # 轮廓管理器，处理BEV和轮廓生成
├── contour_database.py       # 轮廓数据库，KD树和候选匹配
├── correlation.py            # GMM相关性计算和优化
├── evaluator.py              # 评估器，数据加载和性能评估
├── main_loop_closure.py      # 主回环检测流程
└── README.md                 # 本文档
```

## 安装

### 环境要求

- Python 3.7+
- NumPy >= 1.20.0
- OpenCV >= 4.5.0
- SciPy >= 1.7.0
- scikit-learn >= 1.0.0
- PyYAML >= 5.4.0

### 安装依赖

```bash
pip install -r requirements.txt
```

## 使用方法

### 1. 准备数据

准备两个文本文件：

**姿态文件 (poses.txt)**
```
timestamp r11 r12 r13 r21 r22 r23 r31 r32 r33 tx ty tz
0.000 1.000 0.000 0.000 0.000 1.000 0.000 0.000 0.000 1.000 0.000 0.000 0.000
0.100 0.995 -0.100 0.000 0.100 0.995 0.000 0.000 0.000 1.000 1.000 0.100 0.000
...
```

**激光扫描文件 (scans.txt)**
```
timestamp sequence_id path_to_bin_file
0.000 0 /path/to/scan_000000.bin
0.100 1 /path/to/scan_000001.bin
...
```

### 2. 配置参数

复制并修改 `example_config.yaml`：

```yaml
# 数据文件路径
fpath_sens_gt_pose: "/path/to/poses.txt"
fpath_lidar_bins: "/path/to/scans.txt"
fpath_outcome_sav: "/path/to/results.txt"

# 其他参数...
```

### 3. 运行检测

```bash
python main_loop_closure.py --config your_config.yaml --output-dir ./results
```

### 4. 查看结果

结果将保存在指定的输出目录中：
- `loop_closure_results.json`: 详细统计结果
- `outcome-kitti00.txt`: 每个扫描的预测结果

## 算法详解

### 1. BEV生成
- 将3D点云投影到2D网格
- 每个网格单元记录最大高度值
- 应用盲区过滤和范围限制

### 2. 轮廓提取
- 使用多个高度阈值生成不同层级
- 通过连通组件分析提取轮廓
- 计算统计特征：面积、质心、协方差、特征值等

### 3. 检索键生成
- 基于锚点轮廓的环形分布特征
- 使用高斯分布将周围点分配到环形bins
- 生成二进制星座标识(BCI)用于快速匹配

### 4. 数据库查询
- 使用KD树进行高效最近邻搜索
- 多层级检查：星座相似性、成对相似性、后处理
- 动态阈值调整提高检测精度

### 5. GMM优化
- 将轮廓建模为高斯椭圆
- 最大化互相关性估计精确变换
- 使用L-BFGS优化求解最优参数

## 性能优化

### 1. 内存优化
- 清理不必要的BEV图像数据
- 使用压缩的像素存储格式
- 按需重建图像数据

### 2. 计算优化
- 使用sklearn的KDTree进行快速搜索
- 向量化的特征计算
- 智能的候选过滤策略

### 3. 算法优化
- 动态阈值调整
- 早期候选排除
- 多层级渐进式检查

## 配置参数说明

### 轮廓管理器参数
- `lv_grads`: 高度层级阈值列表
- `n_row/n_col`: BEV图像大小
- `reso_row/reso_col`: 网格分辨率
- `roi_radius`: 感兴趣区域半径

### 数据库参数
- `nnk`: KNN搜索的邻居数量
- `max_fine_opt`: 最大精细优化候选数
- `q_levels`: 查询使用的层级

### 阈值参数
- `correlation_thres`: 回环检测的相关性阈值
- `thres_lb/thres_ub`: 各检查阶段的上下界阈值

## 故障排除

### 常见问题

1. **内存不足**
   - 减小BEV图像大小 (`n_row`, `n_col`)
   - 降低ROI半径 (`roi_radius`)
   - 定期清理图像数据

2. **检测精度低**
   - 调整轮廓相似性阈值
   - 增加层级数量
   - 优化GMM参数

3. **处理速度慢**
   - 减少KNN搜索数量 (`nnk`)
   - 限制精细优化候选数 (`max_fine_opt`)
   - 使用更严格的早期过滤

### 调试技巧

1. 启用详细日志输出
2. 可视化中间结果（BEV图像、轮廓等）
3. 分析每个阶段的处理时间
4. 检查参数配置的合理性

## 贡献

欢迎提交问题报告和改进建议！

## 许可证

本项目基于原始contour-context项目改编，请遵循相应的开源许可证。

## 参考文献

如果使用本代码，请引用原始论文：

```bibtex
@article{contour_context,
  title={Contour Context: Abstract Structural Distribution for 3D LiDAR Loop Detection and Metric Pose Estimation},
  author={...},
  journal={...},
  year={...}
}
```

## 联系方式

如有问题或建议，请通过GitHub Issues联系。
