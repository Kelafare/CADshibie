"""赛题常量与运行配置。

数据口径来自赛事 README：
- 前景类别 0-34，35 为背景/忽略；
- 0-29 为 Thing（可数对象，需要实例号），30-34 为 Stuff（不可数，实例号 -1）；
- commands: 0=Line, 1=Arc, 2=circle, 3=ellipse。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

# ---------------------------------------------------------------------------
# 赛题口径
# ---------------------------------------------------------------------------
COMMAND_LINE = 0
COMMAND_ARC = 1
COMMAND_CIRCLE = 2
COMMAND_ELLIPSE = 3
VALID_COMMANDS = (COMMAND_LINE, COMMAND_ARC, COMMAND_CIRCLE, COMMAND_ELLIPSE)

THING_CLASS_MIN = 0
THING_CLASS_MAX = 29
STUFF_CLASS_MIN = 30
STUFF_CLASS_MAX = 34
IGNORE_CLASS = 35
VALID_SEMANTIC_IDS = range(0, 36)

#: dev_final 只给 PNG；默认输出坐标系按赛事样例取 100x100。
DEFAULT_CANVAS_WIDTH = 100
DEFAULT_CANVAS_HEIGHT = 100

#: 模型参数总量上限（5B），见赛题规则第六节。
MAX_MODEL_PARAMETERS = 5_000_000_000

#: 提交物字段顺序，写 JSON 时保持稳定。
OUTPUT_FIELDS = (
    "commands",
    "args",
    "lengths",
    "semanticIds",
    "instanceIds",
    "width",
    "height",
    "obj_cts",
    "boxes",
    "rgb",
    "layerIds",
    "widths",
)


@dataclass(frozen=True)
class InferenceConfig:
    """一次推理的确定性配置。"""

    canvas_width: int = DEFAULT_CANVAS_WIDTH
    canvas_height: int = DEFAULT_CANVAS_HEIGHT

    #: 与背景代表色的 RGB 欧氏距离不大于该值视为背景。
    background_distance: float = 36.0
    #: 颜色量化步长；越小层越细。
    color_quant_step: int = 32
    #: 一个颜色层至少包含的前景像素占比。
    min_layer_fraction: float = 0.0004
    min_layer_pixels: int = 24
    #: 颜色聚类合并距离（RGB 欧氏距离）。
    color_merge_distance: float = 56.0
    #: 单个前景像素离层代表色的最大距离。
    color_assign_distance: float = 90.0

    #: 圆/椭圆拟合的最小半径（像素）。
    min_radius_px: float = 2.5
    #: 圆拟合相对残差阈值。矩形轮廓的残差通常 >0.1，真实圆 <0.05。
    circle_residual: float = 0.06
    #: 椭圆拟合相对残差阈值。
    ellipse_residual: float = 0.08
    #: 圆长宽比超过该值不按圆/椭圆处理。
    max_ellipse_aspect: float = 12.0
    #: 判定整圆的最小角度覆盖比例。
    full_circle_coverage: float = 0.82
    #: 判定圆弧的最小角度覆盖比例。
    min_arc_coverage: float = 0.20

    #: Hough 直线参数（像素）。
    hough_threshold: int = 10
    hough_min_length: int = 8
    hough_line_gap: int = 2
    #: 合并共线线段的角度/距离/间隙容差。
    merge_angle_deg: float = 4.0
    merge_distance_px: float = 2.5
    merge_gap_px: float = 10.0
    #: 用骨架点验证圆/椭圆候选的最近距离容差（像素）。
    skeleton_support_tolerance: float = 3.0

    #: Thing 实例聚合：端点邻近容差（画布单位）。实测 3.0 比 1.5 更能合并
    #: 门/窗的碎片，同时不会把相邻对象粘在一起。
    instance_gap: float = 3.0

    #: 未知颜色的默认语义。
    fallback_semantic: int = IGNORE_CLASS
    #: 颜色先验最近邻最大距离。
    prior_max_distance: float = 70.0

    #: 分类器置信度低于该值时回退为忽略类，减少文字/标注线造成的误检。
    classifier_min_confidence: float = 0.5
    #: 短于该长度的图元视为文字/标注碎片并过滤（画布单位）。
    min_primitive_length: float = 1.5

    #: 图像最长边超过该值时先缩放再检测。1000×1000 的评测图保留全分辨率，
    #: 以免 1px 细线在降采样中断裂（768 下可评估图元召回明显下降）。
    max_detection_size: int = 1200
    #: 随机种子（当前流程是确定性的，保留字段供模型使用）。
    seed: int = 0

    weights_dir: Path = field(default_factory=lambda: Path("weights"))


CANVAS_LIMITS = (16, 8192)
