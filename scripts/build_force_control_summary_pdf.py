# -*- coding: utf-8 -*-
"""Build the PDF counterpart for the current force-control report."""

from __future__ import annotations

import csv
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import Image, PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPORT_DIR = PROJECT_ROOT / "reports"
OUTPUT_PDF = REPORT_DIR / "力控螺旋搜索仿真项目总结_更新版.pdf"


def register_fonts() -> tuple[str, str]:
    regular = "C:/Windows/Fonts/simhei.ttf"
    bold = "C:/Windows/Fonts/msyhbd.ttc"
    pdfmetrics.registerFont(TTFont("CN-Regular", regular))
    pdfmetrics.registerFont(TTFont("CN-Bold", bold))
    return "CN-Regular", "CN-Bold"


FONT, FONT_BOLD = register_fonts()


def read_metrics() -> dict[str, float]:
    metrics: dict[str, float] = {}
    path = REPORT_DIR / "force_control_summary.txt"
    if not path.exists():
        return metrics
    for line in path.read_text(encoding="utf-8").splitlines():
        if ":" not in line or line.startswith("-"):
            continue
        key, value = line.split(":", 1)
        try:
            metrics[key.strip()] = float(value.strip())
        except ValueError:
            pass
    return metrics


def read_phase_summary() -> list[list[str]]:
    path = REPORT_DIR / "force_control_metrics.csv"
    if not path.exists():
        return []
    rows = list(csv.DictReader(path.open("r", encoding="utf-8")))
    groups: dict[str, list[dict[str, str]]] = {}
    for row in rows:
        groups.setdefault(row["phase"], []).append(row)
    mapping = {
        "align gripper center to object center using joint1/joint2 only": "抓取前平面对准",
        "descend to grasp height using joint5 only": "Joint5 下探抓取",
        "Archimedean spiral search using joint1/joint2 only": "阿基米德螺旋搜索",
        "final insertion using joint5 with compensated joint6 screwing": "补偿式 screwing 插入",
    }
    out = []
    for phase, label in mapping.items():
        data = groups.get(phase)
        if not data:
            continue
        duration = float(data[-1]["time"]) - float(data[0]["time"])
        max_force = max(float(r["contact_force_n"]) for r in data)
        xy = float(data[-1]["peg_hole_xy_error_m"]) * 1000.0
        depth = float(data[-1]["inserted_depth_m"]) * 1000.0
        out.append([label, f"{duration:.3f}", f"{max_force:.2f}", f"{xy:.2f}", f"{depth:.2f}"])
    return out


def make_styles():
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle("TitleCN", fontName=FONT_BOLD, fontSize=20, leading=25, alignment=TA_CENTER, textColor=colors.HexColor("#17365D"), spaceAfter=6))
    styles.add(ParagraphStyle("SubtitleCN", fontName=FONT, fontSize=10.5, leading=14, alignment=TA_CENTER, textColor=colors.HexColor("#555555"), spaceAfter=16))
    styles.add(ParagraphStyle("H1CN", fontName=FONT_BOLD, fontSize=15, leading=20, textColor=colors.HexColor("#2E74B5"), spaceBefore=14, spaceAfter=7))
    styles.add(ParagraphStyle("BodyCN", fontName=FONT, fontSize=9.8, leading=14, alignment=TA_LEFT, spaceAfter=6))
    styles.add(ParagraphStyle("SmallCN", fontName=FONT, fontSize=8.5, leading=11, textColor=colors.HexColor("#555555"), spaceAfter=5))
    styles.add(ParagraphStyle("CaptionCN", fontName=FONT, fontSize=8.2, leading=10.5, alignment=TA_CENTER, textColor=colors.HexColor("#555555"), spaceAfter=8))
    return styles


def p(text: str, style):
    return Paragraph(text.replace("\n", "<br/>"), style)


def bullets(items: list[str], style) -> list:
    return [p("• " + item, style) for item in items]


def image_flow(name: str, width: float, caption: str, styles) -> list:
    path = REPORT_DIR / name
    if not path.exists():
        return []
    img = Image(str(path), width=width, height=width * 0.52)
    return [img, p(caption, styles["CaptionCN"])]


def formula_flow(name: str, caption: str, styles) -> list:
    path = REPORT_DIR / name
    if not path.exists():
        return []
    return [Image(str(path), width=6.1 * inch, height=0.55 * inch), p(caption, styles["CaptionCN"])]


def table(data, col_widths):
    t = Table(data, colWidths=col_widths, repeatRows=1)
    t.setStyle(TableStyle([
        ("FONTNAME", (0, 0), (-1, -1), FONT),
        ("FONTNAME", (0, 0), (-1, 0), FONT_BOLD),
        ("FONTSIZE", (0, 0), (-1, -1), 8.3),
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#F2F4F7")),
        ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#C9D1D9")),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("ALIGN", (1, 1), (-1, -1), "CENTER"),
        ("LEFTPADDING", (0, 0), (-1, -1), 5),
        ("RIGHTPADDING", (0, 0), (-1, -1), 5),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ]))
    return t


def build_pdf() -> None:
    REPORT_DIR.mkdir(exist_ok=True)
    metrics = read_metrics()
    phase_rows = read_phase_summary()
    styles = make_styles()
    story = []

    story += [
        p("轴孔装配力控螺旋搜索仿真项目总结", styles["TitleCN"]),
        p("基于 JTC/TCC/FSC 的 6DOF 机械臂力控、螺旋搜索与补偿式 Screwing 插入验证", styles["SubtitleCN"]),
        p("<b>摘要：</b>当前版本将底层控制从刚性位置控制推进到基于 MuJoCo motor actuator 的力/力矩控制，并在 JTC/TCC/FSC 框架下完成抓取、螺旋搜索、Joint5 插入和 Joint6 补偿式 screwing。新的控制逻辑允许接触反力进入闭环，能够用接触力、跟踪误差、插入深度和关节稳定性量化装配过程。", styles["BodyCN"]),
        p("1. 控制框架更新与优势", styles["H1CN"]),
        p("项目当前采用的底层控制不再直接写入关节位置，而是向 MuJoCo 的 motor actuator 写入广义力。对于转动关节，控制量是力矩；对于 Joint5、夹爪等平动关节，控制量是直线力。接触不会被刚性位置命令覆盖，机械臂在孔口、圆柱体和夹爪之间产生的反力可以反映到关节误差和控制输入中。", styles["BodyCN"]),
    ]
    story += bullets([
        "JTC 将任务空间力/力矩映射为关节广义力，适合生成接触力和插入力。",
        "TCC 在末端操作空间中控制位置和姿态，使螺旋搜索轨迹不依赖单纯关节空间插值。",
        "FSC 将插入方向的力生成与平面位置控制分离：x/y 和姿态保持反馈控制，z 方向下压力以 feed-forward 方式施加。",
        "PID/阻抗项允许外力造成有限位姿误差，并通过积分与阻尼消除稳态误差、限制振动。",
    ], styles["BodyCN"])

    story.append(p("2. 不同阶段的力控制公式", styles["H1CN"]))
    story.append(p("控制器的总力矩结构参考论文中的 JTC/TCC/FSC 写法，同时加入 MuJoCo bias 补偿、关节姿态保持和库仑摩擦补偿。", styles["BodyCN"]))
    story += formula_flow("formula_total_tau.png", "式 1  总控制律：任务空间 wrench 经 J^T 映射到关节广义力。", styles)
    story += formula_flow("formula_task_pid.png", "式 2  TCC：任务空间 PID 根据 xd - xc 生成反馈力和力矩。", styles)
    story += formula_flow("formula_fsc.png", "式 3  FSC：运动轴使用反馈 wrench，受力轴使用 feed-forward wrench。", styles)
    story += formula_flow("formula_spiral.png", "式 4  螺旋搜索：圆柱体中心沿阿基米德螺旋在固定平面内搜索孔口。", styles)
    story += formula_flow("formula_screw_comp.png", "式 5  补偿式 screwing：Joint6 自转时由 Joint1/2 抵消偏心项。", styles)

    story.append(p("3. 阶段化控制逻辑", styles["H1CN"]))
    if phase_rows:
        story.append(table([["阶段", "持续时间/s", "最大接触力/N", "末端XY误差/mm", "插入深度/mm"]] + phase_rows, [1.55*inch, 1.0*inch, 1.1*inch, 1.15*inch, 1.1*inch]))
        story.append(Spacer(1, 0.12 * inch))
    story.append(p("螺旋搜索阶段主要消耗时间但接触力较低；插入阶段通过 Joint5 下探和 Joint6 约两圈回转完成最终装配。由于 Joint6 轴线并不穿过圆柱体中心，当前代码在 screwing 阶段引入 Joint1/Joint2 的补偿反解，避免圆柱体中心绕偏置轴形成非物理圆周运动。", styles["BodyCN"]))

    story.append(p("4. 量化仿真结果", styles["H1CN"]))
    metric_rows = [["指标", "数值", "含义"]]
    metric_specs = [
        ("总仿真时长", "duration_s", "s", "包含抓取、移动、螺旋搜索、补偿式 screwing 与保持阶段。"),
        ("最大瞬时接触力", "max_contact_force_n", "N", "接触峰值主要来自闭合夹爪和孔口接触瞬间。"),
        ("最终插入深度", "final_inserted_depth_m", "m", "圆柱体最终进入孔深内的深度。"),
        ("最终孔-轴平面误差", "final_xy_error_m", "m", "最终圆柱体中心与孔中心的平面距离。"),
        ("Joint6 最终回转圈数", "final_joint6_turns", "turns", "补偿式 screwing 保留的末端回转圈数。"),
        ("去除 Joint6 后最大跟踪误差", "max_tracking_error_without_q6", "", "排除多圈 joint6 后的主机械臂跟踪误差。"),
    ]
    for label, key, unit, meaning in metric_specs:
        if key in metrics:
            metric_rows.append([label, f"{metrics[key]:.5g} {unit}".strip(), meaning])
    story.append(table(metric_rows, [1.45*inch, 1.25*inch, 3.7*inch]))

    story.append(PageBreak())
    story.append(p("5. 图表分析", styles["H1CN"]))
    story += image_flow("contact_force_and_depth.png", 6.25 * inch, "图 1  接触力与插入深度。插入深度在最终阶段逐步增加，接触峰值主要来自接触建立和孔口约束。", styles)
    story += image_flow("joint_stability.png", 6.25 * inch, "图 2  Joint3/Joint4 稳定性与 Joint6 回转圈数。Joint6 保留 screwing，Joint3/4 偏移被姿态保持项限制。", styles)
    story += image_flow("tracking_error.png", 6.25 * inch, "图 3  关节跟踪误差分解。完整误差包含 Joint6 多圈回转，去除 Joint6 后可观察主机械臂平动/姿态跟踪质量。", styles)
    story += image_flow("peg_xy_path.png", 5.6 * inch, "图 4  圆柱体中心平面轨迹。轨迹体现先偏置搜索、后靠近孔中心并进行补偿式插入。", styles)

    story.append(p("6. 创新点、不足与局限", styles["H1CN"]))
    story += bullets([
        "创新点：将论文中的 JTC/TCC/FSC 思路落到 MuJoCo motor actuator 层，实现关节力/力矩输入而非直接 qpos 覆盖。",
        "创新点：针对当前 v7 机械臂 Joint6 轴线与 peg 中心不共线的问题，引入 screwing 偏心补偿逻辑，而不是简单关闭 Joint6。",
        "局限：当前仍使用 grasp_lock 稳定夹持，说明纯 STL 接触和摩擦还不足以完全替代真实夹爪物理夹持。",
        "局限：MuJoCo 的软接触模型难以完全复现论文中失稳后三点接触再 wiggling 的真实接触链。",
        "后续方向：批量运行多随机初始位置，统计成功率、平均插入时间、接触力分布，并接入力传感器闭环与自适应下探速度。",
    ], styles["BodyCN"])

    doc = SimpleDocTemplate(
        str(OUTPUT_PDF),
        pagesize=letter,
        rightMargin=0.8 * inch,
        leftMargin=0.8 * inch,
        topMargin=0.8 * inch,
        bottomMargin=0.8 * inch,
        title="力控螺旋搜索仿真项目总结",
    )
    doc.build(story)
    print(OUTPUT_PDF)


if __name__ == "__main__":
    build_pdf()
