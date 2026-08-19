"""
红楼梦 EDA 统计分析与可视化
- 各回字数分布
- 人物出场频次 Top20
- 回目字数趋势
- 关系网络权重 Top20
- 输出 Matplotlib 图表到 report/ 目录
"""

import json
import re
import sys
from pathlib import Path
from collections import Counter

import matplotlib
matplotlib.use("Agg")  # 非交互后端，适合保存图片
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import pandas as pd

# ─── 路径配置 ───
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
REPORT_DIR = PROJECT_ROOT / "report"
REPORT_DIR.mkdir(parents=True, exist_ok=True)


def setup_chinese_font():
    """配置 Matplotlib 中文字体"""
    # 尝试常见中文字体
    font_candidates = [
        "Microsoft YaHei",   # 微软雅黑
        "SimHei",            # 黑体
        "SimSun",            # 宋体
        "KaiTi",             # 楷体
        "FangSong",          # 仿宋
    ]
    for font_name in font_candidates:
        fonts = fm.findSystemFonts()
        for f in fonts:
            try:
                fp = fm.FontProperties(fname=f)
                if font_name in fp.get_name():
                    plt.rcParams["font.sans-serif"] = [font_name]
                    plt.rcParams["axes.unicode_minus"] = False
                    print(f"  字体设置: {font_name}")
                    return font_name
            except Exception:
                continue
    # 兜底
    plt.rcParams["font.sans-serif"] = ["SimHei"]
    plt.rcParams["axes.unicode_minus"] = False
    print("  字体设置: SimHei (兜底)")
    return "SimHei"


def load_data():
    """加载清洗后的数据"""
    with open(PROCESSED_DIR / "chapters_cleaned.json", "r", encoding="utf-8") as f:
        chapters = json.load(f)
    with open(PROCESSED_DIR / "characters_cleaned.json", "r", encoding="utf-8") as f:
        characters = json.load(f)
    with open(PROCESSED_DIR / "edges_cleaned.json", "r", encoding="utf-8") as f:
        edges = json.load(f)
    return chapters, characters, edges


def plot_chapter_char_distribution(chapters: list[dict]):
    """图1: 各回字数分布柱状图"""
    df = pd.DataFrame(chapters)
    df = df.sort_values("chapter_num")

    fig, ax = plt.subplots(figsize=(16, 5))
    colors = ["#E74C3C" if c > df["char_count"].mean() else "#3498DB" for c in df["char_count"]]
    ax.bar(df["chapter_num"], df["char_count"], color=colors, width=0.8, edgecolor="white", linewidth=0.3)
    ax.axhline(y=df["char_count"].mean(), color="#2C3E50", linestyle="--", linewidth=1, label=f"均值: {df['char_count'].mean():.0f}字")
    ax.set_xlabel("回目", fontsize=12)
    ax.set_ylabel("字数", fontsize=12)
    ax.set_title("红楼梦各回字数分布", fontsize=16, fontweight="bold")
    ax.set_xlim(0, 121)
    ax.legend(fontsize=10)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    output = REPORT_DIR / "eda_01_chapter_char_distribution.png"
    plt.savefig(output, dpi=150)
    plt.close()
    print(f"  ✓ 图1: {output.name}")
    return df


def plot_character_frequency(chapters: list[dict], characters: list[dict]):
    """图2: 人物出场频次 Top20"""
    # 统计每个人物在全文中的出现次数
    char_names = [c["name"] for c in characters if len(c["name"]) >= 2]  # 只统计2字及以上名字
    full_text = "\n".join(ch["raw_text"] for ch in chapters)

    freq_counter = Counter()
    for name in char_names:
        count = full_text.count(name)
        if count > 0:
            freq_counter[name] = count

    top20 = freq_counter.most_common(20)
    names = [item[0] for item in top20]
    counts = [item[1] for item in top20]

    fig, ax = plt.subplots(figsize=(12, 8))
    bars = ax.barh(range(len(names)), counts, color="#E67E22", edgecolor="white", height=0.7)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=11)
    ax.invert_yaxis()
    ax.set_xlabel("出场次数", fontsize=12)
    ax.set_title("红楼梦人物出场频次 Top20", fontsize=16, fontweight="bold")
    # 在柱子末端标注数值
    for bar, count in zip(bars, counts):
        ax.text(bar.get_width() + 5, bar.get_y() + bar.get_height() / 2,
                str(count), va="center", fontsize=10, color="#2C3E50")
    ax.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    output = REPORT_DIR / "eda_02_character_frequency_top20.png"
    plt.savefig(output, dpi=150)
    plt.close()
    print(f"  ✓ 图2: {output.name}")
    return freq_counter


def plot_chapter_trend(chapters: list[dict]):
    """图3: 回目字数趋势折线图（含移动平均）"""
    df = pd.DataFrame(chapters).sort_values("chapter_num")

    fig, ax = plt.subplots(figsize=(16, 5))
    ax.plot(df["chapter_num"], df["char_count"], color="#3498DB", alpha=0.5, linewidth=1, label="各回字数")
    # 10回移动平均
    rolling = df["char_count"].rolling(window=10, center=True).mean()
    ax.plot(df["chapter_num"], rolling, color="#E74C3C", linewidth=2.5, label="10回移动平均")
    ax.set_xlabel("回目", fontsize=12)
    ax.set_ylabel("字数", fontsize=12)
    ax.set_title("红楼梦回目字数趋势", fontsize=16, fontweight="bold")
    ax.set_xlim(1, 120)
    ax.legend(fontsize=11)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    output = REPORT_DIR / "eda_03_chapter_trend.png"
    plt.savefig(output, dpi=150)
    plt.close()
    print(f"  ✓ 图3: {output.name}")


def plot_relation_top20(edges: list[dict]):
    """图4: 人物关系权重 Top20"""
    df = pd.DataFrame(edges).sort_values("weight", ascending=False).head(20)
    labels = [f"{row['source']} — {row['target']}" for _, row in df.iterrows()]
    weights = df["weight"].tolist()

    fig, ax = plt.subplots(figsize=(12, 8))
    bars = ax.barh(range(len(labels)), weights, color="#9B59B6", edgecolor="white", height=0.7)
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=10)
    ax.invert_yaxis()
    ax.set_xlabel("交互权重", fontsize=12)
    ax.set_title("红楼梦人物关系权重 Top20", fontsize=16, fontweight="bold")
    for bar, w in zip(bars, weights):
        ax.text(bar.get_width() + 8, bar.get_y() + bar.get_height() / 2,
                f"{w:.0f}", va="center", fontsize=9, color="#2C3E50")
    ax.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    output = REPORT_DIR / "eda_04_relation_top20.png"
    plt.savefig(output, dpi=150)
    plt.close()
    print(f"  ✓ 图4: {output.name}")


def plot_chapter_volume_split(chapters: list[dict]):
    """图5: 前八十回 vs 后四十回字数对比"""
    df = pd.DataFrame(chapters).sort_values("chapter_num")
    first_80 = df[df["chapter_num"] <= 80]["char_count"]
    last_40 = df[df["chapter_num"] > 80]["char_count"]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # 左图: 箱线图对比
    bp = axes[0].boxplot(
        [first_80.values, last_40.values],
        labels=["前80回\n(曹雪芹)", "后40回\n(高鹗续)"],
        patch_artist=True,
        boxprops=dict(facecolor="#3498DB", alpha=0.6),
        medianprops=dict(color="#E74C3C", linewidth=2),
    )
    axes[0].set_ylabel("字数", fontsize=12)
    axes[0].set_title("前后回目字数分布对比", fontsize=14, fontweight="bold")
    axes[0].grid(axis="y", alpha=0.3)

    # 右图: 累计字数
    axes[1].plot(df["chapter_num"], df["char_count"].cumsum(), color="#E67E22", linewidth=2)
    axes[1].axvline(x=80, color="#E74C3C", linestyle="--", linewidth=1.5, label="第80回 (分界)")
    axes[1].set_xlabel("回目", fontsize=12)
    axes[1].set_ylabel("累计字数", fontsize=12)
    axes[1].set_title("红楼梦累计字数增长", fontsize=14, fontweight="bold")
    axes[1].legend(fontsize=11)
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    output = REPORT_DIR / "eda_05_volume_split.png"
    plt.savefig(output, dpi=150)
    plt.close()
    print(f"  ✓ 图5: {output.name}")


def generate_stats_report(chapters: list[dict], characters: list[dict],
                          edges: list[dict], freq_counter: Counter):
    """生成 EDA 统计报告（JSON）"""
    df = pd.DataFrame(chapters)
    report = {
        "总回数": len(chapters),
        "总字数": int(df["char_count"].sum()),
        "平均每回字数": float(df["char_count"].mean()),
        "最短回": {"回目": int(df.loc[df["char_count"].idxmin(), "chapter_num"]),
                    "字数": int(df["char_count"].min())},
        "最长回": {"回目": int(df.loc[df["char_count"].idxmax(), "chapter_num"]),
                    "字数": int(df["char_count"].max())},
        "人物总数": len(characters),
        "关系总数": len(edges),
        "出场频次Top10": [{"人物": name, "次数": count}
                           for name, count in freq_counter.most_common(10)],
        "前80回平均字数": float(df[df["chapter_num"] <= 80]["char_count"].mean()),
        "后40回平均字数": float(df[df["chapter_num"] > 80]["char_count"].mean()),
    }

    output = REPORT_DIR / "eda_stats_report.json"
    with open(output, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"  ✓ 统计报告: {output.name}")
    return report


def run_eda():
    """执行完整 EDA 分析"""
    print("=" * 60)
    print("红楼梦 EDA 统计分析与可视化")
    print("=" * 60)

    print("\n[1/6] 配置中文字体...")
    setup_chinese_font()

    print("\n[2/6] 加载清洗后数据...")
    chapters, characters, edges = load_data()
    print(f"  章回: {len(chapters)}, 人物: {len(characters)}, 关系: {len(edges)}")

    print("\n[3/6] 生成各回字数分布图...")
    plot_chapter_char_distribution(chapters)

    print("\n[4/6] 生成人物出场频次Top20...")
    freq_counter = plot_character_frequency(chapters, characters)

    print("\n[5/6] 生成回目字数趋势 & 前后对比...")
    plot_chapter_trend(chapters)
    plot_relation_top20(edges)
    plot_chapter_volume_split(chapters)

    print("\n[6/6] 生成统计报告...")
    report = generate_stats_report(chapters, characters, edges, freq_counter)

    print("\n" + "=" * 60)
    print("EDA 分析完成汇总:")
    for k, v in report.items():
        if isinstance(v, list):
            print(f"  {k}:")
            for item in v[:5]:
                print(f"    {item}")
            if len(v) > 5:
                print(f"    ... (共{len(v)}项)")
        else:
            print(f"  {k}: {v}")
    print(f"\n  图表输出目录: {REPORT_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    run_eda()
