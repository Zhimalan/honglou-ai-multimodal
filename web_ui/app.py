"""
红楼梦多模态智能解读专家系统 - Gradio 双栏前端

布局:
    ┌──────────────────────────────────────────────────────────┐
    │                    标题 + 系统状态                        │
    ├────────────────────────┬─────────────────────────────────┤
    │  左栏：对话区           │  右栏：可视化区                  │
    │  - 消息历史            │  - Tab1: NetworkX 静态图谱       │
    │  - 输入框              │  - Tab2: Pyvis 交互图谱          │
    │  - 意图/来源/幻觉提示   │  - Tab3: ComfyUI 国风图          │
    │  - 耗时统计            │  - 图片下载按钮                  │
    ├────────────────────────┴─────────────────────────────────┤
    │  底部：模型方案切换(A/B/C) + 文本上传增量更新             │
    └──────────────────────────────────────────────────────────┘

启动:
    python web_ui/app.py
    或 gradio web_ui/app.py
"""

import os
import sys
import time
import json
import shutil
import tempfile
from pathlib import Path

# 设置 HuggingFace 镜像和离线模式（避免连接超时）
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import gradio as gr

# ── 路径常量 ──
REPORT_DIR = PROJECT_ROOT / "report"
HTML_DIR = PROJECT_ROOT / "visualize" / "code_plot" / "html_output"
PROMPT_DIR = PROJECT_ROOT / "visualize" / "comfyui_api" / "prompts"
RAW_TEXT_DIR = PROJECT_ROOT / "data" / "raw_text"

# ── 延迟导入 Agent（避免启动时加载模型太慢）──
_agent = None


def get_agent():
    """懒加载 Agent 单例"""
    global _agent
    if _agent is None:
        from agent_graph.agent import HonglouAgent
        _agent = HonglouAgent()
    return _agent


# ═══════════════════════════════════════════════
# 回调函数
# ═══════════════════════════════════════════════

def chat_respond(message: str, history: list) -> tuple:
    """
    处理用户提问，调用 Agent，返回更新后的对话历史和元信息。

    Returns:
        (history, answer_text, intent_text, sources_text,
         hallucination_text, elapsed_text,
         right_image, right_html, prompt_info)
    """
    if not message.strip():
        return (
            history, "请输入问题", "", "", "",
            None, "", ""
        )

    # 调用 Agent
    agent = get_agent()
    result = agent.ask(message, history)

    answer = result.get("answer", "（无回答）")

    # 更新对话历史
    history = history + [
        {"role": "user", "content": message},
        {"role": "assistant", "content": answer},
    ]

    # 意图
    intent = result.get("intent", "未知")
    intent_map = {
        "plot_qa": "剧情/诗词问答 (RAG检索)",
        "character_relation": "人物关系 (GraphRAG)",
        "drawing": "绘图请求 (ComfyUI)",
        "general": "通用问答",
    }
    intent_text = intent_map.get(intent, intent)

    # 来源引用
    sources = result.get("rag_sources", "")
    if sources:
        sources_text = sources
    else:
        sources_text = "（本次回答未使用 RAG 检索）"

    # 幻觉校验
    score = result.get("hallucination_score", 0)
    flags = result.get("hallucination_flags", [])
    if flags:
        flag_text = "\n".join([f"  - {f}" for f in flags[:5]])
        hallucination_text = (
            f"幻觉评分: {score:.2f} (越低越好)\n"
            f"发现 {len(flags)} 处可疑内容:\n{flag_text}"
        )
    else:
        hallucination_text = f"幻觉评分: {score:.2f} - 未检测到明显幻觉"

    # 耗时
    elapsed = result.get("elapsed_time", {})
    elapsed_text = (
        f"总耗时: {elapsed.get('total', 0):.2f}s\n"
        f"  RAG: {elapsed.get('rag', 0):.2f}s\n"
        f"  GraphRAG: {elapsed.get('graphrag', 0):.2f}s\n"
        f"  绘图: {elapsed.get('draw', 0):.2f}s"
    )

    # 右栏可视化
    right_image = None
    right_html = ""
    prompt_info = ""

    # 如果有 GraphRAG 数据，显示相关子图
    graph_data = result.get("graph_data", {})
    if graph_data:
        detected = result.get("detected_characters", [])
        primary_char = detected[0] if detected else "贾宝玉"

        # 尝试加载对应人物的 Pyvis HTML
        html_path = HTML_DIR / f"subgraph_{primary_char}.html"
        if html_path.exists():
            with open(html_path, "r", encoding="utf-8") as f:
                content = f.read()
            import base64
            encoded = base64.b64encode(content.encode("utf-8")).decode("ascii")
            right_html = (
                f'<iframe src="data:text/html;base64,{encoded}" '
                f'style="width:100%; height:450px; border:none;" '
                f'loading="lazy"></iframe>'
            )

        # 尝试加载对应人物的 NetworkX PNG
        png_path = REPORT_DIR / f"subgraph_{primary_char}.png"
        if png_path.exists():
            right_image = str(png_path)

    # 如果有绘图结果
    draw_result = result.get("draw_result", {})
    if draw_result:
        if draw_result.get("image_path"):
            right_image = draw_result["image_path"]
            prompt_info = f"ComfyUI 已生成图片: {draw_result['image_path']}"
        elif draw_result.get("prompt_path"):
            prompt_path = draw_result["prompt_path"]
            with open(prompt_path, "r", encoding="utf-8") as f:
                prompt_data = json.load(f)
            prompt_info = (
                f"ComfyUI 提示词已保存 (服务未启动)\n"
                f"正向提示词:\n{prompt_data.get('positive_prompt', '')[:200]}...\n"
                f"提示词文件: {prompt_path}"
            )

    return (
        history, intent_text, sources_text,
        hallucination_text, elapsed_text,
        right_image, right_html, prompt_info
    )


def switch_model_scheme(scheme: str) -> str:
    """切换三套对照模型方案"""
    from agent_graph.model_client import get_model_client

    client = get_model_client()

    if scheme == "A - 原生Base(无RAG)":
        client.set_model("qwen2.5:7b", "ollama_base")
        return "已切换到方案A: 原生 Qwen2.5-7B Base（无 RAG、无微调）"
    elif scheme == "B - Base+RAG":
        client.set_model("qwen2.5:7b", "ollama_base")
        return "已切换到方案B: 原生 Qwen2.5-7B Base + 多级增强 RAG"
    elif scheme == "C - 微调+RAG+GraphRAG":
        client.set_model("qwen2.5:7b-lora", "ollama_finetuned")
        return "已切换到方案C: QLoRA微调 + RAG + GraphRAG + Agent\n（注: 微调模型需先完成阶段4训练后导入Ollama）"
    else:
        return "未知方案"


def upload_text_file(file) -> str:
    """上传新增红楼文本，增量更新向量库"""
    if file is None:
        return "未选择文件"

    try:
        # 保存到 raw_text 目录
        filename = os.path.basename(file.name) if hasattr(file, "name") else "uploaded.txt"
        dest = RAW_TEXT_DIR / f"uploaded_{filename}"
        shutil.copy2(file.name, str(dest))

        return (
            f"文件已保存: {dest}\n\n"
            f"增量索引更新说明:\n"
            f"  1. 新文本已存入 data/raw_text/\n"
            f"  2. 需运行 text_chunker.py 重新切片\n"
            f"  3. 需运行 build_index.py 重建向量库\n"
            f"  （完整增量更新脚本待后续集成）"
        )
    except Exception as e:
        return f"上传失败: {e}"


def load_static_image(image_name: str):
    """加载指定的 NetworkX 静态图"""
    path = REPORT_DIR / image_name
    if path.exists():
        return str(path)
    return None


def load_pyvis_html(html_name: str) -> str:
    """加载指定的 Pyvis 交互式 HTML，用 iframe 包裹以确保脚本可执行"""
    path = HTML_DIR / html_name
    if path.exists():
        with open(path, "r", encoding="utf-8") as f:
            content = f.read()
        # 用 base64 编码后通过 iframe srcdoc 加载，确保 <script> 可执行
        import base64
        encoded = base64.b64encode(content.encode("utf-8")).decode("ascii")
        iframe = (
            f'<iframe src="data:text/html;base64,{encoded}" '
            f'style="width:100%; height:450px; border:none;" '
            f'loading="lazy"></iframe>'
        )
        return iframe
    return "<p>HTML 文件不存在</p>"


# ═══════════════════════════════════════════════
# 可选图片列表
# ═══════════════════════════════════════════════

def get_available_images():
    """获取 report 目录下所有 PNG 文件"""
    images = []
    if REPORT_DIR.exists():
        for f in sorted(REPORT_DIR.glob("*.png")):
            images.append(f.name)
    return images


def get_available_htmls():
    """获取 Pyvis HTML 文件列表"""
    htmls = []
    if HTML_DIR.exists():
        for f in sorted(HTML_DIR.glob("*.html")):
            htmls.append(f.name)
    return htmls


# ═══════════════════════════════════════════════
# Gradio 界面构建
# ═══════════════════════════════════════════════

def build_ui():
    """构建 Gradio 双栏界面"""

    available_images = get_available_images()
    available_htmls = get_available_htmls()

    with gr.Blocks(
        title="红楼梦多模态智能解读专家系统",
    ) as app:

        # ── 标题 ──
        gr.Markdown(
            "# 红楼梦多模态智能解读专家系统\n"
            "QLoRA微调 + 多级RAG + GraphRAG知识图谱 + LangGraph Agent + 国风可视化"
        )

        # ── 系统状态 ──
        with gr.Row():
            status_text = gr.Textbox(
                label="系统状态",
                value="点击下方\"初始化系统\"按钮启动 Agent（首次启动需加载模型，约10秒）",
                interactive=False,
                scale=4,
            )
            init_btn = gr.Button("初始化系统", variant="primary", scale=1)

        # ── 主体：双栏 ──
        with gr.Row(equal_height=False):

            # ═══ 左栏：对话区 ═══
            with gr.Column(scale=1):

                chatbot = gr.Chatbot(
                    label="对话",
                    height=420,
                    placeholder="输入关于红楼梦的问题，例如：\n"
                        "- 林黛玉初进贾府时发生了什么？\n"
                        "- 贾宝玉和薛宝钗是什么关系？\n"
                        "- 画一张贾宝玉的人物关系图",
                )

                with gr.Row():
                    msg_input = gr.Textbox(
                        label="你的问题",
                        placeholder="输入问题后按回车或点击发送...",
                        scale=4,
                        lines=2,
                    )
                    send_btn = gr.Button("发送", variant="primary", scale=1)

                # 元信息面板
                with gr.Accordion("检索与校验详情", open=False):
                    intent_box = gr.Textbox(label="意图识别", interactive=False)
                    sources_box = gr.Textbox(label="来源引用", interactive=False, lines=2)
                    hallucination_box = gr.Textbox(
                        label="幻觉校验", interactive=False, lines=4
                    )
                    elapsed_box = gr.Textbox(label="耗时统计", interactive=False, lines=4)

            # ═══ 右栏：可视化区 ═══
            with gr.Column(scale=1):

                with gr.Tabs():
                    # Tab1: NetworkX 静态图
                    with gr.Tab("NetworkX 静态图谱"):
                        with gr.Row():
                            image_selector = gr.Dropdown(
                                choices=available_images,
                                value=available_images[0] if available_images else None,
                                label="选择图谱",
                                scale=3,
                            )
                            load_img_btn = gr.Button("加载", scale=1)

                        static_image = gr.Image(
                            label="人物关系图",
                            value=load_static_image(available_images[0]) if available_images else None,
                            height=400,
                        )

                        gr.Examples(
                            examples=[
                                ["core_network.png", "核心人物关系网络"],
                                ["subgraph_贾宝玉.png", "贾宝玉关系子图"],
                                ["subgraph_王熙凤.png", "王熙凤关系子图"],
                                ["relation_type_pie.png", "关系类型饼图"],
                                ["degree_distribution.png", "度中心性分布"],
                            ],
                            inputs=image_selector,
                            label="快速选择",
                        )

                    # Tab2: Pyvis 交互图谱
                    with gr.Tab("Pyvis 交互图谱"):
                        with gr.Row():
                            html_selector = gr.Dropdown(
                                choices=available_htmls,
                                value=available_htmls[0] if available_htmls else None,
                                label="选择交互图谱",
                                scale=3,
                            )
                            load_html_btn = gr.Button("加载", scale=1)

                        pyvis_html = gr.HTML(
                            value=load_pyvis_html(available_htmls[0]) if available_htmls else "<p>暂无HTML文件</p>",
                            label="交互式人物关系图（可拖拽缩放）",
                        )

                    # Tab3: ComfyUI 国风图
                    with gr.Tab("ComfyUI 国风图"):
                        guofeng_image = gr.Image(
                            label="国风人物关系图（AI生成）",
                            height=400,
                        )
                        prompt_info_box = gr.Textbox(
                            label="提示词信息",
                            interactive=False,
                            lines=6,
                            placeholder="当问题涉及人物关系时，系统会自动生成提示词并显示在此处",
                        )

                        gr.Markdown(
                            "---\n"
                            "**说明**: ComfyUI 服务需单独启动（默认端口 8188）。\n"
                            "服务未运行时，系统自动保存提示词模板，可在 "
                            "`visualize/comfyui_api/prompts/` 目录中查看。"
                        )

        # ── 底部：模型方案切换 + 文本上传 ──
        gr.Markdown("---\n### 系统设置")

        with gr.Row():
            with gr.Column(scale=1):
                scheme_radio = gr.Radio(
                    choices=[
                        "A - 原生Base(无RAG)",
                        "B - Base+RAG",
                        "C - 微调+RAG+GraphRAG",
                    ],
                    value="B - Base+RAG",
                    label="对照实验模型方案",
                )
                switch_btn = gr.Button("切换方案")
                scheme_status = gr.Textbox(
                    label="切换结果", interactive=False, lines=2
                )

            with gr.Column(scale=1):
                upload_file = gr.File(
                    label="上传新增红楼文本（增量更新向量库）",
                    file_types=[".txt", ".md"],
                )
                upload_btn = gr.Button("上传并更新")
                upload_status = gr.Textbox(
                    label="上传结果", interactive=False, lines=3
                )

        # ═══════════════════════════════════════════════
        # 事件绑定
        # ═══════════════════════════════════════════════

        # 初始化系统
        def init_system():
            try:
                agent = get_agent()
                return f"系统已就绪 - 模型: {agent.graph is not None}\nAgent 图构建完成，可以开始提问"
            except Exception as e:
                return f"初始化失败: {e}"

        init_btn.click(fn=init_system, outputs=status_text)

        # 发送消息
        outputs = [
            chatbot,          # history
            intent_box,
            sources_box,
            hallucination_box,
            elapsed_box,
            static_image,     # 右栏图片
            pyvis_html,       # 右栏HTML
            prompt_info_box,  # 提示词信息
        ]

        send_btn.click(
            fn=chat_respond,
            inputs=[msg_input, chatbot],
            outputs=outputs,
        ).then(
            fn=lambda: "",  # 清空输入框
            outputs=msg_input,
        )

        msg_input.submit(
            fn=chat_respond,
            inputs=[msg_input, chatbot],
            outputs=outputs,
        ).then(
            fn=lambda: "",
            outputs=msg_input,
        )

        # 加载静态图
        load_img_btn.click(
            fn=load_static_image,
            inputs=image_selector,
            outputs=static_image,
        )

        # 加载 Pyvis HTML
        load_html_btn.click(
            fn=load_pyvis_html,
            inputs=html_selector,
            outputs=pyvis_html,
        )

        # 切换模型方案
        switch_btn.click(
            fn=switch_model_scheme,
            inputs=scheme_radio,
            outputs=scheme_status,
        )

        # 上传文本
        upload_btn.click(
            fn=upload_text_file,
            inputs=upload_file,
            outputs=upload_status,
        )

    return app


# ═══════════════════════════════════════════════
# 启动
# ═══════════════════════════════════════════════

if __name__ == "__main__":
    app = build_ui()
    app.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True,
        theme=gr.themes.Soft(),
    )
