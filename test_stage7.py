"""
阶段7 Gradio 前端端到端测试

测试内容:
1. 模块导入测试
2. UI 构建测试
3. 回调函数签名与返回值测试
4. 可视化资源加载测试
5. 模型方案切换测试
6. Agent 问答集成测试（需 Ollama 运行）
"""

import os
import sys
import time
import json

# 设置 HuggingFace 离线模式（避免连接超时）
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

# ═══════════════════════════════════════════════
# 测试框架
# ═══════════════════════════════════════════════

passed = 0
failed = 0
errors = []


def test(name, condition, detail=""):
    global passed, failed
    if condition:
        passed += 1
        print(f"  [PASS] {name}")
    else:
        failed += 1
        errors.append(f"{name}: {detail}")
        print(f"  [FAIL] {name} - {detail}")


def section(title):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")


# ═══════════════════════════════════════════════
# 测试开始
# ═══════════════════════════════════════════════

print("\n" + "═" * 60)
print("  红楼梦多模态智能解读专家系统 - 阶段7测试")
print("  Gradio 双栏前端端到端验证")
print("═" * 60)


# ── 测试1: 模块导入 ──
section("测试1: 模块导入")

try:
    from web_ui.app import (
        build_ui,
        chat_respond,
        switch_model_scheme,
        upload_text_file,
        load_static_image,
        load_pyvis_html,
        get_available_images,
        get_available_htmls,
    )
    test("web_ui.app 模块导入", True)
except Exception as e:
    test("web_ui.app 模块导入", False, str(e))

try:
    import gradio as gr
    test(f"Gradio 版本: {gr.__version__}", gr.__version__ is not None)
except Exception as e:
    test("Gradio 导入", False, str(e))


# ── 测试2: 可视化资源 ──
section("测试2: 可视化资源加载")

images = get_available_images()
test("PNG 图片数量 >= 5", len(images) >= 5, f"实际: {len(images)}")
print(f"    可用图片: {images}")

htmls = get_available_htmls()
test("HTML 文件数量 >= 2", len(htmls) >= 2, f"实际: {len(htmls)}")
print(f"    可用HTML: {htmls}")

# 测试加载核心图
img_path = load_static_image("core_network.png")
test("加载 core_network.png", img_path is not None and os.path.exists(img_path))

# 测试加载贾宝玉子图
img_path2 = load_static_image("subgraph_贾宝玉.png")
test("加载 subgraph_贾宝玉.png", img_path2 is not None and os.path.exists(img_path2))

# 测试加载 Pyvis HTML（应包含 iframe）
pyvis_content = load_pyvis_html("subgraph_贾宝玉.html")
test("Pyvis HTML 加载", len(pyvis_content) > 0)
test("Pyvis HTML 含 iframe", "<iframe" in pyvis_content)
test("Pyvis HTML 含 base64", "base64" in pyvis_content)

# 测试不存在的文件
img_none = load_static_image("nonexistent.png")
test("不存在的图片返回 None", img_none is None)

html_none = load_pyvis_html("nonexistent.html")
test("不存在的HTML返回提示", "不存在" in html_none)


# ── 测试3: UI 构建 ──
section("测试3: UI 构建")

try:
    app = build_ui()
    test("build_ui() 返回 Blocks", app is not None)
    test("App 类型为 Blocks", type(app).__name__ == "Blocks")
except Exception as e:
    test("UI 构建", False, str(e))
    import traceback
    traceback.print_exc()


# ── 测试4: 模型方案切换 ──
section("测试4: 模型方案切换")

try:
    result_a = switch_model_scheme("A - 原生Base(无RAG)")
    test("方案A 切换", "方案A" in result_a, result_a)

    result_b = switch_model_scheme("B - Base+RAG")
    test("方案B 切换", "方案B" in result_b, result_b)

    result_c = switch_model_scheme("C - 微调+RAG+GraphRAG")
    test("方案C 切换", "方案C" in result_c, result_c)

    result_unknown = switch_model_scheme("未知方案")
    test("未知方案处理", "未知" in result_unknown, result_unknown)

    # 重置回方案B（默认）
    switch_model_scheme("B - Base+RAG")
except Exception as e:
    test("模型方案切换", False, str(e))


# ── 测试5: 空消息处理 ──
section("测试5: 空消息处理")

try:
    result = chat_respond("", [])
    test("空消息返回8个值", len(result) == 8)
    test("空消息意图提示", "请输入问题" in result[1])
    test("空消息不修改历史", len(result[0]) == 0)
except Exception as e:
    test("空消息处理", False, str(e))


# ── 测试6: Agent 问答集成（需要 Ollama）──
section("测试6: Agent 问答集成")

ollama_running = False
try:
    import ollama
    models = ollama.list()
    model_names = [m.model for m in models.get("models", [])]
    ollama_running = "qwen2.5:7b" in model_names
except Exception:
    pass

if ollama_running:
    print("  Ollama 运行中，开始 Agent 问答测试...")

    test_questions = [
        ("林黛玉初进贾府时发生了什么？", "剧情问答"),
        ("贾宝玉和林黛玉是什么关系？", "人物关系"),
    ]

    for q, desc in test_questions:
        try:
            result = chat_respond(q, [])
            test(f"{desc} - 返回8个值", len(result) == 8)
            test(f"{desc} - 历史更新", len(result[0]) >= 2)
            test(f"{desc} - 有回答内容", len(result[0][-1].get("content", "")) > 10)
            test(f"{desc} - 有意图", len(result[1]) > 0)
            test(f"{desc} - 有耗时", len(result[4]) > 0)
        except Exception as e:
            test(f"{desc} 问答", False, str(e))
else:
    print("  [跳过] Ollama 未运行，跳过 Agent 集成测试")
    print("  启动 Ollama 后重新运行可完成完整测试")


# ── 测试7: 文件上传 ──
section("测试7: 文件上传")

try:
    result = upload_text_file(None)
    test("空文件上传处理", "未选择" in result)
except Exception as e:
    test("空文件上传处理", False, str(e))


# ═══════════════════════════════════════════════
# 测试总结
# ═══════════════════════════════════════════════

print(f"\n{'═'*60}")
print(f"  阶段7测试总结")
print(f"{'═'*60}")
print(f"  通过: {passed}")
print(f"  失败: {failed}")
print(f"  总计: {passed + failed}")
print(f"  通过率: {passed/(passed+failed)*100:.1f}%")

if errors:
    print(f"\n  失败项:")
    for e in errors:
        print(f"    - {e}")

print(f"\n{'═'*60}")
if failed == 0:
    print("  全部测试通过！阶段7 Gradio 前端开发完成。")
else:
    print(f"  有 {failed} 项测试未通过，请检查。")
print(f"{'═'*60}")
