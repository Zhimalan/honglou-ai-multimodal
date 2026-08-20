"""
阶段8 ComfyUI 国风人物关系图端到端测试

测试内容:
1. 模块导入测试
2. 提示词模板构建（正向/反向/ControlNet）
3. 工作流 JSON 结构合规性
4. ComfyUI 服务连接检测
5. 端到端生成 1 张国风关系图（实际调用 ComfyUI API）
6. 生成图片文件校验（存在 + 大小合理 + PNG 头）
7. ComfyUI 不可用时的降级行为（提示词模式）
"""

import os
import sys
import json
import time
import urllib.request
import urllib.error
from pathlib import Path

# 设置 HuggingFace 离线模式
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

REPORT_DIR = os.path.join(PROJECT_ROOT, "report")
PROMPT_DIR = os.path.join(PROJECT_ROOT, "visualize", "comfyui_api", "prompts")
COMFYUI_URL = "http://127.0.0.1:8188"

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
print("  红楼梦多模态智能解读专家系统 - 阶段8测试")
print("  ComfyUI 国风人物关系图生成")
print("═" * 60)


# ── 测试1: 模块导入 ──
section("测试1: 模块导入")

try:
    from visualize.comfyui_api.comfyui_client import (
        ComfyUIClient,
        build_positive_prompt,
        build_negative_prompt,
        build_controlnet_prompt,
        build_workflow,
        generate_character_relation_image,
    )
    test("comfyui_client 模块导入", True)
except Exception as e:
    test("comfyui_client 模块导入", False, str(e))
    import traceback
    traceback.print_exc()
    sys.exit(1)


# ── 测试2: 提示词模板构建 ──
section("测试2: 提示词模板构建")

relations = [
    {"name": "林黛玉", "relation_type": "其他", "description": "木石前盟"},
    {"name": "薛宝钗", "relation_type": "夫妻", "description": "金玉良缘"},
    {"name": "王夫人", "relation_type": "亲子", "description": "母亲"},
    {"name": "袭人", "relation_type": "主仆", "description": "贴身丫鬟"},
]

pos_prompt = build_positive_prompt("贾宝玉", relations)
neg_prompt = build_negative_prompt()
cn_prompt = build_controlnet_prompt("贾宝玉")

test("正向提示词非空", len(pos_prompt) > 50)
test("正向提示词含中心人物", "贾宝玉" in pos_prompt)
test("正向提示词含国风 LoRA 触发词 'gufeng style'", "gufeng style" in pos_prompt,
     "国风 LoRA 需要触发词才能生效")
test("正向提示词含所有关系人", all(r["name"] in pos_prompt for r in relations))
test("正向提示词强调单人物", "solo" in pos_prompt or "single" in pos_prompt or "1girl" in pos_prompt,
     "防 SD1.5 多人物群像问题")
test("正向提示词含国风关键词", any(kw in pos_prompt for kw in ["水墨", "中国", "古典", "传统"]))
test("反向提示词非空", len(neg_prompt) > 30)
test("反向提示词含负面词", "模糊" in neg_prompt and "低质量" in neg_prompt)
test("反向提示词排除多人物", any(kw in neg_prompt for kw in ["multiple", "crowd", "duplicated"]))
test("ControlNet提示词含中心", "贾宝玉" in cn_prompt and "中心" in cn_prompt)

print(f"\n  正向提示词片段: {pos_prompt[:120]}...")
print(f"  ControlNet提示词: {cn_prompt}")


# ── 测试3: 工作流 JSON 结构 ──
section("测试3: 工作流 JSON 结构")

workflow = build_workflow(pos_prompt, neg_prompt, model_name="anything-v5-PrtRE.safetensors")
test("workflow 是 dict", isinstance(workflow, dict))
test("workflow 含必要节点（默认开 LoRA）", set(workflow.keys()) == {"3", "4", "5", "6", "7", "8", "9", "10"})

ksampler = workflow["3"]["inputs"]
test("KSampler.seed 存在", "seed" in ksampler)
test("KSampler.steps=25", ksampler["steps"] == 25, f"实际: {ksampler['steps']}")
test("KSampler.cfg=7", ksampler["cfg"] == 7, f"实际: {ksampler['cfg']}")
test("KSampler.sampler_name=dpmpp_2m", ksampler["sampler_name"] == "dpmpp_2m",
     f"实际: {ksampler['sampler_name']}")
test("KSampler.scheduler=karras", ksampler["scheduler"] == "karras",
     f"实际: {ksampler['scheduler']}")
test("KSampler.denoise=1.0", ksampler["denoise"] == 1.0)
test("KSampler.model 来自 LoraLoader(10)", ksampler["model"] == ["10", 0],
     "LoRA 必须串接在 Checkpoint → KSampler 之间")

checkpoint = workflow["4"]["inputs"]
test("CheckpointLoaderSimple.ckpt_name=anything-v5", "anything-v5" in checkpoint["ckpt_name"])

latent = workflow["5"]["inputs"]
test("EmptyLatentImage.width=768", latent["width"] == 768, f"实际: {latent['width']}")
test("EmptyLatentImage.height=512", latent["height"] == 512, f"实际: {latent['height']}")

save = workflow["9"]["inputs"]
test("SaveImage.filename_prefix 存在", "filename_prefix" in save,
     "新版 ComfyUI 要求 filename_prefix 参数")

# LoraLoader 节点验证
lora = workflow["10"]["inputs"]
test("LoraLoader 存在", "10" in workflow)
test("LoraLoader.lora_name=chinese_style_lora", "chinese_style" in lora["lora_name"],
     f"实际: {lora['lora_name']}")
test("LoraLoader.strength_model=0.8", lora["strength_model"] == 0.8,
     f"实际: {lora['strength_model']}")
test("LoraLoader.model 来自 CheckpointLoader(4)", lora["model"] == ["4", 0])
test("LoraLoader.clip 来自 CheckpointLoader(4)", lora["clip"] == ["4", 1])
# CLIPTextEncode 必须从 LoraLoader 取 CLIP（已施加 LoRA 的 CLIP）
test("CLIPTextEncode.clip 来自 LoraLoader(10)", workflow["6"]["inputs"]["clip"] == ["10", 1],
     "LoRA 必须串接在 CLIPTextEncode 之前")
test("Negative CLIPTextEncode.clip 来自 LoraLoader(10)", workflow["7"]["inputs"]["clip"] == ["10", 1])

# 测试 use_lora=False 时工作流回退
workflow_no_lora = build_workflow(pos_prompt, neg_prompt, use_lora=False)
test("use_lora=False 时不含 LoraLoader(10)", "10" not in workflow_no_lora)
test("use_lora=False 时 KSampler 仍连接 Checkpoint",
     workflow_no_lora["3"]["inputs"]["model"] == ["4", 0])

# JSON 序列化测试
try:
    json.dumps(workflow)
    test("workflow 可 JSON 序列化", True)
except Exception as e:
    test("workflow 可 JSON 序列化", False, str(e))


# ── 测试4: ComfyUI 服务连接检测 ──
section("测试4: ComfyUI 服务连接")

def check_comfyui_alive(url=COMFYUI_URL, timeout=3):
    try:
        req = urllib.request.Request(f"{url}/system_stats", method="GET")
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.status == 200
    except (urllib.error.URLError, OSError, TimeoutError):
        return False


comfyui_alive = check_comfyui_alive()
test(f"ComfyUI 服务可达 ({COMFYUI_URL})", comfyui_alive)

if comfyui_alive:
    # 查询 checkpoints 列表
    try:
        req = urllib.request.Request(f"{COMFYUI_URL}/models/checkpoints", method="GET")
        with urllib.request.urlopen(req, timeout=5) as resp:
            checkpoints = json.loads(resp.read().decode("utf-8"))
        test("anythint-v5 模型可用", "anything-v5-PrtRE.safetensors" in checkpoints,
             f"实际: {checkpoints}")
    except Exception as e:
        test("查询 checkpoints 列表", False, str(e))

    # 查询 loras 列表，验证国风 LoRA 已就位
    try:
        req = urllib.request.Request(f"{COMFYUI_URL}/models/loras", method="GET")
        with urllib.request.urlopen(req, timeout=5) as resp:
            loras = json.loads(resp.read().decode("utf-8"))
        test("国风 LoRA (chinese_style_lora) 已安装",
             "chinese_style_lora_v1.0.safetensors" in loras,
             f"实际: {loras}")
    except Exception as e:
        test("查询 loras 列表", False, str(e))

    # ComfyUIClient 自身连接检测
    client = ComfyUIClient(COMFYUI_URL)
    test("ComfyUIClient.is_available", client.is_available)


# ── 测试5: 端到端生成图片 ──
section("测试5: 端到端生成国风关系图")

if comfyui_alive:
    output_dir = os.path.join(REPORT_DIR, "test_stage8")
    os.makedirs(output_dir, exist_ok=True)

    # 清理旧文件（用 pathlib.Path.unlink，绕过回收站机制）
    for f in os.listdir(output_dir):
        if f.startswith("guofeng_"):
            try:
                Path(os.path.join(output_dir, f)).unlink()
            except OSError:
                pass  # 清理失败不阻塞测试

    print(f"  输出目录: {output_dir}")
    print("  准备生成 林黛玉 国风关系图（25步，约需30秒）...")

    start = time.time()
    result = generate_character_relation_image(
        character="林黛玉",
        relations=relations,
        output_dir=output_dir,
        server_url=COMFYUI_URL,
    )
    elapsed = time.time() - start

    test("生成返回 dict", isinstance(result, dict))
    test("生成 success=True", result.get("success") is True, f"实际: {result.get('message','')}")
    test("生成 image_path 存在", result.get("image_path") is not None)
    test("生成 prompt_path 为 None (服务可用时直接生成)", result.get("prompt_path") is None)
    test("生成耗时 < 120s", elapsed < 120, f"实际: {elapsed:.1f}s")

    img_path = result.get("image_path")
    if img_path and os.path.exists(img_path):
        size_kb = os.path.getsize(img_path) / 1024
        test(f"图片文件存在 ({size_kb:.0f}KB)", size_kb > 50,
             f"文件过小可能生成失败: {size_kb:.0f}KB")
        # 检查 PNG 文件头（89 50 4E 47）
        with open(img_path, "rb") as f:
            header = f.read(8)
        test("图片为 PNG 格式", header[:4] == b"\x89PNG", f"实际头: {header[:8]}")
    else:
        test("图片文件存在", False, "image_path 不存在")

    # positive_prompt 字段检查
    test("返回含正向提示词", "水墨" in result.get("positive_prompt", ""))
    test("返回含反向提示词", "模糊" in result.get("negative_prompt", ""))

    print(f"\n  ✓ 生成成功，耗时 {elapsed:.1f}s")
    if img_path:
        print(f"  ✓ 图片路径: {img_path}")
else:
    print("  [跳过] ComfyUI 不可用，跳过端到端生成测试")


# ── 测试6: 降级行为（ComfyUI 不可用）──
section("测试6: ComfyUI 不可用时的降级行为")

degraded_result = generate_character_relation_image(
    character="测试人物",
    relations=[{"name": "关系1", "relation_type": "其他", "description": ""}],
    output_dir=PROMPT_DIR,
    server_url="http://127.0.0.1:9999",  # 错端口
)
test("降级 success=False", degraded_result["success"] is False)
test("降级 prompt_path 存在", degraded_result["prompt_path"] is not None)
test("降级 image_path=None", degraded_result["image_path"] is None)
test("降级 message 含提示", "不可用" in degraded_result["message"] or "提示词" in degraded_result["message"])

# 检查提示词文件
if degraded_result["prompt_path"] and os.path.exists(degraded_result["prompt_path"]):
    with open(degraded_result["prompt_path"], "r", encoding="utf-8") as f:
        prompt_data = json.load(f)
    test("提示词文件含 character", prompt_data.get("character") == "测试人物")
    test("提示词文件含 positive_prompt", len(prompt_data.get("positive_prompt", "")) > 50)
    test("提示词文件含 negative_prompt", len(prompt_data.get("negative_prompt", "")) > 20)
    test("提示词文件含 controlnet_prompt", "controlnet_prompt" in prompt_data)
    test("提示词文件含 recommended_params", "width" in prompt_data.get("recommended_params", {}))
    print(f"  提示词文件: {degraded_result['prompt_path']}")


# ── 测试7: 与 Agent 集成 ──
section("测试7: 与 LangGraph Agent 集成")

if comfyui_alive:
    # 通过 agent_graph 间接调用绘图
    from agent_graph.nodes import comfyui_tool_node
    state = {
        "question": "画出贾宝玉的关系图",
        "detected_characters": ["贾宝玉"],
        "graph_data": {
            "贾宝玉": [
                {"name": "林黛玉", "type": "其他", "description": ""},
                {"name": "薛宝钗", "type": "夫妻", "description": ""},
            ]
        },
        "graph_summary": "",
        "intent": "drawing",
    }
    print("  通过 Agent 节点测试绘图（25步）...")
    start = time.time()
    new_state = comfyui_tool_node(state)
    elapsed = time.time() - start

    draw = new_state.get("draw_result", {})
    test("Agent 节点返回 draw_result", isinstance(draw, dict))
    test("Agent 节点绘图成功", draw.get("success") is True, f"实际: {draw.get('message','')}")
    test("Agent 节点 image_path", draw.get("image_path") is not None)
    test(f"Agent 绘图耗时 < 120s (实际 {elapsed:.1f}s)", elapsed < 120)

    if draw.get("image_path") and os.path.exists(draw["image_path"]):
        size_kb = os.path.getsize(draw["image_path"]) / 1024
        test(f"Agent 绘图文件有效 ({size_kb:.0f}KB)", size_kb > 50)
else:
    print("  [跳过] ComfyUI 不可用，跳过 Agent 集成测试")


# ═══════════════════════════════════════════════
# 测试总结
# ═══════════════════════════════════════════════

print(f"\n{'═'*60}")
print(f"  阶段8测试总结")
print(f"{'═'*60}")
print(f"  通过: {passed}")
print(f"  失败: {failed}")
print(f"  总计: {passed + failed}")
if passed + failed > 0:
    print(f"  通过率: {passed/(passed+failed)*100:.1f}%")

if errors:
    print(f"\n  失败项:")
    for e in errors:
        print(f"    - {e}")

print(f"\n{'═'*60}")
if failed == 0:
    print("  全部测试通过！阶段8 ComfyUI 国风绘图完成。")
else:
    print(f"  有 {failed} 项测试未通过，请检查。")
print(f"{'═'*60}")
