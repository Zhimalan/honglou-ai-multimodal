"""
ComfyUI 国风人物关系图生成 API 客户端
封装 ComfyUI API 调用，生成红楼人物国风关系图。
当本地无 ComfyUI 服务时，输出提示词模板供手动使用。
"""

import os
import json
import time
import urllib.request
import urllib.error
from typing import Optional
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
REPORT_DIR = PROJECT_ROOT / "report"
PROMPT_DIR = PROJECT_ROOT / "visualize" / "comfyui_api" / "prompts"


class ComfyUIClient:
    """ComfyUI API 客户端"""

    def __init__(self, server_url: str = "http://127.0.0.1:8188"):
        self.server_url = server_url.rstrip("/")
        self.is_available = self._check_connection()

    def _check_connection(self) -> bool:
        """检查 ComfyUI 服务是否可用"""
        try:
            req = urllib.request.Request(f"{self.server_url}/system_stats", method="GET")
            with urllib.request.urlopen(req, timeout=3) as resp:
                return resp.status == 200
        except (urllib.error.URLError, OSError, TimeoutError):
            return False

    def queue_prompt(self, workflow: dict) -> Optional[str]:
        """提交工作流到 ComfyUI，返回 prompt_id"""
        if not self.is_available:
            return None
        try:
            data = json.dumps({"prompt": workflow}).encode("utf-8")
            req = urllib.request.Request(
                f"{self.server_url}/prompt",
                data=data,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=10) as resp:
                result = json.loads(resp.read().decode("utf-8"))
                return result.get("prompt_id")
        except Exception as e:
            print(f"  [error] 提交工作流失败: {e}")
            return None

    def get_history(self, prompt_id: str) -> Optional[dict]:
        """查询生成结果"""
        if not self.is_available:
            return None
        try:
            req = urllib.request.Request(
                f"{self.server_url}/history/{prompt_id}", method="GET"
            )
            with urllib.request.urlopen(req, timeout=5) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except Exception:
            return None

    def wait_for_result(self, prompt_id: str, timeout: int = 120) -> Optional[dict]:
        """等待生成完成，返回结果"""
        start = time.time()
        while time.time() - start < timeout:
            history = self.get_history(prompt_id)
            if history and prompt_id in history:
                return history[prompt_id]
            time.sleep(2)
        return None

    def get_image(self, filename: str, subfolder: str = "", folder_type: str = "output") -> Optional[bytes]:
        """下载生成的图片"""
        if not self.is_available:
            return None
        try:
            params = f"?filename={filename}&subfolder={subfolder}&type={folder_type}"
            req = urllib.request.Request(
                f"{self.server_url}/view{params}", method="GET"
            )
            with urllib.request.urlopen(req, timeout=10) as resp:
                return resp.read()
        except Exception as e:
            print(f"  [error] 下载图片失败: {e}")
            return None


# ──────────────────────────────────────────────
# 提示词模板
# ──────────────────────────────────────────────

def build_positive_prompt(character: str, relations: list[dict]) -> str:
    """
    构建国风人物关系图的正向提示词。
    relations: [{name, relation_type, description}, ...]
    """
    # 人物列表
    char_list = "、".join([character] + [r["name"] for r in relations[:8]])

    # 关系描述
    rel_descs = []
    for r in relations[:6]:
        rtype = r.get("relation_type", "其他")
        rel_descs.append(f"{r['name']}({rtype})")
    rel_text = "，".join(rel_descs)

    prompt = (
        f"中国传统水墨画风格，红楼梦人物关系图，{char_list}，"
        f"水墨晕染，工笔人物，古典园林背景，"
        f"中心人物{character}居中，周围环绕{rel_text}，"
        f"人物之间用纤细墨线连接表示关系，"
        f"红色线条表示夫妻关系，蓝色线条表示亲子关系，绿色线条表示主仆关系，"
        f"人物名称用楷体标注，文字清晰可读，"
        f"整体色调淡雅，宣纸质感，中国传统绘画风格，"
        f"高质量，细节丰富，构图均衡，8k分辨率"
    )
    return prompt


def build_negative_prompt() -> str:
    """构建反向提示词"""
    return (
        "模糊, 扭曲, 文字重叠, 文字不清晰, 低质量, 伪影, "
        "变形, 多余手指, 西方风格, 现代元素, 卡通风格, "
        "3D渲染, 油画风格, 色彩过于鲜艳, 构图混乱, "
        "人物面部模糊, 身体比例失调, 背景杂乱"
    )


def build_controlnet_prompt(character: str) -> str:
    """构建 ControlNet Canny 构图约束提示词"""
    return (
        f"思维导图布局，{character}位于中心，"
        f"分支向四周展开，层级分明，"
        f"每个分支节点为人物头像位置，"
        f"线条简洁清晰，结构对称"
    )


# ──────────────────────────────────────────────
# ComfyUI 工作流模板
# ──────────────────────────────────────────────

def build_workflow(
    positive_prompt: str,
    negative_prompt: str,
    model_name: str = "v1-5-pruned-emaonly.safetensors",
    width: int = 1024,
    height: int = 768,
    steps: int = 30,
    cfg_scale: float = 7.5,
    sampler: str = "DPM++ 2M Karras",
) -> dict:
    """
    构建 ComfyUI 工作流 JSON（文生图基础工作流）。
    实际使用时需根据 ComfyUI 中安装的节点进行调整。
    """
    workflow = {
        "3": {
            "class_type": "KSampler",
            "inputs": {
                "seed": int(time.time()) % (2**32),
                "steps": steps,
                "cfg": cfg_scale,
                "sampler_name": sampler,
                "scheduler": "karras",
                "denoise": 1.0,
                "model": ["4", 0],
                "positive": ["6", 0],
                "negative": ["7", 0],
                "latent_image": ["5", 0],
            },
        },
        "4": {
            "class_type": "CheckpointLoaderSimple",
            "inputs": {"ckpt_name": model_name},
        },
        "5": {
            "class_type": "EmptyLatentImage",
            "inputs": {"width": width, "height": height, "batch_size": 1},
        },
        "6": {
            "class_type": "CLIPTextEncode",
            "inputs": {"text": positive_prompt, "clip": ["4", 1]},
        },
        "7": {
            "class_type": "CLIPTextEncode",
            "inputs": {"text": negative_prompt, "clip": ["4", 1]},
        },
        "8": {
            "class_type": "VAEDecode",
            "inputs": {"samples": ["3", 0], "vae": ["4", 2]},
        },
        "9": {
            "class_type": "SaveImage",
            "inputs": {"images": ["8", 0]},
        },
    }
    return workflow


# ──────────────────────────────────────────────
# 主接口
# ──────────────────────────────────────────────

def generate_character_relation_image(
    character: str,
    relations: list[dict],
    output_dir: str = None,
    server_url: str = "http://127.0.0.1:8188",
) -> dict:
    """
    生成红楼人物国风关系图。

    当 ComfyUI 服务可用时：自动调用 API 生成图片。
    当 ComfyUI 不可用时：保存提示词模板到文件，返回提示词。

    Returns:
        {
            "success": bool,
            "image_path": str | None,       # 生成的图片路径
            "prompt_path": str | None,      # 提示词文件路径
            "positive_prompt": str,
            "negative_prompt": str,
            "message": str,
        }
    """
    if output_dir is None:
        output_dir = str(PROMPT_DIR)
    os.makedirs(output_dir, exist_ok=True)

    # 构建提示词
    positive_prompt = build_positive_prompt(character, relations)
    negative_prompt = build_negative_prompt()

    result = {
        "success": False,
        "image_path": None,
        "prompt_path": None,
        "positive_prompt": positive_prompt,
        "negative_prompt": negative_prompt,
        "message": "",
    }

    # 尝试连接 ComfyUI
    client = ComfyUIClient(server_url)

    if client.is_available:
        print(f"  [comfyui] 服务可用，开始生成 {character} 关系图...")

        # 构建工作流
        workflow = build_workflow(positive_prompt, negative_prompt)

        # 提交生成请求
        prompt_id = client.queue_prompt(workflow)
        if prompt_id is None:
            result["message"] = "ComfyUI 提交工作流失败"
            _save_prompt(character, positive_prompt, negative_prompt, output_dir, result)
            return result

        # 等待结果
        print(f"  [comfyui] 等待生成完成 (prompt_id: {prompt_id})...")
        history = client.wait_for_result(prompt_id, timeout=180)

        if history and "outputs" in history:
            # 下载图片
            for node_id, node_output in history["outputs"].items():
                if "images" in node_output:
                    for img_info in node_output["images"]:
                        filename = img_info["filename"]
                        subfolder = img_info.get("subfolder", "")
                        image_data = client.get_image(filename, subfolder)

                        if image_data:
                            img_path = os.path.join(
                                output_dir,
                                f"guofeng_{character}_{int(time.time())}.png"
                            )
                            with open(img_path, "wb") as f:
                                f.write(image_data)
                            result["success"] = True
                            result["image_path"] = img_path
                            result["message"] = f"图片已保存: {img_path}"
                            print(f"  [comfyui] 图片保存 → {img_path}")
                            return result

        result["message"] = "ComfyUI 生成超时或失败"
        _save_prompt(character, positive_prompt, negative_prompt, output_dir, result)
        return result

    else:
        # ComfyUI 不可用，保存提示词模板
        result["message"] = (
            f"ComfyUI 服务不可用 ({server_url})，已保存提示词模板。"
            "请启动 ComfyUI 后重试，或手动将提示词输入到 ComfyUI 中。"
        )
        _save_prompt(character, positive_prompt, negative_prompt, output_dir, result)
        return result


def _save_prompt(character, positive, negative, output_dir, result):
    """保存提示词到文件"""
    prompt_data = {
        "character": character,
        "positive_prompt": positive,
        "negative_prompt": negative,
        "controlnet_prompt": build_controlnet_prompt(character),
        "recommended_model": "国风/水墨风格 LoRA 模型，如 guofeng_v2.safetensors",
        "recommended_params": {
            "width": 1024,
            "height": 768,
            "steps": 30,
            "cfg_scale": 7.5,
            "sampler": "DPM++ 2M Karras",
        },
    }

    prompt_path = os.path.join(output_dir, f"prompt_{character}.json")
    with open(prompt_path, "w", encoding="utf-8") as f:
        json.dump(prompt_data, f, ensure_ascii=False, indent=2)

    result["prompt_path"] = prompt_path
    print(f"  [prompt] 提示词模板已保存 → {prompt_path}")


# ──────────────────────────────────────────────
# 测试
# ──────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("ComfyUI 国风人物关系图 API 测试")
    print("=" * 60)

    # 模拟贾宝玉的关系数据
    test_relations = [
        {"name": "林黛玉", "relation_type": "其他", "description": "林如海与贾敏之女"},
        {"name": "薛宝钗", "relation_type": "夫妻", "description": "薛姨妈之女"},
        {"name": "王夫人", "relation_type": "亲子", "description": "贾政之妻"},
        {"name": "袭人", "relation_type": "主仆", "description": "贾宝玉的丫鬟"},
        {"name": "晴雯", "relation_type": "主仆", "description": "贾宝玉的丫鬟"},
        {"name": "史太君", "relation_type": "其他", "description": "贾代善之妻"},
        {"name": "王熙凤", "relation_type": "其他", "description": "贾琏之妻"},
        {"name": "贾政", "relation_type": "亲子", "description": "贾宝玉之父"},
    ]

    print("\n[测试] 生成贾宝玉国风关系图...")
    result = generate_character_relation_image("贾宝玉", test_relations)

    print(f"\n  成功: {result['success']}")
    print(f"  消息: {result['message']}")
    if result["prompt_path"]:
        print(f"  提示词文件: {result['prompt_path']}")
    if result["image_path"]:
        print(f"  图片文件: {result['image_path']}")

    print("\n--- 正向提示词 ---")
    print(result["positive_prompt"][:200] + "...")
    print("\n--- 反向提示词 ---")
    print(result["negative_prompt"])

    print("\n" + "=" * 60)
    print("ComfyUI API 测试完成!")
    print("  (ComfyUI 服务未运行时，提示词模板已保存)")
    print("=" * 60)
