# 《人工智能红楼梦多模态智能解读专家系统》

## 项目简介

基于 Qwen2.5-7B Base 的红楼梦领域多模态智能问答平台，集成 QLoRA 领域微调、多级增强 RAG、GraphRAG 知识图谱、LangGraph 多工具 Agent、自动人物关系可视化等功能。

## 技术栈

- 基座模型：Qwen2.5-7B Base
- 微调方式：QLoRA 4bit 量化指令微调（LLaMA-Factory）
- 检索增强：BGE 中文嵌入 + Chroma 向量库 + BM25 多路召回 + Reranker 重排序
- 知识图谱：GraphRAG 三元组抽取 + NetworkX/Pyvis 可视化
- Agent 编排：LangGraph 状态机多工具调度
- 图像生成：ComfyUI 国风人物关系图
- 前端界面：Gradio 双栏交互

## 环境配置

```bash
# 创建 conda 环境（Python 3.10）
conda create -n pytorch_gpu python=3.10
conda activate pytorch_gpu

# 安装 PyTorch（CUDA 12.4）
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

# 安装项目依赖
pip install -r requirements.txt
```

## 项目结构

```
honglou_ai_multimodal/
├── models/                # 模型权重
├── data/
│   ├── raw_text/          # 红楼梦原文、诗词语料
│   ├── kg_triple/         # 人物关系三元组
│   ├── sft_dataset/       # 微调问答数据集
│   ├── test_set/          # 50道评测测试题
│   └── processed/         # 清洗后分块文本、EDA图表
├── rag_pipeline/          # RAG 检索管线
├── graphrag_index/        # GraphRAG 索引
├── lora_train/            # LLaMA-Factory 训练配置与日志
├── agent_graph/           # LangGraph Agent 编排
├── visualize/             # 可视化模块
├── evaluation/            # 评测脚本与结果
├── web_ui/                # Gradio 前端
├── utils/                 # 工具函数
├── report/                # 实验报告
├── requirements.txt
└── README.md
```

## 数据来源

| 数据 | 来源 |
|------|------|
| 红楼梦120回原文 | https://github.com/lixiaolong888888/HongLouMeng |
| 红楼诗词语料 | https://github.com/wangshub/hongloumeng-poetry |
| 人物关系三元组 | https://github.com/PrimerLi/Dream-of-the-Red-Chamber |
| 红楼知识图谱 | https://github.com/echoyoung/HongLouMeng-KnowledgeGraph |
| SFT问答数据集 | https://huggingface.co/datasets/liujch1998/honglou-qa |
| 基座模型 | https://modelscope.cn/models/Qwen/Qwen2.5-7B |

## 使用方式

```bash
# 启动 Gradio 前端
python web_ui/app.py
```
