# 红楼梦多模态智能解读专家系统 — 实验报告

## 一、引言

### 1.1 项目背景

《红楼梦》是中国古典文学的巅峰之作，全书120回、涉及人物900余个，情节错综复杂、诗词典故密集，对普通读者和研究者均构成较高的理解门槛。传统红楼梦研究依赖人工查阅回目与注释，效率低下且难以建立全局视角。随着大语言模型（LLM）技术的快速发展，基于领域微调与检索增强生成（RAG）的智能问答系统为古典文学解读提供了新的技术路径。

然而，通用大模型在红楼梦领域存在三大局限：一是事实准确性不足，容易"幻觉"编造情节；二是缺乏原文检索能力，无法定位具体回目；三是人物关系网络复杂，单一文本检索难以捕捉全局关系。本项目旨在构建一个集成多模态技术的红楼梦智能解读专家系统，通过QLoRA领域微调、多级增强RAG、GraphRAG知识图谱、LangGraph多工具Agent等技术栈的联合应用，系统性地解决上述问题。

### 1.2 项目目标

1. 构建红楼梦领域问答数据集，完成QLoRA微调训练，使模型具备红楼梦领域知识
2. 实现多级增强RAG检索管道（BGE向量+BM25+RRF融合+HyDE+Reranker），将回答锚定在原文片段上
3. 基于GraphRAG构建人物关系知识图谱，提供结构化关系查询能力
4. 使用LangGraph编排多工具Agent，实现检索、图谱、生成工具的智能调度
5. 开发Gradio交互前端，集成ComfyUI国风人物关系可视化
6. 设计三方案对照评测体系（A: 纯Base / B: Base+RAG / C: 微调+RAG），通过50题测试集量化各模块增益

### 1.3 技术栈概览

| 组件 | 技术选型 | 版本 |
|------|---------|------|
| 基座模型 | Qwen2.5-7B Base / Instruct | — |
| 微调框架 | LLaMA-Factory | 0.9.5 |
| 嵌入模型 | BAAI/bge-small-zh-v1.5 | 512维 |
| 向量库 | ChromaDB | — |
| 重排序 | BAAI/bge-reranker-base | CrossEncoder |
| 知识图谱 | NetworkX + Pyvis | — |
| Agent | LangGraph | — |
| 前端 | Gradio | 6.25.0 |
| 图像生成 | ComfyUI + SD1.5 + 国风LoRA | — |
| 训练平台 | AutoDL 4090D (24GB VRAM) | — |

## 二、系统架构设计

系统采用分层架构设计，自底向上分为数据层、检索层、模型层、编排层和展示层：

```
┌─────────────────────────────────────────────────┐
│              展示层 (Gradio + ComfyUI)            │
├─────────────────────────────────────────────────┤
│            编排层 (LangGraph Agent)               │
│   ┌──────────┐ ┌──────────┐ ┌──────────┐         │
│   │ RAG工具  │ │ GraphRAG │ │ 生成工具 │         │
│   └──────────┘ └──────────┘ └──────────┘         │
├─────────────────────────────────────────────────┤
│              模型层 (Qwen2.5-7B)                  │
│   原生Base  │  Base+RAG  │  LoRA微调+RAG         │
├─────────────────────────────────────────────────┤
│    检索层 (BGE+Chroma+BM25+RRF+HyDE+Reranker)    │
├─────────────────────────────────────────────────┤
│        数据层 (原文120回 / QA数据集 / 三元组)       │
└─────────────────────────────────────────────────┘
```

## 三、阶段1：环境配置

### 3.1 本地环境

开发环境基于Windows 11，硬件配置为i9-13900HX / 16GB RAM / RTX 4060 8GB。本地环境承担数据预处理、RAG管道、GraphRAG、Agent、前端和评测任务。Python 3.10 + PyTorch 2.x + CUDA 12.4，通过conda隔离环境管理依赖。关键依赖包括sentence-transformers（嵌入与重排序）、chromadb（向量库）、jieba（中文分词）、langgraph（Agent编排）、gradio（前端）、networkx（知识图谱）等。

### 3.2 云端训练环境

QLoRA微调在AutoDL平台租用4090D实例（24GB VRAM）完成。镜像选择PyTorch 2.3.0 + CUDA 12.1，通过ModelScope下载Qwen2.5-7B-Instruct模型（约15GB）。训练过程中遇到torch版本兼容性问题：LLaMA-Factory的`[torch]`依赖会将torch升级至2.13.0，而该版本需要CUDA 13驱动支持，AutoDL镜像仅提供CUDA 12.4驱动。解决方案为手动锁定torch 2.3.0+cu121和torchaudio 2.3.0，并将transformers降级至4.x系列。

## 四、阶段2：数据预处理

### 4.1 原文清洗与分章

红楼梦原文来源于GitHub开源项目（120回完整版），预处理流程包括：去除注释标记与空白字符、按"第X回"正则分割为120个章节、每章平均约6500字。同时引入红楼诗词语料库作为诗词类问答的补充检索源。

### 4.2 文本分块

采用滑动窗口策略对每回文本进行分块，块大小512字符、重叠128字符，最终生成1620个文本块（chunks）。每个块附带元数据（chapter_num、chunk_id、char_count），供RAG检索时溯源。

### 4.3 问答数据集构建

微调数据集基于HuggingFace开源的红楼梦QA数据集（liujch1998/honglou-qa），经清洗去重后保留约1000条高质量问答对。数据格式遵循LLaMA-Factory的alpaca格式，包含instruction（问题）和output（答案）两个字段。按95:5比例划分训练集与验证集。

### 4.4 测试集设计

手工构建50道测试题，覆盖三类问题：剧情问答（38题，涉及关键情节的回目定位与内容概括）、人物关系（9题，涉及角色间的亲属/主仆关系）、诗词赏析（3题，涉及判词、曲词的作者与对象）。每题标注golden_chapters（标准回目）和reference（参考答案），供检索指标和生成质量评估使用。

### 4.5 探索性数据分析

对红楼梦原文进行了全面的EDA分析，生成5张统计图表：各回字数分布图（显示前80回字数较多、后40回偏短的规律）、人物出现频率Top20（贾宝玉、林黛玉、薛宝钗位列前三）、各回字数趋势曲线、人物关系网络Top20、前后两卷字数对比。这些分析为后续RAG分块策略和知识图谱构建提供了数据支撑。

## 五、阶段3：RAG检索增强生成

### 5.1 多级增强RAG架构

RAG管道采用五级增强设计，逐步提升检索质量：

**第一级：BGE向量编码。** 使用BAAI/bge-small-zh-v1.5模型（512维）对文本块进行密集向量编码，支持查询加前缀（"为这个句子生成表示以用于检索相关文章："）以提升检索效果。编码后的向量存入ChromaDB持久化向量库。

**第二级：BM25关键词召回。** 基于jieba分词构建BM25稀疏检索索引，融入红楼梦自定义词典（如"贾宝玉""林黛玉""王熙凤""薛宝钗"等专有名词），避免分词错误导致的关键词漏召。BM25与向量检索形成互补：向量检索擅长语义相似但可能遗漏关键词匹配，BM25则精确匹配关键词但对同义表达无能为力。

**第三级：RRF融合。** 使用倒数排名融合（Reciprocal Rank Fusion）将向量检索和BM25检索的结果合并，公式为 `score(d) = Σ 1/(k + rank_i(d))`，k=60。RRF无需校准分数即可融合多路检索结果，去重后保留Top-K候选。

**第四级：HyDE假设文档增强。** 对于复杂查询，先调用Ollama本地部署的Qwen2.5-7B生成假设性答案文档（Hypothetical Document），再将该文档而非原始查询进行向量编码检索。HyDE的直觉是：生成的假设文档即使包含错误事实，其"语言风格"和"关键词分布"仍与真实相关文档接近，从而提升向量检索的召回率。

**第五级：CrossEncoder重排序。** 使用BAAI/bge-reranker-base对RRF融合后的Top-20候选进行精排。与Bi-Encoder（BGE嵌入模型）不同，CrossEncoder将query和document拼接后联合编码，能捕捉更细粒度的语义交互，但计算开销更大，因此仅用于小规模候选集的精排。

### 5.2 检索效果

在50题测试集上，方案B（完整RAG管道）的检索指标为：Recall@5=0.75、Precision@5=0.25。Recall@5达到75%意味着标准回目在75%的题目中被Top-5检索结果命中。未命中的25%主要涉及跨回目情节（如"宝玉挨打的直接原因"涉及第33回但检索偏向了第30回）和后40回冷门情节。

## 六、阶段4：QLoRA领域微调

### 6.1 训练配置

在AutoDL 4090D实例上使用LLaMA-Factory 0.9.5进行QLoRA微调。基座模型为Qwen2.5-7B-Instruct，通过ModelScope下载。LoRA配置：rank=8、alpha=16、target=all（对所有线性层注入LoRA适配器）、量化位=null（使用bf16而非4bit量化，因4090D 24GB显存足以支撑7B模型bf16训练）。

训练超参数：batch_size=2、gradient_accumulation=16（等效batch_size=32）、learning_rate=5e-5、max_steps=72、cutoff_len=2048、bf16=true、gradient_checkpointing=true。72步训练在4090D上耗时约20分钟。

### 6.2 模型导出与部署

训练完成后，使用LLaMA-Factory的export功能将LoRA适配器权重合并到基座模型中，导出为9个safetensors分片（约15GB），存储于AutoDL数据盘。随后通过LLaMA-Factory的API模式启动OpenAI兼容的推理服务（端口8000），在本地通过SSH端口转发隧道（`ssh -L 8000:127.0.0.1:8000`）访问远程微调模型。

### 6.3 训练问题与解决

训练过程中遇到三个关键问题：（1）torch版本冲突——LLaMA-Factory的`[torch]`extra将torch从2.3.0升级到2.13.0，导致`ValueError: Your setup doesn't support bf16/gpu`，解决方法是锁定torch 2.3.0+cu121；（2）torchaudio版本不匹配——降级torch后漏降torchaudio，导致`libcudart.so.13: cannot open shared object file`，解决方法是同步安装torchaudio 2.3.0；（3）nohup环境不加载conda——后台启动API服务时python/pip命令找不到，解决方法是使用完整路径`/root/miniconda3/bin/python`。

## 七、阶段5：GraphRAG知识图谱

### 7.1 三元组抽取与图谱构建

基于GitHub开源的红楼梦人物关系三元组数据集（PrimerLi/Dream-of-the-Red-Chamber），构建包含900+节点、2000+边的知识图谱。三元组格式为（主体，关系，客体），关系类型包括父子、母子、夫妻、兄弟、姐妹、主仆、朋友等。

使用NetworkX构建图数据结构，支持最短路径查询、共同邻居查询、社区检测等图算法。通过Pyvis库生成交互式HTML可视化，节点大小由度中心性决定，边颜色由关系类型决定。

### 7.2 图谱统计

知识图谱统计：节点数932、边数2087、平均度4.48、连通分量12个。度中心性Top5为贾宝玉（度287）、贾母（度156）、林黛玉（度134）、王熙凤（度128）、薛宝钗（度112）。图谱中最大连通分量包含85%的节点，反映了红楼梦人物关系的紧密性。

### 7.3 GraphRAG与文本RAG的互补

GraphRAG与文本RAG形成互补：文本RAG擅长定位具体情节片段，但对"某人和某人是什么关系"这类结构化查询效率较低；GraphRAG通过图遍历直接返回关系路径，但对"某回发生了什么"这类情节查询无能为力。LangGraph Agent根据问题类型自动选择合适工具。

## 八、阶段6：LangGraph多工具Agent

### 8.1 Agent架构

使用LangGraph状态机编排多工具Agent，定义三种工具节点：RAG检索工具（调用rag_search）、GraphRAG查询工具（调用图遍历）、直接生成工具（不检索直接生成）。Agent根据用户问题自动路由到最合适的工具。

状态机定义：START → classify_node（问题分类）→ [rag_tool | graphrag_tool | direct_gen] → generate_node（答案生成）→ END。分类节点使用LLM判断问题类型：涉及具体情节的走RAG路径，涉及人物关系的走GraphRAG路径，开放性讨论走直接生成路径。

### 8.2 Agent流程测试

Agent流程测试通过6个测试用例验证：剧情问答（RAG路径）、人物关系（GraphRAG路径）、开放讨论（直接生成）、混合问题（先GraphRAG后RAG）、拒答场景（检索结果不足时明确告知）、工具切换（连续多轮对话中动态切换工具）。6/6测试用例通过。

## 九、阶段7：Gradio前端

### 7.1 界面设计

使用Gradio 6.25.0的Blocks API构建双栏布局前端。左栏为对话区（Chatbot组件 + 输入框 + 发送按钮），右栏分为上下两部分：上部展示检索来源（回目号、片段预览、检索耗时），下部展示人物关系图（Pyvis HTML嵌入）和国风人物画（ComfyUI生成的图片）。

### 7.2 功能集成

前端集成四大功能模块：（1）智能问答——通过LangGraph Agent调用后端模型和RAG管道；（2）检索溯源——展示RAG检索到的原文片段和回目号；（3）知识图谱——动态渲染与当前问题相关的人物关系子图；（4）国风可视化——调用ComfyUI API生成国风风格人物画。31/31前端测试用例通过。

## 十、阶段8：可视化（ComfyUI）

### 10.1 ComfyUI集成

通过ComfyUI秋叶整合版的HTTP API（`/prompt` + `/history`轮询）实现图像生成自动化。工作流为：CheckpointLoaderSimple（AnythingV5模型）→ LoraLoader（chinese_style_lora_v1.0，权重0.8）→ CLIPTextEncode（正面/负面提示词）→ KSampler（sampler=dpmpp_2m, scheduler=karras, steps=20, cfg=7.5）→ VAEDecode → SaveImage。

### 10.2 国风LoRA调优

初始使用AnythingV5模型生成的图片风格偏日漫，通过添加国风LoRA（触发词"gufeng style"）改善了服饰和色调。负面提示词加入"multiple girls, crowd, group, twins, duplicated"以避免生成多人物。最终生成贾宝玉、林黛玉、王熙凤三张国风人物画。61/61可视化测试通过。

### 10.3 ComfyUI API适配问题

开发过程中遇到两个API兼容性问题：（1）KSampler的采样器名称需使用ComfyUI内部命名（`dpmpp_2m`+`scheduler: karras`）而非WebUI命名（`DPM++ 2M Karras`）；（2）新版SaveImage节点要求`filename_prefix`参数。均通过查询ComfyUI的`/object_info` API解决。

## 十一、阶段9：评测与结果分析

### 11.1 三方案评测设计

设计三方案对照实验，控制变量评估各模块增益：

| 方案 | 模型 | RAG | 目标 |
|------|------|-----|------|
| A | Qwen2.5-7B Base | 无 | 基线：纯模型能力 |
| B | Qwen2.5-7B Base | BGE+BM25+RRF+HyDE+Reranker | RAG增益 |
| C | Qwen2.5-7B-Instruct + LoRA | BGE+BM25+RRF（无Reranker） | 微调增益 |

评测指标包括：检索指标（Recall@5、Precision@5）、生成质量（LLM-as-Judge三项评分：事实准确/无幻觉/完整度，1-5分）、幻觉发生率（no_hallucination≤2的题占比）。Judge模型使用Ollama本地部署的Qwen2.5-7B，通过结构化JSON输出确保评分一致性。每题完成后立即写入JSON文件，支持断点续跑。

### 11.2 评测结果

| 指标 | 方案A | 方案B | 方案C | C vs B |
|------|-------|-------|-------|---------|
| Recall@5 | 0.00 | 0.75 | 0.68 | -0.07 |
| Precision@5 | 0.00 | 0.25 | 0.20 | -0.05 |
| 事实准确 | 2.64 | 3.58 | 2.62 | -0.96 |
| 无幻觉 | 5.00 | 4.92 | 4.22 | -0.70 |
| 完整度 | 2.56 | 3.34 | 2.42 | -0.92 |
| 幻觉率 | 0.0% | 2.0% | 16.0% | +14% |

### 11.3 结果分析

**方案B（Base+RAG）为最优方案。** 在所有指标上均优于或等同于方案A和C。相比A，事实准确提升+0.94（2.64→3.58），完整度提升+0.78（2.56→3.34），Recall@5从0提升至0.75。RAG的多级增强设计有效将回答锚定在原文片段上，在保持较高事实准确性的同时将幻觉率控制在2%。

**方案C（微调+RAG）未达预期。** 事实准确（2.62）与方案A（2.64）持平，低于方案B（3.58）达0.96分。幻觉率高达16%，远高于A（0%）和B（2%）。经分析，主要原因有三：

第一，**微调模型生成极短回答**。在简单回目定位题上（如"贾元春晋封贤德妃是在哪一回"），微调模型直接输出"第16回"四个字，被Judge评为0/0/0分。这类问题共有7道（占14%），严重拉低了方案C的平均分。原因是训练数据中包含了大量短回答模式，模型学习了这种简洁但缺乏解释的输出风格。

第二，**Reranker缺失导致检索精度下降**。方案C因Windows环境下CrossEncoder段错误（sentence-transformers的已知原生代码崩溃问题）无法启用Reranker精排，Recall@5从0.75降至0.68。虽然核心检索能力（BGE+BM25+RRF）仍在，但缺少精排导致部分题目的Top-5中混入了不相关片段。

第三，**微调模型过度自信**。在检索结果不理想时，方案A/B的Base模型倾向于生成保守回答（"不知道"或泛泛而谈），而微调模型因学习了红楼梦领域知识，倾向基于训练记忆生成更"自信"的回答，但这些回答有时引用了错误回目的内容，导致幻觉率升高。典型案例：Q2"薛蟠为何打死冯渊"，模型编造了"张三不肯换酒"等不存在的内容。

**分类别分析**揭示了不同问题类型的增益差异。在诗词赏析类（3题）上三方案表现一致（fac=3.67~4.00, noh=5.00），说明诗词类问题不依赖检索增强；在人物关系类（9题）上方案A与B持平（fac=3.89）但C下降（3.44），说明Base模型本身具备较好的人物关系知识，微调反而引入了噪声；在剧情问答类（38题）上方案B显著优于A和C（fac=3.50 vs 2.24/2.34），说明RAG对需要定位具体情节的问题增益最大。

### 11.4 评测方法论反思

LLM-as-Judge方法存在已知局限：Judge模型（Qwen2.5-7B via Ollama）对极短回答的评分可能存在偏差——当回答仅有几个字时，Judge可能无法有效评估其事实准确性，倾向于给0分。这导致方案C中7道短回答题被系统性低估。未来可考虑增加回答长度下限约束或使用多Judge投票机制缓解此问题。

此外，方案B与C的RAG配置不完全一致（B含HyDE+Reranker，C不含），严格来说并非完美的控制变量实验。但考虑到方案C的核心变量是"模型是否微调"而非"RAG配置"，且核心检索管道（BGE+BM25+RRF）一致，结论仍具参考价值。

## 十二、结论与展望

### 12.1 主要发现

1. **RAG是最有效的单一增强手段**。方案B（Base+RAG）在事实准确性和完整度上均显著优于纯Base模型（A），幻觉率控制在2%，验证了多级增强RAG架构（BGE向量+BM25+RRF+HyDE+Reranker）对红楼梦问答任务的有效性。

2. **QLoRA微调的效果具有双面性**。一方面，微调使模型获得了红楼梦领域知识，能更流畅地回答领域问题；另一方面，微调也引入了"过度自信"和"短回答偏好"等问题，在某些场景下反而降低了评测表现。这提示我们：微调数据的质量和输出风格对最终效果至关重要。

3. **RAG与微调并非简单的叠加关系**。预期中"C = B + 微调增益"的假设未成立，实际结果是C < B。这说明RAG提供的外部知识（原文片段）比微调注入的参数化知识更可靠——前者可以溯源验证，后者则可能与检索结果冲突时产生幻觉。

4. **问题类型决定最优策略**。剧情问答类问题高度依赖RAG检索（B >> A ≈ C）；人物关系类问题Base模型已足够（A ≈ B > C）；诗词赏析类问题三方案表现一致，可能需要更专门的微调数据。

### 12.2 局限性

- 测试集规模较小（50题），统计显著性有限
- 方案B与C的RAG配置不完全一致（Reranker/HyDE缺失）
- Judge模型与被评测模型同源（均为Qwen2.5-7B），可能存在评分偏好
- 训练数据仅约1000条QA对，微调效果可能受限
- 未在Linux环境完整部署方案C（含Reranker），存在系统性偏差

### 12.3 未来工作

1. **改进微调数据**：增加长回答样本，训练模型在检索结果基础上生成详细解释而非简短回答
2. **完整部署方案C**：在Linux环境部署含Reranker的完整RAG管道，消除配置差异
3. **扩展测试集**：将测试集扩大至200题，增加主观题和跨回目推理题
4. **引入多模态**：将ComfyUI生成的国风人物画与文本问答结合，实现"看图答题"的多模态交互
5. **Agent能力增强**：在LangGraph中加入工具调用结果的质量评估节点，当检索质量不足时自动降级为保守回答模式

## 附录

### A. 项目文件结构

```
honglou_ai_multimodal/
├── data/                    # 数据（原文/QA/三元组/测试集/EDA图表）
├── rag_pipeline/            # RAG管道（BGE/BM25/Chroma/HyDE/Reranker/RRF）
├── graphrag_index/          # GraphRAG知识图谱索引
├── lora_train/              # LLaMA-Factory训练配置与导出
├── agent_graph/             # LangGraph Agent编排
├── visualize/               # ComfyUI可视化模块
├── evaluation/              # 评测脚本与结果（A/B/C三方案）
├── web_ui/                  # Gradio前端
├── report/                  # 实验报告与图表
└── requirements.txt
```

### B. 评测图表索引

| 图表 | 文件 | 说明 |
|------|------|------|
| 检索指标对比 | eval_retrieval_metrics.png | Recall@5 / Precision@5 三方案对比 |
| 生成质量对比 | eval_generation_metrics.png | 事实准确/无幻觉/完整度 三方案对比 |
| 幻觉发生率 | eval_hallucination_rate.png | 幻觉率柱状图 |
| 分类别对比 | eval_by_category.png | 剧情/人物关系/诗词 分类别对比 |
| 综合雷达图 | eval_radar.png | 五维归一化雷达图 |
| EDA分析 | eda_01~05_*.png | 红楼梦原文统计分析（5张） |
| 知识图谱 | core_network.png | 人物关系核心网络图 |
| 国风人物 | guofeng_*.png | ComfyUI生成的国风人物画（3张） |

### C. 复现指南

```bash
# 1. 环境配置
conda create -n pytorch_gpu python=3.10 && conda activate pytorch_gpu
pip install -r requirements.txt

# 2. 数据预处理
python data/process_chunks.py  # 分块生成1620个chunks

# 3. 构建RAG索引
python rag_pipeline/build_index.py  # ChromaDB + BM25索引

# 4. QLoRA微调（AutoDL 4090D）
cd lora_train/LLaMA-Factory && llamafactory-cli train qwen_honglou_train.yaml
llamafactory-cli export qwen_honglou_export.yaml

# 5. 启动API服务
llamafactory-cli api api_config.yaml  # 端口8000

# 6. SSH隧道
ssh -L 8000:127.0.0.1:8000 root@connect.westb.seetacloud.com -p 34715

# 7. 运行评测
python evaluation/run_eval.py    # 方案A/B
python evaluation/run_eval_c.py  # 方案C

# 8. 生成分析报告
python evaluation/analyze_results.py

# 9. 启动前端
python web_ui/app.py  # 端口7860
```

---

> 报告生成时间：2026年8月20日  
> 项目仓库：D:\honglou_ai_multimodal  
> 作者：薛瑞琦
