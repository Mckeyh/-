# 农田害虫识别与防治助手（Farmland Pest Recognition & Control Assistant）

基于 **FastAPI + 微信小程序** 的「害虫图像识别 + 在线微调训练 + 防治知识问答」一体化系统。
识别模型与语言模型**全部本地部署**（不依赖外部 API），适合内网/离线场景。

> 数据集：**IP102**（102 类农田害虫）｜知识库：农药手册 / 防治指南 / 天敌资料

---

## 一、功能模块

| 模块 | 说明 |
|---|---|
| **害虫识别** | 手机拍照 → 上传 → ResNet50（微调后）返回 Top-1 / Top-5 害虫种类与置信度 |
| **在线微调训练** | 按类别上传样本 → 后台线程微调 → WebSocket 实时推送训练进度、可中途停止；大规模数据集用命令行 `train_cli.py` |
| **知识库管理** | 上传 PDF / DOCX / TXT（农药手册、防治指南等）→ 解析分块 → 本地向量化 → 存入 ChromaDB |
| **防治问答（RAG）** | 提问 → 检索知识库 → 本地 Qwen2.5-1.5B 生成 → SSE 流式返回；回答围绕 **「怎么防治 / 用什么药 / 天敌是什么」** 三方面展开 |

## 二、技术栈

**后端**：Python 3.11 · FastAPI 0.104.1 · Uvicorn 0.24 · Pydantic 2.5 · SQLAlchemy 2.0

**图像模型**：PyTorch 2.1.0+cu121 · torchvision 0.16 · ResNet50 迁移学习（替换为自定义**余弦分类头** `ConsineClassifier`，微调只更新分类头，类别数可动态扩展）

**RAG / LLM**：ChromaDB 0.4.22 · langchain-chroma 0.1.4 · `BAAI/bge-small-zh-v1.5`（本地**中文**句向量，替换掉原先的英文模型 all-MiniLM-L6-v2）· llama-cpp-python 0.2.90（CUDA 版）· Qwen2.5-1.5B-Instruct GGUF（q4_k_m，本地推理，GPU 卸载）

**前端**：微信小程序（首页 / 训练 / 害虫识别 / 知识库 / 智能问答，5 个页面 + WebSocket）

## 三、后端接口

| 方法 | 路径 | 说明 |
|---|---|---|
| POST | `/api/classify/` | 害虫识别（JSON `{"image": "data:image/jpeg;base64,..."}`） |
| POST | `/api/train/finetune` | 上传样本图片 + 类别名，启动微调（JSON `{images, label, clear_old}`） |
| GET | `/api/train/status` | 查询训练状态（`idle/running/done/failed`） |
| POST | `/api/train/stop` | 停止训练 |
| WS | `/api/train/ws` | 训练状态实时推送 |
| POST | `/api/knowledge/upload` | 文档上传并入库（multipart，字段名 `file`） |
| POST | `/api/chat/stream` | 防治问答（SSE 流式：`data: {chunk, source, done}`） |

## 四、目录结构

```
智慧农业/
├── backend/
│   ├── app/
│   │   ├── main.py              # FastAPI 入口（路由挂载 / 启动事件）
│   │   ├── train_cli.py         # 命令行微调训练入口（大数据集用）
│   │   ├── config/config.py     # 全局配置（设备检测、路径、超参）
│   │   ├── routers/             # 接口层：classify(识别) / train(训练) / knowledge(知识库) / chat(问答)
│   │   └── services/            # 业务层：classify_service / train_service / rag_service / llm_service
│   ├── utils/common_utils.py    # 日志
│   └── data/                    # 模型、向量库、数据集、上传文件（不进版本库）
├── knowledge/                   # 知识库资料（防治技术方案 + 自编害虫防治知识库，可入库检索）
├── miniprogram/                 # 微信小程序前端
└── pyproject.toml / uv.lock     # uv 依赖管理
```

## 五、知识库资料（防治问答的"知识"来源）

| 目录 | 内容 |
|---|---|
| `knowledge/农田害虫防治知识库.txt` | 自编的害虫防治知识库：按水稻害虫 / 地下害虫与地老虎 / 旱作与小麦害虫分类，每类都写清 **危害特征、怎么防、用什么药（有效成分）、天敌是什么** |
| `knowledge/防治资料/*.txt` | **14 份政府公开技术文件**（全国农技推广服务中心、农业农村部、陕西省农业农村厅）：2026 年水稻/小麦/玉米重大病虫害防控技术方案、科学安全用药指导意见、地下害虫防治技术、蔬菜害虫绿色防控方案等，详见 `knowledge/防治资料/资料来源说明.md` |

入库后共 **125 个检索段**；回答优先引用这些官方方案的防控指标与推荐药剂，
并在提示词里禁止编造知识库没有的剂量与商品名。

## 六、数据集准备（IP102）

把害虫图片按**类别子目录**放好，子目录名就是识别结果里的类别名（建议用中文，页面直接显示中文）：

```
backend/data/upload/dataset/unzipped/
├── 二化螟/        *.jpg
├── 稻纵卷叶螟/    *.jpg
├── 粘虫/          *.jpg
└── ...（每类一个目录）
```

**本项目实际使用的数据（水稻 + 小麦 + 地下害虫，共 36 类 / 6685 张）**：

| 项 | 说明 |
|---|---|
| 数据集 | IP102（官方：75222 张 / 102 类），本项目取其中 **类别号 0~35** 共 36 类 |
| 下载源 | HuggingFace `EnmmmmOvO/insect-pest-dataset`（train 45095 / val 7508 / test 22619，与官方 IP102 划分完全一致，已验证） |
| 类别名 | 按官方 `classes.txt` 顺序映射为中文（稻纵卷叶螟、二化螟、褐飞虱、蝼蛄、玉米螟、粘虫、蚜虫……），个别类别保留拉丁名 |
| 采样 | 每类上限 200 张（`train-00000/00001` 两个分片足够覆盖 0~35 类，因为分片按类别顺序切分） |
| 训练 | `uv run python train_cli.py --epochs 10`（8GB 显存约 2 分钟/epoch） |

> ⚠️ **标签编号坑**：该 parquet 的 `label` 是 **0 起**（0~101），而官方 `classes.txt` 是 **1 起**（1~102）。
> 映射时若用 `label-1` 会整体错位一类，正确做法是 `中文名[label]`。
> 判定方法：前两个分片的 label 取值范围为 0~51，且同一 label 的图片文件名严格递增
> （00002.jpg → 01115.jpg → 01604.jpg …），说明 label 按原始打包顺序排列、从 0 开始，
> 而不是 ImageFolder 那种「目录名字符串排序」。

**实测结果（本机 RTX 8GB，36 类 / 6685 张，训练集 5348、验证集 1337，随机种子 42）**：

| 训练配置 | Top-1 | Top-5 |
|---|---|---|
| `--epochs 10`（只训分类头，主干冻结） | 32.76% | 65.45% |
| `--epochs 6 --unfreeze`（解冻 layer4） | 32.9% | — |
| **`--epochs 8 --unfreeze-all`（全主干微调，fp32）** | **38.07%** | **73.15%** |

全主干微调（8 epoch）把 Top-1 从 32.8% 提到 **38.1%**、Top-5 提到 **73.2%**；
第 8 轮 train_acc(38.7%) 仍在缓慢上升，说明还没训透，加 epoch / 加数据还能再涨
（瓶颈是近缘种：三种地老虎、三种飞虱、多种蚜虫仅凭一张照片极难区分，且每类只有 200 张）。

**微调踩过的坑（很值得写进答辩：调参不是玄学，是定位问题）**：

1. **fp16 AMP 在本机不可用**：GradScaler 的 scale 从默认 65536 一路跌到 128，
   说明梯度频繁溢出、大量 step 被跳过 —— 表现为 train_acc 长期卡在 25% 甚至从 34% 一路崩到 12%（发散）。
   因此 `fintune(use_amp=...)` 默认关闭（`--amp` 可显式开启）。
2. **BN 必须冻结统计量**：小 batch(16) + 强增广下让 BatchNorm 继续更新 `running_mean/var`，
   会把预训练特征分布带偏；`fintune` 里每轮 `model.train()` 后把 BN 模块单独切回 `eval()`。
3. **BN 的 weight/bias 不能加 weight decay**：优化器按「需正则化 / 不需正则化」分两组
   （后者含 BN 参数与所有 bias），这是解冻主干后准确率反而下降的常见原因。
4. 主干学习率取分类头的 1/10（5e-5），既让主干适配害虫细粒度特征，又不冲掉 ImageNet 预训练表示。

IP102 类别分布极不均衡（部分类别不足百张，最大类别上千张），
这也是可以展开讲的技术点：本项目在采样时按上限截断，训练时用 `WeightedRandomSampler` 做类别加权。

训练方式二选一：

```bash
cd backend/app
uv run python train_cli.py --dry-run          # 先统计数据集（不训练）
uv run python train_cli.py --epochs 10        # 只训分类头（快，适合小样本增量）
uv run python train_cli.py --epochs 6 --unfreeze       # 解冻 layer4 一起微调
uv run python train_cli.py --epochs 8 --unfreeze-all   # 全主干微调（当前最佳：Top-1 38.1%/Top-5 73.2%）
uv run python train_cli.py --epochs 8 --unfreeze-all --amp   # 加 fp16 混合精度（本机实测会发散，慎用）
```
或在小程序「训练」页上传样本（只适合每类几十张的小样本演示）。

## 七、环境搭建

> 依赖已用 [uv](https://docs.astral.sh/uv/) 管理。**注意：Python 必须 3.11**（`numpy==1.24.3` 不支持 3.12+）。

```bash
# 1. 安装 uv（Windows）
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"

# 2. 创建环境并安装依赖（需 GPU 版 torch，见下方说明）
uv sync

# 3. 启动后端（工作目录必须是 backend/app）
cd backend/app
uv run uvicorn main:app --host 0.0.0.0 --port 8000
```

小程序端：打开 `miniprogram/`，修改 `app.js` 中的 `apiBaseUrl`
（开发者工具模拟器用 `http://localhost:8000`；真机调试改为电脑局域网 IP，且需勾选"不校验合法域名"）。

### 大文件说明（**未纳入版本库**）

以下内容体积过大（GitHub 单文件上限 100MB），需自行下载后放到指定位置：

| 内容 | 放置位置 | 来源 |
|---|---|---|
| GPU 版 torch 2.1.0+cu121 / torchvision / torchaudio | uv 从 `wheels/` 安装 | [阿里云 pytorch-wheels 镜像](https://mirrors.aliyun.com/pytorch-wheels/cu121/) |
| llama-cpp-python 0.2.90（**cu121 版**，CPU 版无法用 GPU） | `wheels/` | abetlen GitHub Releases：tag `v0.2.90-cu121` |
| docx2txt 0.8（PyPI 仅源码包） | `wheels/` | 见 `pyproject.toml` 的 `find-links` |
| Qwen2.5-1.5B-Instruct GGUF | `backend/data/models/gguf/LoveSeaW/Qwen2.5-1.5b-instruct-gguf/` | HuggingFace（文件名填到 `config.py` 的 `LLM_MODEL_FILE`） |
| ResNet50 预训练权重 | `backend/data/models/safetensors/microsoft/resnet-50/` | HuggingFace `microsoft/resnet-50` |
| 句向量模型 `BAAI/bge-small-zh-v1.5`（**当前使用**，中文检索） | `backend/data/models/safetensors/BAAI/bge-small-zh-v1.5/` | HuggingFace（可从 `hf-mirror.com` 拉取，仅 95MB） |
| 句向量模型 all-MiniLM-L6-v2（旧，英文，已不用） | `backend/data/models/safetensors/sentence-transformers/all-MiniLM-L6-v2/` | HuggingFace |
| **IP102 害虫数据集** | `backend/data/upload/dataset/unzipped/<类别名>/` | IP102 官方 / 公开镜像 |
| 防治知识文档（农药手册等） | 通过 `/api/knowledge/upload` 上传 | 自备 |

> ⚠️ **`llama-cpp-python` 的 CUDA 版必须先 `import torch` 再 `import llama_cpp`**（CUDA 运行库由 torch 提供），
> 代码中 `services/llm_service.py` 已保证该顺序，请勿调整。

## 八、实现要点

1. **迁移学习方案**：ResNet50 主干冻结，替换为自定义余弦分类头（`ConsineClassifier`），微调只更新分类头；类别数变化时自动扩展分类头并保留已有类别的权重。
2. **训练任务异步化**：训练跑在后台线程，接口立即返回；状态机 + WebSocket 广播（跨线程用 `asyncio.run_coroutine_threadsafe` 投递回主事件循环）。
3. **RAG 上下文治理**：检索结果按内容去重后再拼接（避免重复段落导致模型输出循环），并限制上下文长度；
   分块大小取 **350 字**（句向量模型 bge-small-zh 最大输入 512 token，块过大被截断后各块会互相区分不开、检索跑偏）。
4. **植保问答约束**：系统提示词把模型设定为植保专家，回答强制覆盖「怎么防治 / 用什么药 / 天敌是什么」，并要求提示安全用药（剂量、安全间隔期、轮换用药）。
5. **本地推理**：Qwen2.5-1.5B（q4_k_m）+ GPU 卸载，实测约 45 tokens/s；句向量模型同样本地加载。
6. **流式输出**：后端以 SSE 推送（`chunk` 增量 / `done` 结束 / `error` 错误），前端用 `wx.request` 的 `enableChunked` + `onChunkReceived` 解析。
7. **防止大模型编造农药**：系统提示词明确要求「知识库没给出的用量、稀释倍数、安全间隔期一律不要编造」，
   并禁止写商品名——实测不加该约束时模型会自行编出"吡虫啉（商品名：乐果）""每公顷不超过250克"等错误内容。
8. **混合检索（关键词 + 向量）**：提问里出现知识库小标题或识别模型类别名（36 个害虫名）时，
   先用该名称做一次精确检索并把结果置顶，再叠加向量语义检索结果。
   纯向量检索对「蛴螬/蝼蛄」这类具体虫名不敏感（曾出现问蝼蛄却检索到蚜虫段落），加关键词通道后命中稳定。
9. **识别置信度阈值**：36 类细粒度模型 softmax 很平缓（预测正确时 top-1 也常只有 0.05~0.12），
   阈值沿用 0.25 会让几乎所有照片都显示「未知类型」，故改为 0.05，并始终返回 Top-5 候选表达不确定性。

### 已修复的两个训练相关 Bug（`services/classify_service.py`）

1. **验证集尺寸错误**：`transforms.Resize(224)` 只缩放**短边**、保持长宽比，IP102 图片长宽比不一，
   拼 batch 时报 `stack expects each tensor to be equal size, but got [3,224,346] and [3,224,398]`。
   正确写法是 `transforms.Resize((224,224))`（传**元组**才是缩放到固定尺寸），且必须与推理预处理一致。
   > 注意：写成 `Resize(224, 224)` 也不行——第二个位置参数是 `interpolation`，会报 `KeyError: 224`。
2. **归一化均值笔误**：训练用的是 `mean=[0.485, 0.465, 0.406]`，而推理用的是 ImageNet 标准 `0.456`，
   训练/推理预处理不一致会掉点。已统一为 `[0.485, 0.456, 0.406]`。
