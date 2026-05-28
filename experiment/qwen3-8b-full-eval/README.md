# Qwen3-8B Full Eval (base × v1.0 × DashScope)

完整评测 Qwen3-8B 3 种部署方式 × 2 benchmark × 4 协议 = 24 格。

## 目标

- **base**：本地部署 `/model/Qwen3-8B`（HF 格式）
- **v1.0**：本地部署 RLVR 训练后 ckpt（Stage 15 ckpt-150）
- **API**：阿里云 DashScope 的 `qwen3-8b` 公网服务
- **Benchmark**：ToMBench (5718)、Hi-ToM (600)
- **协议** (4 个):
  - **direct** (no-think): T=0, top_p=1, max_tokens=64, **enable_thinking=false**, 1 sample
  - **direct_think** (default-think): T=0, top_p=1, max_tokens=2048, **enable_thinking=true** (与历史 production_frozen 0.7450 口径对齐)
  - **cot**: T=0.6, top_p=0.95, max_tokens=4096, enable_thinking=true, 1 sample
  - **del_tom**: T=0.7, top_p=0.95, max_tokens=4096, enable_thinking=true, 8 samples 多数投票

## 服务器资源

`h800@172.16.120.191` (hostname `h800-1`)：

| GPU | Model | Port | Container |
|---|---|---|---|
| 0 | 8B base | 8001 | qwen3-eval-base-0 |
| 1 | 8B base | 8002 | qwen3-eval-base-1 |
| 2 | 8B base | 8003 | qwen3-eval-base-2 |
| 3 | 8B base | 8004 | qwen3-eval-base-3 |
| 4 | 8B v1.0 | 8005 | qwen3-eval-v10-0 |
| 5 | 8B v1.0 | 8006 | qwen3-eval-v10-1 |
| 6 | 8B v1.0 | 8007 | qwen3-eval-v10-2 |
| 7 | 8B v1.0 | 8008 | qwen3-eval-v10-3 |

vLLM image: `vllm/vllm-openai:v0.11.0`，TP=1，max_model_len=8192，gpu_util=0.85。

## 目录结构

```
experiment/qwen3-8b-full-eval/
├── README.md                          ← 本文件
├── configs/
│   └── deploy.env.191                 # 服务器参数
├── docker/
│   └── docker-compose.yml             # 8 vLLM 实例
├── scripts/
│   ├── 01_serve_up.sh                 # 起 8 个容器
│   ├── 02_wait_ready.sh               # poll /v1/models 直到全部就绪
│   ├── 04_run_eval.sh                 # 跑全部 18 格评测
│   ├── 05_aggregate_report.py         # 生成 markdown 报告
│   ├── 06_serve_down.sh               # 停容器
│   ├── parallel_eval.py               # 评测引擎（多端点 round-robin）
│   └── prompts.py                     # prompt 模板 + 答案抽取器
├── output/
│   ├── tombench/{base,v10,dashscope}.json
│   ├── hitom/{base,v10,dashscope}.json
│   ├── cache/                         # 单题原始响应缓存
│   └── full_eval_report_qwen3-8b_<DATE>.md
└── logs/
    └── run_{model}_{bench}.log
```

## 运行步骤

1. **同步到服务器**：

   ```bash
   # 在 mac 上
   rsync -avh --exclude='output/cache/' --exclude='output/*.json' \
     experiment/qwen3-8b-full-eval/ \
     h800@172.16.120.191:/home/h800/grj-projects/qwen3-tom/experiment/qwen3-8b-full-eval/
   ```

2. **SSH 进服务器 + 起服务**：

   ```bash
   ssh h800@172.16.120.191
   cd /home/h800/grj-projects/qwen3-tom/experiment/qwen3-8b-full-eval
   bash scripts/01_serve_up.sh
   bash scripts/02_wait_ready.sh   # 约 60-90s
   ```

3. **导出 DashScope key 并跑评测**：

   ```bash
   export DASHSCOPE_API_KEY="sk-..."
   export DASHSCOPE_BASE_URL="https://dashscope.aliyuncs.com/compatible-mode/v1"
   bash scripts/04_run_eval.sh        # 1-1.5 小时
   ```

4. **生成报告**：

   ```bash
   python3 scripts/05_aggregate_report.py --results-dir output
   ```

5. **清理**：

   ```bash
   bash scripts/06_serve_down.sh
   ```

## 冒烟测试

跑 10 题端到端确认链路：

```bash
LIMIT=10 PROTOCOLS=direct bash scripts/04_run_eval.sh
```

## 与历史报告差异

- 本次评测的 cot 温度由历史 v3.4 的 T=0.0 改为 **T=0.6**（用户 2026-05-28 指定）
- direct 温度仍 T=0.0；del_tom 仍 T=0.7
- 8B base 的 ToMBench cot 历史只有 3675/5718 partial，本次将跑满 5718
- 新增第 3 路：DashScope API qwen3-8b（历史报告未有此格）
- 新增 **direct_think** 协议：与历史 production_frozen 8B v1.0 0.7450 评测口径对齐，验证模型在 default thinking 模式下的表现

## 主结果摘要 (2026-05-28)

ToMBench (5718)：

| Model | direct (no-think) | direct_think | cot | del_tom |
|---|---|---|---|---|
| 8B base | 0.7029 | 0.7030 | 0.7387 | — |
| 8B v1.0 | 0.7128 | **0.7462** ⭐ | 0.7559 | **0.7646** |
| DashScope | 0.7020 | 0.7011 | 0.7501 | — |

Hi-ToM (600)：

| Model | direct | direct_think | cot |
|---|---|---|---|
| 8B base | 0.5550 | 0.5150 | 0.6100 |
| 8B v1.0 | 0.5717 | 0.6267 | **0.6883** |
| DashScope | 0.5467 | 0.5517 | 0.6833 |

**关键发现**：
- v10/tombench/direct_think = 0.7462 与历史 production_frozen 0.7450 完美复现（差距 0.12pp）
- 8B v10 强烈依赖 thinking（关 thinking 退化 3.3pp on tombench, 5.5pp on hi-tom）；base 模型不依赖
- DashScope qwen3-8b 行为接近 base，怀疑不是 RLVR 后的版本

详见 `output/full_eval_report_qwen3-8b_2026-05-28.md`。
