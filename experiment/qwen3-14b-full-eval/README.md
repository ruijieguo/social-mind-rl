# Qwen3-14B Full Eval (base × v3.5 × v3.1)

完整评测 Qwen3-14B 的 3 个模型 × 4 benchmark × 3 协议 = 36 格。
参数口径沿用 `experiment/qwen3-8b-full-eval` 的 v2 报告（`max_tokens` 除外，统一设为 **8192**）。

## 模型

| key | 说明 | HF 路径 (host 181) |
|---|---|---|
| `base` | 原始 Qwen3-14B（ModelScope 下载） | `/data_nvme/grj-projects/models/Qwen3-14B` |
| `v35`  | v3.5 = Stage 19 ckpt-120 | `/data_nvme/grj-projects/tom-output/qwen3-14B-tombench-rlvr-stage19-1x8-hf-ckpt120` |
| `v31`  | v3.1 = Stage 14b ckpt-199 | `/data_nvme/grj-projects/tom-output/qwen3-14B-tombench-rlvr-stage14b-1x8-hf-ckpt199` |

## Benchmark

| bench | n | 选项 | 语言 | system prompt |
|---|---|---|---|---|
| ToMBench  | 5718 | 4 (A-D) | zh+en | ToM 专用 |
| EmoBench  | 1200 | 4 (A-D) | en | 通用 MCQ |
| SocialIQA | 1954 | 3 (A-C) | en | 通用 MCQ |
| Hi-ToM    | 600  | 15 (A-O)| en | 通用 MCQ |

## 协议

| protocol | T | top_p | max_tokens | thinking |
|---|---|---|---|---|
| direct        | 0.0 | 1.0  | 64   | false |
| direct_think  | 0.0 | 1.0  | 8192 | true  |
| cot           | 0.6 | 0.95 | 8192 | true  |

## 服务器资源

`h800@172.16.120.181` (h800-3)。**GPUs 0-3 被他人作业占用**，本评测仅用 GPUs 4-7：

| GPU | Port | Container |
|---|---|---|
| 4 | 8001 | qwen3-14b-eval-0 |
| 5 | 8002 | qwen3-14b-eval-1 |
| 6 | 8003 | qwen3-14b-eval-2 |
| 7 | 8004 | qwen3-14b-eval-3 |

每次只服务一个模型（4 端点 TP=1），跑完它的 4 个 benchmark 再换下一个模型。

## 目录结构

```
experiment/qwen3-14b-full-eval/
├── README.md
├── configs/deploy.env.181          # host 参数 + 模型/数据路径
├── docker/docker-compose.yml       # 4× TP=1 vLLM on GPU 4-7（MODEL_PATH 参数化）
├── scripts/
│   ├── 01_serve_up.sh <base|v35|v31>
│   ├── 02_wait_ready.sh
│   ├── 04_run_eval.sh              # 全量驱动（per-model: up→eval→down）
│   ├── 05_aggregate_report.py      # 生成 markdown 报告（含截断分析）
│   ├── 06_serve_down.sh
│   ├── parallel_eval.py            # 评测引擎（记录 finish_reason → 截断检测）
│   └── prompts.py                  # prompt 模板 + 抽取器（4/3/15 选项自适应）
├── output/{tombench,emobench,socialiqa,hitom}/{base,v35,v31}.json
└── logs/run_{model}_{bench}.log
```

## 运行步骤

```bash
# 1. 同步到 host（在 mac 上）
rsync -avh --exclude='output/cache/' --exclude='output/*/' \
  experiment/qwen3-14b-full-eval/ \
  h800@172.16.120.181:/data_nvme/grj-projects/qwen3-tom/experiment/qwen3-14b-full-eval/
# 数据也要同步（gitignored）
rsync -avh data/tom/tombench_eval.jsonl data/eval/{hitom,socialiqa,emobench}_eval.jsonl \
  h800@172.16.120.181:/data_nvme/grj-projects/qwen3-tom/data/...

# 2. 下载 base（ModelScope）
modelscope download --model Qwen/Qwen3-14B --local_dir /data_nvme/grj-projects/models/Qwen3-14B

# 3. 冒烟测试
ssh h800@172.16.120.181
cd /data_nvme/grj-projects/qwen3-tom/experiment/qwen3-14b-full-eval
LIMIT=10 bash scripts/04_run_eval.sh

# 4. 全量（后台）
nohup bash scripts/04_run_eval.sh > logs/full_run.log 2>&1 &

# 5. 生成报告
python3 scripts/05_aggregate_report.py --results-dir output

# 6. 确认 GPU 释放
docker ps | grep qwen3-14b-eval   # 应为空
```

## 与 8B 评测的差异

- 模型从 8B 换成 14B；3 个模型全部本地 vLLM（无 DashScope API 路）。
- benchmark 从 2 个（ToMBench/Hi-ToM）扩到 4 个（+SocialIQA +EmoBench）。
- 协议去掉 del_tom，保留 direct/direct_think/cot；`max_tokens` 统一 8192（direct 除外 64）。
- 评测引擎新增 **逐条 `finish_reason` 记录**，报告含专门的截断检测表。
- 仅用 GPUs 4-7（0-3 为他人作业），TP=1，4 端点。
