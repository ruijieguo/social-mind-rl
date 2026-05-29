# Qwen3-14B 全量评测报告 — base × v3.5 × v3.1 × deepseek-v4-pro

> **日期**: 2026-05-30

> **模型** (4):
>
> - **base** = 原始 Qwen3-14B（ModelScope 下载，本地 vLLM）
> - **v3.5** = Stage 19 ckpt-120（GPT-5.5 蒸馏改进版，本地 vLLM）
> - **v3.1** = Stage 14b ckpt-199（task-weighted 重采样，历史 ToMBench del_tom 最高，本地 vLLM）
> - **deepseek-v4-pro** = DeepSeek 官方 API（参照目标 X）
>
> **Benchmark** (4): ToMBench 5718 / EmoBench 1200 / SocialIQA 1954 / Hi-ToM 600
> **协议** (3):
> - **direct** (no-think): T=0, top_p=1, max_tokens=64, enable_thinking=false
> - **direct_think** (default-think): T=0, top_p=1, **max_tokens=8192**, enable_thinking=true
> - **cot**: T=0.6, top_p=0.95, **max_tokens=8192**, enable_thinking=true
>
> **部署**: `h800@172.16.120.181` (8×H800)，GPUs 4-7 各 1 个 vLLM 实例 (TP=1, max_model_len=16384, gpu_util=0.85)，4 端点 round-robin。GPUs 0-3 为他人作业，未占用。

---

## 0. 主结果表

| Benchmark | Protocol | Qwen3-14B base | v3.5 (Stage 19 ckpt-120) | v3.1 (Stage 14b ckpt-199) | deepseek-v4-pro (API) |
|---|---|---|---|---|---|
| ToMBench (n=5718) | direct (no-think) | 0.7338 | 0.7413 | 0.7454 | 0.0609 |
| ToMBench (n=5718) | direct (default-think) | 0.7597 | 0.7695 | 0.7742 | 0.8199 |
| ToMBench (n=5718) | cot | 0.7609 | 0.7705 | 0.7816 | 0.8286 |
| EmoBench (n=1200) | direct (no-think) | 0.6342 | 0.6483 | 0.6483 | 0.0250 |
| EmoBench (n=1200) | direct (default-think) | 0.6733 | 0.6525 | 0.6400 | 0.7717 |
| EmoBench (n=1200) | cot | 0.6875 | 0.6608 | 0.6425 | 0.7608 |
| SocialIQA (n=1954) | direct (no-think) | 0.7876 | 0.7876 | 0.7871 | 0.0005 |
| SocialIQA (n=1954) | direct (default-think) | 0.7922 | 0.7769 | 0.7789 | 0.8178 |
| SocialIQA (n=1954) | cot | 0.7979 | 0.7799 | 0.7886 | 0.8117 |
| Hi-ToM (n=600) | direct (no-think) | 0.5317 | 0.5883 | 0.5433 | 0.0000 |
| Hi-ToM (n=600) | direct (default-think) | 0.7450 | 0.6833 | 0.6550 | 0.7733 |
| Hi-ToM (n=600) | cot | 0.7950 | 0.7367 | 0.7033 | 0.7217 |


---

## 0.5 每模型每 benchmark 最优协议 + 均值

| Model | ToMBench (n=5718) | EmoBench (n=1200) | SocialIQA (n=1954) | Hi-ToM (n=600) | 均值 |
|---|---|---|---|---|---|
| Qwen3-14B base | 0.7609 (cot) | 0.6875 (cot) | 0.7979 (cot) | 0.7950 (cot) | 0.7603 |
| v3.5 (Stage 19 ckpt-120) | 0.7705 (cot) | 0.6608 (cot) | 0.7876 (dir) | 0.7367 (cot) | 0.7389 |
| v3.1 (Stage 14b ckpt-199) | 0.7816 (cot) | 0.6483 (dir) | 0.7886 (cot) | 0.7033 (cot) | 0.7305 |
| deepseek-v4-pro (API) | 0.8286 (cot) | 0.7717 (dir) | 0.8178 (dir) | 0.7733 (dir) | 0.7979 |


---

## 1. 截断检测 (finish_reason == "length")

评测期间逐条记录 vLLM 的 `finish_reason`。`length` = 输出撑满 max_tokens 被截断，thinking 协议下会导致 `\boxed{}` 丢失 → extractor fallback → 准确率虚低（8B v1 的已知坑）。

| Model | Benchmark | Protocol | truncated / total | 截断率 |
|---|---|---|---|---|
| Qwen3-14B base | tombench | direct | 0/5718 | 0.00% |
| Qwen3-14B base | tombench | direct_think | 77/5718 | 1.35% ⚠️ |
| Qwen3-14B base | tombench | cot | 0/5718 | 0.00% |
| Qwen3-14B base | emobench | direct | 0/1200 | 0.00% |
| Qwen3-14B base | emobench | direct_think | 9/1200 | 0.75% |
| Qwen3-14B base | emobench | cot | 0/1200 | 0.00% |
| Qwen3-14B base | socialiqa | direct | 0/1954 | 0.00% |
| Qwen3-14B base | socialiqa | direct_think | 0/1954 | 0.00% |
| Qwen3-14B base | socialiqa | cot | 0/1954 | 0.00% |
| Qwen3-14B base | hitom | direct | 0/600 | 0.00% |
| Qwen3-14B base | hitom | direct_think | 0/600 | 0.00% |
| Qwen3-14B base | hitom | cot | 2/600 | 0.33% |
| v3.5 (Stage 19 ckpt-120) | tombench | direct | 0/5718 | 0.00% |
| v3.5 (Stage 19 ckpt-120) | tombench | direct_think | 5/5718 | 0.09% |
| v3.5 (Stage 19 ckpt-120) | tombench | cot | 1/5718 | 0.02% |
| v3.5 (Stage 19 ckpt-120) | emobench | direct | 0/1200 | 0.00% |
| v3.5 (Stage 19 ckpt-120) | emobench | direct_think | 0/1200 | 0.00% |
| v3.5 (Stage 19 ckpt-120) | emobench | cot | 0/1200 | 0.00% |
| v3.5 (Stage 19 ckpt-120) | socialiqa | direct | 0/1954 | 0.00% |
| v3.5 (Stage 19 ckpt-120) | socialiqa | direct_think | 0/1954 | 0.00% |
| v3.5 (Stage 19 ckpt-120) | socialiqa | cot | 0/1954 | 0.00% |
| v3.5 (Stage 19 ckpt-120) | hitom | direct | 0/600 | 0.00% |
| v3.5 (Stage 19 ckpt-120) | hitom | direct_think | 0/600 | 0.00% |
| v3.5 (Stage 19 ckpt-120) | hitom | cot | 0/600 | 0.00% |
| v3.1 (Stage 14b ckpt-199) | tombench | direct | 0/5718 | 0.00% |
| v3.1 (Stage 14b ckpt-199) | tombench | direct_think | 5/5718 | 0.09% |
| v3.1 (Stage 14b ckpt-199) | tombench | cot | 0/5718 | 0.00% |
| v3.1 (Stage 14b ckpt-199) | emobench | direct | 0/1200 | 0.00% |
| v3.1 (Stage 14b ckpt-199) | emobench | direct_think | 0/1200 | 0.00% |
| v3.1 (Stage 14b ckpt-199) | emobench | cot | 0/1200 | 0.00% |
| v3.1 (Stage 14b ckpt-199) | socialiqa | direct | 0/1954 | 0.00% |
| v3.1 (Stage 14b ckpt-199) | socialiqa | direct_think | 0/1954 | 0.00% |
| v3.1 (Stage 14b ckpt-199) | socialiqa | cot | 0/1954 | 0.00% |
| v3.1 (Stage 14b ckpt-199) | hitom | direct | 0/600 | 0.00% |
| v3.1 (Stage 14b ckpt-199) | hitom | direct_think | 0/600 | 0.00% |
| v3.1 (Stage 14b ckpt-199) | hitom | cot | 0/600 | 0.00% |
| deepseek-v4-pro (API) | tombench | direct | 5379/5718 | 94.07% ⚠️ |
| deepseek-v4-pro (API) | tombench | direct_think | 38/5718 | 0.66% |
| deepseek-v4-pro (API) | tombench | cot | 87/5718 | 1.52% ⚠️ |
| deepseek-v4-pro (API) | emobench | direct | 1170/1200 | 97.50% ⚠️ |
| deepseek-v4-pro (API) | emobench | direct_think | 0/1200 | 0.00% |
| deepseek-v4-pro (API) | emobench | cot | 4/1200 | 0.33% |
| deepseek-v4-pro (API) | socialiqa | direct | 1952/1954 | 99.90% ⚠️ |
| deepseek-v4-pro (API) | socialiqa | direct_think | 1/1954 | 0.05% |
| deepseek-v4-pro (API) | socialiqa | cot | 6/1954 | 0.31% |
| deepseek-v4-pro (API) | hitom | direct | 600/600 | 100.00% ⚠️ |
| deepseek-v4-pro (API) | hitom | direct_think | 97/600 | 16.17% ⚠️ |
| deepseek-v4-pro (API) | hitom | cot | 144/600 | 24.00% ⚠️ |


---

## 1.5 输出长度统计 (chars)

| Model | Benchmark | Protocol | mean | p95 | max |
|---|---|---|---|---|---|
| Qwen3-14B base | tombench | direct | 9 | 9 | 9 |
| Qwen3-14B base | tombench | direct_think | 2607 | 7771 | 39343 |
| Qwen3-14B base | tombench | cot | 3409 | 11324 | 29531 |
| Qwen3-14B base | emobench | direct | 9 | 9 | 9 |
| Qwen3-14B base | emobench | direct_think | 2940 | 7760 | 27462 |
| Qwen3-14B base | emobench | cot | 4097 | 10789 | 19949 |
| Qwen3-14B base | socialiqa | direct | 9 | 9 | 9 |
| Qwen3-14B base | socialiqa | direct_think | 1879 | 4063 | 9146 |
| Qwen3-14B base | socialiqa | cot | 3181 | 7115 | 12432 |
| Qwen3-14B base | hitom | direct | 9 | 9 | 9 |
| Qwen3-14B base | hitom | direct_think | 7650 | 17951 | 27528 |
| Qwen3-14B base | hitom | cot | 11331 | 25363 | 35583 |
| v3.5 (Stage 19 ckpt-120) | tombench | direct | 9 | 9 | 9 |
| v3.5 (Stage 19 ckpt-120) | tombench | direct_think | 319 | 646 | 33391 |
| v3.5 (Stage 19 ckpt-120) | tombench | cot | 700 | 1762 | 29664 |
| v3.5 (Stage 19 ckpt-120) | emobench | direct | 9 | 9 | 9 |
| v3.5 (Stage 19 ckpt-120) | emobench | direct_think | 344 | 757 | 2198 |
| v3.5 (Stage 19 ckpt-120) | emobench | cot | 845 | 2067 | 5093 |
| v3.5 (Stage 19 ckpt-120) | socialiqa | direct | 9 | 9 | 9 |
| v3.5 (Stage 19 ckpt-120) | socialiqa | direct_think | 416 | 682 | 1878 |
| v3.5 (Stage 19 ckpt-120) | socialiqa | cot | 816 | 1752 | 4592 |
| v3.5 (Stage 19 ckpt-120) | hitom | direct | 9 | 9 | 9 |
| v3.5 (Stage 19 ckpt-120) | hitom | direct_think | 636 | 1347 | 3623 |
| v3.5 (Stage 19 ckpt-120) | hitom | cot | 2600 | 7863 | 15523 |
| v3.1 (Stage 14b ckpt-199) | tombench | direct | 9 | 9 | 9 |
| v3.1 (Stage 14b ckpt-199) | tombench | direct_think | 398 | 962 | 28218 |
| v3.1 (Stage 14b ckpt-199) | tombench | cot | 869 | 2420 | 13491 |
| v3.1 (Stage 14b ckpt-199) | emobench | direct | 9 | 9 | 9 |
| v3.1 (Stage 14b ckpt-199) | emobench | direct_think | 440 | 1035 | 3326 |
| v3.1 (Stage 14b ckpt-199) | emobench | cot | 1082 | 2859 | 9646 |
| v3.1 (Stage 14b ckpt-199) | socialiqa | direct | 9 | 9 | 9 |
| v3.1 (Stage 14b ckpt-199) | socialiqa | direct_think | 518 | 962 | 3215 |
| v3.1 (Stage 14b ckpt-199) | socialiqa | cot | 931 | 2103 | 5239 |
| v3.1 (Stage 14b ckpt-199) | hitom | direct | 9 | 9 | 9 |
| v3.1 (Stage 14b ckpt-199) | hitom | direct_think | 1344 | 3565 | 8105 |
| v3.1 (Stage 14b ckpt-199) | hitom | cot | 4415 | 12932 | 20745 |
| deepseek-v4-pro (API) | tombench | direct | 1 | 9 | 9 |
| deepseek-v4-pro (API) | tombench | direct_think | 9 | 9 | 9 |
| deepseek-v4-pro (API) | tombench | cot | 227 | 551 | 1276 |
| deepseek-v4-pro (API) | emobench | direct | 0 | 0 | 9 |
| deepseek-v4-pro (API) | emobench | direct_think | 9 | 9 | 9 |
| deepseek-v4-pro (API) | emobench | cot | 225 | 509 | 906 |
| deepseek-v4-pro (API) | socialiqa | direct | 0 | 0 | 9 |
| deepseek-v4-pro (API) | socialiqa | direct_think | 9 | 9 | 9 |
| deepseek-v4-pro (API) | socialiqa | cot | 213 | 451 | 748 |
| deepseek-v4-pro (API) | hitom | direct | 0 | 0 | 0 |
| deepseek-v4-pro (API) | hitom | direct_think | 8 | 9 | 9 |
| deepseek-v4-pro (API) | hitom | cot | 337 | 921 | 2007 |


---

## 2. 无法解析的预测 (pred=None) 统计

| Model | Benchmark | Protocol | unparseable / total |
|---|---|---|---|
| Qwen3-14B base | tombench | direct | 0/5718 |
| Qwen3-14B base | tombench | direct_think | 18/5718 |
| Qwen3-14B base | tombench | cot | 0/5718 |
| Qwen3-14B base | emobench | direct | 0/1200 |
| Qwen3-14B base | emobench | direct_think | 1/1200 |
| Qwen3-14B base | emobench | cot | 0/1200 |
| Qwen3-14B base | socialiqa | direct | 0/1954 |
| Qwen3-14B base | socialiqa | direct_think | 0/1954 |
| Qwen3-14B base | socialiqa | cot | 0/1954 |
| Qwen3-14B base | hitom | direct | 0/600 |
| Qwen3-14B base | hitom | direct_think | 0/600 |
| Qwen3-14B base | hitom | cot | 0/600 |
| v3.5 (Stage 19 ckpt-120) | tombench | direct | 0/5718 |
| v3.5 (Stage 19 ckpt-120) | tombench | direct_think | 1/5718 |
| v3.5 (Stage 19 ckpt-120) | tombench | cot | 0/5718 |
| v3.5 (Stage 19 ckpt-120) | emobench | direct | 0/1200 |
| v3.5 (Stage 19 ckpt-120) | emobench | direct_think | 0/1200 |
| v3.5 (Stage 19 ckpt-120) | emobench | cot | 0/1200 |
| v3.5 (Stage 19 ckpt-120) | socialiqa | direct | 0/1954 |
| v3.5 (Stage 19 ckpt-120) | socialiqa | direct_think | 0/1954 |
| v3.5 (Stage 19 ckpt-120) | socialiqa | cot | 0/1954 |
| v3.5 (Stage 19 ckpt-120) | hitom | direct | 0/600 |
| v3.5 (Stage 19 ckpt-120) | hitom | direct_think | 0/600 |
| v3.5 (Stage 19 ckpt-120) | hitom | cot | 0/600 |
| v3.1 (Stage 14b ckpt-199) | tombench | direct | 0/5718 |
| v3.1 (Stage 14b ckpt-199) | tombench | direct_think | 0/5718 |
| v3.1 (Stage 14b ckpt-199) | tombench | cot | 0/5718 |
| v3.1 (Stage 14b ckpt-199) | emobench | direct | 0/1200 |
| v3.1 (Stage 14b ckpt-199) | emobench | direct_think | 0/1200 |
| v3.1 (Stage 14b ckpt-199) | emobench | cot | 0/1200 |
| v3.1 (Stage 14b ckpt-199) | socialiqa | direct | 0/1954 |
| v3.1 (Stage 14b ckpt-199) | socialiqa | direct_think | 0/1954 |
| v3.1 (Stage 14b ckpt-199) | socialiqa | cot | 0/1954 |
| v3.1 (Stage 14b ckpt-199) | hitom | direct | 0/600 |
| v3.1 (Stage 14b ckpt-199) | hitom | direct_think | 0/600 |
| v3.1 (Stage 14b ckpt-199) | hitom | cot | 0/600 |
| deepseek-v4-pro (API) | tombench | direct | 5359/5718 |
| deepseek-v4-pro (API) | tombench | direct_think | 38/5718 |
| deepseek-v4-pro (API) | tombench | cot | 90/5718 |
| deepseek-v4-pro (API) | emobench | direct | 1166/1200 |
| deepseek-v4-pro (API) | emobench | direct_think | 0/1200 |
| deepseek-v4-pro (API) | emobench | cot | 4/1200 |
| deepseek-v4-pro (API) | socialiqa | direct | 1952/1954 |
| deepseek-v4-pro (API) | socialiqa | direct_think | 1/1954 |
| deepseek-v4-pro (API) | socialiqa | cot | 6/1954 |
| deepseek-v4-pro (API) | hitom | direct | 600/600 |
| deepseek-v4-pro (API) | hitom | direct_think | 98/600 |
| deepseek-v4-pro (API) | hitom | cot | 143/600 |


---

## 3. ToMBench 分任务详表


#### tombench / direct

| task | Qwen3-14B base | v3.5 (Stage 19 ckpt-120) | v3.1 (Stage 14b ckpt-199) | deepseek-v4-pro (API) |
|---|---|---|---|---|
| Belief | 0.7289 (207/284) | 0.7218 (205/284) | 0.7148 (203/284) | 0.0246 (7/284) |
| Desire | 0.5833 (210/360) | 0.5889 (212/360) | 0.5889 (212/360) | 0.0056 (2/360) |
| Emotion | 0.7298 (613/840) | 0.7286 (612/840) | 0.7333 (616/840) | 0.0643 (54/840) |
| False Belief | 0.8047 (1191/1480) | 0.8034 (1189/1480) | 0.8142 (1205/1480) | 0.0574 (85/1480) |
| Intention | 0.7691 (523/680) | 0.7956 (541/680) | 0.7912 (538/680) | 0.0279 (19/680) |
| Knowledge | 0.4619 (267/578) | 0.4550 (263/578) | 0.4689 (271/578) | 0.0017 (1/578) |
| Non-literal Comm | 0.7921 (1185/1496) | 0.8135 (1217/1496) | 0.8135 (1217/1496) | 0.1203 (180/1496) |

#### tombench / direct_think

| task | Qwen3-14B base | v3.5 (Stage 19 ckpt-120) | v3.1 (Stage 14b ckpt-199) | deepseek-v4-pro (API) |
|---|---|---|---|---|
| Belief | 0.7641 (217/284) | 0.7359 (209/284) | 0.7394 (210/284) | 0.8592 (244/284) |
| Desire | 0.6056 (218/360) | 0.6167 (222/360) | 0.6306 (227/360) | 0.6083 (219/360) |
| Emotion | 0.7262 (610/840) | 0.7310 (614/840) | 0.7405 (622/840) | 0.8060 (677/840) |
| False Belief | 0.8743 (1294/1480) | 0.8561 (1267/1480) | 0.8730 (1292/1480) | 0.9047 (1339/1480) |
| Intention | 0.8000 (544/680) | 0.8368 (569/680) | 0.8265 (562/680) | 0.8882 (604/680) |
| Knowledge | 0.5208 (301/578) | 0.5035 (291/578) | 0.5052 (292/578) | 0.6574 (380/578) |
| Non-literal Comm | 0.7754 (1160/1496) | 0.8209 (1228/1496) | 0.8168 (1222/1496) | 0.8189 (1225/1496) |

#### tombench / cot

| task | Qwen3-14B base | v3.5 (Stage 19 ckpt-120) | v3.1 (Stage 14b ckpt-199) | deepseek-v4-pro (API) |
|---|---|---|---|---|
| Belief | 0.7711 (219/284) | 0.7324 (208/284) | 0.7394 (210/284) | 0.8627 (245/284) |
| Desire | 0.5778 (208/360) | 0.6111 (220/360) | 0.6028 (217/360) | 0.6333 (228/360) |
| Emotion | 0.7452 (626/840) | 0.7464 (627/840) | 0.7560 (635/840) | 0.8107 (681/840) |
| False Belief | 0.8797 (1302/1480) | 0.8804 (1303/1480) | 0.9020 (1335/1480) | 0.9169 (1357/1480) |
| Intention | 0.7897 (537/680) | 0.8235 (560/680) | 0.8412 (572/680) | 0.8985 (611/680) |
| Knowledge | 0.5190 (300/578) | 0.5294 (306/578) | 0.5519 (319/578) | 0.6626 (383/578) |
| Non-literal Comm | 0.7747 (1159/1496) | 0.7901 (1182/1496) | 0.7894 (1181/1496) | 0.8242 (1233/1496) |

---

## 4. Hi-ToM 分阶 (order_0..order_4)


#### hitom / direct

| task | Qwen3-14B base | v3.5 (Stage 19 ckpt-120) | v3.1 (Stage 14b ckpt-199) | deepseek-v4-pro (API) |
|---|---|---|---|---|
| order_0 | 0.8667 (104/120) | 0.8917 (107/120) | 0.8917 (107/120) | 0.0000 (0/120) |
| order_1 | 0.5583 (67/120) | 0.5917 (71/120) | 0.5833 (70/120) | 0.0000 (0/120) |
| order_2 | 0.4083 (49/120) | 0.5000 (60/120) | 0.4000 (48/120) | 0.0000 (0/120) |
| order_3 | 0.4250 (51/120) | 0.4833 (58/120) | 0.4333 (52/120) | 0.0000 (0/120) |
| order_4 | 0.4000 (48/120) | 0.4750 (57/120) | 0.4083 (49/120) | 0.0000 (0/120) |

#### hitom / direct_think

| task | Qwen3-14B base | v3.5 (Stage 19 ckpt-120) | v3.1 (Stage 14b ckpt-199) | deepseek-v4-pro (API) |
|---|---|---|---|---|
| order_0 | 1.0000 (120/120) | 0.9917 (119/120) | 0.9833 (118/120) | 1.0000 (120/120) |
| order_1 | 0.8000 (96/120) | 0.6833 (82/120) | 0.6833 (82/120) | 0.8667 (104/120) |
| order_2 | 0.7250 (87/120) | 0.6833 (82/120) | 0.6583 (79/120) | 0.7750 (93/120) |
| order_3 | 0.6500 (78/120) | 0.5333 (64/120) | 0.5500 (66/120) | 0.6583 (79/120) |
| order_4 | 0.5500 (66/120) | 0.5250 (63/120) | 0.4000 (48/120) | 0.5667 (68/120) |

#### hitom / cot

| task | Qwen3-14B base | v3.5 (Stage 19 ckpt-120) | v3.1 (Stage 14b ckpt-199) | deepseek-v4-pro (API) |
|---|---|---|---|---|
| order_0 | 1.0000 (120/120) | 1.0000 (120/120) | 1.0000 (120/120) | 1.0000 (120/120) |
| order_1 | 0.8167 (98/120) | 0.6917 (83/120) | 0.6667 (80/120) | 0.8500 (102/120) |
| order_2 | 0.8083 (97/120) | 0.7333 (88/120) | 0.7333 (88/120) | 0.7500 (90/120) |
| order_3 | 0.7000 (84/120) | 0.6000 (72/120) | 0.5750 (69/120) | 0.5583 (67/120) |
| order_4 | 0.6500 (78/120) | 0.6583 (79/120) | 0.5417 (65/120) | 0.4500 (54/120) |

---

## 5. EmoBench 分任务 (EA/EU)


#### emobench / direct

| task | Qwen3-14B base | v3.5 (Stage 19 ckpt-120) | v3.1 (Stage 14b ckpt-199) | deepseek-v4-pro (API) |
|---|---|---|---|---|
| EA | 0.6900 (276/400) | 0.7075 (283/400) | 0.7075 (283/400) | 0.0275 (11/400) |
| EU_cause | 0.7475 (299/400) | 0.7550 (302/400) | 0.7550 (302/400) | 0.0125 (5/400) |
| EU_emotion | 0.4650 (186/400) | 0.4825 (193/400) | 0.4825 (193/400) | 0.0350 (14/400) |

#### emobench / direct_think

| task | Qwen3-14B base | v3.5 (Stage 19 ckpt-120) | v3.1 (Stage 14b ckpt-199) | deepseek-v4-pro (API) |
|---|---|---|---|---|
| EA | 0.7150 (286/400) | 0.6950 (278/400) | 0.7025 (281/400) | 0.7750 (310/400) |
| EU_cause | 0.7725 (309/400) | 0.7650 (306/400) | 0.7500 (300/400) | 0.8450 (338/400) |
| EU_emotion | 0.5325 (213/400) | 0.4975 (199/400) | 0.4675 (187/400) | 0.6950 (278/400) |

#### emobench / cot

| task | Qwen3-14B base | v3.5 (Stage 19 ckpt-120) | v3.1 (Stage 14b ckpt-199) | deepseek-v4-pro (API) |
|---|---|---|---|---|
| EA | 0.7275 (291/400) | 0.7050 (282/400) | 0.6600 (264/400) | 0.7400 (296/400) |
| EU_cause | 0.7725 (309/400) | 0.7775 (311/400) | 0.7675 (307/400) | 0.8475 (339/400) |
| EU_emotion | 0.5625 (225/400) | 0.5000 (200/400) | 0.5000 (200/400) | 0.6950 (278/400) |

---

## 6. ToMBench 中英语言切分


#### ToMBench language split / direct

| Language | Qwen3-14B base | v3.5 (Stage 19 ckpt-120) | v3.1 (Stage 14b ckpt-199) | deepseek-v4-pro (API) |
|---|---|---|---|---|
| en | 0.7251 (2073/2859) | 0.7282 (2082/2859) | 0.7335 (2097/2859) | 0.0073 (21/2859) |
| zh | 0.7426 (2123/2859) | 0.7545 (2157/2859) | 0.7573 (2165/2859) | 0.1144 (327/2859) |

#### ToMBench language split / direct_think

| Language | Qwen3-14B base | v3.5 (Stage 19 ckpt-120) | v3.1 (Stage 14b ckpt-199) | deepseek-v4-pro (API) |
|---|---|---|---|---|
| en | 0.7447 (2129/2859) | 0.7695 (2200/2859) | 0.7698 (2201/2859) | 0.8153 (2331/2859) |
| zh | 0.7747 (2215/2859) | 0.7695 (2200/2859) | 0.7786 (2226/2859) | 0.8244 (2357/2859) |

#### ToMBench language split / cot

| Language | Qwen3-14B base | v3.5 (Stage 19 ckpt-120) | v3.1 (Stage 14b ckpt-199) | deepseek-v4-pro (API) |
|---|---|---|---|---|
| en | 0.7464 (2134/2859) | 0.7611 (2176/2859) | 0.7695 (2200/2859) | 0.8227 (2352/2859) |
| zh | 0.7754 (2217/2859) | 0.7800 (2230/2859) | 0.7936 (2269/2859) | 0.8346 (2386/2859) |

---

## 7. 评测协议与采样参数


| Protocol | temperature | top_p | max_tokens | enable_thinking | system prompt | extractor |
|---|---|---|---|---|---|---|
| **direct**       | 0.0 | 1.0  | 64   | **false** | DIRECT system | `extract_direct`: 第一个 `\boxed{X}`，否则首个有效字母 |
| **direct_think** | 0.0 | 1.0  | 8192 | **true**  | DIRECT system | `extract_cot`: 最后一个 `\boxed{X}`，否则末 200 字符内最后一个有效字母 |
| **cot**          | 0.6 | 0.95 | 8192 | **true**  | COT system    | `extract_cot` |

- 本地 vLLM（base/v3.5/v3.1）通过 `extra_body={"chat_template_kwargs": {"enable_thinking": <bool>}}` 控制 thinking。
- **deepseek-v4-pro**（DeepSeek 官方 API）：所有采样参数（T / top_p / max_tokens）、prompt、extractor 与本地模型**完全一致**，
  以保证公平。`enable_thinking` 是 Qwen/vLLM 专有开关，deepseek 为原生推理模型、无此旋钮，故不传；
  其推理走独立的 `reasoning_content` 字段，可见答案（含 `\boxed{}`）在 `content` 中，我们与本地模型一样从 `content` 抽取。
- ⚠️ **deepseek-v4-pro 的 `direct`（max_tokens=64）几乎必然截断**：推理模型在 64 token 内来不及给出答案，
  reasoning 就耗尽预算、`content` 为空 → pred=None。这是"完全一致参数"作用在推理模型上的必然结果，
  **deepseek 的有效对比应看 `direct_think` / `cot`**（见 §1 截断表）。
- ToMBench 用 ToM 专用 system prompt（含 "theory-of-mind"，4 选项 A-D，ZH 选项自动 strip 重复字母前缀）。
- Hi-ToM / SocialIQA / EmoBench 用通用 MCQ system prompt，字母范围按当题选项数动态生成
  （Hi-ToM 15→A-O，EmoBench 4→A-D，SocialIQA 3→A-C），与 `scripts/eval/run_generic_mcq.py` 一致。


---

## 8. 部署与复现

```bash
# On 172.16.120.181 (h800-3):
cd /data_nvme/grj-projects/qwen3-tom/experiment/qwen3-14b-full-eval

# Full run (per-model: up GPUs 4-7 → eval 4 benches × 3 protocols → down)
bash scripts/04_run_eval.sh

# Smoke test (10 questions)
LIMIT=10 bash scripts/04_run_eval.sh

# Aggregate this report
python3 scripts/05_aggregate_report.py --results-dir output
```

- **模型路径**:
  - base: `/data_nvme/grj-projects/models/Qwen3-14B`
  - v3.5: `/data_nvme/grj-projects/tom-output/qwen3-14B-tombench-rlvr-stage19-1x8-hf-ckpt120`
  - v3.1: `/data_nvme/grj-projects/tom-output/qwen3-14B-tombench-rlvr-stage14b-1x8-hf-ckpt199`
- **数据**: `data/tom/tombench_eval.jsonl`, `data/eval/{hitom,socialiqa,emobench}_eval.jsonl`
- **vLLM**: image `qwen3-tom-serve-eval-dp4:latest`, TP=1, max_model_len=16384, gpu_util=0.85

