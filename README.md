# BIPIA IPI Defense Pipeline (DPO vs GRPO)

本项目提供一个可在单卡 RTX 3090（24GB）上运行的实验骨架，用于比较：

- Base model
- SFT-defense baseline
- DPO
- GRPO

在 BIPIA benchmark 的 IPI（Indirect Prompt Injection）防御表现。

## 快速开始

```bash
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt
```

1. 预处理 BIPIA 并构建训练数据：

```bash
python data/build_dpo_dataset.py --config configs/dpo.yaml
python data/build_grpo_dataset.py --config configs/grpo.yaml
```

2. 训练：

```bash
bash scripts/run_sft.sh
bash scripts/run_dpo.sh
bash scripts/run_grpo.sh
```

3. 评测：

```bash
bash scripts/run_eval.sh
python analysis/summarize_results.py --pred_dir outputs/predictions --out outputs/summary
```

## 数据格式约定

统一样本格式（内部）：

```json
{
  "id": "...",
  "task": "email",
  "question": "...",
  "external_content": "...",
  "reference_answer": "...",
  "attack_type": "html_injection",
  "is_attack": true
}
```

DPO 格式：

```json
{"prompt":"...","chosen":"...","rejected":"...","task":"...","attack_type":"..."}
```

GRPO 格式：

```json
{"id":"...","prompt":"...","reference_answer":"...","task":"...","attack_type":"...","is_attack":true}
```

## 3090 推荐设置

- base 模型：`Qwen/Qwen2.5-0.5B-Instruct` 或 `Qwen/Qwen2.5-1.5B-Instruct`
- QLoRA: 4-bit + LoRA(r=16, alpha=32)
- bf16（若支持）/ fp16
- batch size 小步 + gradient accumulation

> 该仓库是“可跑模板”，默认 reward/judge 是启发式，可替换为更强的 judge 模型或规则系统。
