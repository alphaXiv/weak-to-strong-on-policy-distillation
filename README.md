# Weak-to-Strong On-Policy Distillation: controlled reproduction

This repository is a compact, auditable reproduction of the **scale-contrast** experiment in [Weak-to-Strong On-Policy Distillation](https://arxiv.org/abs/2607.26246). It tests whether a Qwen3-8B student benefits from the logit-space direction between Qwen3-4B and Qwen3-0.6B on GSM8K.

## Result

The local implementation found a strong *relative* W2S-OPD advantage after learning-rate tuning, but it did **not** reproduce an absolute improvement over the untrained Qwen3-8B baseline.

| Setting | Direct OPD | W2S-OPD | Zero-shot |
|---|---:|---:|---:|
| 6 steps × batch 8, 512 eval | 429/512 (83.79%) | 427/512 (83.40%) | 434/512 (84.77%) |
| 25 steps × batch 64, 512 eval | **430/512 (83.98%)** | **429/512 (83.79%)** | 434/512 (84.77%) |
| 100 steps × batch 64, 512 eval | 418/512 (81.64%) | 424/512 (82.81%) | 434/512 (84.77%) |
| 25 steps × batch 64, LR 4e-6 | 423/512 (82.62%) | **434/512 (84.77%)** | 434/512 (84.77%) |

W2S-OPD's best run matched zero-shot and beat its matched direct-OPD control by 11 examples; direct OPD worsened at the same learning rate. The lack of an absolute gain is informative but not a strict refutation. This is a LoRA/FedAvg approximation on one benchmark and one random seed, not the paper's full training stack. See `REPORT.md` for the complete interpretation and limitations.

## What is implemented

- Student-on-policy sampled rollouts.
- Reverse-KL distillation over the teacher's top 32 tokens.
- Direct OPD target: Qwen3-4B.
- W2S target: `Qwen3-8B anchor + alpha * (Qwen3-4B - Qwen3-0.6B)` with `alpha=1`.
- Rank-16 LoRA on attention and MLP projections.
- Eight independent GPU workers followed by mean LoRA-delta aggregation.
- Deterministic prompt/example seeding and exact-match GSM8K evaluation.
- Machine-readable terminal metrics prefixed by `ORX_METRICS`.

## Repository map

| Path | Purpose |
|---|---|
| `experiment.py` | Training, aggregation, and evaluation entrypoint |
| `config.json` | Shipped full-evaluation configuration |
| `configs/` | Frozen representative configurations |
| `run.sh` | Eight-GPU launcher and dependency setup |
| `.orx/k8s.yaml` | Kubernetes job and shared-volume manifest |
| `results.csv` | Run-level terminal-log evidence used in the report |
| `analysis.py` | Self-contained marimo analysis notebook |
| `REPORT.md` | Full reproduction report |

## Run the experiment

The committed Kubernetes manifest requests one node with 8 GPUs, 96 CPUs, and 640 GiB RAM. The launcher expects CUDA-visible devices 0–7 and installs the pinned Python dependencies into the job image.

```bash
# Choose one committed configuration.
cp configs/w2s_25x64.json config.json

# Commit and push the exact code/config that should run, then submit through orx.
orx exp run <experiment-id> --backend k8s
orx exp wait <experiment-id>
orx logs <run-id> --bytes 500000 | rg ORX_METRICS
```

For a direct eight-GPU machine, `bash run.sh` uses the same launcher contract. It expects sufficient local model-cache and shared-volume capacity at `/cache` and `/shared`.

The baseline needs no optimization. Set `"mode": "baseline"` and use the baseline Kubernetes manifest variant documented in `configs/baseline.json`; evaluation can be sharded across multiple pods by setting the manifest's indexed completions and `ORX_NUM_PODS` consistently.

## Explore the evidence

The notebook embeds its own data, so it does not depend on the working tree or network after installation:

```bash
uvx --with pandas --with altair marimo run analysis.py
```

It reconstructs the principal comparison, training-horizon curve, learning-rate sweep, seed sensitivity, and Wilson intervals from terminal run metrics.

## Reproducibility contract

- Models: `Qwen/Qwen3-8B`, `Qwen/Qwen3-4B`, and `Qwen/Qwen3-0.6B`.
- Dataset: `openai/gsm8k`, `main` split.
- Prompt: chat template with thinking disabled and a boxed-answer request.
- Evaluation: one temperature-1 sample per fixed test example; exact numeric match.
- Primary seed: `20260728`.
- Software versions are pinned in `requirements.txt`; the job image is pinned in `.orx/k8s.yaml`.
- Every reported row is tied to an immutable Git commit and Kubernetes run ID in `results.csv`.

## Important limitations

This package covers only the scale-contrast GSM8K setting. It does not reproduce the paper's math/code suite, full-parameter optimizer, post-RL contrasts, hint contrasts, or multi-teacher experiments. The worker updates are independent and averaged rather than gradient-synchronized, and stochastic single-sample evaluation leaves several-example uncertainty. These deviations are large enough that the result should be read as a controlled partial reproduction.

## Citation and license

Citation metadata for this software package and the source paper is in `CITATION.cff`. Code added for this reproduction is MIT licensed; model and dataset use remains governed by their upstream licenses. See `NOTICE.md`.
