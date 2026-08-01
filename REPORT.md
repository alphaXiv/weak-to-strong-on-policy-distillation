# Controlled reproduction of Weak-to-Strong On-Policy Distillation

**Project:** Weak-to-Strong On-Policy Distillation  
**Source:** Fangxu Yu et al., [arXiv:2607.26246](https://arxiv.org/abs/2607.26246)  
**Reproduction date:** 2026-07-31  
**Status:** completed partial reproduction with successful Kubernetes terminal logs

## Executive summary

This study tests the paper's scale-contrast claim in a constrained setting: can an 8B student improve on GSM8K by distilling the capability direction between weaker 4B and 0.6B models? The implementation uses the paper's proxy-teacher construction and top-32 reverse-KL objective, but replaces the paper's full training stack with rank-16 LoRA and mean aggregation of eight independent worker updates.

The strongest local result is mixed:

- Tuned W2S-OPD (25 optimizer steps, effective batch 64, LR 4e-6) scored **434/512 = 84.77%**.
- Its matched direct-OPD control scored **423/512 = 82.62%**, so the W2S construction contributed **+11/512 = +2.15 percentage points** at that learning rate.
- The untrained Qwen3-8B baseline also scored **434/512 = 84.77%**. W2S-OPD recovered the baseline but did not improve it.
- At the default LR 2e-6, direct OPD slightly led W2S-OPD: 430/512 versus 429/512.
- At the paper-scale local schedule of 100×64 rollouts, W2S-OPD led direct OPD by 6/512 but both degraded relative to zero-shot.

The reproduction therefore supports a **relative advantage for the W2S target in some regimes**, especially its robustness at the higher learning rate, but does **not reproduce the paper's claimed absolute scale-contrast gain**. Because this package changes the optimizer geometry, parameterization, benchmark coverage, and evaluation replication, that outcome is evidence about this implementation—not a falsification of the method.

## Research question and success criterion

The source method forms a proxy teacher from a strong student anchor and a weak-model capability direction:

\[
z_T(s_t) = z_{8B}(s_t) + \alpha\left(z_{4B}(s_t) - z_{0.6B}(s_t)\right).
\]

Student-generated tokens are optimized with reverse KL toward this proxy distribution. The direct-OPD control instead uses the 4B logits as its target. The predeclared practical success criterion was a reproducible GSM8K improvement over the same untrained 8B checkpoint, accompanied by a favorable comparison with direct OPD under a matched schedule.

That strict criterion was not met: the best W2S run tied the baseline.

## Implementation

### Models and data

| Role | Resource |
|---|---|
| Student and W2S anchor | `Qwen/Qwen3-8B` |
| Positive weak model / direct teacher | `Qwen/Qwen3-4B` |
| Negative weak model | `Qwen/Qwen3-0.6B` |
| Dataset | `openai/gsm8k`, `main` |

Each example uses the Qwen chat template with thinking disabled and asks for a final `\boxed{}` answer. Evaluation draws one temperature-1 completion per example, seeded by the experiment seed plus the example index. Numeric exact match uses the boxed answer when present and otherwise the last parseable number.

### Optimization

- Student rollouts are generated on policy.
- The loss is reverse KL over the teacher's top 32 tokens.
- LoRA rank 16 and alpha 32 are applied to attention and MLP projections (43,646,976 trainable parameters).
- Each of eight GPUs trains an independent LoRA worker on a disjoint deterministic rollout stream.
- Worker LoRA deltas are averaged once after training, then the averaged adapter is evaluated across the eight GPUs.
- The paper-batch approximation accumulates eight sequential rollouts per worker before each optimizer step, giving an effective batch of 64.
- AdamW uses betas (0.9, 0.95), zero weight decay, and unit gradient clipping.

### Execution and evidence

All measurements were launched as committed Kubernetes jobs through `orx exp run --backend k8s`. A job prints its complete configuration, per-step progress, per-example evaluation records, and a final JSON `ORX_METRICS` block. The immutable run IDs and Git commits are in `results.csv`. Except for two initial smoke runs explicitly flagged in that file, every quantitative claim below is recoverable from persisted `orx logs` terminal output.

## Results

### Primary 512-example comparison

| Recipe | Correct | Accuracy | Difference from zero-shot |
|---|---:|---:|---:|
| Zero-shot Qwen3-8B | 434/512 | 84.77% | — |
| Best direct OPD: 25×64, LR 2e-6 | 430/512 | 83.98% | -4 |
| Best W2S-OPD: 25×64, LR 4e-6 | **434/512** | **84.77%** | 0 |

The 95% Wilson interval is 81.39%–87.62% for 434/512 and 80.56%–86.91% for 430/512. These intervals are descriptive rather than a paired test; the methods use the same evaluation questions and seeds, so independent-binomial significance calculations would be inappropriate without retaining paired predictions.

### Matched learning-rate sweep at 25×64

| Learning rate | Direct OPD | W2S-OPD | W2S − direct |
|---:|---:|---:|---:|
| 1e-6 | 423/512 (82.62%) | 425/512 (83.01%) | +2 |
| 2e-6 | **430/512 (83.98%)** | 429/512 (83.79%) | -1 |
| 4e-6 | 423/512 (82.62%) | **434/512 (84.77%)** | **+11** |

This interaction is the clearest positive evidence for the W2S target. Direct OPD has a narrow optimum at 2e-6, whereas W2S improves at 4e-6 and returns to the zero-shot score. The result is still a single-seed observation and should be replicated before attributing it to greater optimization stability.

### Effective-batch-64 horizon sweep at LR 2e-6

| Optimizer steps | Effective rollouts | Direct OPD | W2S-OPD | W2S − direct |
|---:|---:|---:|---:|---:|
| 12 | 768 | 427 | 428 | +1 |
| 18 | 1,152 | 423 | 426 | +3 |
| 25 | 1,600 | **430** | **429** | -1 |
| 32 | 2,048 | 426 | 422 | -4 |
| 100 | 6,400 | 418 | 424 | +6 |

Neither method exceeds 434/512 at any tested LR-2e-6 horizon. The non-monotonic curve and degradation by 100 steps suggest that early stopping is essential in this approximation. W2S's +6 advantage at 100 steps supports a relative robustness claim, but the absolute outcome remains below the starting checkpoint.

### Small-batch seed check

The inexpensive six-step, batch-8 screen was repeated with three seeds on 128 examples:

| Seed | Zero-shot | Direct OPD | W2S-OPD | Direct delta | W2S delta |
|---:|---:|---:|---:|---:|---:|
| 20260728 | 108 | 108 | 109 | 0 | +1 |
| 20260729 | 108 | 109 | 107 | +1 | -1 |
| 20260730 | 104 | 108 | 105 | +4 | +1 |
| Mean | 106.67 | 108.33 | 107.00 | +1.67 | +0.33 |

At this smoke scale, direct OPD has the larger mean paired gain and W2S changes sign. The result motivated moving to the larger fixed evaluation rather than selecting on 128-example noise.

### Screening observations

- Top-K 32 was best in the six-step screen: K={16, 32, 64, 128} produced {106, 109, 107, 104}/128.
- The 12-step alpha screen did not reveal a gain: alpha={0.5, 1.0, 1.25, 1.5, 2.0} produced {104, 106, 105, 103, 104}/128.
- The six-step horizon screen peaked at 109/128; 3, 6, and 9 steps scored 101, 109, and 107.

These screens were used only to choose plausible settings. Their small evaluation set makes one- or two-example differences weak evidence.

## Comparison with the source paper

Yu et al. report that the 4B–0.6B scale direction yields average absolute gains for the 8B student across math and code tasks. This study preserves the defining proxy-logit equation, on-policy sampling, reverse-KL direction, and top-K teacher approximation. It diverges in several consequential ways:

| Dimension | Source study | This reproduction |
|---|---|---|
| Coverage | Four math and three code benchmarks | GSM8K only |
| Parameter update | Paper training stack | Rank-16 LoRA |
| Distributed optimization | Synchronized training | Eight independent workers, one-shot mean delta |
| Replication | Multi-task reported aggregates | Mostly one seed at 512 examples |
| Evaluation | Paper benchmark protocol | One stochastic completion, local exact-match parser |
| Contrast variants | Scale, RL, hint, multi-teacher | Scale contrast only |

The largest fidelity gap is worker aggregation. Averaging independently optimized LoRA endpoints is not algebraically equivalent to synchronizing gradients at each step. As trajectories become policy-dependent, workers follow increasingly different local policies; averaging their final deltas may attenuate or distort the proxy direction. LoRA also limits how the 8B model can express the transferred capability. Either deviation could explain why W2S shows a relative benefit without the paper's absolute gain.

## Threats to validity

1. **Single-seed model selection.** The best W2S LR was selected on the same 512 examples used for reporting. It is an exploratory best-observed result, not a held-out estimate.
2. **Stochastic evaluation.** One sample per problem adds generation variance. The seed check shows several-example variation even before optimization.
3. **No paired prediction artifact.** Aggregate logs preserve correctness counts but not a public per-example paired file, preventing McNemar or paired bootstrap analysis.
4. **Approximate distributed optimizer.** One-shot FedAvg-like adapter aggregation differs from synchronized OPD.
5. **Parameter-efficient tuning.** Rank-16 LoRA may underfit the intended capability direction or alter the optimal schedule.
6. **Limited benchmark scope.** GSM8K alone cannot test the paper's cross-domain or out-of-domain claims.
7. **Model revision drift.** Hugging Face repository names are pinned, but immutable model revision hashes were not recorded.
8. **Parser sensitivity.** Exact numeric extraction may disagree with a benchmark's canonical evaluator for unusual response formats.

## Conclusion

This controlled partial reproduction finds that W2S-OPD can be substantially better than direct distillation from the weak positive model under a matched aggressive learning rate: +11 correct answers out of 512. It also degrades less than direct OPD at the longest tested schedule. However, the best W2S result only ties the zero-shot 8B checkpoint, so the paper's central absolute-improvement claim is not reproduced here.

The most scientifically useful next experiment is a faithful synchronized-gradient implementation with the same three model checkpoints, followed by a pre-registered multi-seed comparison of zero-shot, direct OPD, and W2S-OPD at the two promising learning rates. That experiment is intentionally not launched in this package.

## Reproduction artifacts

- `results.csv`: immutable run/commit provenance and terminal metrics.
- `analysis.py`: self-contained marimo notebook with embedded evidence and plots.
- `configs/`: representative frozen configurations.
- `experiment.py`, `run.sh`, `.orx/k8s.yaml`: executable training and cluster contract.
- `CITATION.cff`, `LICENSE`, `NOTICE.md`, `CONTRIBUTING.md`: public repository metadata.
