#!/usr/bin/env python3
"""Controlled smoke-scale reproduction of weak-to-strong on-policy distillation."""

from __future__ import annotations

import json
import os
import random
import re
import time
from pathlib import Path

import torch
from datasets import load_dataset
from huggingface_hub import snapshot_download
from transformers import AutoModelForCausalLM, AutoTokenizer


ROOT = Path(__file__).resolve().parent
CFG = json.loads((ROOT / "config.json").read_text())
LOCAL_RANK = int(os.environ["LOCAL_RANK"])
LOCAL_WORLD_SIZE = int(os.environ["LOCAL_WORLD_SIZE"])
POD_INDEX = int(os.environ.get("JOB_COMPLETION_INDEX", "0"))
GLOBAL_SHARD_RANK = POD_INDEX * LOCAL_WORLD_SIZE + LOCAL_RANK
GLOBAL_SHARDS = int(os.environ.get("ORX_NUM_PODS", "2")) * LOCAL_WORLD_SIZE
DEVICE = torch.device("cuda", 0)
CACHE = os.environ.get("HF_HOME", "/cache/hf")


def setup() -> None:
    torch.cuda.set_device(DEVICE)
    seed = int(CFG["seed"]) + GLOBAL_SHARD_RANK
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


def wait_for(paths: list[Path], timeout: int = 7200) -> None:
    deadline = time.time() + timeout
    while time.time() < deadline:
        if all(path.exists() for path in paths):
            return
        time.sleep(2)
    missing = [str(path) for path in paths if not path.exists()]
    raise TimeoutError(f"Timed out waiting for files: {missing}")


def atomic_torch_save(payload, path: Path) -> None:
    """Publish a torch checkpoint only after all bytes have been written."""
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def atomic_write_text(path: Path, text: str) -> None:
    """Publish small coordination files atomically on the shared volume."""
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(text)
    os.replace(temporary, path)


def download(repo_id: str) -> None:
    ready = Path("/cache") / ("ready-" + repo_id.replace("/", "--"))
    if LOCAL_RANK == 0:
        print(f"DOWNLOAD_START model={repo_id}", flush=True)
        snapshot_download(repo_id=repo_id, cache_dir=CACHE, token=os.environ.get("HF_TOKEN"))
        ready.touch()
        print(f"DOWNLOAD_DONE model={repo_id}", flush=True)
    else:
        wait_for([ready])


def load_tokenizer():
    return AutoTokenizer.from_pretrained(
        CFG["student_model"], cache_dir=CACHE, token=os.environ.get("HF_TOKEN")
    )


def load_data():
    ready = Path("/cache/dataset-ready")
    if LOCAL_RANK == 0:
        data = load_dataset(CFG["dataset"], CFG["dataset_config"], cache_dir=CACHE)
        ready.touch()
        return data
    wait_for([ready])
    return load_dataset(CFG["dataset"], CFG["dataset_config"], cache_dir=CACHE)


def load_model(repo_id: str, trainable: bool = False):
    model = AutoModelForCausalLM.from_pretrained(
        repo_id,
        cache_dir=CACHE,
        token=os.environ.get("HF_TOKEN"),
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",
        low_cpu_mem_usage=True,
    ).to(DEVICE)
    model.config.use_cache = not trainable
    return model


def prompt_ids(tokenizer, question: str) -> torch.Tensor:
    messages = [
        {
            "role": "user",
            "content": question
            + "\nPlease reason step by step, and put your final answer within \\boxed{}.",
        }
    ]
    return tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        enable_thinking=False,
        tokenize=True,
        return_tensors="pt",
    ).to(DEVICE)


def generate(model, tokenizer, question: str, seed: int) -> tuple[torch.Tensor, int, str]:
    ids = prompt_ids(tokenizer, question)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    old_cache = model.config.use_cache
    model.config.use_cache = True
    with torch.inference_mode():
        full = model.generate(
            ids,
            do_sample=True,
            temperature=float(CFG["temperature"]),
            top_p=float(CFG["top_p"]),
            max_new_tokens=int(CFG["max_new_tokens"]),
            pad_token_id=tokenizer.eos_token_id,
        )
    model.config.use_cache = old_cache
    text = tokenizer.decode(full[0, ids.shape[1] :], skip_special_tokens=True)
    return full, ids.shape[1], text


def normalize_number(text: str) -> str | None:
    text = text.replace(",", "").replace("$", "").strip()
    try:
        value = float(text)
    except ValueError:
        return None
    if value.is_integer():
        return str(int(value))
    return f"{value:.8g}"


def predicted_answer(text: str) -> str | None:
    boxed = re.findall(r"\\boxed\s*\{([^{}]+)\}", text)
    if boxed:
        candidate = normalize_number(boxed[-1])
        if candidate is not None:
            return candidate
    numbers = re.findall(r"[-+]?\d[\d,]*(?:\.\d+)?", text)
    return normalize_number(numbers[-1]) if numbers else None


def gold_answer(answer: str) -> str | None:
    return normalize_number(answer.split("####")[-1])


def evaluate(model, tokenizer, test_rows, shard_rank: int, shard_count: int) -> list[dict]:
    model.eval()
    records = []
    for idx in range(shard_rank, int(CFG["eval_examples"]), shard_count):
        row = test_rows[idx]
        _, _, text = generate(model, tokenizer, row["question"], int(CFG["seed"]) + idx)
        pred = predicted_answer(text)
        gold = gold_answer(row["answer"])
        records.append(
            {
                "index": idx,
                "prediction": pred,
                "gold": gold,
                "correct": pred == gold,
                "response_chars": len(text),
            }
        )
        print(
            f"EVAL_RECORD pod={POD_INDEX} rank={LOCAL_RANK} index={idx} "
            f"correct={pred == gold} pred={pred} gold={gold}",
            flush=True,
        )
    return records


def write_records(records: list[dict], prefix: str) -> Path:
    path = Path("/shared") / f"{prefix}-pod-{POD_INDEX}-rank-{LOCAL_RANK}.json"
    path.write_text(json.dumps(records))
    return path


def metric_block(records: list[dict], mode: str, method: str) -> dict:
    records = sorted(records, key=lambda x: x["index"])
    correct = sum(int(row["correct"]) for row in records)
    return {
        "mode": mode,
        "method": method,
        "student_model": CFG["student_model"],
        "positive_model": CFG["positive_model"],
        "negative_model": CFG["negative_model"],
        "eval_examples": len(records),
        "correct": correct,
        "accuracy": correct / max(1, len(records)),
        "seed": CFG["seed"],
    }


def run_baseline(tokenizer, test_rows) -> None:
    download(CFG["student_model"])
    model = load_model(CFG["student_model"])
    records = evaluate(model, tokenizer, test_rows, GLOBAL_SHARD_RANK, GLOBAL_SHARDS)
    out = write_records(records, "baseline")
    print(f"SHARD_RESULT_WRITTEN path={out} records={len(records)}", flush=True)
    if POD_INDEX == 0 and LOCAL_RANK == 0:
        expected = [
            Path("/shared") / f"baseline-pod-{pod}-rank-{rank}.json"
            for pod in range(int(os.environ.get("ORX_NUM_PODS", "2")))
            for rank in range(LOCAL_WORLD_SIZE)
        ]
        wait_for(expected)
        all_records = [item for path in expected for item in json.loads(path.read_text())]
        metrics = metric_block(all_records, "baseline", "none")
        print("ORX_METRICS " + json.dumps(metrics, sort_keys=True), flush=True)


def response_logits(model, full_ids: torch.Tensor, prompt_len: int) -> torch.Tensor:
    output = model(input_ids=full_ids, attention_mask=torch.ones_like(full_ids), use_cache=False)
    sliced = output.logits[:, prompt_len - 1 : full_ids.shape[1] - 1].detach().clone()
    del output
    return sliced


def train_distillation(tokenizer, train_rows, test_rows) -> None:
    from peft import LoraConfig, get_peft_model, get_peft_model_state_dict, set_peft_model_state_dict

    method = CFG["method"]
    required = [CFG["student_model"], CFG["positive_model"]]
    if method == "w2s":
        required.append(CFG["negative_model"])
    for repo_id in required:
        download(repo_id)

    student = load_model(CFG["student_model"], trainable=True)
    lora = LoraConfig(
        r=int(CFG["lora_rank"]),
        lora_alpha=int(CFG["lora_alpha"]),
        lora_dropout=0.0,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )
    student = get_peft_model(student, lora)
    student.enable_input_require_grads()
    student.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})

    positive = load_model(CFG["positive_model"])
    positive.eval()
    anchor = negative = None
    if method == "w2s":
        anchor = load_model(CFG["student_model"])
        negative = load_model(CFG["negative_model"])
        anchor.eval()
        negative.eval()

    optimizer = torch.optim.AdamW(
        [p for p in student.parameters() if p.requires_grad],
        lr=float(CFG["learning_rate"]),
        betas=(0.9, 0.95),
        weight_decay=0.0,
    )
    if LOCAL_RANK == 0:
        trainable = sum(p.numel() for p in student.parameters() if p.requires_grad)
        print(f"TRAIN_START method={method} trainable_parameters={trainable}", flush=True)

    losses = []
    tokens_seen = 0
    rollouts_per_step = int(CFG.get("rollouts_per_step", 1))
    for step in range(int(CFG["train_steps"])):
        optimizer.zero_grad(set_to_none=True)
        rollout_losses = []
        step_tokens = 0
        for rollout in range(rollouts_per_step):
            sample_index = (step * rollouts_per_step + rollout) * LOCAL_WORLD_SIZE + LOCAL_RANK
            row_idx = sample_index % len(train_rows)
            student.eval()
            full_ids, prompt_len, _ = generate(
                student,
                tokenizer,
                train_rows[row_idx]["question"],
                int(CFG["seed"]) + 100000 + sample_index,
            )
            response_tokens = full_ids.shape[1] - prompt_len
            if response_tokens == 0:
                raise RuntimeError("Student generated an empty response")

            with torch.no_grad():
                if method == "w2s":
                    proxy = response_logits(anchor, full_ids, prompt_len)
                    pos = response_logits(positive, full_ids, prompt_len)
                    proxy.add_(pos, alpha=float(CFG["alpha"]))
                    del pos
                    neg = response_logits(negative, full_ids, prompt_len)
                    proxy.add_(neg, alpha=-float(CFG["alpha"]))
                    del neg
                    teacher_logits = proxy
                elif method == "opd":
                    teacher_logits = response_logits(positive, full_ids, prompt_len)
                else:
                    raise ValueError(f"Unknown training method: {method}")
                top_values, top_indices = torch.topk(
                    teacher_logits, k=int(CFG["teacher_top_k"]), dim=-1
                )
                teacher_log_probs = torch.log_softmax(top_values.float(), dim=-1)
                del teacher_logits, top_values

            student.train()
            output = student(
                input_ids=full_ids, attention_mask=torch.ones_like(full_ids), use_cache=False
            )
            student_logits = output.logits[:, prompt_len - 1 : full_ids.shape[1] - 1]
            student_top = torch.gather(student_logits, -1, top_indices)
            student_log_probs = torch.log_softmax(student_top.float(), dim=-1)
            student_probs = student_log_probs.exp()
            loss = (student_probs * (student_log_probs - teacher_log_probs)).sum(dim=-1).mean()
            (loss / rollouts_per_step).backward()
            rollout_losses.append(float(loss.detach()))
            step_tokens += response_tokens
            del (
                full_ids,
                output,
                student_logits,
                student_top,
                student_log_probs,
                student_probs,
                teacher_log_probs,
                top_indices,
                loss,
            )

        grad_norm = torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
        optimizer.step()
        step_loss = sum(rollout_losses) / len(rollout_losses)
        losses.append(step_loss)
        tokens_seen += step_tokens
        if LOCAL_RANK == 0:
            print(
                f"TRAIN_STEP step={step + 1}/{CFG['train_steps']} loss={step_loss:.6f} "
                f"grad_norm={float(grad_norm):.6f} rollouts_rank0={rollouts_per_step} "
                f"response_tokens_rank0={step_tokens} rank0_tokens_seen={tokens_seen}",
                flush=True,
            )

    adapter_dir = Path("/shared/adapters")
    adapter_dir.mkdir(parents=True, exist_ok=True)
    adapter_path = adapter_dir / f"{method}-rank-{LOCAL_RANK}.pt"
    atomic_torch_save(
        {key: value.detach().cpu() for key, value in get_peft_model_state_dict(student).items()},
        adapter_path,
    )
    stats_path = adapter_dir / f"{method}-stats-{LOCAL_RANK}.json"
    atomic_write_text(
        stats_path,
        json.dumps({"mean_loss": sum(losses) / len(losses), "final_loss": losses[-1]}),
    )
    adapter_paths = [adapter_dir / f"{method}-rank-{rank}.pt" for rank in range(LOCAL_WORLD_SIZE)]
    stats_paths = [
        adapter_dir / f"{method}-stats-{rank}.json" for rank in range(LOCAL_WORLD_SIZE)
    ]
    wait_for(adapter_paths + stats_paths)
    averaged_path = adapter_dir / f"{method}-averaged.pt"
    if LOCAL_RANK == 0:
        states = [torch.load(path, map_location="cpu", weights_only=True) for path in adapter_paths]
        averaged = {}
        for key in states[0]:
            averaged[key] = torch.stack([state[key].float() for state in states]).mean(0).to(
                states[0][key].dtype
            )
        atomic_torch_save(averaged, averaged_path)
        print(f"ADAPTER_AVERAGED workers={LOCAL_WORLD_SIZE} path={averaged_path}", flush=True)
    wait_for([averaged_path])
    averaged = torch.load(averaged_path, map_location="cpu", weights_only=True)
    set_peft_model_state_dict(student, averaged)

    records = evaluate(student, tokenizer, test_rows, LOCAL_RANK, LOCAL_WORLD_SIZE)
    out = write_records(records, method)
    print(f"SHARD_RESULT_WRITTEN path={out} records={len(records)}", flush=True)
    if LOCAL_RANK == 0:
        expected = [
            Path("/shared") / f"{method}-pod-{POD_INDEX}-rank-{rank}.json"
            for rank in range(LOCAL_WORLD_SIZE)
        ]
        wait_for(expected)
        all_records = [item for path in expected for item in json.loads(path.read_text())]
        stats = [json.loads(path.read_text()) for path in stats_paths]
        metrics = metric_block(all_records, "distillation", method)
        metrics.update(
            {
                "alpha": CFG["alpha"] if method == "w2s" else None,
                "teacher_top_k": CFG["teacher_top_k"],
                "train_steps": CFG["train_steps"],
                "rollouts_per_step": rollouts_per_step,
                "parallel_workers": LOCAL_WORLD_SIZE,
                "effective_rollouts": (
                    int(CFG["train_steps"]) * LOCAL_WORLD_SIZE * rollouts_per_step
                ),
                "adapter_aggregation": "mean_of_independent_worker_deltas",
                "mean_worker_loss": sum(row["mean_loss"] for row in stats) / len(stats),
                "mean_final_worker_loss": sum(row["final_loss"] for row in stats) / len(stats),
            }
        )
        print("ORX_METRICS " + json.dumps(metrics, sort_keys=True), flush=True)


def main() -> None:
    setup()
    if LOCAL_RANK == 0:
        print(
            f"PROCESS_GROUP pod={POD_INDEX} local_world={LOCAL_WORLD_SIZE} "
            f"global_shard_rank_base={POD_INDEX * LOCAL_WORLD_SIZE}",
            flush=True,
        )
    download(CFG["student_model"])
    tokenizer = load_tokenizer()
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    data = load_data()
    test_rows = data["test"]
    if CFG["mode"] == "baseline":
        run_baseline(tokenizer, test_rows)
    else:
        train_distillation(tokenizer, data["train"], test_rows)


if __name__ == "__main__":
    main()
