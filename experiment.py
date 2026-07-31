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
import torch.distributed as dist
from datasets import load_dataset
from huggingface_hub import snapshot_download
from torch.nn.parallel import DistributedDataParallel as DDP
from transformers import AutoModelForCausalLM, AutoTokenizer


ROOT = Path(__file__).resolve().parent
CFG = json.loads((ROOT / "config.json").read_text())
LOCAL_RANK = int(os.environ["LOCAL_RANK"])
LOCAL_WORLD_SIZE = int(os.environ["LOCAL_WORLD_SIZE"])
POD_INDEX = int(os.environ.get("JOB_COMPLETION_INDEX", "0"))
GLOBAL_SHARD_RANK = POD_INDEX * LOCAL_WORLD_SIZE + LOCAL_RANK
GLOBAL_SHARDS = int(os.environ.get("ORX_NUM_PODS", "2")) * LOCAL_WORLD_SIZE
DEVICE = torch.device("cuda", LOCAL_RANK)
CACHE = os.environ.get("HF_HOME", "/cache/hf")


def setup() -> None:
    torch.cuda.set_device(DEVICE)
    dist.init_process_group("nccl", device_id=DEVICE)
    seed = int(CFG["seed"]) + GLOBAL_SHARD_RANK
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


def barrier() -> None:
    dist.barrier(device_ids=[LOCAL_RANK])


def download(repo_id: str) -> None:
    if LOCAL_RANK == 0:
        print(f"DOWNLOAD_START model={repo_id}", flush=True)
        snapshot_download(repo_id=repo_id, cache_dir=CACHE, token=os.environ.get("HF_TOKEN"))
        print(f"DOWNLOAD_DONE model={repo_id}", flush=True)
    barrier()


def load_tokenizer():
    return AutoTokenizer.from_pretrained(
        CFG["student_model"], cache_dir=CACHE, token=os.environ.get("HF_TOKEN")
    )


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
    generator = torch.Generator(device=DEVICE).manual_seed(seed)
    base = model.module if isinstance(model, DDP) else model
    old_cache = base.config.use_cache
    base.config.use_cache = True
    with torch.inference_mode():
        full = base.generate(
            ids,
            do_sample=True,
            temperature=float(CFG["temperature"]),
            top_p=float(CFG["top_p"]),
            max_new_tokens=int(CFG["max_new_tokens"]),
            pad_token_id=tokenizer.eos_token_id,
            generator=generator,
        )
    base.config.use_cache = old_cache
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


def gather_local(records: list[dict]) -> list[dict] | None:
    gathered = [None for _ in range(LOCAL_WORLD_SIZE)] if LOCAL_RANK == 0 else None
    dist.gather_object(records, gathered, dst=0)
    if LOCAL_RANK != 0:
        return None
    return [item for rank_records in gathered for item in rank_records]


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
    local_records = gather_local(records)
    if LOCAL_RANK == 0:
        shared = Path("/shared")
        shared.mkdir(parents=True, exist_ok=True)
        out = shared / f"baseline-pod-{POD_INDEX}.json"
        out.write_text(json.dumps(local_records))
        print(f"POD_RESULT_WRITTEN path={out} records={len(local_records)}", flush=True)
        if POD_INDEX == 0:
            other = shared / "baseline-pod-1.json"
            deadline = time.time() + 7200
            while not other.exists() and time.time() < deadline:
                time.sleep(5)
            if not other.exists():
                raise TimeoutError("Timed out waiting for second baseline shard")
            all_records = local_records + json.loads(other.read_text())
            metrics = metric_block(all_records, "baseline", "none")
            print("ORX_METRICS " + json.dumps(metrics, sort_keys=True), flush=True)
    barrier()


def response_logits(model, full_ids: torch.Tensor, prompt_len: int) -> torch.Tensor:
    output = model(input_ids=full_ids, attention_mask=torch.ones_like(full_ids), use_cache=False)
    sliced = output.logits[:, prompt_len - 1 : full_ids.shape[1] - 1].detach().clone()
    del output
    return sliced


def train_distillation(tokenizer, train_rows, test_rows) -> None:
    from peft import LoraConfig, get_peft_model

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
    student = DDP(student, device_ids=[LOCAL_RANK], find_unused_parameters=False)

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
    for step in range(int(CFG["train_steps"])):
        row_idx = (step * LOCAL_WORLD_SIZE + LOCAL_RANK) % len(train_rows)
        student.eval()
        full_ids, prompt_len, _ = generate(
            student,
            tokenizer,
            train_rows[row_idx]["question"],
            int(CFG["seed"]) + 100000 + step * LOCAL_WORLD_SIZE + LOCAL_RANK,
        )
        response_tokens = full_ids.shape[1] - prompt_len
        if response_tokens == 0:
            raise RuntimeError("Student generated an empty response")

        with torch.inference_mode():
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
        optimizer.zero_grad(set_to_none=True)
        output = student(input_ids=full_ids, attention_mask=torch.ones_like(full_ids), use_cache=False)
        student_logits = output.logits[:, prompt_len - 1 : full_ids.shape[1] - 1]
        student_top = torch.gather(student_logits, -1, top_indices)
        student_log_probs = torch.log_softmax(student_top.float(), dim=-1)
        student_probs = student_log_probs.exp()
        loss = (student_probs * (student_log_probs - teacher_log_probs)).sum(dim=-1).mean()
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
        optimizer.step()
        losses.append(float(loss.detach()))
        tokens_seen += response_tokens * LOCAL_WORLD_SIZE
        if LOCAL_RANK == 0:
            print(
                f"TRAIN_STEP step={step + 1}/{CFG['train_steps']} loss={loss.item():.6f} "
                f"grad_norm={float(grad_norm):.6f} response_tokens_rank0={response_tokens} "
                f"global_tokens_seen={tokens_seen}",
                flush=True,
            )
        del output, student_logits, student_top, student_log_probs, student_probs, loss

    records = evaluate(student, tokenizer, test_rows, LOCAL_RANK, LOCAL_WORLD_SIZE)
    all_records = gather_local(records)
    if LOCAL_RANK == 0:
        metrics = metric_block(all_records, "distillation", method)
        metrics.update(
            {
                "alpha": CFG["alpha"] if method == "w2s" else None,
                "teacher_top_k": CFG["teacher_top_k"],
                "train_steps": CFG["train_steps"],
                "global_batch_size": LOCAL_WORLD_SIZE,
                "mean_rank0_loss": sum(losses) / len(losses),
                "final_rank0_loss": losses[-1],
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
    data = load_dataset(CFG["dataset"], CFG["dataset_config"], cache_dir=CACHE)
    test_rows = data["test"]
    if CFG["mode"] == "baseline":
        run_baseline(tokenizer, test_rows)
    else:
        train_distillation(tokenizer, data["train"], test_rows)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
