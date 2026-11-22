import json
import random

import numpy as np

from .job import encode_row, load_rows

PREFERENCE = ("dpo", "ipo", "simpo", "orpo")


def _load(data):
    if isinstance(data, str):
        if data.endswith((".jsonl", ".json")) and _is_preference_file(data):
            with open(data, encoding="utf-8") as f:
                return [json.loads(line) for line in f if line.strip()] if data.endswith(".jsonl") else json.load(f)
        return load_rows(data)
    if hasattr(data, "to_list"):
        return data.to_list()
    return list(data)


def _is_preference_file(path: str) -> bool:
    with open(path, encoding="utf-8") as f:
        first = f.readline() if path.endswith(".jsonl") else f.read(4096)
    return '"chosen"' in first and '"rejected"' in first


def _pack(rows: list, tokenizer, seq: int, assistant_only: bool) -> tuple[np.ndarray, np.ndarray]:
    stream, targets = [], []
    for row in rows:
        if isinstance(row, dict) and "messages" in row:
            row = row["messages"]
        elif isinstance(row, dict) and "text" in row:
            row = row["text"]
        ids, labels = encode_row(tokenizer, row, assistant_only)
        stream.extend(ids)
        targets.extend(labels)
    count = len(stream) // seq
    if count == 0:
        raise ValueError(f"the data has {len(stream)} tokens, fewer than one sequence of {seq}")
    return np.array(stream[:count * seq]).reshape(count, seq), np.array(targets[:count * seq]).reshape(count, seq)


def _messages(prompt, reply) -> list[dict]:
    head = prompt if isinstance(prompt, list) else [{"role": "user", "content": prompt}]
    tail = reply if isinstance(reply, list) else [{"role": "assistant", "content": reply}]
    return head + tail


def _pair(tokenizer, row: dict, seq: int) -> tuple:
    out = []
    for side in ("chosen", "rejected"):
        ids, labels = encode_row(tokenizer, _messages(row["prompt"], row[side]))
        out.append((ids[:seq], labels[:seq]))
    return out


def _tell_length(run, total: int):
    try:
        run.pool.set_run(expected_steps=total)
    except Exception:
        pass


def train(run, data, method="sft", steps=None, epochs=1.0, batch_size=8, seq=None, max_grad_norm=1.0, beta=None, on_step=None, seed=0,
          assistant_only=True, gamma=0.5, lam=0.1, start_step=0, reference=None, **kw):
    from . import losses
    if kw:
        raise TypeError(f"train() got unknown settings: {', '.join(sorted(kw))}")
    if beta is None:
        beta = 2.0 if method == "simpo" else 0.1
    tok = run.tokenizer
    if tok is None:
        raise ValueError("this run has no tokenizer; pass tokenized batches to forward_backward instead")
    seq = seq or run.job.get("seq", 512)
    rows = _load(data)
    rng = random.Random(seed)
    history = []
    if method in ("sft", "cpt"):
        ids, labels = _pack(rows, tok, seq, assistant_only and method == "sft")
        per_epoch = max(1, len(ids) // batch_size)
        total = steps or max(1, int(per_epoch * epochs))
        _tell_length(run, total)
        order = list(range(len(ids)))
        for step in range(total):
            if step % per_epoch == 0:
                rng.shuffle(order)
            if step < start_step:
                continue
            pick = [order[(step * batch_size + i) % len(order)] for i in range(batch_size)]
            out = run.forward_backward(ids[pick], labels[pick])
            record = run.optim_step(lr=run.schedule(step, total), max_grad_norm=max_grad_norm, loss=out.loss)
            record.update(at=step, total=total)
            history.append(record)
            if on_step:
                on_step(record)
        return history
    if method not in PREFERENCE:
        raise ValueError(f"method must be one of sft, cpt, {', '.join(PREFERENCE)}")
    import torch
    pairs = [_pair(tok, row, seq) for row in rows]
    pad = run.job.get("pad_id", 0)

    def batch(index):
        chosen = [pairs[i][0] for i in index]
        rejected = [pairs[i][1] for i in index]
        both = chosen + rejected
        length = max(len(ids) for ids, _ in both)
        ids = np.full((len(both), length), pad, np.int64)
        lab = np.full((len(both), length), -100, np.int64)
        for r, (i, l) in enumerate(both):
            ids[r, :len(i)], lab[r, :len(l)] = i, l
        return ids, lab

    if method in ("dpo", "ipo") and reference is not None:
        reference = np.asarray(reference, np.float64)
        if reference.shape != (len(pairs), 2):
            raise ValueError(f"the saved reference log-probabilities are for {len(reference)} pairs, the data has {len(pairs)}")
    elif method in ("dpo", "ipo"):
        reference = np.zeros((len(pairs), 2), np.float64)
        for start in range(0, len(pairs), batch_size):
            index = list(range(start, min(len(pairs), start + batch_size)))
            ids, lab = batch(index)
            lp = run.forward(ids, lab, adapter=run.job.get("mode", "lora") == "full")
            mask = (lab[:, 1:] != -100)
            seqlp = (lp * mask).sum(-1)
            n = len(index)
            reference[index, 0], reference[index, 1] = seqlp[:n], seqlp[n:]
    run.reference = reference
    per_epoch = max(1, len(pairs) // batch_size)
    total = steps or max(1, int(per_epoch * epochs))
    _tell_length(run, total)
    order = list(range(len(pairs)))
    for step in range(total):
        if step % per_epoch == 0:
            rng.shuffle(order)
        if step < start_step:
            continue
        index = [order[(step * batch_size + i) % len(order)] for i in range(batch_size)]
        ids, lab = batch(index)
        n = len(index)
        ref = None if reference is None else torch.tensor(reference[index])
        stats = {}

        def loss_fn(lp, mask, _):
            total_c = losses.sequence_logprob(lp[:n], mask[:n], average=method in ("simpo", "orpo"))
            total_r = losses.sequence_logprob(lp[n:], mask[n:], average=method in ("simpo", "orpo"))
            if method == "dpo":
                value = losses.dpo(total_c, total_r, ref[:, 0], ref[:, 1], beta)
                stats["reward_margin"] = float(beta * ((total_c.detach() - ref[:, 0]) - (total_r.detach() - ref[:, 1])).mean())
            elif method == "ipo":
                value = losses.ipo(total_c, total_r, ref[:, 0], ref[:, 1], beta)
            elif method == "simpo":
                value = losses.simpo(total_c, total_r, beta, gamma)
            else:
                value = losses.orpo(total_c, total_r, -total_c, lam)
            stats["accuracy"] = float((total_c.detach() > total_r.detach()).float().mean()) if method in ("simpo", "orpo") else \
                float(((total_c.detach() - ref[:, 0]) > (total_r.detach() - ref[:, 1])).float().mean())
            return value
        out = run.forward_backward(ids, lab, loss=loss_fn)
        record = run.optim_step(lr=run.schedule(step, total), max_grad_norm=max_grad_norm, loss=out.loss)
        record.update(stats)
        record.update(at=step, total=total)
        run.log(**stats, loss=out.loss)
        history.append(record)
        if on_step:
            on_step(record)
    return history
