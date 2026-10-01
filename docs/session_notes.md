# Session notes: toolchain pins, Colab landmines, small patches

Condensed from the project handoff. Everything here was learned by losing time to it.

## Pins

- Model: `CohereLabs/tiny-aya-l2-thinker`, revision `e9feb287c4a4755ac5ee4d085b8788a78811604f` (gated, CC-BY-NC). Every row records this sha.
- llama.cpp `5f436dddb440a288ee5611d7d1eca564a6aca9f4`; llama-cpp-python 0.3.35; Colab Python 3.13 with the CUDA-13 runtime (the CUDA-fix cell runs every session).
- Harness: `ayastaga/tinyaya-quant-safety`, package `tinyaya_eval`; `l2.py` must contain `prompt_seed` and `_LID.f.predict` (the version in `vendor/`). `vocabtrim.py` does select/apply/verify.
- Decoding for all trim comparisons: `TINYAYA_DECODING=greedy`; `TINYAYA_MAX_NEW=8192`, `TINYAYA_N_CTX=10240`.

## GGUF conversion: three deviations from the stock converter (report in any write-up)

1. Pre-tokenizer hash `9e079518a3ea5a3fe148bd4a570f4962db5db05a20e0cf74632de683e054d5b7` aliased to `tiny_aya` in `conversion/base.py:get_vocab_base_pre`. The tokenizer is identical to tiny-aya-global except that L2-Thinker's appends EOS on plain `encode()`; GGUF `add_eos_token` came out `false`.
2. Weights ship as `worker-000-00{0..3}.safetensors`; the converter only finds `model*.safetensors` and otherwise silently writes a tensor-less 12 MB GGUF. Symlink to `model-0000i-of-00004.safetensors` and rewrite `model.safetensors.index.json`.
3. `--no-lazy`: the checkpoint has `input_layernorm.bias` tensors and the Cohere converter's zero-bias check crashes in lazy mode. Open item: confirm every bias tensor is zero, because llama.cpp's Cohere2 graph is bias-free.

Sanity: f16 ≈ 6.7 GB, tensor count > 0, GPU load test prints `LOAD OK`.

## Colab landmines

- Interrupting a cell kills the background generation jobs it started. Watch progress from the Terminal instead: `wc -l /content/tinyaya-eval/runs_l2/*/*/*.jsonl`, `pgrep -fa "[t]inyaya_eval.l2 gen"`, `nvidia-smi`.
- Disconnecting or changing the runtime type wipes `/content` (GGUFs, builds, installs). "Restart session" keeps files. Drive survives. Rows are mirrored to Drive as they are written; GGUFs must be copied once by hand.
- Paired comparisons must come from one GPU type. Readouts assert `gpu` is unique. T4 rows (`tag=trim`) are excluded.
- T4 (15 GB) cannot hold two f16 processes; use the memory-aware runner (`BUDGET_GB` 14 on T4, 36 on A100). T4 is also ~4× slower (18 MGSM rows/hour vs ~80).
- Never start a runner while old `gen` processes live; duplicates append to the same files. The runner's `pgrep` must use the `[t]inyaya_eval` form or it matches its own shell.
- A runner without `time.sleep()` busy-spins a core and the notebook lags; print only on start/finish.
- Control symlinks (`l2thinker_ctrl-*.gguf`) live in `/content` and must be recreated every session, before any runner. A missing link fails the job within seconds with `missing; run prepare first`.
- Aya jobs that have nothing left to generate on resume ("0 done, 0 to go") do not exit: the streaming dataset iterator leaves non-daemon threads alive. Kill just that PID; the rows are complete. MGSM jobs exit normally. Fix in `l2.py`: end `main()` with `os._exit(0)` after `gen()` returns.
- `fasttext-wheel` has no Python-3.13 wheel; the fasttext cell builds 0.9.3 from source (~2 min).
- The `prefix` column in rows is the *variant* prefix (empty for `L2`), not the prompt. To replay a row, rebuild the prompt through the GGUF chat template and tokenize with `add_bos=True`; without BOS the model emits `<|END_RESPONSE|>` forever.
- `Llama.eval()` in llama-cpp-python 0.3.35 does not populate `scores`; use `Llama.generate(temp=0.0)` for greedy replays, or `create_chat_completion(logprobs=True, top_logprobs=k)` for logprobs.

## Code-switch loader for E3 (add to `load_items` in `l2.py`, before the final `else`)

```python
    elif dataset == "codeswitch":
        import json, os
        f = Path(os.environ.get("TINYAYA_CODESWITCH", "/content/tinyaya-eval/calib/codeswitch_prompts.jsonl"))
        rows = [json.loads(l) for l in f.read_text(encoding="utf-8").splitlines() if l.strip()]
        for lang in langs:
            for r in [r for r in rows if r["lang"] == lang][:n]:
                items.append({"id": r["id"], "lang": lang, "prompt": r["prompt"], "gold": None, "aux": {"category": r["category"]}})
```

Then `jobs = [{"model": m, "precision": "f16", "dataset": "codeswitch", "langs": ["en", "hi"], "n": 50} for m in (TRIM, CTRL)]` with `tag="trim_a100_cs"`.

## Standard session start (fresh runtime)

Part 0 → Restart session → Part 1 session cell → Drive mount → 1b restore → CUDA fix → fasttext → `l2.py` update (greedy) → `PRECISIONS` → Part 3 preflight (every block `PASS`) → runner + helpers → Part 10 → control links. Then whatever runner is next. Back up rows after each runner.
