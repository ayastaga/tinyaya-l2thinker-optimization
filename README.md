https://github.com/user-attachments/assets/4b90b456-a3f7-4a0a-9ba8-24d80a0e7eaf

# l2thinker-trim

Regional vocabulary trimming of **Tiny Aya L2-Thinker** (Cohere Labs, 3.35B, multilingual reasoning) for on-device use, with a lossless-ness protocol borrowed from speculative decoding: the trimmed model must reproduce the untrimmed model's greedy output.

**Result.** A South Asia trim (262,144 → 138,105 tokens) makes the 4-bit file 9.995% smaller and CPU decoding 8.1% faster, with output character-identical to the untrimmed model on 1,442 of 1,450 paired generations across Bengali, Hindi, Gujarati, Tamil, Telugu and English, and every quality metric unchanged to the decimal. Every one of the 8 differences involves a token that was removed. The same trim applied outside its region halves Swahili math accuracy (70.4% → 35.6%): trims are region-specific.

| | Untrimmed | South Asia trim | Δ |
| --- | --- | --- | --- |
| Vocabulary | 262,144 | 138,105 | −47.3% |
| q4_k_m file | 2.144 GB | 1.930 GB | −9.995% |
| CPU decode, tg100 | 16.1 tok/s | 17.4 tok/s | +8.1% |
| Paired greedy identity, in-region (f16) | — | 1,442 / 1,450 | 99.45% |
| MGSM bn accuracy, f16 / q4_k_m | 64.0 / 55.6 | 64.0 / 55.6 | 0.0 |
| MGSM te accuracy, f16 | 60.8 | 60.8 | 0.0 |
| Control-vs-control identity (noise floor) | 900 / 900 | — | 0 |
| **MGSM sw accuracy (out of region), f16** | **70.4** | **35.6** | **−34.8, p < 10⁻¹⁸** |

Full results, pre-registered criteria, divergence analysis, external checks and next steps: [`docs/writeup.md`](docs/writeup.md).

## Reproduce

**Verify every number, no GPU (1 minute).** The generation rows are shipped. `notebooks/01_verify_from_rows.ipynb` or:

```bash
pip install pandas numpy
python scripts/readout.py          # all tables + pre-registered checks; writes results/tables/*.csv
```

**Rebuild everything on Colab (A100, ~10 GPU hours).** `notebooks/02_full_pipeline.ipynb`: GGUF conversion with the three documented converter fixes, selection corpus, keep file, trimmed GGUF, tokenization drift, size and speed, paired generation vs a same-session control, readout, divergence replay, KV-cache probe. Requires a Hugging Face token with the L2-Thinker licence accepted, Google Drive for the mirror, and the harness repo [`ayastaga/tinyaya-quant-safety`](https://github.com/ayastaga/tinyaya-quant-safety) (`tinyaya_eval/l2.py` at the version in `vendor/`, plus `vocabtrim.py`).

The shipped keep file (`keep/l2thinker_trim_keep.json`, sha `bfadfaf070959033…`) rebuilds the exact trimmed GGUF evaluated here in ~15 minutes from the f16 base, so the corpus step can be skipped.

## Method in one paragraph

Keep every token that appears in a ~650 MB regional corpus (FineWeb-2 and Aya Collection for bn, hi, mr, gu, pa, ta, te, ne, ur; FineWeb-Edu English; code; L2-Thinker's released reasoning traces), every token written entirely in the region's scripts, all special, byte and no-letter tokens, and the BPE-merge ancestors of everything kept. Drop the rest from the tied embedding and the merge table; touch nothing else. Logits over kept tokens are then identical by construction, so the model's output can only change where the original model would have emitted a removed token, or where a word in the prompt or earlier output was built from a removed piece. Evaluate with greedy decoding against the untrimmed model re-run in the same session on the same GPU type (the "control"), paired by prompt: identity rate, MGSM accuracy with an exact McNemar test, truncation, wrong-language rate and a 4-gram loop score, with thresholds fixed before each run. Replay every divergence through the base model to name the removed token responsible. Regenerate the control once more to measure the runtime's own noise floor (it was zero).

## What the divergences look like

All eight in-region differences are the original model touching a language outside the region: Danish pastry names in an English brainstorm, a Portuguese and a Malay word, and stray Chinese or Korean tokens inside Bengali and Hindi text. In two of those the trimmed model's output is arguably *better* (it writes "पाम बीच" where the original wrote "पाम贝ल"). On a language outside the region (Swahili), ~14% of every prompt's tokens are gone, the model reads the prompt through different pieces from the first token, and reasoning collapses into loops and truncation.

## Layout

```
README.md                  this file
docs/writeup.md            results, criteria, divergence analysis, external checks, next steps
docs/session_notes.md      Colab landmines and the toolchain pins (from the project handoff)
docs/session_notebook.ipynb  the notebook as actually run, with outputs (record; not for re-running)
notebooks/01_verify_from_rows.ipynb   recompute everything from shipped rows (no GPU)
notebooks/02_full_pipeline.ipynb      rebuild from scratch on Colab A100
scripts/readout.py         the readout as a CLI
keep/l2thinker_trim_keep.json         the keep set (138,105 token ids + stats)
results/rows/<model>/<tag>/*.jsonl    every generation row: prompt id, trace, answer, metrics, GPU, revision
results/tables/*.csv       the tables readout.py produces
data/codeswitch_prompts.jsonl         50 en/hi prompts built to hit removed tokens (E3, not yet run)
vendor/l2.py               the harness module the notebooks depend on, pinned
```

Row files carry `decoding`, `revision`, `gpu`, `max_new`, `n_ctx`, `n_prompt/n_trace/n_answer`, `finish`, `truncated`, `think_closed`, `doomloop`, and `correct` for MGSM. Tags: `trim_a100` round 1 · `trim_a100_rerun` E0 control second pass · `trim_a100_r2` E1/E2 · `trim_a100_r2x` exploratory MGSM sw. The earlier T4 rows (`tag=trim`) are excluded; paired comparisons must come from one GPU type.

## Licences

Code (`scripts/`, `notebooks/`, `vendor/l2.py`): MIT. The generation rows in `results/` are outputs of `CohereLabs/tiny-aya-l2-thinker`, released under CC-BY-NC, and are provided for research only under the same terms; see `DATA_LICENSE.md`. Model weights and GGUFs are not redistributed; accept the licence on the model page and convert locally.

## Citation

Sharma, A. (2026). *Regional vocabulary trimming of Tiny Aya L2-Thinker with greedy-identity verification.* github.com/ayastaga/l2thinker-trim.

Prior work this builds on: Ushio, Zhou, Camacho-Collados (2023), *Efficient Multilingual Language Model Compression through Vocabulary Trimming*, Findings of EMNLP · Bogoychev, Chen, Haddow, Birch (2024), *The Ups and Downs of LLM Inference with Vocabulary Trimming by Language Heuristics* · Mofakhami et al. (2026), *Tiny Aya L2-Thinker*, arXiv 2609.10445 · Leviathan, Kalman, Matias (2023), *Fast Inference from Transformers via Speculative Decoding*, ICML (the identity criterion) · He and Thinking Machines Lab (2025), *Defeating Nondeterminism in LLM Inference* (the noise-floor control).
