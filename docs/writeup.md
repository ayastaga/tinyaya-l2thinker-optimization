# L2-Thinker vocabulary trim: results, methodology and next steps

Agastya Sharma (`ayastaga`) · last updated 2026-10-01

## Summary

Trimming L2-Thinker's vocabulary from 262,144 to 138,105 tokens (52.7% kept) for South Asia gives a 4-bit file 9.995% smaller, 8.1% faster CPU decoding, and output that is character-identical to the untrimmed model on 1,442 of 1,450 paired greedy generations across six South Asian languages and English (Bengali, Hindi, Gujarati, Tamil, Telugu, Urdu-script corpus coverage; Bengali and Telugu MGSM, Bengali/English/Hindi/Gujarati/Tamil Aya). MGSM accuracy, truncation, wrong-language rate and loop score are unchanged to the decimal at both f16 and q4_k_m. Every one of the 8 in-region divergences involves a token that was removed; the 5 from round 1 are replay-verified (3 at the point of divergence, 2 upstream via respelling of a Danish word). A second full pass of the 900 round-1 control rows reproduced byte-for-byte, so the identity test has a zero noise floor and every divergence is attributable to the trim. One pre-registered criterion is missed by 0.005 points: the size target was 10%.

Outside its region the same trim is harmful. On Swahili MGSM, accuracy falls from 70.4% to 35.6% (McNemar p < 0.001, 107 discordant pairs), truncation rises from 5.6% to 54.8% and the loop score from 0.29 to 0.62. Trims are region-specific: one base model, one trimmed file per region.

Separately, one KV-cache setting (`swa_full=False`) cuts memory at 32K context from 5.38 GB to 3.88 GB with 100% identical output. A lossy setting (flash attention + 8-bit KV) reaches 2.92 GB but changes numerics and still needs a quality evaluation.

## Setup

Model: `CohereLabs/tiny-aya-l2-thinker` (3.35B, Cohere2: 36 layers, GQA 16/4, head 128, 3:1 sliding-window 4096 / global NoPE, tied 262,144-token embedding, 32K context, reasoning mode), revision `e9feb287c4a4755ac5ee4d085b8788a78811604f`, CC-BY-NC research licence. Toolchain: llama.cpp `5f436dddb440a288ee5611d7d1eca564a6aca9f4`, llama-cpp-python 0.3.35, Colab A100-SXM4-40GB, Python 3.13.

Trim rule (v2): keep every token used in the selection corpus, all special and control tokens, the 256 byte tokens, every token with no letters, every token entirely in the region's scripts (Bengali, Devanagari, Gujarati, Gurmukhi, Tamil, Telugu, Arabic), plus BPE-ancestor closure. The four think delimiters are asserted kept. Kept embedding rows are untouched, so logits over kept tokens are identical by construction; output can only differ where the untrimmed model's next token was removed, or where a word in the prompt or earlier output was built from a removed piece and is now tokenized differently.

Selection corpus (~650 MB): FineWeb-2 (40 MB) and Aya Collection (10 MB) per region language (bn, hi, mr, gu, pa, ta, te, ne, ur); FineWeb-Edu English (80 MB); code (40 MB); L2-Thinker's released translated reasoning traces, train split (20 MB per language). Held-out: the traces' test split, MGSM bn/en prompts, and separate web slices (3 MB each). Evaluation prompts were never in the corpus.

Comparison protocol: greedy decoding (identity is the test; with sampling, a smaller vocabulary maps the same random draw to a different token even for identical probabilities), same session, same GPU type, trimmed vs a control that is the untrimmed GGUF under a second name. Prompts go through the GGUF's own chat template, which prepends Cohere's default system preamble and appends ` /think`. q4_k_m was built without an imatrix, so per-row quantization of kept rows is also identical between trimmed and control. Max new tokens 8,192; n_ctx 10,240.

Three deviations from the stock converter were needed: a pre-tokenizer hash alias (`9e079518…` → `tiny_aya`), safetensors shard renaming (`worker-*` → `model-*`), and `--no-lazy`. See the pipeline notebook.

## Pre-registered results (round 1: Bengali, English)

| Criterion | Threshold | Result | Verdict |
| --- | --- | --- | --- |
| q4_k_m file size | ≥ 10% smaller | 2.144 → 1.930 GB, −9.995% (f16 −7.7%, q4_0 −10.5%) | Miss by 0.005 pt |
| CPU decode speed, tg100 | ≥ 5% faster | 16.1 → 17.4 tok/s, +8.1% | Pass |
| Held-out fertility increase | < 1% per in-region language, incl. reasoning traces | max +0.03% (web_ur); traces ≤ +0.01%; MGSM bn 0.00% | Pass |
| f16 greedy identity vs control | ≥ 98% | Aya bn 100.0, Aya en 98.0, MGSM bn 99.6 (q4_k_m: 100.0, 98.0, 100.0) | Pass |
| Truncation, L2 rate, MGSM accuracy | within ±1 pt | all deltas 0.0 | Pass |
| Doomloop score | within ±0.01 | deltas 0.000 to −0.004 | Pass |
| MGSM accuracy paired test | McNemar n.s. | 64.0 vs 64.0 (f16), 55.6 vs 55.6 (q4_k_m); p = 1.0, zero discordant pairs | Pass |
| Divergences explained by a removed token | ≈ all | 5 of 5, replay-verified | Pass, rule refined |

Sample sizes: MGSM bn 250, Aya bn 100, Aya en 100, at f16 and q4_k_m, trimmed and control: 900 paired rows per side. Keep set: 138,105 of 262,144 tokens (125,014 seen in corpus, the rest by script and closure rules); keep file sha `bfadfaf070959033`.

Side finding, not this project's question but the largest effect in the table: quantizing to q4_k_m drops Bengali MGSM accuracy from 64.0% to 55.6% under greedy decoding, identically for trimmed and control. Truncation rises 16.8 → 22.4%, accounting for about two thirds of the loss.

## The five round-1 divergences

Each divergent pair was replayed greedily with the untrimmed model (prompt rebuilt through the GGUF chat template with BOS) to recover its real token sequence. Retokenizing the text after the fact is approximate and mislabeled one case.

| Row | Precision | Divergence at | Mechanism | Removed token |
| --- | --- | --- | --- | --- |
| aya:en:75 | f16 | char 310 of the trace, a list of Danish pastry names | Upstream respelling: control used `ød` in "Rugbrød"; trimmed model spelled it with kept pieces, so its state differed before the text did | `ød` (16413) |
| aya:en:75 | q4_k_m | char 339, same prompt | Same; drifted apart at a later low-confidence spot | `ød` (16413) |
| aya:en:81 | q4_k_m | char 668, Portuguese "tronco de árvore" | Removed token at the divergence | ` árv` (213325) |
| aya:en:81 | f16 | answer char 13 (trace identical, 986 chars) | Removed token at the divergence, in a Malay answer | ` banyak` (7562) |
| mgsm:bn:53 | f16 | char 903, Bengali arithmetic | Control emitted a stray Korean token mid-Bengali; the trimmed model could not and continued in Bengali. Both answers correct | `보다` (18565) |

Refined rule as tested: a removed token occurs in the prompt or in the control's generated sequence at or before the first differing character. All five meet it by replay. Per-token logprobs differ slightly between the two models (−0.3574 vs −0.3565 on the same token) even where raw logits agree, because softmax renormalizes over the smaller vocabulary; this does not affect greedy output.

The mechanism matters more than the count: the trim changes internal state whenever the model reads or writes a word built from removed pieces, even when the visible text stays the same. On English prompts that brush against Danish or Portuguese it cost one brainstorm list. On a language whose words are mostly removed pieces it happens in nearly every response (see E2 and the MGSM sw result below).

## Round 2 (pre-registered before the run)

| Experiment | Language | n | Identity | Δ truncation | Δ L2 | Δ doomloop | Control base rates (trunc / L2 / loop) | Accuracy ctrl → trim | Verdict |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| E0 noise floor | bn, en (MGSM, Aya), f16 + q4_k_m | 900 | 100.0 (control vs control) | — | — | — | — | — | Floor is zero |
| E1 | Hindi (tier 2), Aya | 100 | 98.0 | 0.0 | 0.0 | 0.000 | 0.0 / 73.0 / 0.142 | — | Pass |
| E1 | Gujarati (tier 4), Aya | 100 | 100.0 | 0.0 | 0.0 | 0.000 | 4.0 / 100.0 / 0.221 | — | Pass |
| E1 | Tamil, Aya | 100 | 99.0 | 0.0 | 0.0 | 0.000 | 10.0 / 100.0 / 0.256 | — | Pass |
| E1 | Telugu, MGSM | 250 | 100.0 | 0.0 | 0.0 | 0.000 | 15.6 / 99.6 / 0.400 | 60.8 → 60.8, McNemar p = 1.0 | Pass |
| E2 | Swahili (out of region), Aya | 100 | 0.0 | +5.0 | −1.0 | +0.063 | 12.0 / 96.0 / 0.276 | — | Misses bounds; not significant (truncation McNemar p = 0.41; Δ loop 95% CI −0.003 to 0.13) |
| Exploratory | Swahili (out of region), MGSM | 250 | 0.0 | +49.2 | — | +0.329 | 5.6 / — / 0.294 | 70.4 → 35.6, McNemar p < 0.001, 107 discordant | Severe degradation |

All runs f16, greedy, A100, trimmed vs same-session control.

E0: the 900 round-1 control prompts were regenerated in a second session on a different A100 instance with a different mix of concurrent jobs; all 900 matched the first pass byte-for-byte (trace and answer). The identity test therefore has a zero noise floor in this configuration.

E1: 547 of 550 paired outputs identical (99.45%); all 3 divergences have a removed token at or before the divergence (2 Hindi, 1 Tamil, by retokenization). The 73% L2 rate on Hindi is a property of the base model (it often reasons in English for Hindi prompts) and is identical in the trimmed model.

E2 and MGSM sw: a median of 14.4% of each Swahili prompt's tokens are removed, so the trimmed model reads every prompt through different pieces from the first token. On free-text Aya the trend toward more truncation and looping was not significant at n = 100 and the outputs were fluent Swahili in both models. On MGSM, where answers can be scored, the effect is unambiguous: accuracy halves, half the responses hit the token cap, and the loop score doubles. The trimmed model loses the ability to reason in Swahili, not just to spell it. The MGSM sw run was added after E2 and is labelled exploratory; its thresholds were not pre-registered.

Greedy base rates are high on MGSM (truncation 15.6% te, 16.8% bn; loop score ~0.40), because greedy decoding loops more than the paper's sampled setting. These rates are not comparable to the paper's.

MGSM parser spot check: 10 random Bengali f16 control rows read by eye; all 10 scored correctly (7 correct incl. `$1{,}596` → 1596; 3 wrong: one empty truncated answer, two genuinely wrong numbers, one in Bengali digits).

## KV-cache probe (exploratory)

`swa_full=False` is the deployment setting: at 32K context it cuts VRAM from 5.38 GB to 3.88 GB with 100% identical output on 20 Bengali Aya prompts, because 27 of 36 layers only need a 4,096-token window. It saves nothing at 4K and is independent of vocabulary trimming, so the two savings add.

| Setting | VRAM at 4K (GB) | VRAM at 32K (GB) | Output vs default |
| --- | --- | --- | --- |
| Default (full KV, no flash) | 2.78 | 5.38 | identical |
| swa_full=False | 2.78 | 3.88 | identical, 20/20 |
| flash attention only | 2.78 | 4.74 | diverges, median char 316 of 1,568 |
| q8_0 KV + flash | 2.64 | 3.69 | diverges, median char 324 |
| q8_0 KV + flash + swa_full=False | 2.64 | 2.92 | diverges, median char 324 |

Per stream, including about 2.1 GB of q4_k_m weights; measured on the untrimmed base. Nearly all the non-identity in the lossy rows comes from flash attention's numerics, not from 8-bit KV. Divergence within a few hundred characters under greedy decoding is normal for any numerical change and says nothing about quality by itself; flash and q8_0 KV need the same paired-metric evaluation quantization got before they are cleared.

## External checks

| Our figure | Reference | Agreement |
| --- | --- | --- |
| 52.7% of vocabulary kept, quality retained | Ushio et al. 2023 report that keeping about 50% of a multilingual vocabulary retains original performance | Consistent; theirs is task accuracy after fine-tuning, not free-generation identity |
| swa_full=False saves 1.50 GB at 32K | llama.cpp sizes the SWA cache at n_swa + n_ubatch padded to 256; 27 layers × 2,048 B/token × (32,768 − 4,608) = 1.56 GB expected | Within 4% |
| Full f16 KV at 32K ≈ 2.4 GB | 2 × 36 × 4 × 128 × 2 B = 73.7 KB/token × 32,768 = 2.42 GB | Arithmetic |
| MGSM bn 64.0% (f16, greedy) | The L2-Thinker paper reports 68.0 ± 14.1 averaged over non-English languages under sampling | Bengali is tier 3; a few points under the mean is expected |
| Bengali answer parsing | Mind the Gap (arXiv 2511.05162): Bengali is the one MGSM language where models often output native numerals | Parser converts native digits to ASCII; verified by the 10-row spot check |
| Chat template, system preamble, ` /think`, `<|START_THINKING|>` generation prompt | Model card | Matches |

## Open items

- Confirm every `input_layernorm.bias` tensor in the checkpoint is zero (llama.cpp's Cohere2 graph is bias-free). Predates the trim.
- E3 code-switch set (`data/codeswitch_prompts.jsonl`): identity ≥ 90% pre-registered, plus a human read for misspelled respellings.
- E4 Earth trim (sw/ha/yo): the Latin-script stress test, with MGSM sw as the recovery metric against the 70.4 → 35.6 baseline above.
- Tier-2 evaluation of flash attention and q8_0 KV.

## References

Ushio, Zhou, Camacho-Collados (2023), Efficient Multilingual Language Model Compression through Vocabulary Trimming, Findings of EMNLP · Bogoychev, Chen, Haddow, Birch (2024), The Ups and Downs of Large Language Model Inference with Vocabulary Trimming by Language Heuristics, Insights from Negative Results · Gurgurov et al. (2025), arXiv 2505.16956 · Mofakhami et al. (2026), Tiny Aya L2-Thinker, arXiv 2609.10445 · Shi et al. (2022), MGSM, arXiv 2210.03057 · Mind the Gap (2025), arXiv 2511.05162 · He and Thinking Machines Lab (2025), Defeating Nondeterminism in LLM Inference · Leviathan, Kalman, Matias (2023), Fast Inference from Transformers via Speculative Decoding, ICML · llama.cpp PR 13194 and PR 13833 (SWA cache).
