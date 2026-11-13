# Phase B1: gold-supervised first-action QLoRA

## Scope and provenance

B1 starts from `f3ddc151d514dbade049332d52d021ddd30fbff0` on the isolated `feature/lifemate-action-sft` branch. It uses the official TRAIN partition and the existing registry, router, ActionAgent and ToolEnv; no ToolLoop synthetic data, held-out training labels, hidden state inspection, ASR changes or public/private challenge labels are used for fitting. Generated data, model files, adapter weights, checkpoints, predictions, IDs and traces remain under ignored `outputs/action_sft/`.

The official trainable checkpoint is [Qwen/Qwen3-4B-Instruct-2507](https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507), revision `cdbee75f17c01a7cc42f958dc650907174af0554`. All three downloaded safetensor shard SHA-256 values matched official Hub metadata. The existing Ollama GGUF was not used as a trainable model. GPU: NVIDIA RTX 5060 Laptop, 8,151 MiB reported; Windows driver 577.05. Python 3.11.18; torch 2.11.0+cu128; Transformers 4.57.6; PEFT 0.18.1; bitsandbytes 0.49.2; Accelerate 1.15.0. The dedicated `.venv` inherits the pre-existing base Python's system packages and contains its own PEFT/Accelerate/bitsandbytes installation; it is not a hermetic environment. Available project-drive space was approximately 480 GB before download. A direct Windows NF4 operation succeeded. WSL was available but its default Python had no pip, so the existing CUDA-enabled Windows Python was used instead. Initial HF cache symlink creation failed with WinError 1314; downloading directly into ignored local storage succeeded.

## Dataset contract and leakage safeguards

`python -m scripts.audit_action_sft --output-dir outputs/action_sft/b1-data-audit` verified the dataset and catalog hashes against the existing manifest and audited all 1,236 TRAIN IDs: 89 zero-call, 295 single-call, 852 multi-call, 173 with missing-information calls. There were no unknown or unavailable tool names, malformed calls, or basic structural failures in this audit. All 1,147 first calls in nonempty examples passed official `env.validate` without being executed. All 1,236 call/result lists had equal lengths and all 2,828 tool results were objects. Results do not identify their own tool names; length agreement alone does not independently establish semantic or temporal alignment.

Zero-call examples supervise the official no-call `finish` target, not a claim that every request is out of scope. All 89 have nonempty transcripts, 42 have history, and a coarse verb-overlap heuristic flagged 43; this is not a semantic scope classification. This uncertainty matters when interpreting finish behavior.

`python -m scripts.prepare_action_sft --output-dir outputs/action_sft/b1-data-1536-v2 --max-length 1536` produced 1,034 fit and 202 TRAIN-internal dev examples, zero skipped, via a deterministic seed-42 hash of normalized transcripts. All targets are the first official gold call or the no-call `finish` envelope. Candidate sets are gold-positive during preparation and TRAIN-internal controlled evaluation, with shuffled installed distractors and official full schemas; normal runtime routing has no gold injection. The production ActionAgent `SYSTEM` and JSON envelope are retained. No gold tool results, expected final state or fabricated observations are input to the model. Original metadata and gold labels stay local.

`python -m scripts.verify_action_sft --dataset outputs/action_sft/b1-data-1536-v2` checked all 1,236: gold JSON and masks matched, assistant EOS appeared in the response span, no target was truncated, gold-positive candidates were present, prompt keys were restricted and observations empty, and no normalized transcript group crossed fit/dev. Prompts and targets follow the downloaded tokenizer's actual Qwen chat template. The template emits a newline after `<|im_end|>`; both are kept while the prompt tokens are masked. Each training batch is a single unpadded sequence with an all-one attention mask. To remain within 1,536 tokens, 812 examples lost some past history; this limits context-dependent learning and can create a mismatch with routed evaluation. Fit median/p95 sequence lengths were 1,461/1,528 tokens.

## Training and inference method

The bounded run uses NF4 with double quantization, BF16 compute, LoRA rank 8, alpha 16, dropout 0.05 on `q_proj` and `v_proj`, AdamW learning rate 0.0001, microbatch one, gradient accumulation four, gradient checkpointing, seed 42, and 50 optimizer steps. The first two steps form a smoke stage; the final checkpoint is selected by loss on the first eight TRAIN-internal dev records. This small selection set makes the checkpoint choice noisy. The run is intentionally bounded and does not complete a full pass over all fit examples. No loss decrease alone establishes task quality.

The source includes checked optimizer/RNG resume support for newly saved checkpoints; the active run started before that support was added, so its adapter checkpoints do not include resume-state files and cannot be resumed exactly. The resume-state helper has a CPU unit test but a real GPU resume has not been demonstrated.

The primary comparison uses a single local quantized Transformers/PEFT model with the adapter explicitly disabled or active, identical Qwen tokenization, greedy decoding, 256 output tokens and the same candidate lists. It does not compare the trained PEFT model to the Ollama GGUF. JSON syntax and schema validity are counted separately. Transformers generation uses a token-boundary time limit; it is not a hard process-level deadline and is not grammar-constrained like the earlier Ollama endpoint. The adapter must exist, identify the local base, load as PEFT, and change probe logits before evaluation continues.

## Measured training outcome

The real run completed **50 optimizer steps** (200 fit presentations from 1,034 fit examples), not a full epoch. It trained 2,949,120 LoRA parameters. The final four-example mean training loss was 0.0431; checkpoint-selection dev loss fell from 2.0901 at step 2 to 0.1373 at step 50 on the same eight records, so step 50 was selected. Mean optimizer-step time was 24.59 s; total time including load and checkpoint checks was 1,269.3 s. Peak PyTorch-allocated GPU memory was 6.42 GiB; this is not total WDDM process usage. The selected local adapter is `outputs/action_sft/b1-train/adapter-step-50/`. No OOM occurred. The direct model-download retry avoided Windows symlink restrictions.

The adapter was actually reloaded into PEFT for inference. Its `lora_B` absolute sum was nonzero (1,448.54), and enabling it changed a same-prompt next-token logit by a maximum absolute 11.09375. Base and adapter produced different raw-output hashes for all eight controlled examples. The following results do not rely on the declining training loss as a proxy for accuracy.

## Paired first-action evaluation

The following numbers use the same locally loaded quantized model, greedy Transformers decoding, prompts, candidate lists, 2,048-token context limit and response checker for each pair. Base means PEFT adapter disabled; adapter means the selected step-50 adapter enabled. JSON syntax validity and conformance to the required action envelope are different metrics. The base's 8/8 controlled and 8/8 fresh-routed outputs parsed as JSON but all failed the required envelope; this largely explains its zero scored actions. The comparison does **not** show how the base would perform with Ollama's structured-output constraints, which the earlier C1 Agent used.

| Metric | Base | Adapter |
|---|---:|---:|
| Controlled TRAIN-dev JSON syntax valid | 8/8 | 8/8 |
| Controlled TRAIN-dev action schema valid | 0/8 | 8/8 |
| Controlled TRAIN-dev first-tool exact | 0/8 | 5/8 |
| Controlled TRAIN-dev first arguments exact (including correct tool) | 0/8 | 5/8 |
| Controlled TRAIN-dev complete first-action exact | 0/8 | 5/8 |
| Fresh routed validation JSON syntax valid | 8/8 | 8/8 |
| Fresh routed validation action schema valid | 0/8 | 8/8 |
| Fresh routed validation first-tool exact | 0/8 | 4/8 |
| Fresh routed validation first arguments exact (including correct tool) | 0/8 | 1/8 |
| Fresh routed validation complete first-action exact | 0/8 | 1/8 |
| Fresh routed validation mean latency | 5.00 s | 4.33 s |

The controlled TRAIN-dev examples have gold-positive candidates and exclude the first eight dev records used to select the checkpoint. They have no zero-call cases and should not be treated as production performance. Under realistic routed TRAIN-dev candidates, complete first-action exact match was 0/8 base and 3/8 adapter; four gold first tools were absent from the actual shortlist. The fresh validation set was selected before evaluation, excludes the older ten C1 pilot IDs and also excludes the first eight B1 validation IDs used during an initial evaluator run. In the fresh routed set, three of eight gold first tools were absent after label-blind context compaction. This is a retrieval limitation as well as a model-selection limitation.

The initial B1 evaluator tried the untrimmed routed prompt directly: 7/8 TRAIN-dev and 6/8 validation examples exceeded its 2,048-token backend budget. Those validation labels were not used to tune the model or router. The evaluator was corrected to apply a deterministic label-blind history/shortlist compaction mirroring the Agent, and a **different predeclared validation subset** was selected before the corrected run. The corrected comparison had zero prompt-budget failures; seven of eight routed TRAIN-dev prompts and three of eight fresh validation prompts needed compaction. These are small samples, and reducing the candidate list can itself reduce retrieval recall.

## Finish, missing information and autonomous execution

A separate deliberately label-stratified, gold-positive TRAIN-dev diagnostic used four zero-call and four first-call `sys.ask_missing_info` records outside the checkpoint-selection and main controlled subsets. It is not representative. The adapter generated valid action envelopes for all eight, but chose `finish` for **0/4 zero-call records** and selected `sys.ask_missing_info` for **0/4 missing-information records**, despite the system tool being available. The base also scored 0/4 for each category because its JSON did not match the action envelope. Zero-call records are not automatically classified as out of scope.

The same ActionAgent and official ToolEnv were exercised with both models on three TRAIN-dev and three fresh-validation records, with the same four-step and 30-second settings. On the three fresh-validation records, both models had **0/3 exact call sequences**. The base made no calls because its model responses failed the envelope checker; its exact-call precision and F1 are undefined, with recall zero. The adapter made five calls, all five passed SDK validation, four returned the SDK's apparent success status, and one exact call matched among seven gold calls: precision 0.20, recall 0.143, F1 0.167. It missed four named actions and added two; all three samples stopped on the repeated-call guard. In the three TRAIN-dev Agent records, the adapter executed three SDK-validated calls and matched one exact call; both models again had 0/3 full sequences. Schema-valid or SDK-successful calls are not equivalent to task completion. No final-state-hash match or official score is claimed.

The selected adapter is therefore useful for further research on the direct JSON action format and first-action grounding, but it is **not safe to prefer for deployment** given the observed no-call and missing-information failures. Retain the base as a reference; prefer neither checkpoint for C2. The evidence supports a targeted follow-up on no-call/missing-information supervision and then multi-step training, but the full semantics and alignment of intermediate results must be independently verified first. B2 multi-step supervision is a reasonable research direction after that validation; C2 ASR integration is not ready.

## Reproduction and checks

The model may be downloaded from the official Hub at the revision above into `outputs/action_sft/model/`; the checked local shards and all generated outputs must remain untracked. The experiment used:

```
python -m scripts.audit_action_sft --output-dir outputs/action_sft/b1-data-audit
python -m scripts.prepare_action_sft --output-dir outputs/action_sft/b1-data-1536-v2 --max-length 1536
python -m scripts.verify_action_sft --dataset outputs/action_sft/b1-data-1536-v2
python -m scripts.train_action_qlora --dataset outputs/action_sft/b1-data-1536-v2 --output-dir outputs/action_sft/b1-train --steps 50
python -m scripts.evaluate_action_sft --train-run outputs/action_sft/b1-train --dataset outputs/action_sft/b1-data-1536-v2 --selection outputs/action_sft/b1-evaluation-selection-v2.json --output-dir outputs/action_sft/b1-eval-compact --samples 8 --agent-samples 3
python -m scripts.evaluate_action_sft_scenarios --train-run outputs/action_sft/b1-train --dataset outputs/action_sft/b1-data-1536-v2 --output-dir outputs/action_sft/b1-eval-scenarios --per-category 4 --val-selection outputs/action_sft/b1-evaluation-selection-v2.json --val-agent-samples 3
```

`python -m scripts.select_action_sft_evaluation --help` describes label-blind selection. The earlier C1 pilot manifest must be locally available to exclude its IDs. Selection and run directories refuse overwrite. The real checkpoint, selection and sample manifests are deliberately ignored. The selected regression run passed 110 tests, including a real local Qwen-tokenizer formatting test and a CPU checkpoint/RNG unit test; this is distinct from the real training, SDK and model evaluations. The SFT run used one fixed configuration, not a hyperparameter search.
