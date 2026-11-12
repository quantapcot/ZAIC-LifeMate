# ToolLoop Phase 2: local synthesis pilot

## Environment and setup

The pipeline reuses the Phase 1 official `ToolRegistry` and `TrainOnlyDataset`. It never falls back to reading unfiltered records for synthesis; the accessor checks dataset/catalog hashes and split assignments and exposes only training records. The accessor internally scans the source dataset to verify complete split coverage; it retains and exposes only training records. It is an accidental-leakage boundary, not an operating-system sandbox. The pilot reads installed-app tool names and a single-call label from the training view to sample targets; it does not send train queries, labels, history, context, profiles or user identifiers to the model. The only model inputs are official catalog definitions and synthetic query text. Outputs are private local artifacts, not competition submissions.

Tested on Windows, Python 3.13.13 for the CLI and fixture tests; the full project test run used the existing Python 3.11 ASR environment. Ollama 0.40.16 was already running; the specifically authorized `qwen3:4b-instruct-2507-q4_K_M` model was pulled. Its local manifest digest was `0edcdef34593eac1aa2be9c7d06c432dcf81945adca5eca2f27662c18f168ba0`, file size 2,497,293,803 bytes. This is the non-thinking Qwen3-4B-Instruct-2507 Q4_K_M Ollama build for local generation, not the full-precision Hugging Face fine-tuning checkpoint. Official references: [Ollama model](https://ollama.com/library/qwen3:4b-instruct-2507-q4_K_M), [Qwen model card](https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507), [Ollama chat API](https://docs.ollama.com/api/chat).

The RTX 5060 Laptop GPU was detected with 8,151 MiB total. After the pilot Ollama's `/api/ps` reported 3,180,247,121 bytes `size_vram`, and `nvidia-smi` showed 3,162 MiB device memory in use at that instant. These are snapshots, not measured peak memory; model file size is not GPU usage and combined ASR workload was not tested. No inference service or system-wide software was installed.

## Pipeline

The [ToolLoop paper](https://aclanthology.org/2026.emnlp-main.1262/) Section 3.3.2 and Figure 6 motivate generating a complete and natural user request from selected targets and signatures. Here:

1. **Ground truth:** seeded selection from single-call train examples whose tool is available in that sample's installed apps. `simple` uses the target alone; `multiple` adds a same-app installed tool as distractor. Same-app membership is a heuristic, not proof of semantic similarity. Targets with `x-ref`, `x-match` or `x-one-of` are excluded conservatively. Parallel and parallel-multiple return unsupported; no multi-call example is treated as independent.
2. **Backward query:** the local model receives official target definitions and instructions for a natural Vietnamese request with explicit values, no invented state and no extra actions. No actual train text is sent. An exact normalized match to any training query is rejected; paraphrase copying is not detected. The language and meaning are not proven by rules.
3. **Forward calls:** the model sees the generated query and all candidate definitions, without the expected target names as a separate answer. It returns internal ordered `{name, arguments}` calls. These are not an SDK execution or a competition submission. Deterministic validation checks selected names against the target sequence only after generation.

Existing `GroundTruth`, `BackwardQuery`, `ToolCall`, `ForwardCalls` and `VerificationFeedback` dataclasses are reused. Local records preserve intermediate stage outputs, stage feedback, model configuration, seed and timing. Status is `provisional`, not verified or SFT-ready. The strict deterministic validator checks basic structure, supported scenarios, target/candidate membership, call counts, argument names, required fields, types, enums, numeric bounds, arrays and `x-one-of`. Its stricter unknown-argument policy is separate from the SDK's acceptance behavior. `x-ref`, `x-match`, `format`, schema alternatives, semantic alignment, grounding and SDK/state execution are recorded as not verified when applicable. No ToolEnv actions, LLM judge, regeneration loop or fine-tuning are implemented.

Ollama uses non-streaming `/api/chat`, structured JSON response schemas, a fixed seed, temperature 0.1, 4,096-token context, maximum 512 output tokens and 180-second timeout by default. The context cap is 16,384; configuration is adjustable via CLI. The client allows only local HTTP origins and disables proxy and redirect handling. Malformed responses and transport failures are separate errors. Requests are sequential with no automatic retries. `--limit` is capped at 10 attempts; rejected results are retained locally without printing their contents.

## Verification and pilot (2026-11-13)

The real local smoke test returned valid JSON, six output tokens, in 0.249 seconds with the model already loaded. The final 5-attempt pilot used seed 42 and the same loaded model. The test suite ran concurrently, so these wall-clock timings are not an isolated benchmark.

| Engineering metric | Result |
|---|---:|
| Attempts / provisional / rejected | 5 / 5 / 0 |
| Simple / multiple | 3 / 2 |
| Ground truth checks valid | 5 / 5 |
| Backward JSON/structural checks valid | 5 / 5 |
| Forward JSON and strict schema checks valid | 5 / 5 |
| Mean ground truth latency | 0.00023 s |
| Mean backward latency | 0.393 s |
| Mean forward latency | 0.456 s |
| Backward / forward output tokens | 126 / 160 |
| Peak local GPU memory | unavailable |

These numbers are engineering checks, not model accuracy, semantic verification or proof that a tool executes successfully. Failure categories in this pilot: none. The sampled target set is intentionally narrow. A separate no-network, no-GPU fixture suite covers malformed responses, timeouts, local-only configuration, schema failures, target leakage, stage order, bounded failure, deterministic sampling and local output protection. Full discovery passed 30 tests, including the Phase 1 tests and the existing ASR helper, in the compatible Python environment.

## Running

From `.worktrees/toolloop-synthesis` with the Phase 1 inputs and manifests already available:

```powershell
python -m scripts.synthesize_toolcalls --help
python -m scripts.synthesize_toolcalls --dry-run
python -m scripts.synthesize_toolcalls --smoke
python -m scripts.synthesize_toolcalls --generate --limit 5 --seed 42 --output-dir outputs/toolloop/synthesis/my-new-pilot
```

Dry-run validates paths, catalog, split and configuration without contacting Ollama. Smoke sends a single generic JSON request. Generation writes `records.json`, `summary.json` and an outcome marker under a new Git-ignored directory and refuses overwrite. The `COMPLETE`, `PARTIAL` and `FAILED` markers describe the run, not data verification; a missing marker denotes an interrupted write. Never commit these outputs. The model must already be present locally; on an authorized installation use `ollama pull qwen3:4b-instruct-2507-q4_K_M`.

## Remaining Phase 3 work

Validate Vietnamese naturalness and target semantics, value provenance and copy leakage, compare against official SDK rules without executing state-changing calls, handle additional constraints and stateful observations, establish independent parallel scenarios, and implement bounded stage-specific feedback/refinement. Initial same-app distractors and rule checks are not a semantic verifier; direct filesystem access could bypass the train-only API. Do not use these provisional samples for SFT until the later validation process is designed and reviewed.


## Phase 2.1 ? Pipeline Hardening

Recursive schema inspection now visits schema-bearing properties, nested objects, array items, definitions, alternatives and other supported schema nodes. It leaves the official schema untouched and excludes targets containing `x-ref`, `x-match`, `x-one-of`, or conditional schema alternatives from this static pipeline. `x-one-of` can be checked deterministically by the validator, but target sampling remains conservative. Plain nested types, enums and arrays remain eligible. On the current 147-tool catalog, **62** have no excluded schema constraints. Exclusion reason counts are 65 `UNSUPPORTED_X_REF`, 32 `UNSUPPORTED_X_MATCH` and 6 `UNSUPPORTED_CONDITIONAL_SCHEMA`; categories overlap. These are catalog-schema counts, not counts of eligible training records. Dry-run finds 79 simple and 72 multiple eligible training records. No counts are hardcoded.

Generation and validation failures now share structured `VerificationFeedback.issues`: `error_code`, `field`, `message`, `retryable`, `source` and a compatibility `reason`. Timeout and transport errors are distinct technical failures; invalid JSON, malformed envelopes and response-schema mismatches are distinct model-output failures. Argument type, missing/unknown arguments, enum, numeric range, tool name, target mismatch and copied query have stable issue codes. Schema rejections carry the tool name and schema paths, and the registry retains the original schema. Error messages omit generated content; when available, at most 4,096 characters of invalid model output are retained only in ignored local run artifacts. The technical transport path is not a semantic retry; unsupported schema/data problems are not marked correctable by generation. Semantic alignment and grounding remain `not_verified`, and retries remain disabled. Prompts are reproducible from the existing templates and preserved stage inputs; no refinement prompt has been implemented.

Dry-run and smoke no longer check whether a generation output directory already exists. Dry-run contacts no model and writes nothing; smoke issues one generic structured request and writes no records. Generation continues to reject existing or unsafe destinations. A run summary separates `requested` (limit), `eligible` (distinct training records usable for at least a simple target), `attempted` (selected records actually processed), `provisional`, `rejected`, and `shortfall` (`requested - attempted`); exclusion and multiple-to-simple fallback reasons are counted separately. Fewer eligible records produce a `partial` outcome and nonzero exit while preserving provisional records; zero eligible or a transport failure produces `failed`. A `completed` run may still contain rejected records. `COMPLETE`, `PARTIAL` and `FAILED` markers indicate only the run outcome; they are published after successful record and summary writes. An interrupted or failed write has no outcome marker. None of these markers makes data SFT-ready.

Verification on 2026-11-13: **42 tests passed, 0 failed** in the compatible Python environment, including Phase 1 and Phase 2 regression tests. Mock tests cover nested schema cases and immutability, structured errors and retry policy, existing output directories, deterministic selection/shortfalls and marker failure. Real `--dry-run` and one generic Ollama `--smoke` both passed with an existing pilot directory; smoke took 2.572 seconds. The original Qwen model configuration was preserved. No synthetic pilot or additional model download was performed in this hardening phase; dataset/catalog hashes still match.

The pipeline is ready to begin integrating a separate semantic verifier, stage-specific feedback and bounded regeneration. These components and acceptance policy are not yet implemented. Phase 3 must define safe retry rules, retain feedback history, evaluate semantic/argument provenance, address unsupported stateful and parallel scenarios, and avoid exposing local generated content.
