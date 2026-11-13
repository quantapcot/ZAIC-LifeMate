# ToolLoop Phase 3: semantic verification and bounded feedback

## Scope and evidence

This development-only layer adapts [ToolLoop Section 3.3 and Appendix B](https://aclanthology.org/2026.emnlp-main.1262.pdf): sample ground truth, derive a query backward, derive calls forward, and use stage-specific feedback. It adds no inference-time submission model, ToolEnv execution, SFT or QLoRA. Phase 2's no-verifier path still produces `provisional` records.

The intended generator remains `qwen3:4b-instruct-2507-q4_K_M`, local digest `0edcdef34593eac1aa2be9c7d06c432dcf81945adca5eca2f27662c18f168ba0`. The separately configured development verifier defaults to `qwen3:8b-q4_K_M`; it was **not installed or run**, so its local digest and actual VRAM footprint are unavailable. The [Qwen release announcement](https://qwenlm.github.io/blog/qwen3/) dates Qwen3-8B to 2025-04-29 and lists Apache 2.0; the [Ollama tag](https://ollama.com/library/qwen3:8b-q4_K_M) lists Q4_K_M and about 5.2 GB. Those facts do not independently establish competition eligibility.

The organizer's [LifeMate portal](https://challenge.zalo.ai/portal/lifemate) was not readable through the available web fetch and the browser access host was unavailable, so the current official development-model rules and any relevant submission obligations could not be independently verified. The task-provided limits (under 6B for the submitted system, under 5B for its LLM, 16,384-token context and offline inference) are therefore treated as requirements to confirm, not as independently verified rules. The 8B verifier must never be included in the submitted inference system. No model was downloaded during this task: the existing system-drive model store had about 8.9 GiB free, while the requested download is about 5.2 GB, and the organizer terms could not be checked. Another explicitly configured verifier may be used only after its terms and competition eligibility are established; it must have a different local digest from the generator.

On this machine the RTX 5060 Laptop GPU reported 8,151 MiB total, 40 MiB in use and 7,771 MiB free before testing. This is a snapshot, not a peak or an 8B compatibility test. Verified mode uses separate localhost-only Ollama clients, sequential calls, `keep_alive=0` on both models, explicit unload requests before generation, and `think=false` for the verifier. OOM/transport failures stop the run rather than trigger semantic retries. The Phase 3 byte-based prompt guard was replaced in Phase 3.1 by a configurable Unicode-codepoint token estimate and a separate payload-byte cap; it remains a heuristic, not a tokenizer. User-specified context is still capped at 16,384 tokens. It cannot control other applications concurrently using the GPU.

## Implementation and acceptance

The sampler now excludes unsupported state-dependent or conditional schemas from distractors as well as targets. Same-app distractors are not considered semantically validated until the separate verifier evaluates them. Oversized Ollama responses receive `RESPONSE_TOO_LARGE` and are not refined.

`SemanticVerifier` sends only generated outputs, official candidate definitions and scenario metadata to the local verifier. Its structured verdict must be `pass`, `fail`, or `uncertain`; each required rubric check must be present exactly once and pass. The ground-truth rubric covers coherence, feasibility, distractors and unknown state. The query rubric covers Vietnamese naturalness, intent, required information and unknown state. The forward-call rubric covers tool selection, completeness, extra actions, argument consistency and unknown state. A verifier explanation or a confidence score is not proof; uncertain or missing checks fail closed.

The deterministic validator runs first, and a structural failure is never overridden by the LLM. For every generated argument leaf, the verifier must report a matching field path and provenance. For query evidence, the quoted span must occur literally in the query and contain the literal argument value as a bounded token. For a schema default, the actual official default and value must match in type and value. Context and intermediate observations are not supported in this static policy. This deliberately rejects many legitimate paraphrases and time conversions rather than claiming they are grounded. It does not prove that a literal match entails the intended action. Used `format`, `x-ref`, `x-match`, and schema alternatives that remain unverified cause rejection in verified mode. SDK execution and state correctness are not verified.

The `lifemate-static-v1` policy grants `verified_candidate` only after all three stages have complete histories and pass deterministic, semantic and evidence checks. `sft_ready` is always false. Uncertain verdicts become `manual_review_required`; failures become `rejected`. These are experimental judgments, not gold labels or proof of real-world correctness.

## Feedback and audit

Each stage permits one initial output and up to three revisions. The ground-truth sampler uses bounded seeded re-sampling of the same train record and records the rejection reasons; this is an adaptation, not LLM ground-truth rewriting. Query revisions keep the accepted ground truth fixed; call revisions keep both accepted ground truth and query fixed. Downstream work is constructed only after the upstream stage passes. Correctable structural/semantic failures carry the previous output, stage objective, official definitions, structured issues and the original structured-output schema into the next prompt. Instructions are bounded and model outputs are untrusted. The forward refinement feedback redacts literal expected tool names; the unredacted verifier output is retained only in local audit history. Missing verifier, unsupported schema, server failure, timeout, OOM and oversized response do not trigger semantic refinement.

The protected local stage history records each output, deterministic feedback, verifier verdict and issues, retry instruction, attempt number, usage and latency. Each record includes the policy version, seed, model IDs, digests and options. A `review.json` worksheet contains only local record indices/status and marks every entry `not_reviewed`; it does not assert human review. Output directories remain ignored and must not be committed.

## Running and verification

```powershell
python -m scripts.synthesize_toolcalls --help
python -m scripts.synthesize_toolcalls --dry-run --verify
python -m scripts.synthesize_toolcalls --smoke
# Only after verifying organizer terms and installing a distinct local verifier:
python -m scripts.synthesize_toolcalls --smoke --verify
python -m scripts.synthesize_toolcalls --generate --verify --limit 3 --seed 42 --max-refinements 3 --output-dir outputs/toolloop/synthesis/phase3-pilot
```

`--dry-run --verify` validates paths, splits, catalog and configuration without contacting models. A verified run requires both local model digests, rejects equal digests and saves results only locally. Set `--max-refinements 0` for generate/verify/filter; use the same seed and limit with three refinements for the refinement mode. No-verifier generation remains the Phase 2 generate-only mode. Model determinism and matching scenario selection must be checked when interpreting an ablation.

The compatible Python 3.11 environment passed 58 tests (0 failures); the existing Phase 1/2 tests are included. The local Qwen3-4B generic JSON smoke passed in 5.109 seconds; verified smoke reported that the 8B verifier was unavailable. Mock tests exercise rubric failures, missing and fabricated evidence, uncertainty, missing or malformed verifier output, 0–3 revisions and the four-attempt cap, transport failures, deterministic gate ordering, revised-query propagation, model identity/unloading, OOM and unsupported distractors. These are not measures of verifier quality. The real generator smoke was run separately; verified smoke and all real verified pilots/ablation modes A/B/C were **not run** because the required 8B model was unavailable and the organizer rules could not be verified. Accordingly acceptance rates, initial semantic pass rate, recovery rate, verifier tokens/latency, verifier peak VRAM and human review findings are unavailable. Existing Phase 2's five provisional samples are not a Phase 3 ablation. The CLI records engineering counters when a verified run is eventually performed; these are not LifeMate accuracy.

## Remaining work before SFT

Confirm the current organizer rules, development-model usage and resource budget; make adequate disk space available or configure an approved model store, then install and smoke-test the 8B verifier. Validate structured verdict reliability, runtime memory and sequential unloading. Run the same-seed small A/B/C experiment and conduct real, local human review of accepted, rejected, repaired and uncertain examples, including false-acceptance analysis. Resolve the conservative provenance limitations, leakage and semantic calibration. No current synthetic record has been established as SFT-ready.
