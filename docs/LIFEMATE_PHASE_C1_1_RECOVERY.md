# Phase C1.1 recovery

## Pre-change diagnosis

The preserved C1 pilot has no proposal-level traces. Its runtime records establish that nine of ten samples never reached SDK validation, four exhausted invalid retries, one hit the prompt budget, and one stopped after a repeated call. Only one call was executed. The exact causes of invalid proposals, premature finishing, tool choice and argument errors cannot be reconstructed. The old prompt removed nested parameter descriptions and sometimes exceeded its character-based context guard. These are plausible contributing factors, not proven root causes. The following work uses training-partition development runs and step traces to distinguish them before a frozen validation comparison.

## Implementation and safety

A deterministic accent-normalized lexical router scores request, permitted context and recent observations against installed catalog names, descriptions and parameter definitions. It shows the top eight application tools plus `sys.ask_missing_info`. It re-ranks on each step and can try one bounded alternate window after an initial finish. No gold data enters the router. This simple retrieval is not semantic search, and there is no guarantee that prerequisites will be retrieved.

Shortlisted tools retain the original catalog descriptions and complete schemas, including nested descriptions, enums, formats and custom constraints. Certain missing fields and primitive type mismatches are checked locally; enum, cross-field and advanced constraints are delegated to SDK validation. Rejections include the proposal, field-specific error and schema in the next model context. Calls rejected by either layer never reach `env.call`. The agent records observations and permits an identical call after an intervening different call, but stops an immediate repeated call; this is a heuristic for ineffective repetition, not proof of state change. It can still finish too early or choose an unsuitable missing-information call.

Prompt construction keeps the current request and schemas and first drops old history, then old observations, then reduces the shortlist on a prompt-budget exception. These are bounded measures; dropping old observations can remove information needed for a later dependency. No schema constraints are removed from the remaining candidates. The Ollama budget is a conservative Unicode-codepoint heuristic, not an exact tokenizer, and the socket timeout is not a hard process deadline. Oversized residual prompts and model errors have separate stop reasons.

Per-step traces record shortlist and scores, model proposal and argument values, validation stages and SDK errors, result status, timing, prompt estimate, retries and termination. They are written exclusively to ignored local `outputs/agent/.../traces.jsonl`; they can contain sensitive arguments and should not be published. The evaluator handles `train` and `val` separately, verifies split coverage and source hashes, and loads the selected partition's labels only after all predictions. It does not decode internal-test labels. Candidate recall is measured offline: the primary figure counts each gold call whose tool appeared in the union of candidate lists for that sample; it is optimistic relative to step-aligned recall. First-target initial recall is also shown. A finish with fewer calls than the gold is merely a count-based premature-finish indicator, not proof of semantic incompleteness. The official score, final-state hash match, dependency and out-of-scope categories remain unverified.

## Experiments

- Mock/regression: `python -m unittest tests.test_action_agent_c1 tests.test_action_agent_c1_1 -v` passed 22 tests. The selected combined ToolLoop regression suite is also run separately below.
- Actual SDK: `python -m scripts.evaluate_action_agent --sdk-smoke` initialized the official environment, rejected an invalid missing-info proposal without executing it, then validated and executed a valid safe call and returned recorded calls and an authoritative hash.
- Training-only development: `python -m scripts.evaluate_action_agent --split train --limit 8 --seed 42 --output-dir outputs/agent/c1-1-train-dev3`. The random fixed sample contained one zero-call, one single-call and six multi-call gold records; dependency and out-of-scope categories were not independently established. Final development run: exact sequence 0/8, invalid proposals 2/10, SDK validation 8/9, 17 missing named actions and 4 extra; union candidate recall 14/21 (66.7%), initial first-gold recall 3/7 (42.9%), count-based early finish 6/8, no timeout or prompt-budget failure. One earlier development run had 5/11 invalid proposals, including wrapper-shaped arguments in place of actual parameter fields, and one invalid JSON failure. These traces substantiate a specific argument-envelope failure; they do not establish it as the sole root cause. Development iteration changed the action instructions; it did not use validation labels.
- Frozen validation: `python -m scripts.evaluate_action_agent --split val --seed 42 --ids-from-manifest outputs/agent/c1-pilot-final/run_manifest.json --output-dir outputs/agent/c1-1-val-pilot`. The exact same ten IDs, model name and model digest as C1 were verified; temperature 0, seed 42, context 16,384 and output limit 256 were retained. Fixed seeds do not ensure deterministic model output. Prompts and candidate routing changed.

| Diagnostic | C1 | C1.1 |
| --- | ---: | ---: |
| Tool-name sequence exact | 0/10 | 0/10 |
| Call sequence exact | 0/10 | 0/10 |
| Exact-call precision / recall / F1 | 0 / 0 / 0 | 0 / 0 / 0 |
| Missing / extra named actions | 24 / 0 | 20 / 1 |
| Invalid proposals | 15 | 0 |
| SDK validations passing | 1/1 | 6/6 |
| Reported successful executions | 1/1 | 5/6 |
| Prompt-budget failures | 1 | 0 |
| Model-output failures | 0 recorded as model failures | 1 invalid JSON |
| Mean / p50 / p95 latency, seconds | 2.42 / 2.71 / 3.43 | 2.14 / 1.70 / 3.90 |

C1.1 validation candidate recall across all shown lists was 17/25 (68%), with first gold tool in the initial shortlist for 5/10. The model's first proposed tool matched the first gold name in 1/10; one sample proposed a different first tool even though the target was in the initial shortlist. Eight samples finished with fewer recorded calls than gold (count heuristic), one had invalid JSON and no timeout occurred. Six calls were made (0.6 per sample); one aligned tool had incorrect arguments. There were no local structural or SDK validation rejections in this validation pilot. At least eight gold tool occurrences were never retrieved during their samples; several retrieved targets still did not result in complete calls. This evidence implicates both retrieval coverage and action/finish behavior, but cannot distinguish all errors from small-model limitations. Improvement in syntax validity alone is not an improvement in complete-task accuracy.

The C1.1 router averaged nine candidates including the system tool, approximately 0.062 s total routing time for ten samples. A post-run `nvidia-smi` snapshot showed RTX 5060 Laptop GPU, 8,151 MiB total, 5,012 MiB in use; it is neither a peak measurement nor a model-only allocation. Local artifacts, including the fixed training IDs, are ignored and existing C1 artifacts were not overwritten. No full validation was run because the pilot still has critical quality failures.

## Readiness and next experiment

C1.1 quality recovery is **not complete** and C2 ASR integration is **not ready**. The data/evaluation foundation can support a controlled future supervised fine-tuning experiment, but it is not evidence that SFT inputs or a model are ready, and no training was run. First improve retrieval recall and control flow on fixed training-only examples, including dependent reads and multiple requested actions; independently measure step-aligned candidate recall and diagnose why the model finishes after only part of a request. Preserve a new validation comparison without repeatedly tuning against these same validation labels. Obtain authoritative official scoring and submission requirements before claiming official performance.
