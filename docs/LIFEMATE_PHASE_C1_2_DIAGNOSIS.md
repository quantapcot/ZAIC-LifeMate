# Phase C1.2: routing and oracle diagnosis

## Scope and diagnosis

C1.1 (`3b52382102f647a2df8195668c36fae960d4352d`) obtained 0/10 exact validation sequences; 17/25 gold tool occurrences appeared somewhere in the union of shown candidates, while eight samples finished with fewer calls than their labels. The old traces do not identify all causal mechanisms. Training-only traces showed wrapper-shaped arguments, missing required fields and premature finishes. Validation labels were never used to modify the router or prompt in C1.2.

The confirmed router bug concatenated two copies of a name, causing adjacent tokens to merge. The fix separates copies. The deterministic ranker also weights request tokens more than context and recent observations; names, descriptions and complete parameter metadata remain available. This is still lexical retrieval and may miss Vietnamese paraphrases or English-only concepts. No validation-derived alias table, new model or SDK modification was introduced.

## Independent retrieval experiment

`python -m scripts.evaluate_tool_router --ids-from-manifest outputs/agent/c1-1-train-dev3/run_manifest.json --output-dir outputs/agent/c1-2-router-train` ranked all installed application tools for eight fixed TRAIN records without Ollama before labels were loaded for scoring. K means K application candidates plus the SDK system tool, when available. Relevant gold names are counted per call, including duplicates; all-tool coverage is per nonempty sample. Full schema JSON Unicode codepoints are a prompt-size proxy, not actual tokens. The system tool is always included rather than ranked alongside applications.

| K | Call recall | First-tool recall | All-gold coverage | Mean schema codepoints |
|---|---:|---:|---:|---:|
| 4 | 10/21 (47.6%) | 3/7 (42.9%) | 2/7 (28.6%) | 2,233 |
| 8 | 14/21 (66.7%) | 3/7 (42.9%) | 2/7 (28.6%) | 3,732 |
| 12 | 18/21 (85.7%) | 5/7 (71.4%) | 4/7 (57.1%) | 5,163 |
| 16 | 18/21 (85.7%) | 5/7 (71.4%) | 4/7 (57.1%) | 6,223 |

Mean routing time was approximately 1.52 ms/sample. Single-call recall was 1/1 at every K; multi-call recall was 9/20, 13/20, 17/20 and 17/20. These tiny groups are not representative. Categories for English-dominant descriptions, paraphrases and actual dependencies were not independently labeled, so no performance claims are made for them. Per-sample rankings, scores and target ranks remain in ignored local artifacts.

Step-aligned recall uses the first shortlist at each prefix where *executed complete calls including arguments* match the gold prefix, and stops at divergence; it does not count a target merely because it appeared earlier or later. C1.1 and C1.2 TRAIN runs each measured 3/7; the corresponding previously examined validation runs measured 5/10 and 6/10. All available observations were first-action prefixes, because no subsequent executed call matched the complete gold prefix. This is not a measurement of later dependent steps.

## TRAIN-only first-action oracle

`python -m scripts.diagnose_action_selection --ids-from-manifest outputs/agent/c1-1-train-dev3/run_manifest.json --output-dir outputs/agent/c1-2-oracle-train` is explicitly `DIAGNOSTIC_ORACLE_TRAIN_ONLY`. The independent script hard-codes the verified TRAIN partition, rejects manifest IDs outside it, and cannot be enabled in normal Agent, validation or submission inference. The seven nonempty TRAIN records were compared under A: the C1.1 router including its original concatenation behavior; B: the C1.2 router; and C: B's shortlist with the first gold tool inserted among shuffled distractors. The SDK system tool is retained. C uses the first TRAIN gold **name only** for candidate inclusion; neither gold arguments, later gold order, final state nor hash is included in the model prompt. There is no multi-step oracle or teacher forcing. Each mode uses the same legitimate input and Qwen3-4B digest, seed 42, temperature 0, context 16,384 and output bound 256. Each mode creates a fresh real TRAIN ToolEnv and validates eligible proposals; this first-action diagnostic does not execute them.

| Metric | A: C1.1 | B: C1.2 | C: TRAIN oracle |
|---|---:|---:|---:|
| First target present | 3/7 | 3/7 | 7/7 |
| First tool selected correctly | 2/7 | 2/7 | 3/7 |
| Correct given target present | 2/3 | 2/3 | 3/7 |
| Exact first arguments | 0/7 | 0/7 | 0/7 |
| Finish before first call | 3/7 | 2/7 | 3/7 |
| Wrong tool proposed | 2/7 | 3/7 | 1/7 |
| Model errors | 0 | 0 | 0 |
| SDK validation passed / checked | 2/4 | 4/5 | 2/3 |
| Mean latency, s | 1.31 | 0.74 | 0.74 |
| Mean prompt estimate | 7,339 | 7,153 | 6,970 |

These are paired but very small, stochastic local-model diagnostics. Mode ordering may affect warm-up and latency; the oracle's shuffled candidate order and replacement also change the prompt, so 3/7 versus 2/7 does not isolate a causal effect or establish a substantial improvement. Correct candidates do not guarantee correct action or arguments. Oracle results are not deployable performance.

## Multi-step behavior and safety

The agent retains the official validate-before-call flow, one ToolEnv per autonomous sample and authoritative calls and hash. When the last successful execution looks like a lookup by tool name, or its result explicitly reports failure, one bounded progress recheck asks the same model to reconsider whether the request is finished. It does not force another call, infer a gold subtask or assert completion. The name-based lookup heuristic may miss prerequisites or prompt a needless recheck. Normal initial fallback remains bounded. Prompt compaction retains a recent identifier-bearing result as well as the latest observations where possible, but this is a heuristic, does not identify all dependencies and cannot guarantee retention of every important identifier. Official results remain the only source of execution observations.

Unit tests cover read-before-write, a returned identifier, failure followed by a progress recheck, completion, out-of-scope, unsupported oracle partitions, repeat protection, deterministic routing, accent variation, candidate recall and step alignment. Real dependent task completion was not established by the experiments. Traces and predictions can contain private arguments and stay under ignored `outputs/agent/`.

## Autonomous paired comparison

`python -m scripts.evaluate_action_agent --split train --seed 42 --ids-from-manifest outputs/agent/c1-1-train-dev3/run_manifest.json --output-dir outputs/agent/c1-2-train-ab` compared the frozen C1.1 run to C1.2 using the same eight TRAIN IDs, model, seed and settings. Seeded decoding is not necessarily deterministic.

| TRAIN diagnostic | C1.1 | C1.2 |
|---|---:|---:|
| Exact sequences | 0/8 | 0/8 |
| Missing / extra named actions | 17 / 4 | 17 / 2 |
| Union candidate recall | 14/21 | 17/21 |
| Step-aligned candidate recall | 3/7 | 3/7 |
| First proposed tool matches gold | 2/7 | 2/7 |
| Invalid proposals | 2 | 3 |
| SDK validated / checked | 8/9 | 6/7 |
| Reported successful executions | 7/8 | 5/6 |
| Count-based incomplete finish | 6/8 | 6/8 |
| Mean / p50 / p95 seconds | 2.08 / 1.75 / 3.74 | 1.73 / 1.40 / 3.47 |

The C1.2 run had three progress rechecks, no prompt-budget or model errors, and no timeouts; all eight ultimately finished. One of eight gold sequences was empty and seven were nonempty. More union coverage with no improvement in first selection or exact sequences does not establish a meaningful task-quality gain.

## Previously examined validation confirmation

After freezing implementation, `python -m scripts.evaluate_action_agent --split val --seed 42 --ids-from-manifest outputs/agent/c1-1-val-pilot/run_manifest.json --output-dir outputs/agent/c1-2-val-confirm` ran once on the same ten validation IDs already examined in C1 and C1.1. This is continuity data, not an untouched unbiased test. C1.2 obtained exact tool-name and call sequence 0/10, exact-call precision/recall/F1 0, missing/extra actions 24/2, invalid proposals 8, SDK validation 3/3 and reported successful executions 3/3. Union candidate recall was 17/25, step-aligned recall 6/10 (all first prefixes), and first-proposal tool accuracy 2/10. Eight finished with fewer calls than their labels; one progress recheck occurred. There were no invalid JSON errors, prompt-budget failures or timeouts, and two invalid-limit stops. Mean/p50/p95 latency was 2.16/1.86/3.34 s, model calls averaged 2.7/sample. Compared with C1.1 on the same IDs, invalid proposals worsened from 0 to 8 and missing actions from 20 to 24; the first-tool metric improved from 1/10 to 2/10, too small to establish a reliable improvement. Official score, hash match and objective task completion remain unavailable.

## Verification and decision

The real SDK smoke test validated a valid missing-information call, rejected an invalid call without executing it, recorded one execution and returned an SDK hash. `python -m unittest tests.test_action_agent_c1 tests.test_action_agent_c1_1 tests.test_action_agent_c1_2 tests.test_toolloop_ablation tests.test_toolloop_data tests.test_toolloop_feedback tests.test_toolloop_hardening tests.test_toolloop_synthesis -q` passed 97 tests. A post-run GPU snapshot showed an RTX 5060 Laptop GPU, 8,151 MiB total and 5,022 MiB in use, not a peak or a per-model measurement.

Retrieval omissions, first-action errors even with the target present, argument mismatches, and incomplete trajectories are all observed; these small experiments do not identify a dominant bottleneck. Research quality recovery is **not complete**, and C2 ASR integration is **not ready**. A controlled small SFT experiment is justified as a research test of action selection and argument grounding, not as an assertion that training will solve the problem. In the eight TRAIN records inspected, label `tool_calls` and `tool_results` were lists of equal lengths in every record; this does not establish dataset-wide result semantics or all intermediate observation alignment. Validate the full TRAIN data contract and trace availability first, use only legitimate observed outputs, and never fabricate intermediate reasoning. The single next experiment is a controlled TRAIN-only supervised first-action/argument baseline against the same retrieval conditions, with a fresh reserved validation subset and separately reported autonomous multi-step results; no training is performed here.
