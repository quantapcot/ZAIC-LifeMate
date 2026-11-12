# ToolLoop Phase 1: data and schema preparation

## Repository and scope

Work is isolated in `.worktrees/toolloop-synthesis` on `feature/toolloop-synthesis`, based on the locally available `origin/main` (`fe7f368`). At initial Phase 1 preparation, the original `feature/asr-chunkformer-vie` worktree had a pending merge and staged work; this hardening task did not switch or modify it. The new worktree does not contain the uncommitted ChunkFormer changes. No commits, pushes, model downloads, ToolEnv calls, LLM generation or fine-tuning were performed.

Inputs inspected read-only from the original workspace: `train/train.jsonl`, `train/train_tools_catalog.json`, `train/zaic_tools/README.md`, `train/zaic_tools/validate.py`, `train/zaic_tools/executor.py`, and the SDK tool registry; project README and existing benchmark/test conventions were also inspected. No environment state files or public-test labels were read. The local worktree automatically finds the sibling original `train/`; `--train-dir` can explicitly select a local train directory. The [ToolLoop paper](https://aclanthology.org/2026.emnlp-main.1262/) describes three stages and dynamic feedback; this phase defines interfaces only.

## Dataset audit

There are 1,545 valid records and no duplicate IDs. Every record has non-null `id`, `audio`, `audio_text` (strings), `context`, `label` (objects), and `history` (array). Null/missing counts are zero for these top-level fields; empty values and nested missingness are not exhaustively characterized. All labels have arrays `tool_calls` and `tool_results` and an object `final_state`; result and call counts match in the aggregate. Calls have ordered `name` and `arguments`. Stored final-state objects are labels, not an execution trace or proof of a particular dependency. Each context includes string `current_time`, `timezone`, and object `location`, `user_profile`, `installed_apps` in all records. History is empty in 527 records and contains alternating user/assistant entries otherwise; user entries have audio and audio text, and assistant entries may have text, calls and results.

There is one extra exact duplicate query and one extra normalized duplicate (NFC, case folding, punctuation/symbol removal and whitespace collapse); these are counts of records beyond the first, not published queries. No fuzzy/semantic near-duplicate search was performed. Counts by label calls: zero 111, one 369, two 428, three 359, four 161, five 107, six 10. There are 146 distinct labeled tool names, all in the catalog. `sys.ask_missing_info` appears in 217 calls across 217 records; its `tool_name` and `params` describe the missing information. A zero-call label is observable, but its cause is not inferred.

Scenario taxonomy for future work: `no_tool`, `single_tool`, `multi_call_unknown`, `parallel`, `multiple`, `parallel_multiple`, `sequential_dependent`, `missing_information`, `context_dependent`, `stateful`, and `similar_tool_disambiguation`. These dimensions can overlap. The only reliable automatic count classification here is zero/single/multiple (111/369/1,065); the 1,065 multi-call examples remain **unknown** for parallelism/dependency without additional evidence. A catalog description explicitly says `smart_home.light_set`'s `device_id` can come from `smart_home.get_status`; that documents a possible dependency, not a claim about every pair of calls in labels. No train examples are copied into this report.

## Catalog and runtime

The original catalog is version 2.7.0, with 46 apps and 147 tools: 62 read, 84 write and one meta. It contains 150 required and 96 optional top-level parameters: 204 string, 25 integer, 10 array, six boolean and one number. The registry provides exact descriptions, kinds, original parameter schemas, app/category filtering and required/optional lists; `get_tool` and `get_schema` return copies. It rejects duplicate names and structurally malformed definitions but does not claim full JSON Schema validation. Constraint occurrences across nested schema objects include 32 `enum`, 11 `items`, six `x-one-of`, 68 `x-ref` and 40 `x-match`; other and future keys remain untouched. Counts by app and full labeled tool-name counts are in ignored local `outputs/toolloop/audit.json`. App counts by catalog group: home 8, information 7, communication 6, planning 5, finance 5, mobility 3, media 3, shopping 2, health 2, personal 2, and one each in food, travel and control.

Catalog names and kind annotations matched the SDK's `TOOLS` registry (147 on each side). Runtime validation is authoritative for actual acceptance: the split catalog plus SDK `schema_error` and installed-app allowlist. The SDK validates its documented subset (including `x-one-of`, types, range, formats and array items); `x-ref`/`x-match` are preserved metadata and are not enforced as generic JSON Schema rules. Its validator checks known properties and does not reject unknown argument keys, so a future stricter verifier must distinguish project policy from runtime behavior. Structural registry validation is not a replacement for SDK validation or semantic/state checks. Runtime tool availability is sample-specific. No state was loaded or mutated.

## Splits and leakage

Seed 42; records are grouped by normalized query, with nonempty duplicates kept together. Seeded tie ordering and a deterministic size/scenario allocation minimize deviations from 80/10/10 and zero/one/multiple call proportions. Input order does not affect the result. Local results: train 1,236 (89 zero, 295 one, 852 multiple); validation 155 (11, 37, 107); test 154 (11, 37, 106). All IDs are unique and pairwise intersections of IDs and normalized queries are zero. The generated manifest records input and catalog SHA-256 and schema version; recomputation matched both source files after preparation.

Exact grouping does not prevent paraphrase, shared history, entity, template or state leakage; these need a stronger reviewed grouping strategy before treating the internal test as a robust held-out estimate. `outputs/toolloop/` is ignored and contains IDs, the manifest and aggregate audit, not copied records. No synthesis code exists in this phase. The train-only accessor now provides a checked train-only view; future generators must use it and must not be passed validation/test IDs or records. It is not an OS-level security boundary. Do not use official public test as training data.

## Contracts and future integration

`src/toolcall/data/contracts.py` defines serializable dataclasses: ground truth (`scenario_type`, candidates, targets and metadata), backward query (targets, natural Vietnamese query, allowed context and metadata), forward calls (query, ordered `{name, arguments}` and metadata), feedback (stage, validity, status, checks, issues, retry limits, failure reasons and history), and a separate trajectory with calls, optional observations, argument provenance and optional final state. Metadata tracks dataset version, source and validation status. These are designs, not runtime semantic validators. Later verification must establish Vietnamese naturalness, target/query consistency, required values and their allowed sources, schema defaults, and the absence of invented state or observations. Provenance should identify query, allowed context, earlier observation or actual schema default; it is not automatically enforced here.

The internal ordered call envelope resembles SDK call objects but is not itself a submission. Runtime requires installed-app checks and ordered `ToolEnv.call(name, arguments)` execution; results and final `get_tool_calls()`/`get_state_hash()` are produced by the SDK. A submission also needs the sample ID and must follow the current competition instructions. Phase 2 must separately design safe execution and verification, without inventing identifiers or results.

## Reproduction and tests

From `.worktrees/toolloop-synthesis`:

```powershell
python -m scripts.prepare_toolloop_data --help
python -m scripts.prepare_toolloop_data --dry-run
python -m scripts.prepare_toolloop_data --prepare
python -m unittest discover -s tests -v
```

`--dry-run` writes nothing. `--prepare` creates a new `outputs/toolloop` exclusively and refuses to overwrite an existing output directory or write over inputs. It reports aggregate stats and creates `splits/{train,val,test}_ids.json`, `splits/split_manifest.json`, and `audit.json`. Preserve or move existing local output before running preparation again. Missing/invalid source data, duplicate IDs, invalid catalogs and unknown labeled tool names fail rather than fabricate results.

Created `src/toolcall/data/{__init__,schema_registry,contracts}.py`, `src/toolcall/__init__.py`, `scripts/prepare_toolloop_data.py`, `tests/test_toolloop_data.py`, and this report. The existing `.gitignore` already ignores `/outputs/toolloop/`. Seven fixture tests pass on system Python without network, GPU or checkpoint. Full discovery passes eight tests using the existing `lifemate-chunkformer` Python environment; system Python's full discovery fails because the pre-existing ASR helper imports NumPy, which is not installed there. No ASR files were changed in this worktree.

## Phase 2 readiness and risks

The sampler needs to use the train-only data access boundary and select representative tools and scenarios. The backward generator needs controlled context, Vietnamese quality checks and no private data transmission without authorization. The forward generator needs the official schema and sample-installed tools; the verifier needs SDK-compatible deterministic checks plus semantic checks, explicit retry limits and audit history. Stateful trajectories require a separately reviewed execution policy and provenance. Fuzzy leakage, reliable dependency/parallel labels, stronger input validation, semantic constraints and a genuinely isolated held-out evaluation remain open; Phase 1 does not claim to solve them.


## Phase 1 Hardening ? Final Review

The original preparation did not provide a train-only accessor, performed only partial dataset and catalog validation, did not identify cross-split similarities, and could leave a partial-looking output after failure. Hardening addresses these points without changing the split algorithm or official tool schemas.

`TrainOnlyDataset.from_prepared_split` checks both input SHA-256 hashes, catalog version, valid ID lists and counts, uniqueness and pairwise disjointness, complete dataset coverage, seed/strategy, and exact deterministic split assignments. It validates the dataset against the catalog, checks the source hashes again after reading, retains only training records and returns copies. Its public `get` rejects validation, test and unknown IDs; its iterator returns only training records. Missing or corrupt metadata fails closed. The hashes and deterministic assignment checks detect accidental inconsistency, not malicious coordinated tampering, and this Python API cannot prevent direct file reads or introspection. For a future sampler:

```python
from pathlib import Path
from src.toolcall.data import TrainOnlyDataset

train_data = TrainOnlyDataset.from_prepared_split(
    train_path=Path("../../train/train.jsonl"),
    catalog_path=Path("../../train/train_tools_catalog.json"),
    splits_dir=Path("outputs/toolloop/splits"),
)
for record in train_data.iter_records():
    pass  # consume training records only
```

Preparation now validates required audio/text, context, history, label, call/result/final-state types and duplicate IDs, plus unknown tool names in calls. Optional history fields are validated when present. Errors refer to line numbers and fields without printing record contents. Registry checks duplicate app IDs and required names, basic nested schema field types and required-property membership; it preserves custom constraints and does not claim full JSON Schema compliance. The SDK and split catalog remain the runtime reference; a future stricter synthetic-data validator may reject unknown arguments even though the SDK currently ignores them.

The leakage audit examines cross-split pairs sharing a token: exact normalized query matches; equal templates after replacing digit runs; and lexical warnings with token-set Jaccard >= 0.75 and character SequenceMatcher >= 0.88. Counts can overlap; empty/token-disjoint pairs, paraphrases, history and context leakage are not covered. This is not semantic deduplication and does not reassign records. On the local data: **0 exact normalized cross-split pairs, 0 digit-template pairs, 34 lexical warnings**. No text or ID pairs are emitted.

Preparation validates before writing, stages and verifies JSON in a new temporary sibling under ignored `outputs/`, rechecks source hashes, and publishes with an atomic no-replace directory rename. Failure cleans only its own temporary directory; an interrupted process may leave a clearly named `.toolloop-preparing-*` temporary directory, which is not a valid published result. Existing destinations are never intentionally overwritten. Windows and Linux with `renameat2` support are supported; platforms lacking atomic no-replace rename fail closed. To preserve the prior output while preparing another local copy:

```powershell
python -m scripts.prepare_toolloop_data --prepare --output-dir outputs/toolloop/hardening-20261113
```

The 2026-11-13 local run again found 1,545 records and 147 tools. Splits remain 1,236/155/154, and every ID matches the prior manifest. Both old and new manifests passed train-only checks; validation and test lookups were rejected, and source dataset and catalog hashes matched. The new output is ignored by Git. Fourteen fixture tests pass on system Python without GPU or network; full discovery in the existing compatible ASR environment passes 15 tests, including the existing ASR helper. System Python still lacks the optional NumPy dependency needed by that unrelated pre-existing test.

The data foundation is ready for a controlled Ground Truth Sampler using the accessor. Backward/forward generation still requires implementation and evaluation of semantic validation, Vietnamese quality, provenance, stateful execution policy and the 34 lexical warnings; no generation has been run. The access API is not a sandbox and does not establish an unbiased held-out set.
