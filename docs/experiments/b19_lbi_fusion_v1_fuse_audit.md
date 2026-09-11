# LBI-Fusion v1 fuse preflight audit

The former lifecycle gate compared `expected[0]` and `fused(x)[0]` by row. In this pinned Detect implementation,
`[B,300,6]` is the postprocessed table: `get_topk_index()` chooses original grid IDs, then `postprocess()` gathers
their boxes. A row number is a score rank, not a stable candidate identity. The defective predicate has been replaced
with the three checks below; its original error statistics and traceback are still recorded.

This is a confirmed local verifier defect. The supplied server error transcription reports a CUDA B1 lifecycle
failure with maximum positional difference approximately 145.8067627, 735 elements outside tolerance, before native
B32 update auditing. The server screenshot/input/raw tensors were not available during this fix. The local
reproductions below establish the mechanism but do not establish the root cause of that particular server run.

## Comparison ownership and state

The original `expected` comes from the eval-mode candidate after setting its output projection to a nonzero test
fixture, before any optimizer update. Reload, EMA and native half-save controls use separate instances. The old
`fused` was also a deep copy of that same candidate. No evidence showed an EMA/FP16 or updated-state mismatch.
Cached reference outputs are now cloned immediately, and lifecycle failures own their own receipt rather than
depending on a successful return to `model_checks`.

`fuse_audit()` creates two same-source eval copies and records both states, the input and inference attributes.
It checks exact pre-fuse state equality, unchanged unfused state/input, eval status of all modules including BN,
head stride/anchors/shape/export/end2end/max_det/dtype/device, feature grids, and all four unchanged LBI weights.
Native Conv-BN folding removes one2many; only the retained one2one branch is compared across folding.
The source model is never fused or modified. LBI zero and nonzero projections are both exercised, with a native
b19 control on the same input and device. One failing variant does not prevent collecting the other controls;
any failure still rejects the lifecycle.

## Gate and retained diagnostics

1. Compare **every original candidate** in raw one2one boxes and classification logits, then all decoded boxes and
   sigmoid scores, using the unchanged `atol=rtol=1e-4`. Box and score units have separate statistics. A failure
   rejects the audit and triggers per-layer observations, including complete Conv+BN+activation blocks, layer15,
   downstream blocks and one2one output convolutions. The first layer outside the existing tolerance is recorded
   as a localization result, not automatically labeled an implementation bug.
2. Capture actual candidate IDs and class IDs from the very `get_topk_index()` call that generated each final table.
   This does not re-run top-k to guess IDs. Require valid unique IDs, class 0 for fixed b19 nc=1, exact correspondence
   between decoded candidates and final gathered values, descending scores and the correct kth threshold over all
   candidates. This catches an omitted higher-scoring candidate even if both compared tables omit it. There is no
   confidence filter, coordinate sort or IoU matching.
3. Record changed rank positions, dropped/added IDs, both scores, both kth margins and measured perturbations.
   Compare the **union** of selected IDs against the corresponding full candidate tensor on the other side, including
   candidates absent from that side's final table. For every dropped ID `d` and added ID `a`, require
   `0 <= score_before[d] - score_before[a] <= abs(delta[d]) + abs(delta[a])`. This bound uses actual score changes,
   not an extra tolerance. Correct top-k membership is independently required on both sides. Exact ties may choose
   different identities; unexplained changes and genuine raw/decode discrepancies fail.

The helper is intentionally limited to the fixed single-class b19 head, and rejects an incompatible head configuration.
Production Detect, model layers, native fuse and all precision/environment policies are unchanged. Temporary observers
run only on diagnostic copies and restore their bound methods/hooks in `finally`. Evidence tensors are detached CPU
clones, preventing downstream in-place aliases and persistent GPU references.

## Evidence layout

In each existing preflight output directory, `cpu/` and `cuda_0/` contain `lifecycle_checks.json`, plus
`fuse_native/`, `fuse_lbi_zero/` and `fuse_lbi_nonzero/`. Each fuse directory has:

- `source.pt`: original CPU input, unfused state, model YAML/names and attributes sufficient for reconstruction.
- `fused_state.pt`: the independently folded state and attributes.
- `before.pt`, `after.pt`: raw one2one tensors, all decoded candidates, actual indices/classes/selected scores,
  final predictions, input after inference, grids, original device/dtype and head caches.
- `audit.json`: actual Git SHA, source hashes/dirty paths, seed, Python/PyTorch/CUDA/cuDNN/Ultralytics versions,
  GPU, backend precision settings, fixed tolerances, numerical/rank summaries and complete failure traceback.
- On raw/decode failure, `layer_outputs.pt` plus per-layer errors and the first differing layer in `audit.json`.

The source, fused state and complete before/after tensors are written before equality assertions. The initial
lifecycle source/reference and partial receipt also survive a reload/EMA failure. Canonical packaging now retains
regular `.pt` evidence under provenance; the former blanket exclusion would otherwise silently strip these tensors.
Attempt-directory and symlink exclusions remain in place. Checksums and archive round trips cover included evidence.
Failed preflight directories remain separate and are never replaced by later attempts.

## Local reproduction and regression scope

On local Python 3.11.15 / torch 2.7.1+cu118 / RTX 2060, using the original pretrained checkpoint adapted to nc=1:

- CPU native and nonzero LBI each exhibited two exchanged ranks with the same 300-candidate set. The former
  positional maximum errors were about 87.0895 and 91.0680; all raw/decode and candidate identity checks passed.
- A bounded CUDA search over input seeds 100–131 found a nonzero LBI boundary change at seed 107. Original ID 294
  was replaced by ID 396 at rank 300. Their initial scores were exactly `0.0009350699256174266`; ID 396 increased by
  `4.656612873077393e-10`, while ID 294 was unchanged. The positional maximum box error was `60.672950744628906`.
  All 525 raw/decoded candidates passed, and the selected union contained 301 audited candidates. The same-input
  native and zero-LBI controls passed with unchanged selected identities. These are hypothesis reproductions,
  not a measured frequency of failures and not server input replay.

Local evidence is kept in `runs/lbi_local/fuse_fix/initial/` and `cuda_reproduction/`, including the first observed
CUDA failure, its input seed, search ledger, actual tensors and reconstructable model states. Existing evidence from
before this fix is preserved.

Regressions cover tied and nearly tied permutations, boundary replacements, unselected-candidate box/score corruption,
wrong gather/class/duplicate IDs, omission of a clearly better candidate, real CPU/CUDA fuse for native and both LBI
states, capture cloning/restoration, and injected head corruption with failure artifacts and layer localization.
Existing zero-init/shared-state, staged MuSGD, AMP, cross-process reload, EMA and package dry-run checks remain.

Local B1/160 lifecycle checks and B2 synthetic learning do not authorize a server B32/640 preflight PASS. The exact
server runtime has not been rerun here. The runner remains unchanged: after a matching successful independent
preflight, a fresh formal trainer initializes model/optimizer/EMA/GradScaler/RNG. No formal training was started.
The fixed server worktree and `bash tools/experiments/server_b19_lbi_fusion_v1.sh train` interface remain unchanged;
this repair does not execute that command.

## Review and references

Deleted: the positional final-table pass/fail predicate and the blanket exclusion of provenance `.pt` files.
Reused: native decode/top-k/fuse, `common.assert_close_tree`, inference-attribute/backend/source utilities and the
existing lifecycle/package pipeline. Net additions are necessary because deletion alone cannot supply the required
candidate identities, boundary proofs, reconstructable failure evidence and negative regression controls.
No added condition suppresses a true numerical failure: numerical and postprocess correctness are independently gated.

The actual repository source is authoritative. PyTorch documents that tied top-k indices can vary in
[torch.topk](https://docs.pytorch.org/docs/2.8/generated/torch.topk.html), and mathematically equivalent floating-point
calculations need not be bitwise identical in [numerical accuracy](https://docs.pytorch.org/docs/2.8/notes/numerical_accuracy.html).
The online [Detect](https://docs.ultralytics.com/reference/nn/modules/head/) and
[BaseModel](https://docs.ultralytics.com/reference/nn/tasks/) references were also consulted, without replacing the
pinned local implementation or using those general facts to waive a failed candidate-level check.
