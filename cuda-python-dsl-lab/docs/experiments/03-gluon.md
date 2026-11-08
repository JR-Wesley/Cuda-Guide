# 03 — Gluon G3.1–G3.8: ownership, communication, and real async copies

## Scope and status

Uses **only the existing `.venvs/triton`**, with Triton **3.9.0**; no packages
were installed or changed. The recorded GPU run used an NVIDIA GeForce RTX 4060
Laptop GPU, compute capability **8.9**, as recorded in the
[GPU environment log](../../runs/20261105T060458Z-learn-cbx4cfro/02-gluon-gpu.log).

**Recorded functional result: 34/34 GPU cases passed, with two correctness
repetitions per case.** Separately, **10/10 ownership probes passed**, each with
two poisoned-output GPU runs, together with their CPU mapping checks.

Evidence from `runs/20261105T060458Z-learn-cbx4cfro/`:

- [Functional results](../../runs/20261105T060458Z-learn-cbx4cfro/gluon/results.json)
  and [run summary](../../runs/20261105T060458Z-learn-cbx4cfro/SUMMARY.md).
- [Ownership summary](../../runs/20261105T060458Z-learn-cbx4cfro/G3.2-layouts/summary.json):
  four 1-D, two 2-D order, two transpose-layout, and two reduction-layout probes.
- [Manifest and command exit codes](../../runs/20261105T060458Z-learn-cbx4cfro/manifest.json).
- [Static report](06-static.md#gluon-layout-conversion-and-asynchronous-copy-groups)
  analyzing actual exported PTX/cubins, including SASS disassembly.

The [ownership log](../../runs/20261105T060458Z-learn-cbx4cfro/03-G3.2-layouts.log)
and [functional log](../../runs/20261105T060458Z-learn-cbx4cfro/04-gluon-functional.log)
both contain **nanobind shutdown/leak warnings**, followed by **exit code 0 (rc=0)**.
The warnings are retained as observed, not treated as absent or as a sanitizer result.
**Sanitizer and performance acceptance remain pending**: the recorded run was
functional-only, with no timing or hardware-counter measurements. These passes
validate the recorded inputs/configurations, not every learning-plan acceptance item.

| Exercise | Implementation | Validation so far |
| --- | --- | --- |
| G3.1 | Add with two layouts; row sum with two layouts | All corresponding tiny GPU cases pass, two repetitions each |
| G3.2 | 128/256-element 1-D derivation with warp/lane/logical-slot CSVs | CPU checks and GPU ownership probes pass within the 10-probe mapping record |
| G3.3 | SPT=2, both 2-D orders, actual `%tid.x` owner probes | Add cases and mapping probes pass; physical register assignment not measured |
| G3.4 | One versus four warps per row | Both layouts pass; shuffle/shared/barrier instructions inspected; no timing |
| G3.5 | Direct transpose versus explicit `convert_layout` | Both pass; actual PTX/SASS conversion lowering inspected |
| G3.6 | Register copy, synchronous shared copy, Ampere async shared copy | All modes pass; actual async PTX/SASS inspected |
| G3.7 | One shared slot versus two-slot prefetch/consume pipeline | Both pass; distinct wait depths verified in PTX/SASS; sanitizer/performance pending |
| G3.8 | Compiled-kernel artifact closure | Actual runtime compiler artifacts exported; selected cubins disassembled in the static report |

Files:

- `experiments/gluon_suite.py`: `Experiment`/`Workload` adapters and artifact closure.
- `experiments/vector/gluon_ops.py`: Add, copy variants, 1-D owner probe.
- `experiments/reduction/gluon_ops.py`: row sum.
- `experiments/transpose/gluon_ops.py`: transpose and 2-D owner probe.
- `experiments/gluon_layouts.py`: standalone, standard-library-only CPU derivation CLI;
  imports Torch/Triton only when `--verify-gpu` is explicitly requested.

The suite is integrated with the shared runner and the `python3 commands/learn/run.py` entry point.
The recorded run preserves its source snapshot, manifest, logs, results, and actual
compiler artifacts; those records, rather than development cache files, are the
functional-validation evidence.

## Commands

Run from `cuda-python-dsl-lab`. The independent action replaces its own
`commands/learn/log/` and defaults to **functional validation only, with no benchmarks**.
The ownership module writes CSV mappings, diagrams and the check summary, without
copying its source or generating `source_sha256`. Historical snapshots above remain unchanged:

```bash
python3 commands/learn/run.py gluon
```

This runs the 34 tiny functional cases and the separate ownership module with GPU
verification. Run GPU suites serially rather than concurrently.

For just the **CPU-only** mapping derivation, use a new output directory:

```bash
.venvs/triton/bin/python -m experiments.gluon_layouts --output runs/gluon-layouts-cpu
```

It writes `summary.json`, a source snapshot/checksum, ten complete mapping CSVs,
and `maps.md` (full 1-D maps, labelled 2-D crops). The summary records the original
invocation, interpreter, and completion status. Checks reject omissions, duplicates,
out-of-bounds coordinates, and mismatched inverse/closed-form mappings. Replicated/sub-tile layouts are explicitly outside
this CPU derivation, rather than being incorrectly described as unique ownership.


The suite uses the shared `run_cli` unchanged. No `--bench` means correctness only;
all outputs are preallocated, with shared references/tolerances and the runner's two
poisoned-output launches. No configuration or large-size scan runs by default.
The suite has 34 cases, ordered numerically by their leading learning-plan ID:

| Prefix | Cases | Example ID |
| --- | --- | --- |
| `G3.1` | 4 baseline Add + 3 baseline row sums | `G3.1-add-spt1-255`, `G3.1-sum-w1-3x129` |
| `G3.3` | 4 Add cases with SPT 2 | `G3.3-add-spt2-255` |
| `G3.4` | 3 row sums with four warps per row | `G3.4-sum-w4-3x129` |
| `G3.5` | 4 direct/converted transposes | `G3.5-transpose-convert-37x53` |
| `G3.6` | 12 register/synchronous-shared/async-single copies | `G3.6-copy-async_single-4099` |
| `G3.7` | 4 async-double copies | `G3.7-copy-async_double-4099` |

Add/copy sizes remain `1,255,256,4099`; Add uses block 256, copy block 128 and
four tiles/CTA. Row-sum shapes remain `1x127,3x129,2x4097`; transpose shapes remain
`32x32,37x53` with `32x32` tiles. All 34 cases passed in the linked functional run;
benchmark timing was not enabled.

Every case exposes `plan_ids` in its catalog config and resulting Workload metadata.
All include `G3.8` for compiler artifact export. Add variants cover `G3.1/G3.3`,
row-sum variants cover `G3.1/G3.4`, and both async copy variants cover `G3.6/G3.7`.
These overlapping IDs retain comparison baselines without duplicating launches.
`G3.2` and the `G3.3` CPU mapping exercises remain in the separate mapping CLI;
the full `learn gluon` command also invokes its GPU probes. `G3.8` is artifact
coverage rather than an additional GPU case.
Old `gluon-*` suite IDs are replaced, not retained as aliases.

For a targeted functional rerun (these commands launch GPU kernels):

```bash
python3 commands/learn/run.py gluon --ids G3.7
python3 commands/learn/run.py gluon --ids G3.6,G3.7
```

Plan-prefix matching now accepts the hyphenated case IDs: `--ids G3.7` selects the
four `G3.7-copy-async_double-*` cases. Use `G3.6,G3.7` to include the single-buffer
and other copy baselines. Filtered runs do not invoke the separate ownership module.

Performance is opt-in via `--bench`; none is requested by the commands above.
Sanitizer and performance acceptance remain pending despite the functional passes.
These tiny shapes are not a throughput study.

## G3.1–G3.4: what becomes explicit

The Add algorithm is unchanged from a Triton implementation: form offsets, mask the
last tile, load A/B, add, store. Gluon additionally specifies how logical elements
are distributed over registers, lanes, and warps. Both Add layouts use four warps:

| Use | size_per_thread | threads_per_warp | warps_per_cta | order |
| --- | --- | --- | --- | --- |
| Add SPT 1 | `[1]` | `[32]` | `[4]` | `[0]` |
| Add SPT 2 | `[2]` | `[32]` | `[4]` | `[0]` |
| Sum: warp per row | `[1,1]` | `[1,32]` | `[4,1]` | `[1,0]` |
| Sum: four warps per row | `[1,1]` | `[1,32]` | `[1,4]` | `[1,0]` |
| 2-D derivation | `[1,2]` | `[4,8]` | `[2,2]` | `[1,0]` or `[0,1]` |

### One-dimensional derivation

Let `w` be warp 0–3, `l` lane 0–31, `q` a logical per-thread element slot.
For SPT 1:

```text
index = l + 32*w + 128*q
128 elements: q=0, one element/thread
256 elements: q=0,1, two elements/thread
warp 0, slot 0: 0..31;   warp 3, slot 0: 96..127
warp 0, slot 1: 128..159; warp 3, slot 1: 224..255
```

For SPT `s`:

```text
index = s*(l + 32*w) + (q % s) + 128*s*(q // s)
```

At SPT 2 and 256 elements, lane 0 owns `0,1`, lane 1 owns `2,3`, and
warp 3/lane 31 owns `254,255`. The 512-element exercise adds the next CTA tile
in the same thread's logical slots. Logical slots are **not** physical `%rN`
register names; register allocation/liveness is a separate compiler decision.

### Two-dimensional derivation

`order[0]` is the fastest-changing dimension, at each local/lane/warp/tile level.
For each dimension `d`, decode the lane and warp into mixed-radix coordinates:

```text
CTA_tile[d] = SPT[d] * TPW[d] * WPC[d]
coordinate[d] = local[d]
              + SPT[d] * (lane_coordinate[d] + TPW[d]*warp_coordinate[d])
              + CTA_tile[d]*repeat_coordinate[d]
```

For the `16x64` examples, the CTA tile is `8x32`; each of 128 threads has eight
logical slots. With `q` in 0–7, the two explicit closed forms are:

```text
order [1,0]:
  row = l//8 + 4*(w//2) + 8*(q//4)
  col = 2*(l%8 + 8*(w%2)) + q%2 + 32*((q//2)%2)
order [0,1]:
  row = l%4 + 4*(w%2) + 8*((q//2)%2)
  col = 2*(l//4 + 8*(w//2)) + q%2 + 32*(q//4)
```

The CPU module checks the entire domain, not just printed samples. Optional GPU
probes store **actual PTX `%tid.x`** for each element; the expected value is
`32*warp + lane`. They compare two independently poisoned outputs against the CPU
model and export their compiled artifacts. All ten probes passed both runs in the
linked ownership summary. They check physical thread ownership, not physical
register-slot assignment or a separate atomic write-count metric.

### Reduction communication

Both variants process four rows per CTA, zero-pad columns to the next power of two,
mask invalid rows/columns, and call `gl.sum(value, 1)` with FP32 accumulation.
`SliceLayout` constructs row and column index vectors consistently with the parent
layout; row indices use `SliceLayout(1, parent)`, columns `SliceLayout(0, parent)`.

- **Warp per row:** each thread first reduces its local column elements; a row is
  contained within a warp, so warp shuffle communication suffices. No row partial
  sum needs to cross warps.
- **Four warps per row:** each thread has contributions for several rows; each
  warp reduces its partials, then the four warp partials must communicate.
  On the inspected SM89 compilation this uses shared scratch and a CTA barrier.

This is a layout-only comparison of the same masked reduction algorithm, not a
claim that either is faster. Compared with implicit Triton layouts, explicit
ownership is the main source-level change; reduction order can still change FP32
rounding. Shared tolerances are not weakened.

## G3.5: transpose conversion

Both kernels read a logical `32x32` tile with
`BlockedLayout([1,1], [4,8], [4,1], [1,0])` and write address `Y[col*M + row]`.
The direct baseline keeps that ownership at the store. The conversion variant
performs `gl.convert_layout(value, BlockedLayout([1,1], [8,4], [1,4], [0,1]))`
and rebuilds the row/column index tensors in the destination layout.

The logical tensor axes are unchanged by conversion; **ownership** changes, while
the transposed address expression implements transpose. Separate masks cover the
partial `37x53` input tile and the contiguous `53x37` output.

Both direct and converted variants passed on `32x32` and `37x53`. This particular
conversion lowers to shared stores, a barrier, and shared loads in the actual
PTX/SASS inspected by [the static report](06-static.md#gluon-layout-conversion-and-asynchronous-copy-groups).
That observation must not be generalized to all conversions: some are register-only,
warp shuffles, or eliminated entirely.

## G3.6–G3.7: shared and asynchronous movement

Installed sources inspected:

- `triton.experimental.gluon.language.nvidia.ampere.async_copy`
- `gl.allocate_shared_memory`, `gl.shared_memory_descriptor.index/load/store`
- `gl.barrier`, `SwizzledSharedLayout`
- `triton.experimental.gluon._runtime.GluonASTSource` for offline compilation

`async_copy.async_load(smem, pointers, mask)` is the supported API.
The similarly named `async_copy_global_to_shared` exists but **raises an error**
in this version; it is not used. No Hopper TMA, mbarrier, or warp specialization
is required. Runtime adapters reject non-NVIDIA/pre-SM80 devices.

Each copy CTA handles four consecutive 128-element tiles with identical masks:

| Mode | Data path | Shared slots | Async groups |
| --- | --- | --- | --- |
| `registers` | global → registers → global | 0 | none |
| `shared_sync` | global → registers → shared → registers → global | 1 | none |
| `async_single` | global → shared via `cp.async`, then registers → global | 1 | commit then wait 0 for each tile |
| `async_double` | same async data path, next tile prefetched | 2 | oldest-group waits 1; final drain 0 |

The rank-1 shared layout is `SwizzledSharedLayout(1,1,1,[0])`. A double-buffer
allocation has shape `[2,128]` but keeps the **rank-1 layout**: the leading extent
selects buffers and `index(slot)` returns a rank-1 shared descriptor. A rank-2
layout on this allocation was rejected by the installed descriptor API during
development and was corrected before the successful compile checks.

Responsibilities are distinct:

1. `async_load` issues global-to-shared transfers; masked lanes do not read invalid
   input, and no masked output is stored.
2. `commit_group` closes one async group; it does not wait for data.
3. `wait_group(k)` completes the older groups while allowing at most `k` to remain.
4. A CTA barrier after waiting coordinates readiness across participating threads.
5. Loading the current slot consumes its values into registers; another CTA barrier
   ensures all readers finish before any thread reuses that shared slot.

Double-buffer schedule (S0 and S1 are different 512-byte regions):

```text
issue tile 0 → S0; commit
issue tile 1 → S1; commit
wait 1; barrier; read S0; barrier/release S0; store tile 0
issue tile 2 → S0; commit
wait 1; barrier; read S1; barrier/release S1; store tile 1
issue tile 3 → S1; commit
wait 1; barrier; read S0; barrier/release S0; store tile 2
wait 0; barrier; read S1; barrier/release S1; store tile 3
```

The next group can remain in flight while the current slot is consumed. This is
real two-buffer prefetching, not two synchronous copies renamed as a pipeline.
Conservative barriers are intentional for this teaching example. There is little
computation to hide latency, so neither overlap magnitude nor a speedup is claimed.
The number of explicit shared slots is independent of compiler `num_stages` defaults.
All four copy modes passed the recorded small functional cases twice. The actual
single-/double-buffer PTX and SASS confirm different wait depths, but these passes
and static observations do not replace sanitizer acceptance or measure overlap.

## G3.8: compiler evidence and limits

`Workload.artifacts` captures the actual compiled kernel returned by `launch`.
Calling it before the first launch raises a clear error. Afterwards it returns all
text/byte entries of `kernel.asm` under their real stage names, plus `compiler.json`
with actual compiler metadata and available register/spill information. It does not
fabricate missing stages. The recorded run exports these actual compiler artifacts:

```text
kernel.source  kernel.glir  kernel.ttgir  kernel.llir  kernel.ptx  kernel.cubin
compiler.json
```

The authoritative runtime artifacts are under the per-case evidence directories
linked from the [functional run summary](../../runs/20261105T060458Z-learn-cbx4cfro/SUMMARY.md).
The [static report](06-static.md#gluon-layout-conversion-and-asynchronous-copy-groups)
selects six Gluon cases from that successful run: the four copy modes at `N=4099`
and direct/converted transpose at `37x53`. It preserves the actual PTX/cubins and
raw disassembly logs; it does not infer machine code from source or re-run kernels.
Vector/reduction cases were outside that report's bounded selection.

Observed **static** properties of those actual artifacts:

| Case | Actual PTX | Actual SASS |
| --- | --- | --- |
| `G3.6-copy-registers-4099` | Global load/store; no matched shared/async instructions | `LDG.E`, `STG.E` |
| `G3.6-copy-shared_sync-4099` | Shared stores, `ldmatrix`, barriers; no `cp.async` | `STS`, `LDSM.16.M88`, barriers |
| `G3.6-copy-async_single-4099` | Four copies/commits; wait depths `0,0,0,0` | Four `LDGSTS.E`, four `LDGDEPBAR`, four `DEPBAR.LE SB0, 0x0` |
| `G3.7-copy-async_double-4099` | Four copies/commits; wait depths `1,1,1,0` | Four `LDGSTS.E`, four `LDGDEPBAR`; three `DEPBAR.LE SB0, 0x1`, final `0x0` |
| `G3.5-transpose-direct-37x53` | Global loads/stores; no matched shared/barrier instructions | No matched shared-memory exchange |
| `G3.5-transpose-convert-37x53` | Two `st.shared.v4.b32`, two `ld.shared.v4.b32`, one `bar.sync` | Two `STS.128`, two `LDS.128`, one `BAR.SYNC.DEFER_BLOCKING` |

For direct evidence, see the preserved
[single-buffer PTX](../../runs/20261105T061056Z-static-3xunxkwj/evidence/cases/15-gluon-copy/artifacts/kernel.ptx),
[double-buffer PTX](../../runs/20261105T061056Z-static-3xunxkwj/evidence/cases/16-gluon-copy/artifacts/kernel.ptx),
[double-buffer SASS](../../runs/20261105T061056Z-static-3xunxkwj/evidence/commands/019.log),
and [conversion SASS](../../runs/20261105T061056Z-static-3xunxkwj/evidence/commands/023.log).
Use the static report's corrected `analysis-v2` summary and linked compiler/resource
metadata, not development cache files or assumed resource counts.

In these copy specializations the shared **loads lower to `ldmatrix`**, not to
literal `ld.shared` instructions. Searching only for `ld.shared` would incorrectly
report no shared loads. Static instruction counts are not dynamic traffic, cache
hit rates, throughput, or runtime synchronization proofs.

**Completed evidence:** small GPU functional correctness (34/34 cases, two repeats),
CPU mapping checks and actual GPU thread ownership (10/10 probes, two poisoned runs
each), compiler artifact export, and selected PTX/SASS analysis.
**Still pending:** sanitizer and performance acceptance, physical-register mapping,
latency/overlap/bandwidth measurements, measured occupancy, and any speedup claim.
Functional success and rc=0 do not resolve the observed nanobind shutdown warnings
or replace those remaining checks. The optional advanced MMA/GEMM extension of the
learning plan is outside G3.1–G3.8 and is not implemented here.
