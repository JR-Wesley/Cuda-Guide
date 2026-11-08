# 06 — Offline static evidence from the four successful learning runs

## Scope and evidence entry points

This experiment reads existing artifacts only. **No GPU queries, kernel launches,
compilation/JIT, profiling, package installation, or performance measurement occurred.**
Historical PASS results identify eligible inputs; they are not new correctness tests.
The historical collector was subsequently migrated to commands/static/run.py. It does not import backend GPU modules.

Evidence root: `runs/20261105T061056Z-static-3xunxkwj/`.

- **[Corrected static summary](../../runs/20261105T061056Z-static-3xunxkwj/analysis-v2/SUMMARY.md)**
  and [structured JSON](../../runs/20261105T061056Z-static-3xunxkwj/analysis-v2/summary.json).
- [Original collection summary](../../runs/20261105T061056Z-static-3xunxkwj/evidence/SUMMARY.md),
  [invocation](../../runs/20261105T061056Z-static-3xunxkwj/evidence/invocation.json),
  [exit status](../../runs/20261105T061056Z-static-3xunxkwj/evidence/exit.json).
- [Every tool command, rc and raw-log hash](../../runs/20261105T061056Z-static-3xunxkwj/evidence/commands.json),
  [input snapshots, original relative paths and hashes](../../runs/20261105T061056Z-static-3xunxkwj/evidence/files.json),
  [historical source and original collector hashes](../../runs/20261105T061056Z-static-3xunxkwj/evidence/source-sha256.json).
- [Final collector snapshot](../../runs/20261105T061056Z-static-3xunxkwj/analysis-v2/learning_static.py),
  [revision invocation](../../runs/20261105T061056Z-static-3xunxkwj/analysis-v2/invocation.json),
  [revision validation](../../runs/20261105T061056Z-static-3xunxkwj/analysis-v2/analysis.log).

There was **one collector/disassembly invocation**, exit 0. Inspection then exposed a
PTX parser omission: CuTe prints the MMA opcode on one line and its operands on subsequent
lines. The parser now accepts an opcode at end of line. A **CPU-only analysis revision**
reparsed the preserved PTX, correcting the two CuTe MMA counts to 1 and 8. It issued
**zero additional tool commands**. `analysis-v2/` preserves the final script, revised
JSON/Markdown and exact revision command; `evidence/` remains unchanged, including its
original script and incomplete first-pass PTX counts. All original file hashes were
checked before/after; see
[original evidence hashes](../../runs/20261105T061056Z-static-3xunxkwj/analysis-v2/original-evidence-sha256.json).
Use the corrected summary for instruction counts, not the first-pass summary.

Final collector SHA256:
`14414b27bbf9f4b351e4fb482a71e8bff0ad9310de6dec6a7c3fc3a1688f8362`.

## Inputs and bounded selection

| Backend | Input run root | Input cases | Selected cases | Cubins / PTX / TileIR |
| --- | --- | ---: | ---: | ---: |
| Triton | `runs/20261105T060430Z-learn-ne9a7tkh` | 40 | 6 | 7 / 7 / 0 |
| Gluon | `runs/20261105T060458Z-learn-cbx4cfro` | 34 | 6 | 6 / 6 / 0 |
| CuTe | `runs/20261105T060524Z-learn-9jj6_yq0` | 23 | 6 | 6 / 6 / 0 |
| cuTile | `runs/20261105T060543Z-learn-osusvmpm` | 15 | 7 | 0 / 0 / 7 |
| **Total** | **4 runs** | **112** | **25** | **19 / 19 / 7** |

Selection groups PASS cases by backend, operation and full config except `plan_ids`.
For each group it selects the largest shape product, then breaks ties lexically by
run path and evidence directory. IDs alone are not unique in Triton, so original
`backend/case/artifacts/filename` paths remain part of the identity.

Focus: Triton GEMM, transpose and fusion; Gluon transpose and all four copy modes;
CuTe shared-copy value layouts, transpose and MMA/GEMM; cuTile all operations/configs.
This intentionally excludes vector/reduction examples from the machine-code backends.
It does not scan every shape or optimize/tune configurations. The full 112-case
selection inventory is in `summary.json`.

The hard limit is **20 cubin files across all positional run roots**. The collector
rejects an oversized selection rather than truncating it or dropping half of an
unfused pair. Selected cases retain **all** their artifacts, including IR/compiler
metadata; it does not search caches or mistake historical snapshots for live source.

## Current independent CLI

From cuda-python-dsl-lab:

```bash
python3 -B commands/static/run.py commands/learn/log
```

Supply one or more existing roots containing backend/results.json. Historical runs may also be supplied read-only. Output is fixed at commands/static/log/ and replaced on each invocation; there is no --output option, source snapshot, artifact copying, hashing, timestamp directory or archive. The script reads existing PTX/cubin files in place and fails on missing inputs without compiling or launching kernels. CUOBJDUMP can select the tool; the default is cuobjdump from PATH.

The following counts and logs describe the historical collection, not a new validation of this migrated entry.

The collector uses only Python's standard library and `/opt/cuda/bin/cuobjdump`:

- One `--version` command.
- `--dump-sass` and `--dump-resource-usage` for each of 19 snapshotted cubins.
- **39 commands, all rc=0; zero failed commands and zero recorded collection errors.**
- Tool version actually observed: **CUDA 13.4, V13.4.92**;
  [raw version output](../../runs/20261105T061056Z-static-3xunxkwj/evidence/commands/000.log).
- `nvdisasm` is not invoked. PTX is read from the exported `.ptx` files, not fabricated
  or reconstructed from SASS. Observed PTX/SASS targets are `sm_89`; this is a binary
  property, not a fresh device/driver compatibility test.

Each command has an argv array, cwd, timestamps, 60-second timeout, actual return code
(or null if process creation failed), status, input provenance and raw combined
stdout/stderr log. The command record is created before execution and finalized
with the log hash. Failures/timeouts are retained and cause a nonzero collector exit.
No profiler, GPU utility or DSL runtime is a fallback.

The original collection preserved **329 files**: **200 historical source files**
(50 per run), **1 collector script**, **12 input manifests** (run manifest, source
hash manifest and backend results for each run), and **116 selected artifact files**.
Of these, **316 hashes matched existing source/artifact manifests**; the other 13
files have newly recorded hashes without a prior expected digest. The revision
adds a final collector snapshot; it does not replace any original bytes.

## Observed static facts

Counts below are **instruction sites in the exported text/disassembly**, not dynamic
execution counts. Every selected artifact has line-numbered matched instructions in
`analysis-v2/summary.json`; raw files and complete SASS/resource logs are linked from
its Markdown summary.

### Triton: MMA, transpose and fusion

- **GEMM**, input `(37,53,29)`: tile `(16,32,16)` has 2
  `mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32` PTX sites and 2
  `HMMA.16816.F32` SASS sites. Tile `(32,32,16)` has 4 of each.
  Both contain `ldmatrix`/`LDSM`. Evidence:
  [16×32 PTX](../../runs/20261105T061056Z-static-3xunxkwj/evidence/cases/21-triton-gemm/artifacts/gemm.ptx),
  [SASS](../../runs/20261105T061056Z-static-3xunxkwj/evidence/commands/031.log),
  [32×32 PTX](../../runs/20261105T061056Z-static-3xunxkwj/evidence/cases/22-triton-gemm/artifacts/gemm.ptx),
  [SASS](../../runs/20261105T061056Z-static-3xunxkwj/evidence/commands/033.log).
- **Naive and tiled transpose**, input `(37,53)`: both contain shared loads/stores
  and a barrier. Naive has 2 `st.shared.b32`, 2 `ld.shared.b32` and one `bar.sync`;
  tiled has 8, 8 and one respectively. SASS has corresponding `STS`, `LDS` and
  `BAR.SYNC.DEFER_BLOCKING` sites. **The name “naive” does not imply no shared-memory
  lowering.** Evidence:
  [naive PTX](../../runs/20261105T061056Z-static-3xunxkwj/evidence/cases/23-triton-transpose/artifacts/naive_transpose.ptx),
  [SASS](../../runs/20261105T061056Z-static-3xunxkwj/evidence/commands/035.log),
  [tiled PTX](../../runs/20261105T061056Z-static-3xunxkwj/evidence/cases/24-triton-transpose/artifacts/tiled_transpose.ptx),
  [SASS](../../runs/20261105T061056Z-static-3xunxkwj/evidence/commands/037.log).
- **Fusion**, `N=4099`: unfused retains `scale` and `add` cubin/PTX pairs, each with
  one entry/function; fused retains one `vector` pair with one entry/function.
  The separate scale SASS contains `FMUL`, add contains `FADD`, and fused contains
  both `FMUL` and `FADD` (not an observed `FFMA`). This is **two exported kernels
  versus one**, consistent with historical `launches_per_call=2/1`; no launches
  were traced here. Evidence:
  [scale SASS](../../runs/20261105T061056Z-static-3xunxkwj/evidence/commands/029.log),
  [add SASS](../../runs/20261105T061056Z-static-3xunxkwj/evidence/commands/027.log),
  [fused SASS](../../runs/20261105T061056Z-static-3xunxkwj/evidence/commands/025.log).
  Kernel fusion is not synonymous with floating-point instruction contraction.

### Gluon: layout conversion and asynchronous copy groups

For transpose `(37,53)`, direct-layout PTX has global loads/stores but no matched
shared-memory or barrier instructions. The conversion variant has two
`st.shared.v4.b32`, two `ld.shared.v4.b32` and one `bar.sync`; SASS has two `STS.128`,
two `LDS.128` and one `BAR.SYNC.DEFER_BLOCKING`.
See [direct SASS](../../runs/20261105T061056Z-static-3xunxkwj/evidence/commands/021.log),
[conversion PTX](../../runs/20261105T061056Z-static-3xunxkwj/evidence/cases/18-gluon-transpose/artifacts/kernel.ptx)
and [conversion SASS](../../runs/20261105T061056Z-static-3xunxkwj/evidence/commands/023.log).
This supports a shared-memory exchange in this conversion, not a universal cost
model for `convert_layout`.

Copy representatives all use `N=4099`, block 128 and four tiles:

| Mode | Actual PTX evidence | Actual SASS evidence |
| --- | --- | --- |
| registers | Global load/store; no matched shared/async instructions | `LDG.E`, `STG.E` |
| shared_sync | Shared store, `ldmatrix`, barriers; no `cp.async` | `STS`, `LDSM.16.M88`, barriers |
| async_single | 4 `cp.async.ca.shared.global`, 4 commit groups, **4 wait_group 0** | 4 `LDGSTS.E`, 4 `LDGDEPBAR`, 4 `DEPBAR.LE SB0, 0x0` |
| async_double | 4 copies, 4 commit groups, **3 wait_group 1 + 1 wait_group 0** | 4 `LDGSTS.E`, 4 `LDGDEPBAR`; 3 `DEPBAR.LE SB0, 0x1`, final `0x0` |

Sources:
[single-buffer PTX](../../runs/20261105T061056Z-static-3xunxkwj/evidence/cases/15-gluon-copy/artifacts/kernel.ptx),
[SASS](../../runs/20261105T061056Z-static-3xunxkwj/evidence/commands/017.log),
[double-buffer PTX](../../runs/20261105T061056Z-static-3xunxkwj/evidence/cases/16-gluon-copy/artifacts/kernel.ptx),
[SASS](../../runs/20261105T061056Z-static-3xunxkwj/evidence/commands/019.log).
These are static group-control differences, not measured overlap, bandwidth or a
proof that every runtime synchronization path is correct.

### CuTe: tiled copy widths and MMA

For the shared-copy variants at `N=4099`, v1 has scalar `ld.global.b32`,
`st.shared.b32`, `ld.shared.b32`, `st.global.b32` and corresponding scalar SASS
loads/stores. V4 additionally has one each of `ld.global.v4.b32`, `st.shared.v4.b32`,
`ld.shared.v4.b32`, `st.global.v4.b32`, with `LDG.E.128`, `STS.128`, `LDS.128`,
`STG.E.128` in SASS. **Scalar paths remain in the v4 artifact**; it is not entirely
128-bit transfers. Both have a barrier.
See [v1 PTX](../../runs/20261105T061056Z-static-3xunxkwj/evidence/cases/01-cute-copy/artifacts/kernel.ptx),
[SASS](../../runs/20261105T061056Z-static-3xunxkwj/evidence/commands/003.log),
[v4 PTX](../../runs/20261105T061056Z-static-3xunxkwj/evidence/cases/00-cute-copy/artifacts/kernel.ptx),
[SASS](../../runs/20261105T061056Z-static-3xunxkwj/evidence/commands/001.log).

Minimal `(16,8,16)` MMA contains **one**
`mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32` (PTX line 231) and one
`HMMA.16816.F32`. The `(32,32,16)` tiled GEMM on input `(37,53,29)` contains
**eight** of each (PTX opcode lines 551, 556, 561, 566, 571, 576, 581, 586).
These observations come from actual exported PTX and disassembly, not merely the
compiler metadata's `mma_sync_in_ptx` boolean.
See [minimal PTX](../../runs/20261105T061056Z-static-3xunxkwj/evidence/cases/02-cute-gemm/artifacts/kernel.ptx),
[SASS](../../runs/20261105T061056Z-static-3xunxkwj/evidence/commands/005.log),
[tiled PTX](../../runs/20261105T061056Z-static-3xunxkwj/evidence/cases/03-cute-gemm/artifacts/kernel.ptx),
[SASS](../../runs/20261105T061056Z-static-3xunxkwj/evidence/commands/007.log).
No `cp.async` was found in these selected CuTe PTX files; this is not a multistage
pipeline validation.

The extra CuTe transpose control pair also differs: naive SASS contains global
load/store, while shared transpose has `STS`, `LDS` and a barrier. It does not
establish a bank-conflict rate or speedup.

### cuTile: explicit missing machine-code evidence

Seven representative cases preserve seven `.tileirbc` files plus their export
reports. **No PTX, cubin or SASS was present in these selected case artifacts.**
Entry/function counts remain null, not zero-valued measurements. No disassembler
was called on TileIR.

The export reports identify `cuda.tile.compilation.export_kernel`,
`output_format="tileir_bytecode"`, bytecode version `13.3`, scope
“separate frontend export from the same kernel and launch arguments”, and
`jit_binary_identity="not verified"`.
See [GEMM export report](../../runs/20261105T061056Z-static-3xunxkwj/evidence/cases/08-cutile-gemm/artifacts/compiler-artifacts.json).
This is not a captured JIT binary. Hardware MMA instructions, scheduling,
registers, shared-memory consumption and performance are **unavailable** for
cuTile in this evidence; none are inferred from `ct.mma` or successful prior runs.

## Resource evidence: reported fields, not inferred occupancy

Only actual `cuobjdump --dump-resource-usage` output is retained. Examples:

| Artifact | Reported REG | Raw resource log |
| --- | ---: | --- |
| Triton GEMM, either selected tile | 40 | [16×32](../../runs/20261105T061056Z-static-3xunxkwj/evidence/commands/032.log), [32×32](../../runs/20261105T061056Z-static-3xunxkwj/evidence/commands/034.log) |
| Gluon async single / double | 14 / 16 | [single](../../runs/20261105T061056Z-static-3xunxkwj/evidence/commands/018.log), [double](../../runs/20261105T061056Z-static-3xunxkwj/evidence/commands/020.log) |
| CuTe copy v1 / v4 | 16 / 14 | [v1](../../runs/20261105T061056Z-static-3xunxkwj/evidence/commands/004.log), [v4](../../runs/20261105T061056Z-static-3xunxkwj/evidence/commands/002.log) |
| CuTe minimal / tiled MMA | 37 / 128 | [minimal](../../runs/20261105T061056Z-static-3xunxkwj/evidence/commands/006.log), [tiled](../../runs/20261105T061056Z-static-3xunxkwj/evidence/commands/008.log) |

The logs report `SHARED:0`, including kernels with shared instructions. **Do not
interpret this as zero total shared memory at launch.** The exported PTX declares
external shared storage (for example CuTe's `__dynamic_shmem__0[]` and Gluon's
`global_smem[]`). This binary resource field does not supply the dynamic launch
allocation. The collector does not compute occupancy or promote metadata/IR
estimates into measured runtime resource use. Unavailable quantities stay unavailable.

## Validation and limits

- CPU AST parsing, representative selection, real PTX parsing, comment/directive
  exclusion, predicated SASS parsing, wait-group operands, hard cubin-cap rejection
  and existing-output rejection passed.
- The single collection completed with 39 successful tool commands; source/artifact
  hash comparisons passed. The later multiline-opcode regression and reparse passed;
  only the two CuTe MMA PTX analyses changed, and all original evidence hashes stayed
  identical.
- Counts are text-site counts from a line-oriented extractor, not a complete PTX
  grammar, CFG analysis or dynamic instruction census. Complete raw files remain
  authoritative. Missing matches alone are not a general proof of absence.
- No timings, bandwidth, cache hits, DRAM traffic, bank conflicts, occupancy,
  sanitizer results, GPU compatibility checks or performance rankings were measured.
- Existing PASS results, one observed opcode, and a disassembler's rc=0 do not prove
  numerical correctness, race freedom, ordering or performance for other inputs.
