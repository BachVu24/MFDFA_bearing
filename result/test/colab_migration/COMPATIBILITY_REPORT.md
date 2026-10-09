# Colab/Linux compatibility report

## Status

Execution migration is prepared and locally tested. Step 5 regression tests pass 8/8. Execution portability/resume/fallback tests pass 12/12 on Windows. No full Step 4 or Step 5 experiment was rerun. A single representative full recording was used for benchmarks; a separate one-recording execution/resume check was performed in a new generated directory.

This machine has no working Linux/Colab runtime. Actual Linux shared-library compilation, Linux numerical parity, mounted Google Drive behavior and Colab timings remain to be checked by notebook cells. Do not interpret Windows checks as Linux execution evidence.

## Scope and invariants

`code/core/mfdfa.py`, `code/core/vmd.py`, `code/core/emd.py`, the practical EMD implementation, Step 4 scaling/QC code, C++ recurrence and historical accelerator wrapper are byte-for-byte unchanged. All 13 locked dependency SHA256 values are retained. q, actual scales, DFA order 2, fit interval 64–512, QC, features, EMD settings and VMD K=7/alpha=500/tau=0/init=1/tol=1e-7/max_iter=2000 remain locked. Data, VidData, papers and validated feature/spectrum/report files were not edited.

## Findings and remedies

| Finding | Remedy |
|---|---|
| Windows separators in JSON/CSV provenance | New paths use `relative_to(ROOT).as_posix()`; historical CSV/JSON paths resolve through a strict project-relative POSIX parser in memory. |
| Locked config hash includes path spelling | Explicit migration normalizes only dependency paths and recomputes the original sorted-JSON hash; historical config identity is retained in a migration ledger. |
| Historical snapshots overwritten during verification | Existing snapshots are verified read-only. Missing/changed input hashes raise an error; no silent rebaseline. |
| Source-relative ROOT depended on output location | ROOT derives from the script file; execution output root is independently configurable. |
| Historical scripts write into validated folders | The new runner restricts output to `result/test/colab_runs`; direct Step 5 execution/analysis protects the historical result folder. |
| `.dll`, MinGW pthread DLL, Windows DLL search directory | Historical binaries remain archived/hash-verified, never loaded on Linux. Linux uses a newly compiled `.so` or NumPy. |
| C++ export uses `__declspec(dllexport)` | Linux build adds `-D__declspec(x)=`, `-fPIC`, `-shared`, without editing locked C++ source. No fast-math. |
| Legacy accelerator module loads a DLL on import | Portable adapter extracts and compiles only its original `vmd` function AST; excludes its Windows loader. This retains a single numerical implementation and all locked source hashes. Step 5 and its transitive sensitivity import use this adapter. |
| Checkpoints trusted existence alone | Content-hash ledger plus config/source signatures; atomic completion JSON; per-channel recovery and recording-level Drive commit. |
| Working-directory assumptions | Runner uses resolved project paths; notebook selects runtime directory explicitly; subprocesses use project cwd. |
| Python/package reproducibility | Pinned NumPy/SciPy/Matplotlib/psutil; isolated Colab venv; thread count 1. Python >=3.12 is required by pinned SciPy. |
| Encoding | All Python sources parse as UTF-8/UTF-8 BOM; CSV reads/writes explicitly handle UTF-8 BOM; JSON UTF-8. Raw three-column numeric text uses existing reader unchanged. |

All project Python files were scanned and parsed; see `source_compatibility_inventory.json`. Archived original pilot/Step 4 scripts and their Windows-format manifests remain historical inputs. Use `execution.runner` for Colab computation; do not invoke legacy optimization/sensitivity scripts to rerun scientific stages. Retaining those immutable sources is necessary to keep the locked dependency hashes valid.

## Path migration and historical provenance

Old config SHA256: `681840f1de3d6ca6c0b7875e63e1a53800dcd80e9d6566d945a09c2fa904889b`

Portable config SHA256: `801a6da803068a9c2671c7db6d095a5ee5966dc525152c583c20e0b2bd162097`

The 13 dependency digests and scientific JSON fields are identical. `path_migration.json` records both config identities, old/new snapshot file hashes and the unchanged scientific configuration. Historical feature tables/checkpoints keep their original identity; reuse accepts it only when the ledger proves the same scientific configuration and dependencies. Tuple/list equivalence is compared using JSON serialization, matching the original hash semantics.

`runtime_provenance.json`, `source_inventory.json` and `reused_step4_array_manifest.csv` remain unchanged on disk. Their paths normalize in memory, so their historic digests remain valid. New snapshots use `/`. Existing output manifests are retained as historic audit; `validated_results_integrity.json` explicitly reconciles authorized script/config changes rather than rewriting their recorded hashes. No missing/changed dependency is accepted.

385 checked references = 13 locked inputs + 4 runtime/reference inputs + 288 Step 4 arrays + 80 raw recordings. Each exists and matches SHA256.

## Running the notebook

1. Upload the updated Project folder to Drive, including `result/test/step4`, `step4_corrected`, `step5` and `colab_migration/path_migration.json`. These provide immutable references for strict validation/reuse. Source and dataset may stay on Drive permanently.
2. Open `colab_runner.ipynb`; select a CPU runtime and set DRIVE_PROJECT, a separate DRIVE_RESULTS, STEP and RUN_ID in the form.
3. Run mount/copy/install/verification/test cells. Copying uses local runtime for computation and never moves/deletes inputs. Python >=3.12 is checked; numerical packages install into an isolated venv.
4. Choose `verify` first, then `benchmark`, `step4` or `step5`. Default reuse skips hash-attested validated recordings. `step4` means the frozen final pipeline for the eight 50 Hz/High helical recordings, not parameter optimization. `step5` selects the 80-recording inventory. MAX_RECORDINGS limits an execution batch; later increase it using the same RUN_ID.
5. Keep RUN_ID to resume. Both channels remain independent analysis channels, with one acquisition as the recording unit. New results go to `result/test/colab_runs/<step>/<run_id>` and a separate Drive directory.
6. Optional GENERATE_STEP5_ANALYSIS runs the original analysis/report only for a full Step 5 inventory. It never updates the historical validated folder. Step 4 locked extraction currently produces features/QC and arrays; historical Step 4 analysis remains the reference.

Drive commits copy/check payloads first and completion markers last. Each checkpoint carries full recording source path/hash/sample count/channels, config hash, execution-script hash and all array/channel-metadata hashes. On restart the dedicated generated output is copied back, hashes are verified, completed recordings skipped, and missing/corrupt recordings repaired. A disconnected runtime during a Drive copy cannot turn partial payloads into a validated skip. Completion is verified from actual file hashes rather than trusting a marker.

Historical reused arrays remain referenced in the uploaded Project; they are not duplicated into every new Drive output. Keep that source Project available to reproduce/revalidate the experiment.

## Native VMD and benchmark

`execution/accelerator.py` compiles the immutable C++ kernel with GCC on Linux; validates modes, spectra, frequency histories, iterations and convergence against NumPy (even/odd lengths, tau 0/0.5, DC, init 0/1/2, and a locked-parameter max_iter=2000 case). Relative numerical error limit is 1e-9. Compile/load/parity failure disables the native library and records the reason. It never changes parameters to force convergence. Windows native parity was checked locally; Linux parity runs before every experiment initialization and is recorded in `_runtime/accelerator_status.json`.

The one-recording Windows benchmark is in `benchmark/BENCHMARK.md` and `benchmark/benchmark.json`. It measures five requested workloads in isolated subprocesses, CPU and wall time, sampled peak RSS, text-reading and compressed-output I/O. Compile/parity warm-up is outside compute timing. VMD/EMD+MF-DFA includes decomposition, every mode, sum and residue. One timing per workload is descriptive; cache/CPU effects can make combined timings smaller than decomposition-only timings. No Colab speedup is claimed. Drive copy timing is printed separately by the notebook.

## Step 6 readiness

Environment, data staging, locked-input verification, native/NumPy fallback, persistent outputs and recording-level resume are prepared for Colab. Execute the notebook's Linux checks first. Step 6's scientific protocol/experiment is not yet implemented by this migration; it can be registered on this infrastructure once specified. Existing Step 4–5 parameters must not be altered to develop it.

References: [Colab FAQ](https://research.google.com/colaboratory/faq.html) documents runtime deletion and local staging recommendations; [SciPy 1.18.1 metadata](https://pypi.org/project/scipy/1.18.1/) requires Python >=3.12.

## Files changed or added

| File | Purpose |
|---|---|
| `result/test/step5/scripts/pipeline.py` | POSIX provenance, strict snapshot verification, portable reads, output isolation, atomic/hash-attested caches, OS-aware VMD adapter. |
| `result/test/step5/scripts/analysis.py` | Portable diagnostic paths, runtime plotting cache, historical-output protection. Analysis formulas unchanged. |
| `result/test/step5/scripts/presentation.py` | Portable input/output paths and verified historical config alias; QC/statistical/report definitions unchanged. |
| `result/test/step5/scripts/tests.py` | Test logs go to a separate generated directory; eight scientific regression tests unchanged. |
| `result/test/step5/config/locked_step4_configuration.json` | Only dependency separator normalization and corresponding config hash. |
| `colab_runner.ipynb` | Mount, input staging, isolated dependencies, verification/tests, step selection, execution/resume and Drive persistence. |
| `requirements.txt` | Pinned numerical dependencies. |
| `execution/__init__.py`, `execution/provenance.py` | Execution package and strict portable paths/migration/lineage. |
| `execution/accelerator.py` | Linux compile, parity-gated native loader and NumPy fallback, original numerical function retained. |
| `execution/runner.py` | Per-recording/channel cache recovery, content hashes, immutable historical reuse and Drive commit. |
| `execution/benchmark.py` | Five isolated full-recording workloads, CPU/wall/RSS/I/O report. |
| `execution/tests.py` | Portability/resume/failure-path regression tests. |
| `result/test/colab_migration/` | Compatibility/benchmark reports, migration ledger, dependency/source/result-integrity audits and test results. |
| `result/test/colab_runs/step5/resume_check/` | One-recording generated execution/resume smoke-check metadata; original scientific results were not overwritten. |

Final original-manifest audit: Step 4 corrected 376 unchanged files; Step 5 2,796 unchanged files and exactly five authorized source/config changes. Zero unexpected changes. The original manifests are preserved.
