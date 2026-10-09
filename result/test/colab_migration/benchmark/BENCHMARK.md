# Benchmark: one full recording

Platform: Windows-10-10.0.19045-SP0. CPU only; single numerical thread.

Recording: helical_1_50hz_High_1, column 2. Locked Step 4 configuration. Separate subprocess per workload; one measurement, not a performance distribution.

| Workload | CPU s | Compute wall s | Peak RSS MiB | Read I/O s | Write I/O s |
|---|---:|---:|---:|---:|---:|
| raw_mfdfa | 0.641 | 0.651 | 127.082 | 0.202 | 0.124 |
| vmd_decomposition | 20.516 | 20.873 | 247.629 | 0.203 | 0.854 |
| vmd_mfdfa | 18.594 | 18.618 | 247.641 | 0.205 | 0.902 |
| emd_decomposition | 11.562 | 11.723 | 137.562 | 0.205 | 2.160 |
| emd_mfdfa | 20.625 | 21.072 | 143.340 | 0.265 | 1.046 |

VMD/EMD + MF-DFA timings include decomposition, all modes, oscillatory_sum and residue. Peak RSS is sampled, not an exact allocation maximum. I/O measures local text loading and compressed decomposition serialization; Drive transfer is timed separately in the notebook.

No Colab speed comparison is available until this same benchmark runs there. Compilation/parity initialization is outside compute timing. No Step 4–5 outputs were replaced.
