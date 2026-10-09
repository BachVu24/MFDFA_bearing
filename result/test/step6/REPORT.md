# STEP 6 — Full helical validation

## 1. Dataset

Đủ **120/120 recordings**; missing=0, duplicate keys=0, duplicate content=0, unexpected=0. H1–H6 × 5 speeds (30/35/40/45/50 Hz) × High/Low × 2 acquisitions. 120 recording units, 240 channel signals; input/output cùng recording không được coi là hai acquisitions độc lập. Không chia windows thành samples.
Full length 245648–266656 samples, 3.6847–3.9998 s; fs=66,666.7 Hz. Output voltage cột 2 là primary, input voltage cột 1 là secondary.

| State | Fault configuration | VidData.xls nguyên văn (non-Good entries) |
|---|---|---|
| H1 | Healthy | All Good |
| H2 | Gear fault | 24T: Chipped |
| H3 | Gear + Bearing + Shaft | 24T: Broken; IS:OS: Combination; ID:OS: Inner; Input: Bent Shaft |
| H4 | Bearing + Shaft | IS:OS: Combination; ID:OS: ball; Input: Imbalnce |
| H5 | Gear + Bearing | 24T: Broken; ID:OS: Inner |
| H6 | Shaft | Input: Bent Shaft |

Labels đọc từ Sheet1 hàng 16–22; giữ cả lỗi chính tả Imbalnce. Các vị trí trong workbook là metadata, không phải kết luận localization từ feature. VidData ghi Total Runs=140 nhưng grid H1–H6 được yêu cầu và kiểm kê ở đây là 120.

## 2. Configuration

Giữ q=-5…5, step 0.5; DFA order=2; 40 scales gốc; fit 64–512 samples (17 actual points, 64–468). Không tune bằng H3/H4. Δα=max(alpha)−min(alpha); α0=alpha(q=0); Δh=max(hq)−min(hq).
QC: ≥8 fit points; min R²≥0.90; ≥80% q có R²≥0.95; ≥80% q có adjacent slope CV≤0.5; min hq>0.05. Analysis valid = scaling valid AND decomposition valid. Spectrum shape warnings giữ audit, không thêm gate.
VMD K=7, alpha=500, tau=0, DC=False, init=1, tol=1e-7, max_iter=2000. EMD practical approximation giữ max_imfs=6, max_siftings=200, sd_threshold=0.2, envelope_tol=0.05, extrema_relative_tolerance=0.005; strict IMF audit riêng. Raw chỉ mean removal bên trong MF-DFA; decomposition dùng full signal bỏ global mean.
Step 5 snapshot được giữ; bốn JSON config khác hash chỉ do CRLF→LF. Mã MF-DFA hiện tại khác historical hash; vì vậy **toàn bộ 120 recordings được chạy lại**, không reuse numerical features Step 5. Không thay mã core/config cũ. Portable C++ VMD chỉ bỏ Windows export annotation; recurrence giữ nguyên, không fast-math; kiểm tra parity với NumPy reference. Xem config/dependency_audit.json và diagnostics/tests.json.

## 3. QC

| Channel | Pipeline | Main raw/sum valid signals | All components valid | Complete main cells |
|---|---|---:|---:|---:|
| output_voltage | RAW | 109/120 (90.8%) | 109/120 | 53/60 |
| output_voltage | VMD | 111/120 (92.5%) | 207/1080 | 54/60 |
| output_voltage | EMD | 103/120 (85.8%) | 178/960 | 51/60 |
| input_voltage | RAW | 115/120 (95.8%) | 115/120 | 57/60 |
| input_voltage | VMD | 115/120 (95.8%) | 201/1080 | 57/60 |
| input_voltage | EMD | 110/120 (91.7%) | 160/960 | 54/60 |

Không dùng tổng mode-valid rate để xếp hạng Raw/VMD/EMD: số components khác nhau. Invalid features và Fq giữ audit; mọi kết luận chỉ dùng valid/matched recordings. QC breakdown theo state/speed/load/component ở CSV.

## 4. H1–H6 feature results

Các bảng và plots dùng recording hoặc mean±sample SD của đúng hai acquisitions tại cell. Không dùng pooled fault mean để tuyên bố separation khi operating conditions khác nhau.

| Channel | Pipeline | Feature | Evaluable pair-condition comparisons /150 | Disjoint / evaluable |
|---|---|---|---:|---:|
| output_voltage | RAW | delta_alpha | 122/150 | 111/122 (91.0%) |
| output_voltage | RAW | alpha0 | 122/150 | 115/122 (94.3%) |
| output_voltage | RAW | delta_h | 122/150 | 107/122 (87.7%) |
| output_voltage | VMD | delta_alpha | 126/150 | 108/126 (85.7%) |
| output_voltage | VMD | alpha0 | 126/150 | 117/126 (92.9%) |
| output_voltage | VMD | delta_h | 126/150 | 105/126 (83.3%) |
| output_voltage | EMD | delta_alpha | 113/150 | 98/113 (86.7%) |
| output_voltage | EMD | alpha0 | 113/150 | 97/113 (85.8%) |
| output_voltage | EMD | delta_h | 113/150 | 98/113 (86.7%) |
| input_voltage | RAW | delta_alpha | 135/150 | 113/135 (83.7%) |
| input_voltage | RAW | alpha0 | 135/150 | 111/135 (82.2%) |
| input_voltage | RAW | delta_h | 135/150 | 113/135 (83.7%) |
| input_voltage | VMD | delta_alpha | 135/150 | 109/135 (80.7%) |
| input_voltage | VMD | alpha0 | 135/150 | 111/135 (82.2%) |
| input_voltage | VMD | delta_h | 135/150 | 111/135 (82.2%) |
| input_voltage | EMD | delta_alpha | 123/150 | 104/123 (84.6%) |
| input_voltage | EMD | alpha0 | 123/150 | 110/123 (89.4%) |
| input_voltage | EMD | delta_h | 123/150 | 110/123 (89.4%) |

Feature summaries bên dưới là mean±sample SD của valid recordings gộp observed conditions, chỉ mô tả phân bố. Missingness và operating effects có thể làm pooled means lệch; fault separation luôn dùng cell-matched ranges ở mục 5.

| Channel | Pipeline | State | Valid N/20 | delta_alpha | alpha0 | delta_h |
|---|---|---|---:|---:|---:|---:|
| output_voltage | RAW | H1 | 20/20 | 0.6377±0.2032 | 1.0283±0.1734 | 0.3748±0.1191 |
| output_voltage | RAW | H2 | 20/20 | 0.5848±0.2177 | 1.0581±0.1744 | 0.3590±0.1378 |
| output_voltage | RAW | H3 | 16/20 | 0.2058±0.0992 | 0.9196±0.2279 | 0.0984±0.0601 |
| output_voltage | RAW | H4 | 18/20 | 0.6181±0.2198 | 1.1310±0.2899 | 0.3506±0.1457 |
| output_voltage | RAW | H5 | 17/20 | 0.2067±0.0900 | 0.9401±0.2249 | 0.0988±0.0576 |
| output_voltage | RAW | H6 | 18/20 | 0.5784±0.1805 | 1.1149±0.1887 | 0.3555±0.1258 |
| output_voltage | VMD | H1 | 20/20 | 0.6517±0.2015 | 1.0476±0.1810 | 0.3825±0.1190 |
| output_voltage | VMD | H2 | 20/20 | 0.5972±0.2186 | 1.0740±0.1790 | 0.3660±0.1390 |
| output_voltage | VMD | H3 | 17/20 | 0.2487±0.1334 | 0.9167±0.3059 | 0.1239±0.0758 |
| output_voltage | VMD | H4 | 18/20 | 0.6333±0.2270 | 1.1524±0.2974 | 0.3592±0.1526 |
| output_voltage | VMD | H5 | 18/20 | 0.2275±0.1233 | 0.9538±0.2978 | 0.1134±0.0700 |
| output_voltage | VMD | H6 | 18/20 | 0.5883±0.1795 | 1.1345±0.1945 | 0.3606±0.1256 |
| output_voltage | EMD | H1 | 20/20 | 0.5689±0.2043 | 0.9883±0.1871 | 0.3311±0.1198 |
| output_voltage | EMD | H2 | 20/20 | 0.5237±0.2055 | 1.0220±0.1808 | 0.3164±0.1311 |
| output_voltage | EMD | H3 | 14/20 | 0.2066±0.0647 | 0.8667±0.2299 | 0.1118±0.0474 |
| output_voltage | EMD | H4 | 16/20 | 0.5116±0.2103 | 1.1155±0.2986 | 0.2830±0.1336 |
| output_voltage | EMD | H5 | 15/20 | 0.1552±0.0431 | 0.8791±0.2178 | 0.0799±0.0327 |
| output_voltage | EMD | H6 | 18/20 | 0.5168±0.1818 | 1.0788±0.1959 | 0.3133±0.1239 |
| input_voltage | RAW | H1 | 16/20 | 0.9040±0.2276 | 0.9488±0.1690 | 0.4962±0.1299 |
| input_voltage | RAW | H2 | 20/20 | 0.8156±0.1808 | 1.0131±0.1513 | 0.4677±0.1097 |
| input_voltage | RAW | H3 | 20/20 | 0.5767±0.2070 | 0.8833±0.1676 | 0.3101±0.1127 |
| input_voltage | RAW | H4 | 19/20 | 0.7751±0.2379 | 0.9264±0.1794 | 0.4258±0.1309 |
| input_voltage | RAW | H5 | 20/20 | 0.5930±0.1579 | 0.9431±0.1863 | 0.3278±0.0827 |
| input_voltage | RAW | H6 | 20/20 | 0.7348±0.2219 | 1.0463±0.1611 | 0.4318±0.1407 |
| input_voltage | VMD | H1 | 16/20 | 0.9263±0.2294 | 0.9606±0.1750 | 0.5069±0.1317 |
| input_voltage | VMD | H2 | 20/20 | 0.8307±0.1826 | 1.0261±0.1567 | 0.4741±0.1108 |
| input_voltage | VMD | H3 | 20/20 | 0.5906±0.1964 | 0.8937±0.1780 | 0.3179±0.1083 |
| input_voltage | VMD | H4 | 19/20 | 0.7758±0.2375 | 0.9407±0.1832 | 0.4267±0.1306 |
| input_voltage | VMD | H5 | 20/20 | 0.6038±0.1639 | 0.9622±0.1963 | 0.3337±0.0845 |
| input_voltage | VMD | H6 | 20/20 | 0.7469±0.2163 | 1.0618±0.1654 | 0.4368±0.1389 |
| input_voltage | EMD | H1 | 16/20 | 0.7940±0.1872 | 0.8915±0.1672 | 0.4318±0.1213 |
| input_voltage | EMD | H2 | 20/20 | 0.7361±0.2035 | 0.9796±0.1494 | 0.4130±0.1292 |
| input_voltage | EMD | H3 | 18/20 | 0.4444±0.1619 | 0.8477±0.1074 | 0.2206±0.0908 |
| input_voltage | EMD | H4 | 18/20 | 0.6571±0.2513 | 0.9045±0.1554 | 0.3549±0.1473 |
| input_voltage | EMD | H5 | 18/20 | 0.4394±0.1454 | 0.8896±0.0965 | 0.2227±0.0889 |
| input_voltage | EMD | H6 | 20/20 | 0.6137±0.2613 | 0.9952±0.1557 | 0.3548±0.1646 |

## 5. Fault-pair separation

Giữ disjoint repeat ranges: max(A)<min(B) hoặc max(B)<min(A), strict inequality, cùng speed/load; cả bốn acquisitions phải valid. Overlap không tách; missing/invalid là NOT EVALUABLE, không đếm như overlap. Đây là descriptive separation, **không phải classification accuracy** hoặc kiểm định significance.

Mỗi ô dưới là disjoint/evaluable/10 expected conditions. Bảng bao gồm toàn bộ 15 cặp cho cả hai channels và ba primary representations; CSV lưu từng condition, direction và gap.

### output_voltage — RAW

| Pair | delta_alpha | alpha0 | delta_h |
|---|---:|---:|---:|
| H1-H2 | 10/10/10 | 9/10/10 | 9/10/10 |
| H1-H3 | 7/7/10 | 7/7/10 | 7/7/10 |
| H1-H4 | 9/9/10 | 9/9/10 | 9/9/10 |
| H1-H5 | 8/8/10 | 7/8/10 | 8/8/10 |
| H1-H6 | 9/9/10 | 8/9/10 | 9/9/10 |
| H2-H3 | 7/7/10 | 7/7/10 | 7/7/10 |
| H2-H4 | 8/9/10 | 9/9/10 | 5/9/10 |
| H2-H5 | 8/8/10 | 8/8/10 | 8/8/10 |
| H2-H6 | 7/9/10 | 7/9/10 | 6/9/10 |
| H3-H4 | 7/7/10 | 6/7/10 | 7/7/10 |
| H3-H5 | 2/7/10 | 6/7/10 | 3/7/10 |
| H3-H6 | 7/7/10 | 7/7/10 | 7/7/10 |
| H4-H5 | 8/8/10 | 8/8/10 | 8/8/10 |
| H4-H6 | 6/9/10 | 9/9/10 | 6/9/10 |
| H5-H6 | 8/8/10 | 8/8/10 | 8/8/10 |

### output_voltage — VMD

| Pair | delta_alpha | alpha0 | delta_h |
|---|---:|---:|---:|
| H1-H2 | 9/10/10 | 9/10/10 | 9/10/10 |
| H1-H3 | 8/8/10 | 8/8/10 | 8/8/10 |
| H1-H4 | 9/9/10 | 9/9/10 | 9/9/10 |
| H1-H5 | 8/8/10 | 8/8/10 | 8/8/10 |
| H1-H6 | 9/9/10 | 8/9/10 | 8/9/10 |
| H2-H3 | 7/8/10 | 7/8/10 | 7/8/10 |
| H2-H4 | 6/9/10 | 9/9/10 | 6/9/10 |
| H2-H5 | 7/8/10 | 7/8/10 | 7/8/10 |
| H2-H6 | 7/9/10 | 7/9/10 | 7/9/10 |
| H3-H4 | 8/8/10 | 7/8/10 | 7/8/10 |
| H3-H5 | 2/7/10 | 6/7/10 | 2/7/10 |
| H3-H6 | 7/8/10 | 8/8/10 | 8/8/10 |
| H4-H5 | 8/8/10 | 8/8/10 | 7/8/10 |
| H4-H6 | 6/9/10 | 9/9/10 | 5/9/10 |
| H5-H6 | 7/8/10 | 7/8/10 | 7/8/10 |

### output_voltage — EMD

| Pair | delta_alpha | alpha0 | delta_h |
|---|---:|---:|---:|
| H1-H2 | 8/10/10 | 9/10/10 | 8/10/10 |
| H1-H3 | 6/7/10 | 7/7/10 | 7/7/10 |
| H1-H4 | 8/8/10 | 7/8/10 | 7/8/10 |
| H1-H5 | 7/7/10 | 7/7/10 | 7/7/10 |
| H1-H6 | 9/9/10 | 7/9/10 | 9/9/10 |
| H2-H3 | 7/7/10 | 5/7/10 | 7/7/10 |
| H2-H4 | 4/8/10 | 8/8/10 | 4/8/10 |
| H2-H5 | 7/7/10 | 6/7/10 | 7/7/10 |
| H2-H6 | 7/9/10 | 6/9/10 | 6/9/10 |
| H3-H4 | 6/6/10 | 5/6/10 | 5/6/10 |
| H3-H5 | 4/7/10 | 4/7/10 | 5/7/10 |
| H3-H6 | 7/7/10 | 7/7/10 | 7/7/10 |
| H4-H5 | 6/6/10 | 4/6/10 | 6/6/10 |
| H4-H6 | 5/8/10 | 8/8/10 | 6/8/10 |
| H5-H6 | 7/7/10 | 7/7/10 | 7/7/10 |

### input_voltage — RAW

| Pair | delta_alpha | alpha0 | delta_h |
|---|---:|---:|---:|
| H1-H2 | 6/8/10 | 5/8/10 | 7/8/10 |
| H1-H3 | 7/8/10 | 5/8/10 | 7/8/10 |
| H1-H4 | 5/7/10 | 6/7/10 | 5/7/10 |
| H1-H5 | 6/8/10 | 6/8/10 | 6/8/10 |
| H1-H6 | 8/8/10 | 6/8/10 | 8/8/10 |
| H2-H3 | 9/10/10 | 9/10/10 | 10/10/10 |
| H2-H4 | 9/9/10 | 8/9/10 | 9/9/10 |
| H2-H5 | 10/10/10 | 7/10/10 | 9/10/10 |
| H2-H6 | 6/10/10 | 8/10/10 | 8/10/10 |
| H3-H4 | 8/9/10 | 8/9/10 | 9/9/10 |
| H3-H5 | 6/10/10 | 7/10/10 | 4/10/10 |
| H3-H6 | 10/10/10 | 10/10/10 | 9/10/10 |
| H4-H5 | 8/9/10 | 8/9/10 | 7/9/10 |
| H4-H6 | 7/9/10 | 9/9/10 | 8/9/10 |
| H5-H6 | 8/10/10 | 9/10/10 | 7/10/10 |

### input_voltage — VMD

| Pair | delta_alpha | alpha0 | delta_h |
|---|---:|---:|---:|
| H1-H2 | 6/8/10 | 4/8/10 | 7/8/10 |
| H1-H3 | 7/8/10 | 5/8/10 | 7/8/10 |
| H1-H4 | 5/7/10 | 6/7/10 | 4/7/10 |
| H1-H5 | 6/8/10 | 7/8/10 | 6/8/10 |
| H1-H6 | 8/8/10 | 6/8/10 | 8/8/10 |
| H2-H3 | 9/10/10 | 9/10/10 | 10/10/10 |
| H2-H4 | 8/9/10 | 9/9/10 | 9/9/10 |
| H2-H5 | 10/10/10 | 7/10/10 | 9/10/10 |
| H2-H6 | 6/10/10 | 8/10/10 | 8/10/10 |
| H3-H4 | 7/9/10 | 8/9/10 | 9/9/10 |
| H3-H5 | 6/10/10 | 8/10/10 | 4/10/10 |
| H3-H6 | 9/10/10 | 10/10/10 | 8/10/10 |
| H4-H5 | 8/9/10 | 8/9/10 | 7/9/10 |
| H4-H6 | 6/9/10 | 8/9/10 | 8/9/10 |
| H5-H6 | 8/10/10 | 8/10/10 | 7/10/10 |

### input_voltage — EMD

| Pair | delta_alpha | alpha0 | delta_h |
|---|---:|---:|---:|
| H1-H2 | 7/8/10 | 8/8/10 | 8/8/10 |
| H1-H3 | 7/7/10 | 7/7/10 | 7/7/10 |
| H1-H4 | 6/6/10 | 5/6/10 | 5/6/10 |
| H1-H5 | 7/7/10 | 6/7/10 | 7/7/10 |
| H1-H6 | 7/8/10 | 7/8/10 | 7/8/10 |
| H2-H3 | 8/9/10 | 8/9/10 | 9/9/10 |
| H2-H4 | 7/8/10 | 8/8/10 | 7/8/10 |
| H2-H5 | 8/9/10 | 9/9/10 | 8/9/10 |
| H2-H6 | 7/10/10 | 6/10/10 | 9/10/10 |
| H3-H4 | 7/8/10 | 7/8/10 | 8/8/10 |
| H3-H5 | 6/9/10 | 7/9/10 | 6/9/10 |
| H3-H6 | 7/9/10 | 8/9/10 | 8/9/10 |
| H4-H5 | 6/8/10 | 8/8/10 | 8/8/10 |
| H4-H6 | 6/8/10 | 7/8/10 | 6/8/10 |
| H5-H6 | 8/9/10 | 9/9/10 | 7/9/10 |

output_voltage/RAW: feature có nhiều disjoint conditions nhất là alpha0; counts delta_alpha=111, alpha0=115, delta_h=107. Đây là within-condition descriptive ranking; đọc operating sensitivity ở mục 6 cùng với nó.

output_voltage/VMD: feature có nhiều disjoint conditions nhất là alpha0; counts delta_alpha=108, alpha0=117, delta_h=105. Đây là within-condition descriptive ranking; đọc operating sensitivity ở mục 6 cùng với nó.

input_voltage/RAW: feature có nhiều disjoint conditions nhất là delta_alpha, delta_h; counts delta_alpha=113, alpha0=111, delta_h=113. Đây là within-condition descriptive ranking; đọc operating sensitivity ở mục 6 cùng với nó.

input_voltage/VMD: feature có nhiều disjoint conditions nhất là alpha0, delta_h; counts delta_alpha=109, alpha0=111, delta_h=111. Đây là within-condition descriptive ranking; đọc operating sensitivity ở mục 6 cùng với nó.

Feature tốt hơn được đánh giá bằng coverage và matched disjoint counts; α0 tách nhiều cặp trong cell vẫn có thể nhạy operating condition. Không suy ra feature tốt nhất cho classifier.

## 6. Speed/load effects

Giữ Step 5: speed range=(max−min)/abs(mean) trên đủ 5 speeds; load difference=2|High−Low|/(|High|+|Low|) tại cùng fault/speed; thiếu repeat/cell thì không đánh giá. Full-grid fault/operating ratio yêu cầu tất cả 6 states đủ 10 conditions; partial grid chỉ dùng common valid operating cells, không interpolation.

| Channel | Pipeline/feature | Speed range median (N/12) | Load difference median (N/30) | Common grid (N/10), fault/operating SD ratio |
|---|---|---:|---:|---:|
| output_voltage | RAW/delta_alpha | 56.5% (6/12) | 54.4% (23/30) | 7/10, 1.414 |
| output_voltage | RAW/alpha0 | 22.4% (6/12) | 23.0% (23/30) | 7/10, 0.416 |
| output_voltage | RAW/delta_h | 61.7% (6/12) | 56.8% (23/30) | 7/10, 1.470 |
| output_voltage | VMD/delta_alpha | 57.0% (6/12) | 53.9% (24/30) | 7/10, 1.050 |
| output_voltage | VMD/alpha0 | 22.7% (6/12) | 27.8% (24/30) | 7/10, 0.415 |
| output_voltage | VMD/delta_h | 61.5% (6/12) | 55.2% (24/30) | 7/10, 1.365 |
| output_voltage | EMD/delta_alpha | 77.9% (6/12) | 50.3% (21/30) | 6/10, 1.423 |
| output_voltage | EMD/alpha0 | 24.0% (6/12) | 23.2% (21/30) | 6/10, 0.430 |
| output_voltage | EMD/delta_h | 82.1% (6/12) | 58.4% (21/30) | 6/10, 1.401 |
| input_voltage | RAW/delta_alpha | 64.0% (10/12) | 35.4% (27/30) | 7/10, 1.161 |
| input_voltage | RAW/alpha0 | 21.4% (10/12) | 14.0% (27/30) | 7/10, 0.393 |
| input_voltage | RAW/delta_h | 61.2% (10/12) | 41.8% (27/30) | 7/10, 1.260 |
| input_voltage | VMD/delta_alpha | 60.4% (10/12) | 34.5% (27/30) | 7/10, 1.194 |
| input_voltage | VMD/alpha0 | 21.2% (10/12) | 13.9% (27/30) | 7/10, 0.392 |
| input_voltage | VMD/delta_h | 59.3% (10/12) | 38.4% (27/30) | 7/10, 1.305 |
| input_voltage | EMD/delta_alpha | 89.9% (8/12) | 40.9% (24/30) | 6/10, 1.141 |
| input_voltage | EMD/alpha0 | 18.1% (8/12) | 9.2% (24/30) | 6/10, 1.180 |
| input_voltage | EMD/delta_h | 93.9% (8/12) | 52.6% (24/30) | 6/10, 1.223 |

Categorical additive fault+speed+load: sum-to-zero Helmert contrasts, partial η²=incremental SS/(incremental SS+residual SS), không cộng η² thành tổng variance. Full fault*speed*load chỉ evaluable nếu đủ 60 cells, full rank và ≥20 residual df. Additive effects với missing cells không chứng minh absence of interactions hoặc causality.
Giữ OLS/HC3, null-imposed Rademacher wild bootstrap 1,999 draws, seed 6509, HC2-scaled reduced residuals và studentized HC3 Wald; BH riêng mỗi channel trên toàn family representations/features/models/terms. Wild/HC3 exploratory, classical p chỉ reference. Shapiro và heteroskedasticity LM trong diagnostics.

| Channel | Pipeline/feature | η² fault/speed/load | Wild p BH fault/speed/load |
|---|---|---:|---:|
| output_voltage | RAW/delta_alpha | 0.550/0.099/0.016 | 0.0010/0.0608/0.2322 |
| output_voltage | RAW/alpha0 | 0.320/0.510/0.629 | 0.0018/0.0010/0.0010 |
| output_voltage | RAW/delta_h | 0.565/0.105/0.037 | 0.0010/0.0445/0.0844 |
| output_voltage | VMD/delta_alpha | 0.510/0.067/0.001 | 0.0010/0.1141/0.7305 |
| output_voltage | VMD/alpha0 | 0.312/0.496/0.648 | 0.0010/0.0010/0.0010 |
| output_voltage | VMD/delta_h | 0.525/0.073/0.012 | 0.0010/0.0958/0.2975 |
| output_voltage | EMD/delta_alpha | 0.490/0.069/0.056 | 0.0010/0.1867/0.0711 |
| output_voltage | EMD/alpha0 | 0.363/0.528/0.573 | 0.0010/0.0010/0.0010 |
| output_voltage | EMD/delta_h | 0.489/0.079/0.102 | 0.0010/0.1503/0.0127 |
| input_voltage | RAW/delta_alpha | 0.289/0.046/0.089 | 0.0008/0.1183/0.0071 |
| input_voltage | RAW/alpha0 | 0.242/0.557/0.325 | 0.0008/0.0008/0.0008 |
| input_voltage | RAW/delta_h | 0.286/0.044/0.045 | 0.0008/0.1660/0.0487 |
| input_voltage | VMD/delta_alpha | 0.298/0.047/0.088 | 0.0008/0.1125/0.0071 |
| input_voltage | VMD/alpha0 | 0.237/0.557/0.336 | 0.0008/0.0008/0.0008 |
| input_voltage | VMD/delta_h | 0.289/0.044/0.044 | 0.0008/0.1584/0.0487 |
| input_voltage | EMD/delta_alpha | 0.368/0.240/0.075 | 0.0008/0.0008/0.0189 |
| input_voltage | EMD/alpha0 | 0.290/0.472/0.167 | 0.0008/0.0008/0.0008 |
| input_voltage | EMD/delta_h | 0.368/0.235/0.041 | 0.0008/0.0008/0.0487 |

**Kiểm tra kết luận Step 5 (từ dữ liệu Step 6):**

- output_voltage, Raw delta_alpha: fault mạnh hơn từng main speed/load effect.
- output_voltage, Raw alpha0: ít nhất một operating main effect mạnh hơn fault.
- output_voltage, Raw delta_h: fault mạnh hơn từng main speed/load effect.
- input_voltage, Raw delta_alpha: fault mạnh hơn từng main speed/load effect.
- input_voltage, Raw alpha0: ít nhất một operating main effect mạnh hơn fault.
- input_voltage, Raw delta_h: fault mạnh hơn từng main speed/load effect.

## 7. Raw vs VMD

| Channel | Feature | Paired N | Pearson r | Median relative difference | Mean signed change | RMSE |
|---|---|---:|---:|---:|---:|---:|
| output_voltage | delta_alpha | 109 | 0.9891 | 3.5% | 0.02207 | 0.04352 |
| output_voltage | alpha0 | 109 | 0.9903 | 2.0% | 0.01171 | 0.04848 |
| output_voltage | delta_h | 109 | 0.9922 | 3.3% | 0.01286 | 0.02430 |
| input_voltage | delta_alpha | 115 | 0.9980 | 1.8% | 0.01223 | 0.01903 |
| input_voltage | alpha0 | 115 | 0.9992 | 1.5% | 0.01407 | 0.01696 |
| input_voltage | delta_h | 115 | 0.9989 | 1.5% | 0.00599 | 0.00869 |

Matched valid comparisons only; sensitivity ratios VMD/Raw <1 biểu thị giảm sensitivity.

| Channel | Feature | Common pairs | Raw/VMD disjoint | Gained/lost | Speed ratio (N) | Load ratio (N) |
|---|---|---:|---:|---:|---:|---:|
| output_voltage | delta_alpha | 122 | 111/104 | 0/7 | 0.993 (6) | 0.969 (23) |
| output_voltage | alpha0 | 122 | 115/113 | 1/3 | 1.013 (6) | 1.039 (23) |
| output_voltage | delta_h | 122 | 107/101 | 2/8 | 0.994 (6) | 0.973 (23) |
| input_voltage | delta_alpha | 135 | 113/109 | 0/4 | 0.968 (10) | 0.969 (27) |
| input_voltage | alpha0 | 135 | 111/111 | 5/5 | 1.011 (10) | 1.012 (27) |
| input_voltage | delta_h | 135 | 113/111 | 0/2 | 0.983 (10) | 0.984 (27) |

output_voltage: **gần như chỉ giữ nguyên Raw; không có cải thiện nhất quán**; separation giảm tổng thể trên matched feature-pair conditions (3 gained, 18 lost).
input_voltage: **gần như chỉ giữ nguyên Raw; không có cải thiện nhất quán**; separation giảm tổng thể trên matched feature-pair conditions (5 gained, 11 lost).

Tau=0 không ép oscillatory sum bằng Raw. Residue và reconstruction error giữ riêng; modes+residue identity theo construction không chứng minh sum-only reconstruction tốt.

## 8. VMD modes

Giữ bands 0–1000, 1000–4000, 4000–10000, 10000–20000, 20000–33334 Hz. Chọn mode energy lớn nhất trong band **trước QC**, không cứu invalid bằng mode khác; all two/four centers ratio≤1.5. Across-condition recurrence cũng kiểm tra center ratio≤1.5. Mode index không xác lập mode vật lý; không tune band bằng H3/H4.

| Channel | Band Hz | Complete cells /60 | Evaluable feature-pair comparisons /450 | Disjoint | Extra vs Raw |
|---|---|---:|---:|---:|---:|
| output_voltage | 0–1000 | 34/60 | 177/450 | 156 | 7 |
| output_voltage | 1000–4000 | 5/60 | 3/450 | 3 | 2 |
| output_voltage | 4000–10000 | 0/60 | 0/450 | 0 | 0 |
| output_voltage | 10000–20000 | 0/60 | 0/450 | 0 | 0 |
| output_voltage | 20000–33334 | 0/60 | 0/450 | 0 | 0 |
| input_voltage | 0–1000 | 36/60 | 210/450 | 149 | 22 |
| input_voltage | 1000–4000 | 0/60 | 0/450 | 0 | 0 |
| input_voltage | 4000–10000 | 0/60 | 0/450 | 0 | 0 |
| input_voltage | 10000–20000 | 0/60 | 0/450 | 0 | 0 |
| input_voltage | 20000–33334 | 0/60 | 0/450 | 0 | 0 |

Extra nghĩa cùng cell Raw evaluable nhưng overlap, band evaluable và disjoint. Recurrence cần cùng pair/feature/band qua ≥2 conditions và centers compatible; một extra ở mỗi feature khác nhau không được gộp thành repeated evidence.

| Channel | Band | Pair/feature | Extra conditions | Speeds/loads | Across-condition compatible | Conditions |
|---|---|---|---:|---:|---|---|
| input_voltage | band_0_1000_Hz | H1-H2/alpha0 | 3 | 2/2 | True | 30/High;30/Low;50/High |
| input_voltage | band_0_1000_Hz | H1-H3/alpha0 | 2 | 2/1 | True | 30/High;45/High |
| input_voltage | band_0_1000_Hz | H1-H5/delta_alpha | 1 | 1/1 | True | 30/High |
| input_voltage | band_0_1000_Hz | H1-H5/alpha0 | 1 | 1/1 | True | 30/High |
| input_voltage | band_0_1000_Hz | H1-H5/delta_h | 1 | 1/1 | True | 30/High |
| input_voltage | band_0_1000_Hz | H2-H3/alpha0 | 1 | 1/1 | True | 30/High |
| input_voltage | band_0_1000_Hz | H4-H5/delta_alpha | 1 | 1/1 | True | 30/High |
| input_voltage | band_0_1000_Hz | H4-H5/delta_h | 1 | 1/1 | True | 30/High |
| input_voltage | band_0_1000_Hz | H4-H6/delta_h | 1 | 1/1 | True | 30/High |
| input_voltage | band_0_1000_Hz | H5-H6/delta_h | 1 | 1/1 | True | 30/High |
| input_voltage | band_0_1000_Hz | H1-H6/alpha0 | 1 | 1/1 | True | 35/Low |
| input_voltage | band_0_1000_Hz | H2-H5/alpha0 | 1 | 1/1 | True | 35/Low |
| input_voltage | band_0_1000_Hz | H2-H6/delta_alpha | 1 | 1/1 | True | 35/Low |
| input_voltage | band_0_1000_Hz | H2-H6/delta_h | 1 | 1/1 | True | 35/Low |
| input_voltage | band_0_1000_Hz | H3-H5/delta_alpha | 1 | 1/1 | True | 45/High |
| input_voltage | band_0_1000_Hz | H3-H5/delta_h | 1 | 1/1 | True | 45/High |
| input_voltage | band_0_1000_Hz | H1-H3/delta_h | 1 | 1/1 | True | 50/High |
| input_voltage | band_0_1000_Hz | H1-H4/delta_alpha | 1 | 1/1 | True | 50/High |
| input_voltage | band_0_1000_Hz | H1-H4/alpha0 | 1 | 1/1 | True | 50/High |
| output_voltage | band_0_1000_Hz | H1-H5/alpha0 | 1 | 1/1 | True | 30/High |
| output_voltage | band_0_1000_Hz | H4-H6/delta_alpha | 1 | 1/1 | True | 35/High |
| output_voltage | band_0_1000_Hz | H4-H6/delta_h | 1 | 1/1 | True | 35/High |
| output_voltage | band_0_1000_Hz | H2-H6/alpha0 | 1 | 1/1 | True | 35/Low |
| output_voltage | band_0_1000_Hz | H3-H5/delta_alpha | 1 | 1/1 | True | 40/High |
| output_voltage | band_0_1000_Hz | H1-H6/alpha0 | 1 | 1/1 | True | 45/High |
| output_voltage | band_0_1000_Hz | H2-H4/delta_h | 1 | 1/1 | True | 45/High |
| output_voltage | band_1000_4000_Hz | H3-H5/delta_alpha | 1 | 1/1 | True | 40/Low |
| output_voltage | band_1000_4000_Hz | H3-H5/delta_h | 1 | 1/1 | True | 40/Low |

Có 2 pair/feature/band groups có repeated compatible extra separation; xem vmd_extra_separation_recurrence.csv. Đây là candidate evidence trên cùng dataset, không xác nhận generalized improvement hoặc physical fault localization.

## 9. EMD

EMD chỉ đối chiếu phụ, giữ cùng QC và practical approximation Step 5; không sửa sifting để tăng valid coverage. Strict IMF validity tách riêng khỏi practical convergence và MF-DFA QC.

| Channel | Method | Decomposition valid /120 | Sum reconstruction error median/max | Strict IMF valid |
|---|---|---:|---:|---:|
| output_voltage | VMD | 120/120 | 7.12%/16.52% | N/A |
| output_voltage | EMD | 120/120 | 21.29%/42.22% | 0/120 |
| input_voltage | VMD | 120/120 | 5.94%/14.37% | N/A |
| input_voltage | EMD | 120/120 | 27.31%/60.08% | 0/120 |

EMD main features/separation/effects trình bày cùng các bảng trên; individual band coverage nằm trong qc_coverage.csv. Không gọi approximate EMD là benchmark strict IMF.

## 10. Limitations

- Hai acquisitions/cell tạo repeat ranges hẹp nhưng không phải validation của classifier. Không train classifier; không classification accuracy.
- QC missingness giới hạn full-grid conclusions; additive inference exploratory, interactions chưa evaluable nếu thiếu cells. Partial grid không đại diện cells invalid.
- Fixed scaling range <1 decade, fixed sample count ứng với số vòng quay khác nhau khi speed đổi. Scaling pass không chứng minh physical multifractality.
- Fault configurations gộp nhiều thay đổi và loại lỗi khác nhau. Không bearing/gear/shaft localization; H2 vs H5 không phải pure bearing effect.
- Cùng recording có hai channels tương quan; chỉ fit mỗi channel riêng. Independence giữa acquisitions là thiết kế giả định, chưa kiểm chứng thực nghiệm.
- Numerical source hash lịch sử không khớp nên rerun toàn bộ; kết luận Step 5 được kiểm tra về mặt xu hướng, không tuyên bố đây là binary-identical replay.
- Band matching chỉ theo center/energy, không chứng minh cùng nguồn cơ học. Sparse mode coverage và extra cases cùng dataset không phải validation độc lập.

## 11. Final conclusion

MF-DFA phân biệt fault configuration theo operating condition trong các comparisons valid; bảng 15 cặp chỉ rõ cặp/feature/condition tách hoặc overlap và coverage chưa đánh giá. Không suy rộng separation cho toàn bộ operating grid khi QC loại cells.
output_voltage — delta_alpha: fault > speed/load; alpha0: operating effect ≥ fault; delta_h: fault > speed/load.
input_voltage — delta_alpha: fault > speed/load; alpha0: operating effect ≥ fault; delta_h: fault > speed/load.

H3–H5 là điểm hạn chế rõ ở primary Raw: delta_alpha tách 2/7 conditions evaluable, alpha0 tách 6/7 conditions evaluable, delta_h tách 3/7 conditions evaluable. Không gọi đây là accuracy.

Repeated extra separation từ VMD modes: input_voltage H1-H2/alpha0 (band_0_1000_Hz, 30/High;30/Low;50/High); input_voltage H1-H3/alpha0 (band_0_1000_Hz, 30/High;45/High). Các extra ở output channel trong dữ liệu này chưa lặp lại cùng pair/feature/band qua nhiều conditions.
output_voltage: **gần như chỉ giữ nguyên Raw; không có cải thiện nhất quán**; separation giảm tổng thể trên matched feature-pair conditions (3 gained, 18 lost).
input_voltage: **gần như chỉ giữ nguyên Raw; không có cải thiện nhất quán**; separation giảm tổng thể trên matched feature-pair conditions (5 gained, 11 lost).

Giữ descriptive robustness rule Step 5: ≥75% complete-cell coverage; ROBUST khi ≥2 features vừa có disjoint fraction≥75% vừa có full-grid fault/operating ratio≥1; NOT ROBUST khi ≥2 features có fraction≤25% và ratio<1; còn lại CONDITION-DEPENDENT. Full-grid ratio không evaluable không được thay bằng partial-grid ratio để cứu verdict.

| Channel | Pipeline | Complete cells | Full-grid evaluable features | Verdict |
|---|---|---:|---:|---|
| output_voltage | RAW | 53/60 | 0/3 | CONDITION-DEPENDENT |
| output_voltage | VMD | 54/60 | 0/3 | CONDITION-DEPENDENT |
| output_voltage | EMD | 51/60 | 0/3 | CONDITION-DEPENDENT |
| input_voltage | RAW | 57/60 | 0/3 | CONDITION-DEPENDENT |
| input_voltage | VMD | 57/60 | 0/3 | CONDITION-DEPENDENT |
| input_voltage | EMD | 54/60 | 0/3 | CONDITION-DEPENDENT |

Không kết luận localization hoặc isolated bearing contribution. Individual low-frequency modes chỉ bổ sung bằng evidence matched và recurrence được ghi riêng.

Reproduce: `PHM_OUTPUT_DIR=/workspace/MFDFA_bearing/result/test/step6 /workspace/shared/mfdfa-venv/bin/python -B result/test/step6/scripts/pipeline.py`, sau đó `tests.py` và `finish.py` cùng env. Scripts phân tích reuse cách tính Step 5; mở rộng fault levels từ 4 lên 6 và expected grid từ 40 lên 60 cells. Dữ liệu/source cũ không sửa.


![feature_heatmaps output_voltage](figures/feature_heatmaps_output_voltage.png)

![paired_raw_vmd output_voltage](figures/paired_raw_vmd_output_voltage.png)

![fault_pair_separation output_voltage](figures/fault_pair_separation_output_voltage.png)

![qc_coverage output_voltage](figures/qc_coverage_output_voltage.png)

![feature_vs_load output_voltage](figures/feature_vs_load_output_voltage.png)

![vmd_frequency_modes output_voltage](figures/vmd_frequency_modes_output_voltage.png)

![Speed raw output_voltage](figures/feature_vs_speed_raw_output_voltage.png)

![Speed vmd output_voltage](figures/feature_vs_speed_vmd_output_voltage.png)

![feature_heatmaps input_voltage](figures/feature_heatmaps_input_voltage.png)

![paired_raw_vmd input_voltage](figures/paired_raw_vmd_input_voltage.png)

![fault_pair_separation input_voltage](figures/fault_pair_separation_input_voltage.png)

![qc_coverage input_voltage](figures/qc_coverage_input_voltage.png)

![feature_vs_load input_voltage](figures/feature_vs_load_input_voltage.png)

![vmd_frequency_modes input_voltage](figures/vmd_frequency_modes_input_voltage.png)

![Speed raw input_voltage](figures/feature_vs_speed_raw_input_voltage.png)

![Speed vmd input_voltage](figures/feature_vs_speed_vmd_input_voltage.png)

![Raw speed](figures/feature_vs_speed_raw.png)

![VMD speed](figures/feature_vs_speed_vmd.png)
