# STEP 7 — Full Spur validation

## 1. Dataset

**160/160 recordings**; missing=0, duplicate acquisition keys=0, duplicate numeric signal contents=0, unexpected files=0. S1–S8 × 30/35/40/45/50 Hz × High/Low × acquisitions 1/2.
Full lengths 266650–266656 samples (3.999748–3.999838 s), fs=66,666.7 Hz. 160 recording units, 320 analyzed channel signals. Cột 1=input_voltage, cột 2=output_voltage (primary), cột 3 giữ trong source audit nhưng không phân tích. Hai channels cùng recording không phải hai acquisitions độc lập. Không windows, resampling hoặc truncation.
Labels lấy trực tiếp từ VidData.xls, Sheet1 hàng 5–13. Mỗi component, vị trí và nguyên văn fault type được giữ; cột Good lưu đầy đủ trong config/fault_labels_from_workbook.json. Vị trí metadata không phải localization suy ra từ MF-DFA.

| State | Metadata nguyên văn — non-Good components |
| --- | --- |
| S1 | Healthy |
| S2 | Gear 32T=Chipped; Gear 48T=Eccentric |
| S3 | Gear 48T=Eccentric |
| S4 | Gear 48T=Eccentric; Gear 80T=Broken; Bearing IS:IS=Ball |
| S5 | Gear 32T=Chipped; Gear 48T=Eccentric; Gear 80T=Broken; Bearing IS:IS=Inner; Bearing ID:IS=Ball; Bearing OS:IS=Outer |
| S6 | Gear 80T=Broken; Bearing IS:IS=Inner; Bearing ID:IS=Ball; Bearing OS:IS=Outer; Shaft Input=Imbalance |
| S7 | Bearing IS:IS=Inner; Shaft Output=Keyway Sheared |
| S8 | Bearing ID:IS=Ball; Bearing OS:IS=Outer; Shaft Input=Imbalance |

Workbook có Total Runs=140 nhưng roster Spur được yêu cầu và audit thực tế là 160; không dùng con số tổng chung đó để loại recordings.

## 2. Locked configuration

Bằng đúng Step 6: q=-5…5, step 0.5; DFA order=2; cùng 40 scales; scaling fit 64–512 samples, actual 17 scales 64–468. Δα=max(alpha)−min(alpha), α0=alpha(q=0), Δh=max(hq)−min(hq). Không tune tham số hoặc range trên Spur.
QC giữ nguyên: ≥8 scale points; min R²≥0.90; ≥80% q có R²≥0.95; ≥80% q có local adjacent-slope CV≤0.5; min h(q)>0.05. analysis_valid=scaling_valid AND decomposition_valid. Shape warning giữ riêng; không biến thành gate mới.
VMD: K=7, alpha=500, tau=0, DC=False, init=1, tol=1e-7, max_iter=2000; cùng fused recurrence, no fast-math. EMD practical approximation: max_imfs=6, max_siftings=200, sd_threshold=0.2, envelope_tol=0.05, extrema_relative_tolerance=0.005; strict IMF validity audit riêng. Raw mean removal bên trong MF-DFA; decompositions dùng full signal bỏ global mean.
Chỉ đổi roster/fault contrasts từ 6 Helical states sang 8 Spur states và expected grid 60→80 cells, 15→28 pairs. Config equality, source hashes, method source hashes và Step 6 reference hashes được lưu. Không dùng Helical features làm Spur samples.
RAW=raw/raw; VMD_SUM=vmd/oscillatory_sum; EMD comparator=emd/oscillatory_sum. VMD sum-only reconstruction không bị ép bằng raw khi tau=0; residue giữ audit. Sum+residue identity theo construction không chứng minh sum-only reconstruction tốt.

## 3. QC


| Channel | Pipeline | Main signals valid | All components valid | Complete main cells |
| --- | --- | --- | --- | --- |
| output_voltage | RAW | 48/160 (30.0%) | 48/160 | 24/80 |
| output_voltage | VMD | 45/160 (28.1%) | 69/1440 | 22/80 |
| output_voltage | EMD | 43/160 (26.9%) | 186/1280 | 20/80 |
| input_voltage | RAW | 76/160 (47.5%) | 76/160 | 36/80 |
| input_voltage | VMD | 69/160 (43.1%) | 123/1440 | 33/80 |
| input_voltage | EMD | 75/160 (46.9%) | 197/1280 | 35/80 |

| output_voltage state | RAW valid/20; cells/10 | VMD_SUM valid/20; cells/10 | EMD_SUM valid/20; cells/10 |
| --- | --- | --- | --- |
| S1 | 8/20; 4/10 | 8/20; 4/10 | 7/20; 3/10 |
| S2 | 2/20; 1/10 | 2/20; 1/10 | 2/20; 1/10 |
| S3 | 0/20; 0/10 | 0/20; 0/10 | 0/20; 0/10 |
| S4 | 0/20; 0/10 | 0/20; 0/10 | 0/20; 0/10 |
| S5 | 2/20; 1/10 | 2/20; 1/10 | 1/20; 0/10 |
| S6 | 6/20; 3/10 | 6/20; 3/10 | 3/20; 1/10 |
| S7 | 16/20; 8/10 | 13/20; 6/10 | 16/20; 8/10 |
| S8 | 14/20; 7/10 | 14/20; 7/10 | 14/20; 7/10 |

| input_voltage state | RAW valid/20; cells/10 | VMD_SUM valid/20; cells/10 | EMD_SUM valid/20; cells/10 |
| --- | --- | --- | --- |
| S1 | 17/20; 8/10 | 15/20; 7/10 | 17/20; 8/10 |
| S2 | 2/20; 1/10 | 2/20; 1/10 | 2/20; 1/10 |
| S3 | 3/20; 1/10 | 3/20; 1/10 | 4/20; 2/10 |
| S4 | 0/20; 0/10 | 0/20; 0/10 | 0/20; 0/10 |
| S5 | 3/20; 1/10 | 2/20; 1/10 | 5/20; 2/10 |
| S6 | 18/20; 9/10 | 17/20; 8/10 | 17/20; 8/10 |
| S7 | 18/20; 9/10 | 18/20; 9/10 | 17/20; 8/10 |
| S8 | 15/20; 7/10 | 12/20; 6/10 | 13/20; 6/10 |

Không xếp hạng methods bằng all-components valid rate: số components khác nhau. Invalid numerical features/arrays vẫn giữ audit; mọi effect/separation summary loại invalid. Full QC theo recording/state/speed/load/band ở diagnostics và summaries/qc_coverage.csv.

## 4. S1–S8 feature results

Bảng pooled mean±sample SD chỉ mô tả valid recordings qua observed conditions; không dùng nó để suy ra separation giữa faults ở conditions khác nhau. Cell means±SD và repeat ranges theo exact speed/load nằm trong cell_feature_summary.csv và figures.

| Channel | Pipeline | State | Valid N/20 | delta_alpha | alpha0 | delta_h |
| --- | --- | --- | --- | --- | --- | --- |
| output_voltage | RAW | S1 | 8 | 0.6330±0.1292 | 0.7802±0.0439 | 0.3695±0.0646 |
| output_voltage | RAW | S2 | 2 | 0.8131±0.0525 | 0.7562±0.0042 | 0.6009±0.0467 |
| output_voltage | RAW | S3 | 0 | NE | NE | NE |
| output_voltage | RAW | S4 | 0 | NE | NE | NE |
| output_voltage | RAW | S5 | 2 | 0.9558±0.0957 | 0.8246±0.0153 | 0.7382±0.0648 |
| output_voltage | RAW | S6 | 6 | 0.4364±0.2617 | 0.8051±0.0553 | 0.2214±0.1687 |
| output_voltage | RAW | S7 | 16 | 0.7312±0.2736 | 0.7336±0.0386 | 0.4184±0.1635 |
| output_voltage | RAW | S8 | 14 | 0.3510±0.0908 | 0.7064±0.1641 | 0.1799±0.0565 |
| output_voltage | VMD | S1 | 8 | 0.6397±0.1326 | 0.7947±0.0443 | 0.3742±0.0663 |
| output_voltage | VMD | S2 | 2 | 0.8275±0.0529 | 0.7592±0.0062 | 0.6121±0.0469 |
| output_voltage | VMD | S3 | 0 | NE | NE | NE |
| output_voltage | VMD | S4 | 0 | NE | NE | NE |
| output_voltage | VMD | S5 | 2 | 0.9654±0.1036 | 0.8264±0.0160 | 0.7391±0.0671 |
| output_voltage | VMD | S6 | 6 | 0.4453±0.2860 | 0.8284±0.0588 | 0.2268±0.1816 |
| output_voltage | VMD | S7 | 13 | 0.6370±0.1804 | 0.7537±0.0282 | 0.3632±0.0998 |
| output_voltage | VMD | S8 | 14 | 0.3570±0.0979 | 0.7180±0.1741 | 0.1822±0.0610 |
| output_voltage | EMD | S1 | 7 | 0.5368±0.1276 | 0.7447±0.0495 | 0.3207±0.0683 |
| output_voltage | EMD | S2 | 2 | 0.6956±0.0433 | 0.7347±0.0164 | 0.5099±0.0240 |
| output_voltage | EMD | S3 | 0 | NE | NE | NE |
| output_voltage | EMD | S4 | 0 | NE | NE | NE |
| output_voltage | EMD | S5 | 1 | NE | NE | NE |
| output_voltage | EMD | S6 | 3 | 0.1988±0.0401 | 0.6935±0.0691 | 0.0546±0.0269 |
| output_voltage | EMD | S7 | 16 | 0.6642±0.2895 | 0.6840±0.0275 | 0.3822±0.1681 |
| output_voltage | EMD | S8 | 14 | 0.2954±0.1040 | 0.6668±0.1499 | 0.1485±0.0655 |
| input_voltage | RAW | S1 | 17 | 0.4987±0.1807 | 0.5196±0.2066 | 0.2622±0.1058 |
| input_voltage | RAW | S2 | 2 | 0.7090±0.0138 | 0.9152±0.0080 | 0.5169±0.0214 |
| input_voltage | RAW | S3 | 3 | 0.6094±0.0729 | 0.8406±0.0723 | 0.3748±0.0330 |
| input_voltage | RAW | S4 | 0 | NE | NE | NE |
| input_voltage | RAW | S5 | 3 | 1.2100±0.2794 | 0.5841±0.0802 | 0.9032±0.2159 |
| input_voltage | RAW | S6 | 18 | 0.3131±0.1700 | 0.7318±0.2282 | 0.1574±0.0821 |
| input_voltage | RAW | S7 | 18 | 0.5701±0.0901 | 0.8075±0.1101 | 0.3095±0.0627 |
| input_voltage | RAW | S8 | 15 | 0.3998±0.1431 | 0.5111±0.2071 | 0.1954±0.0509 |
| input_voltage | VMD | S1 | 15 | 0.5352±0.1655 | 0.5509±0.2136 | 0.2827±0.0959 |
| input_voltage | VMD | S2 | 2 | 0.7200±0.0150 | 0.9221±0.0082 | 0.5262±0.0231 |
| input_voltage | VMD | S3 | 3 | 0.6185±0.0821 | 0.8437±0.0759 | 0.3765±0.0355 |
| input_voltage | VMD | S4 | 0 | NE | NE | NE |
| input_voltage | VMD | S5 | 2 | 1.3591±0.0921 | 0.5245±0.0066 | 1.0196±0.0585 |
| input_voltage | VMD | S6 | 17 | 0.3401±0.1965 | 0.7248±0.2445 | 0.1748±0.0990 |
| input_voltage | VMD | S7 | 18 | 0.5740±0.0919 | 0.8283±0.1227 | 0.3123±0.0648 |
| input_voltage | VMD | S8 | 12 | 0.4442±0.1565 | 0.5429±0.2341 | 0.2166±0.0637 |
| input_voltage | EMD | S1 | 17 | 0.4286±0.1226 | 0.4884±0.1778 | 0.2244±0.0755 |
| input_voltage | EMD | S2 | 2 | 0.5766±0.0470 | 0.8910±0.0117 | 0.4176±0.0250 |
| input_voltage | EMD | S3 | 4 | 0.4604±0.1404 | 0.8252±0.0722 | 0.2618±0.0589 |
| input_voltage | EMD | S4 | 0 | NE | NE | NE |
| input_voltage | EMD | S5 | 5 | 1.0130±0.3160 | 0.5726±0.0993 | 0.7369±0.2852 |
| input_voltage | EMD | S6 | 17 | 0.2275±0.1085 | 0.6794±0.1901 | 0.1117±0.0546 |
| input_voltage | EMD | S7 | 17 | 0.4935±0.1075 | 0.6969±0.0933 | 0.2708±0.0526 |
| input_voltage | EMD | S8 | 13 | 0.3946±0.1281 | 0.4656±0.1898 | 0.1869±0.0368 |

## 5. Fault-pair separation

**28 pairs × 10 conditions × 3 features**, riêng mỗi channel/representation. Cùng speed/load, cả 4 independent acquisitions phải valid; modes còn cần compatible frequency centers. Disjoint iff max(A)<min(B) hoặc max(B)<min(A), strict inequality. Overlap là không tách; invalid/missing là NOT EVALUABLE, không gộp thành overlap. Đây là descriptive separation, không phải classification accuracy hay statistical significance.
Direction consistency giữ baseline 50 Hz/High từ Step 6: sign(mean(B)−mean(A)) giống baseline trên conditions khác khi cả baseline và condition valid. Baseline không valid thì baseline consistency NE; không chọn baseline khác theo kết quả. Ghi thêm positive/negative counts và uniform direction trên các conditions evaluable, không thay baseline rule.

| Channel | Pipeline | Feature | Evaluable/280 | Disjoint/evaluable | Same baseline direction/evaluable others |
| --- | --- | --- | --- | --- | --- |
| output_voltage | RAW | delta_alpha | 31/280 | 27/31 (87.1%) | 11/16 |
| output_voltage | RAW | alpha0 | 31/280 | 23/31 (74.2%) | 10/16 |
| output_voltage | RAW | delta_h | 31/280 | 30/31 (96.8%) | 11/16 |
| output_voltage | VMD | delta_alpha | 30/280 | 28/30 (93.3%) | 10/15 |
| output_voltage | VMD | alpha0 | 30/280 | 24/30 (80.0%) | 10/15 |
| output_voltage | VMD | delta_h | 30/280 | 29/30 (96.7%) | 10/15 |
| output_voltage | EMD | delta_alpha | 18/280 | 14/18 (77.8%) | 7/8 |
| output_voltage | EMD | alpha0 | 18/280 | 14/18 (77.8%) | 7/8 |
| output_voltage | EMD | delta_h | 18/280 | 17/18 (94.4%) | 7/8 |
| input_voltage | RAW | delta_alpha | 55/280 | 41/55 (74.5%) | 29/37 |
| input_voltage | RAW | alpha0 | 55/280 | 52/55 (94.5%) | 30/37 |
| input_voltage | RAW | delta_h | 55/280 | 45/55 (81.8%) | 29/37 |
| input_voltage | VMD | delta_alpha | 49/280 | 33/49 (67.3%) | 23/31 |
| input_voltage | VMD | alpha0 | 49/280 | 45/49 (91.8%) | 25/31 |
| input_voltage | VMD | delta_h | 49/280 | 37/49 (75.5%) | 24/31 |
| input_voltage | EMD | delta_alpha | 59/280 | 49/59 (83.1%) | 14/17 |
| input_voltage | EMD | alpha0 | 59/280 | 55/59 (93.2%) | 14/17 |
| input_voltage | EMD | delta_h | 59/280 | 46/59 (78.0%) | 13/17 |

### output_voltage — RAW

Mỗi ô = disjoint/evaluable/10 expected conditions. Direction chi tiết lưu ở fault_pair_summary.csv.

| Pair | delta_alpha | alpha0 | delta_h |
| --- | --- | --- | --- |
| S1-S2 | 1/1/10 | 0/1/10 | 1/1/10 |
| S1-S3 | 0/0/10 | 0/0/10 | 0/0/10 |
| S1-S4 | 0/0/10 | 0/0/10 | 0/0/10 |
| S1-S5 | 1/1/10 | 1/1/10 | 1/1/10 |
| S1-S6 | 2/3/10 | 2/3/10 | 3/3/10 |
| S1-S7 | 3/4/10 | 3/4/10 | 3/4/10 |
| S1-S8 | 4/4/10 | 2/4/10 | 4/4/10 |
| S2-S3 | 0/0/10 | 0/0/10 | 0/0/10 |
| S2-S4 | 0/0/10 | 0/0/10 | 0/0/10 |
| S2-S5 | 1/1/10 | 1/1/10 | 1/1/10 |
| S2-S6 | 1/1/10 | 0/1/10 | 1/1/10 |
| S2-S7 | 1/1/10 | 1/1/10 | 1/1/10 |
| S2-S8 | 1/1/10 | 0/1/10 | 1/1/10 |
| S3-S4 | 0/0/10 | 0/0/10 | 0/0/10 |
| S3-S5 | 0/0/10 | 0/0/10 | 0/0/10 |
| S3-S6 | 0/0/10 | 0/0/10 | 0/0/10 |
| S3-S7 | 0/0/10 | 0/0/10 | 0/0/10 |
| S3-S8 | 0/0/10 | 0/0/10 | 0/0/10 |
| S4-S5 | 0/0/10 | 0/0/10 | 0/0/10 |
| S4-S6 | 0/0/10 | 0/0/10 | 0/0/10 |
| S4-S7 | 0/0/10 | 0/0/10 | 0/0/10 |
| S4-S8 | 0/0/10 | 0/0/10 | 0/0/10 |
| S5-S6 | 1/1/10 | 1/1/10 | 1/1/10 |
| S5-S7 | 1/1/10 | 1/1/10 | 1/1/10 |
| S5-S8 | 1/1/10 | 1/1/10 | 1/1/10 |
| S6-S7 | 2/3/10 | 3/3/10 | 3/3/10 |
| S6-S8 | 2/3/10 | 2/3/10 | 3/3/10 |
| S7-S8 | 5/5/10 | 5/5/10 | 5/5/10 |

### output_voltage — VMD

Mỗi ô = disjoint/evaluable/10 expected conditions. Direction chi tiết lưu ở fault_pair_summary.csv.

| Pair | delta_alpha | alpha0 | delta_h |
| --- | --- | --- | --- |
| S1-S2 | 1/1/10 | 0/1/10 | 1/1/10 |
| S1-S3 | 0/0/10 | 0/0/10 | 0/0/10 |
| S1-S4 | 0/0/10 | 0/0/10 | 0/0/10 |
| S1-S5 | 1/1/10 | 1/1/10 | 1/1/10 |
| S1-S6 | 3/3/10 | 2/3/10 | 3/3/10 |
| S1-S7 | 3/4/10 | 3/4/10 | 3/4/10 |
| S1-S8 | 4/4/10 | 2/4/10 | 4/4/10 |
| S2-S3 | 0/0/10 | 0/0/10 | 0/0/10 |
| S2-S4 | 0/0/10 | 0/0/10 | 0/0/10 |
| S2-S5 | 1/1/10 | 1/1/10 | 1/1/10 |
| S2-S6 | 1/1/10 | 1/1/10 | 1/1/10 |
| S2-S7 | 1/1/10 | 1/1/10 | 1/1/10 |
| S2-S8 | 1/1/10 | 1/1/10 | 1/1/10 |
| S3-S4 | 0/0/10 | 0/0/10 | 0/0/10 |
| S3-S5 | 0/0/10 | 0/0/10 | 0/0/10 |
| S3-S6 | 0/0/10 | 0/0/10 | 0/0/10 |
| S3-S7 | 0/0/10 | 0/0/10 | 0/0/10 |
| S3-S8 | 0/0/10 | 0/0/10 | 0/0/10 |
| S4-S5 | 0/0/10 | 0/0/10 | 0/0/10 |
| S4-S6 | 0/0/10 | 0/0/10 | 0/0/10 |
| S4-S7 | 0/0/10 | 0/0/10 | 0/0/10 |
| S4-S8 | 0/0/10 | 0/0/10 | 0/0/10 |
| S5-S6 | 1/1/10 | 1/1/10 | 1/1/10 |
| S5-S7 | 1/1/10 | 1/1/10 | 1/1/10 |
| S5-S8 | 1/1/10 | 1/1/10 | 1/1/10 |
| S6-S7 | 3/3/10 | 3/3/10 | 3/3/10 |
| S6-S8 | 2/3/10 | 2/3/10 | 3/3/10 |
| S7-S8 | 4/4/10 | 4/4/10 | 4/4/10 |

### output_voltage — EMD

Mỗi ô = disjoint/evaluable/10 expected conditions. Direction chi tiết lưu ở fault_pair_summary.csv.

| Pair | delta_alpha | alpha0 | delta_h |
| --- | --- | --- | --- |
| S1-S2 | 1/1/10 | 0/1/10 | 1/1/10 |
| S1-S3 | 0/0/10 | 0/0/10 | 0/0/10 |
| S1-S4 | 0/0/10 | 0/0/10 | 0/0/10 |
| S1-S5 | 0/0/10 | 0/0/10 | 0/0/10 |
| S1-S6 | 1/1/10 | 1/1/10 | 1/1/10 |
| S1-S7 | 2/3/10 | 3/3/10 | 2/3/10 |
| S1-S8 | 1/3/10 | 3/3/10 | 3/3/10 |
| S2-S3 | 0/0/10 | 0/0/10 | 0/0/10 |
| S2-S4 | 0/0/10 | 0/0/10 | 0/0/10 |
| S2-S5 | 0/0/10 | 0/0/10 | 0/0/10 |
| S2-S6 | 1/1/10 | 1/1/10 | 1/1/10 |
| S2-S7 | 1/1/10 | 1/1/10 | 1/1/10 |
| S2-S8 | 1/1/10 | 1/1/10 | 1/1/10 |
| S3-S4 | 0/0/10 | 0/0/10 | 0/0/10 |
| S3-S5 | 0/0/10 | 0/0/10 | 0/0/10 |
| S3-S6 | 0/0/10 | 0/0/10 | 0/0/10 |
| S3-S7 | 0/0/10 | 0/0/10 | 0/0/10 |
| S3-S8 | 0/0/10 | 0/0/10 | 0/0/10 |
| S4-S5 | 0/0/10 | 0/0/10 | 0/0/10 |
| S4-S6 | 0/0/10 | 0/0/10 | 0/0/10 |
| S4-S7 | 0/0/10 | 0/0/10 | 0/0/10 |
| S4-S8 | 0/0/10 | 0/0/10 | 0/0/10 |
| S5-S6 | 0/0/10 | 0/0/10 | 0/0/10 |
| S5-S7 | 0/0/10 | 0/0/10 | 0/0/10 |
| S5-S8 | 0/0/10 | 0/0/10 | 0/0/10 |
| S6-S7 | 1/1/10 | 1/1/10 | 1/1/10 |
| S6-S8 | 0/1/10 | 0/1/10 | 1/1/10 |
| S7-S8 | 5/5/10 | 3/5/10 | 5/5/10 |

### input_voltage — RAW

Mỗi ô = disjoint/evaluable/10 expected conditions. Direction chi tiết lưu ở fault_pair_summary.csv.

| Pair | delta_alpha | alpha0 | delta_h |
| --- | --- | --- | --- |
| S1-S2 | 1/1/10 | 1/1/10 | 1/1/10 |
| S1-S3 | 0/1/10 | 1/1/10 | 1/1/10 |
| S1-S4 | 0/0/10 | 0/0/10 | 0/0/10 |
| S1-S5 | 1/1/10 | 1/1/10 | 1/1/10 |
| S1-S6 | 4/8/10 | 8/8/10 | 5/8/10 |
| S1-S7 | 5/7/10 | 7/7/10 | 5/7/10 |
| S1-S8 | 5/6/10 | 5/6/10 | 5/6/10 |
| S2-S3 | 0/0/10 | 0/0/10 | 0/0/10 |
| S2-S4 | 0/0/10 | 0/0/10 | 0/0/10 |
| S2-S5 | 0/0/10 | 0/0/10 | 0/0/10 |
| S2-S6 | 1/1/10 | 1/1/10 | 1/1/10 |
| S2-S7 | 1/1/10 | 1/1/10 | 1/1/10 |
| S2-S8 | 0/1/10 | 1/1/10 | 1/1/10 |
| S3-S4 | 0/0/10 | 0/0/10 | 0/0/10 |
| S3-S5 | 0/0/10 | 0/0/10 | 0/0/10 |
| S3-S6 | 0/1/10 | 1/1/10 | 1/1/10 |
| S3-S7 | 0/1/10 | 0/1/10 | 1/1/10 |
| S3-S8 | 1/1/10 | 1/1/10 | 1/1/10 |
| S4-S5 | 0/0/10 | 0/0/10 | 0/0/10 |
| S4-S6 | 0/0/10 | 0/0/10 | 0/0/10 |
| S4-S7 | 0/0/10 | 0/0/10 | 0/0/10 |
| S4-S8 | 0/0/10 | 0/0/10 | 0/0/10 |
| S5-S6 | 1/1/10 | 1/1/10 | 1/1/10 |
| S5-S7 | 1/1/10 | 1/1/10 | 1/1/10 |
| S5-S8 | 1/1/10 | 1/1/10 | 1/1/10 |
| S6-S7 | 7/8/10 | 7/8/10 | 7/8/10 |
| S6-S8 | 6/7/10 | 7/7/10 | 5/7/10 |
| S7-S8 | 6/7/10 | 7/7/10 | 6/7/10 |

### input_voltage — VMD

Mỗi ô = disjoint/evaluable/10 expected conditions. Direction chi tiết lưu ở fault_pair_summary.csv.

| Pair | delta_alpha | alpha0 | delta_h |
| --- | --- | --- | --- |
| S1-S2 | 1/1/10 | 1/1/10 | 1/1/10 |
| S1-S3 | 0/1/10 | 1/1/10 | 0/1/10 |
| S1-S4 | 0/0/10 | 0/0/10 | 0/0/10 |
| S1-S5 | 1/1/10 | 1/1/10 | 1/1/10 |
| S1-S6 | 3/6/10 | 6/6/10 | 4/6/10 |
| S1-S7 | 3/6/10 | 6/6/10 | 4/6/10 |
| S1-S8 | 4/5/10 | 4/5/10 | 3/5/10 |
| S2-S3 | 0/0/10 | 0/0/10 | 0/0/10 |
| S2-S4 | 0/0/10 | 0/0/10 | 0/0/10 |
| S2-S5 | 0/0/10 | 0/0/10 | 0/0/10 |
| S2-S6 | 1/1/10 | 1/1/10 | 1/1/10 |
| S2-S7 | 1/1/10 | 1/1/10 | 1/1/10 |
| S2-S8 | 0/1/10 | 1/1/10 | 1/1/10 |
| S3-S4 | 0/0/10 | 0/0/10 | 0/0/10 |
| S3-S5 | 0/0/10 | 0/0/10 | 0/0/10 |
| S3-S6 | 0/1/10 | 1/1/10 | 1/1/10 |
| S3-S7 | 0/1/10 | 1/1/10 | 1/1/10 |
| S3-S8 | 1/1/10 | 1/1/10 | 1/1/10 |
| S4-S5 | 0/0/10 | 0/0/10 | 0/0/10 |
| S4-S6 | 0/0/10 | 0/0/10 | 0/0/10 |
| S4-S7 | 0/0/10 | 0/0/10 | 0/0/10 |
| S4-S8 | 0/0/10 | 0/0/10 | 0/0/10 |
| S5-S6 | 1/1/10 | 1/1/10 | 1/1/10 |
| S5-S7 | 1/1/10 | 1/1/10 | 1/1/10 |
| S5-S8 | 1/1/10 | 0/1/10 | 1/1/10 |
| S6-S7 | 6/8/10 | 6/8/10 | 6/8/10 |
| S6-S8 | 5/6/10 | 6/6/10 | 4/6/10 |
| S7-S8 | 4/6/10 | 6/6/10 | 5/6/10 |

### input_voltage — EMD

Mỗi ô = disjoint/evaluable/10 expected conditions. Direction chi tiết lưu ở fault_pair_summary.csv.

| Pair | delta_alpha | alpha0 | delta_h |
| --- | --- | --- | --- |
| S1-S2 | 1/1/10 | 1/1/10 | 1/1/10 |
| S1-S3 | 1/2/10 | 2/2/10 | 0/2/10 |
| S1-S4 | 0/0/10 | 0/0/10 | 0/0/10 |
| S1-S5 | 2/2/10 | 2/2/10 | 2/2/10 |
| S1-S6 | 6/7/10 | 7/7/10 | 5/7/10 |
| S1-S7 | 5/6/10 | 6/6/10 | 4/6/10 |
| S1-S8 | 4/5/10 | 3/5/10 | 4/5/10 |
| S2-S3 | 1/1/10 | 0/1/10 | 1/1/10 |
| S2-S4 | 0/0/10 | 0/0/10 | 0/0/10 |
| S2-S5 | 1/1/10 | 1/1/10 | 1/1/10 |
| S2-S6 | 1/1/10 | 1/1/10 | 1/1/10 |
| S2-S7 | 1/1/10 | 1/1/10 | 1/1/10 |
| S2-S8 | 1/1/10 | 1/1/10 | 1/1/10 |
| S3-S4 | 0/0/10 | 0/0/10 | 0/0/10 |
| S3-S5 | 1/1/10 | 1/1/10 | 1/1/10 |
| S3-S6 | 2/2/10 | 1/2/10 | 2/2/10 |
| S3-S7 | 0/2/10 | 2/2/10 | 0/2/10 |
| S3-S8 | 2/2/10 | 2/2/10 | 1/2/10 |
| S4-S5 | 0/0/10 | 0/0/10 | 0/0/10 |
| S4-S6 | 0/0/10 | 0/0/10 | 0/0/10 |
| S4-S7 | 0/0/10 | 0/0/10 | 0/0/10 |
| S4-S8 | 0/0/10 | 0/0/10 | 0/0/10 |
| S5-S6 | 2/2/10 | 2/2/10 | 2/2/10 |
| S5-S7 | 2/2/10 | 2/2/10 | 2/2/10 |
| S5-S8 | 2/2/10 | 2/2/10 | 2/2/10 |
| S6-S7 | 6/7/10 | 7/7/10 | 7/7/10 |
| S6-S8 | 4/6/10 | 6/6/10 | 4/6/10 |
| S7-S8 | 4/5/10 | 5/5/10 | 4/5/10 |

Descriptive feature ranking:

output_voltage/RAW: nhiều disjoint conditions nhất = delta_h; delta_alpha=27, alpha0=23, delta_h=30. Ranking within-condition không chứng minh operating invariance.

output_voltage/VMD: nhiều disjoint conditions nhất = delta_h; delta_alpha=28, alpha0=24, delta_h=29. Ranking within-condition không chứng minh operating invariance.

input_voltage/RAW: nhiều disjoint conditions nhất = alpha0; delta_alpha=41, alpha0=52, delta_h=45. Ranking within-condition không chứng minh operating invariance.

input_voltage/VMD: nhiều disjoint conditions nhất = alpha0; delta_alpha=33, alpha0=45, delta_h=37. Ranking within-condition không chứng minh operating invariance.

## 6. Speed/load effects

Giữ Step 6 metrics: normalized speed range=(max−min)/abs(mean) trên đủ 5 complete cells; load difference=2|High−Low|/(|High|+|Low|) cùng fault/speed; thiếu valid/matched repeat/cell thì trajectory/comparison NE. Full-grid fault/operating ratio=median matched between-fault mean gap / median within-fault SD của 10 cell means; phải đủ 8 states. Supplemental partial-grid dùng common conditions đầy đủ tất cả 8 states, ≥3 conditions, matching trước kết quả, không interpolation hoặc thay full-grid verdict bằng partial grid.

| Channel | Pipeline/feature | Speed range median (N/16) | Load difference median (N/40) | Full-grid fault/operating ratio | Common grid; partial ratio |
| --- | --- | --- | --- | --- | --- |
| output_voltage | RAW/delta_alpha | 73.4% (1/16) | 28.8% (6/40) | NE | 0/10; NE |
| output_voltage | RAW/alpha0 | 42.7% (1/16) | 8.8% (6/40) | NE | 0/10; NE |
| output_voltage | RAW/delta_h | 97.4% (1/16) | 32.8% (6/40) | NE | 0/10; NE |
| output_voltage | VMD/delta_alpha | 80.7% (1/16) | 26.8% (4/40) | NE | 0/10; NE |
| output_voltage | VMD/alpha0 | 44.6% (1/16) | 21.5% (4/40) | NE | 0/10; NE |
| output_voltage | VMD/delta_h | 104.5% (1/16) | 37.7% (4/40) | NE | 0/10; NE |
| output_voltage | EMD/delta_alpha | 107.1% (1/16) | 38.1% (6/40) | NE | 0/10; NE |
| output_voltage | EMD/alpha0 | 46.0% (1/16) | 3.6% (6/40) | NE | 0/10; NE |
| output_voltage | EMD/delta_h | 140.5% (1/16) | 48.8% (6/40) | NE | 0/10; NE |
| input_voltage | RAW/delta_alpha | 32.4% (3/16) | 59.8% (13/40) | NE | 0/10; NE |
| input_voltage | RAW/alpha0 | 35.4% (3/16) | 57.4% (13/40) | NE | 0/10; NE |
| input_voltage | RAW/delta_h | 41.9% (3/16) | 46.3% (13/40) | NE | 0/10; NE |
| input_voltage | VMD/delta_alpha | 33.2% (2/16) | 34.1% (10/40) | NE | 0/10; NE |
| input_voltage | VMD/alpha0 | 28.4% (2/16) | 57.7% (10/40) | NE | 0/10; NE |
| input_voltage | VMD/delta_h | 32.7% (2/16) | 43.0% (10/40) | NE | 0/10; NE |
| input_voltage | EMD/delta_alpha | 17.3% (1/16) | 30.3% (12/40) | NE | 0/10; NE |
| input_voltage | EMD/alpha0 | 18.6% (1/16) | 54.7% (12/40) | NE | 0/10; NE |
| input_voltage | EMD/delta_h | 23.7% (1/16) | 35.5% (12/40) | NE | 0/10; NE |

Categorical fault+speed+load dùng sum-to-zero Helmert contrasts; additive partial η²=incremental SS/(incremental SS+residual SS), không cộng thành total variance. Full fault*speed*load chỉ fit khi đủ 80 cells, full rank và ≥20 residual df. Với partial coverage, additive main effect không loại trừ interactions và không phải causal effect.
Giữ OLS, HC3 Wald, null-imposed Rademacher wild bootstrap 1,999 draws, seed 6509, HC2-scaled reduced residuals, studentized HC3 Wald; BH riêng mỗi channel trên family representations/features/models/terms. Wild/HC3 exploratory, classical p chỉ reference. Shapiro, heteroskedasticity LM và residual plots giữ diagnostics; 2 repeats/cell hạn chế inference.

| Channel | Pipeline/feature | η² fault/speed/load | Wild p BH fault/speed/load | Full interactions |
| --- | --- | --- | --- | --- |
| output_voltage | RAW/delta_alpha | NE | NE | NOT EVALUABLE |
| output_voltage | RAW/alpha0 | NE | NE | NOT EVALUABLE |
| output_voltage | RAW/delta_h | NE | NE | NOT EVALUABLE |
| output_voltage | VMD/delta_alpha | NE | NE | NOT EVALUABLE |
| output_voltage | VMD/alpha0 | NE | NE | NOT EVALUABLE |
| output_voltage | VMD/delta_h | NE | NE | NOT EVALUABLE |
| output_voltage | EMD/delta_alpha | NE | NE | NOT EVALUABLE |
| output_voltage | EMD/alpha0 | NE | NE | NOT EVALUABLE |
| output_voltage | EMD/delta_h | NE | NE | NOT EVALUABLE |
| input_voltage | RAW/delta_alpha | NE | NE | NOT EVALUABLE |
| input_voltage | RAW/alpha0 | NE | NE | NOT EVALUABLE |
| input_voltage | RAW/delta_h | NE | NE | NOT EVALUABLE |
| input_voltage | VMD/delta_alpha | NE | NE | NOT EVALUABLE |
| input_voltage | VMD/alpha0 | NE | NE | NOT EVALUABLE |
| input_voltage | VMD/delta_h | NE | NE | NOT EVALUABLE |
| input_voltage | EMD/delta_alpha | NE | NE | NOT EVALUABLE |
| input_voltage | EMD/alpha0 | NE | NE | NOT EVALUABLE |
| input_voltage | EMD/delta_h | NE | NE | NOT EVALUABLE |

| Channel | Main pipeline | Valid recording N | Represented states/8 | Missing states | Additive rank/13 | Observed cells/80 |
| --- | --- | --- | --- | --- | --- | --- |
| input_voltage | EMD | 75 | 7 | S4 | 12 | 40 |
| output_voltage | EMD | 43 | 6 | S3;S4 | 11 | 23 |
| input_voltage | RAW | 76 | 7 | S4 | 12 | 40 |
| output_voltage | RAW | 48 | 6 | S3;S4 | 11 | 24 |
| input_voltage | VMD | 69 | 7 | S4 | 12 | 36 |
| output_voltage | VMD | 45 | 6 | S3;S4 | 11 | 23 |

Không bỏ states QC-invalid để fit một fault model với ít classes hơn. Khi thiếu states/rank, toàn bộ S1–S8 factorial result là NOT EVALUABLE; không diễn giải thành fault yếu hơn speed/load. Main QC failure reasons có recording-level trace tại diagnostics/main_qc_failure_reasons.csv.

**Kiểm tra kết luận Helical trên Spur độc lập:**

- output_voltage, RAW delta_alpha: NE.
- output_voltage, RAW alpha0: NE.
- output_voltage, RAW delta_h: NE.
- input_voltage, RAW delta_alpha: NE.
- input_voltage, RAW alpha0: NE.
- input_voltage, RAW delta_h: NE.

## 7. Raw vs VMD

Paired valid recordings only; relative feature difference=2|Raw−VMD|/(|Raw|+|VMD|). Correlation cao không tự chứng minh improvement.

| Channel | Feature | Paired N | Pearson r | Median relative difference | Mean signed change | RMSE |
| --- | --- | --- | --- | --- | --- | --- |
| output_voltage | delta_alpha | 45 | 0.9990 | 1.8% | 0.00824 | 0.01532 |
| output_voltage | alpha0 | 45 | 0.9979 | 1.7% | 0.01344 | 0.01605 |
| output_voltage | delta_h | 45 | 0.9992 | 1.6% | 0.00475 | 0.00911 |
| input_voltage | delta_alpha | 69 | 0.9895 | 2.8% | 0.01380 | 0.03626 |
| input_voltage | alpha0 | 69 | 0.9900 | 2.9% | 0.00569 | 0.03968 |
| input_voltage | delta_h | 69 | 0.9927 | 2.1% | 0.00798 | 0.02176 |

| Channel | Feature | Common pair-conditions | Raw/VMD disjoint | Gained/lost | Speed ratio (N) | Load ratio (N) |
| --- | --- | --- | --- | --- | --- | --- |
| output_voltage | delta_alpha | 30 | 26/28 | 2/0 | 1.100 (1) | 1.060 (4) |
| output_voltage | alpha0 | 30 | 22/24 | 2/0 | 1.044 (1) | 1.033 (4) |
| output_voltage | delta_h | 30 | 29/29 | 0/0 | 1.073 (1) | 1.059 (4) |
| input_voltage | delta_alpha | 49 | 37/33 | 0/4 | 1.044 (2) | 1.054 (10) |
| input_voltage | alpha0 | 49 | 46/45 | 1/2 | 1.073 (2) | 1.090 (10) |
| input_voltage | delta_h | 49 | 40/37 | 1/4 | 0.958 (2) | 1.053 (10) |

Sensitivity ratios VMD/Raw <1 là giảm sensitivity, tính trên đúng intersection trajectories/load comparisons. Separation chỉ trên common valid pair-conditions, không trộn coverage gain thành separation gain. Giữ rule diễn giải Step 6: consistent improvement đòi tăng separation và giảm speed/load sensitivity trên cả 3 features; preservation dựa trên r≥0.95, đọc kèm feature differences và gains/losses.

output_voltage: **gần như giữ nguyên RAW; không có cải thiện nhất quán**. Matched feature-pair cases gained=4, lost=0; separation tăng tổng thể.

input_voltage: **gần như giữ nguyên RAW; không có cải thiện nhất quán**. Matched feature-pair cases gained=2, lost=10; separation kém đi tổng thể.

## 8. VMD modes

Giữ fixed bands 0–1000, 1000–4000, 4000–10000, 10000–20000, 20000–33334 Hz. Đại diện = mode energy lớn nhất trong band **trước QC**, không chọn class-specific hoặc rescue bằng mode khác. Repeat/cross-fault centers phải ratio≤1.5; across-condition recurrence cũng ratio≤1.5. Mode rank không chứng minh same physical mode.
Extra separation: RAW cùng condition evaluable nhưng overlap; VMD band evaluable và disjoint. Repeated extra: cùng channel/band/pair/feature qua ≥2 conditions và frequency compatible. Báo riêng distinct speeds và loads; không gộp extras khác feature thành một kết quả lặp.

| Channel | Band Hz | Complete cells/80 | Evaluable feature-pair conditions/840 | Disjoint | Extra vs RAW |
| --- | --- | --- | --- | --- | --- |
| output_voltage | 0–1000 | 7/80 | 3/840 | 1 | 1 |
| output_voltage | 1000–4000 | 0/80 | 0/840 | 0 | 0 |
| output_voltage | 4000–10000 | 0/80 | 0/840 | 0 | 0 |
| output_voltage | 10000–20000 | 0/80 | 0/840 | 0 | 0 |
| output_voltage | 20000–33334 | 0/80 | 0/840 | 0 | 0 |
| input_voltage | 0–1000 | 16/80 | 12/840 | 10 | 2 |
| input_voltage | 1000–4000 | 1/80 | 0/840 | 0 | 0 |
| input_voltage | 4000–10000 | 0/80 | 0/840 | 0 | 0 |
| input_voltage | 10000–20000 | 0/80 | 0/840 | 0 | 0 |
| input_voltage | 20000–33334 | 0/80 | 0/840 | 0 | 0 |

| Channel | Band | Pair/feature | Extra conditions | Speeds/loads | Conditions |
| --- | --- | --- | --- | --- | --- |
| — | — | No compatible repeated extra | 0 | — | — |

0 repeated compatible groups. Toàn bộ single/repeated/incompatible extras ghi trong vmd_extra_separation_recurrence.csv. Coverage thấp và same-dataset discovery không chứng minh generalized improvement hoặc localization.

## 9. EMD

Comparator phụ, practical approximation và QC giữ nguyên; không thay EMD để tăng coverage. Main EMD features/separation/effects được báo với cùng denominator/matching rules; individual-band coverage ở qc_coverage.csv. Strict IMF audit khác practical convergence và scaling validity.

| Channel | Method | Decomposition valid/160 | Sum-only reconstruction error median/max | Strict IMF valid |
| --- | --- | --- | --- | --- |
| output_voltage | VMD | 160/160 | 5.69%/19.63% | N/A |
| output_voltage | EMD | 160/160 | 12.00%/45.93% | 0/160 |
| input_voltage | VMD | 160/160 | 5.56%/17.18% | N/A |
| input_voltage | EMD | 160/160 | 11.82%/43.10% | 0/160 |

## 10. Helical vs Spur comparison

So sánh sau khi phân tích Spur độc lập. Không gộp Helical/Spur thành cùng class, không coi Hn tương ứng Sn, không so sánh raw feature magnitudes để tuyên bố cùng physical fault. Cùng speed/load/procedure nhưng topology, gear tooth counts và fault definitions thay đổi cùng nhau: không cô lập causal topology effect.
Bảng dưới so sánh pattern và descriptive metrics với denominator riêng. Partial η² khác số fault levels/coverage; các giá trị không phải kiểm định sự bằng nhau giữa gearbox types. BH families cũng riêng mỗi dataset/channel.

| Channel | RAW feature | Helical pattern | Helical η² F/S/L | Spur pattern | Spur η² F/S/L | Transfer |
| --- | --- | --- | --- | --- | --- | --- |
| output_voltage | delta_alpha | FAULT > each operating main effect | 0.550/0.099/0.016 | NOT EVALUABLE | NE/NE/NE | pattern differs / incomplete |
| output_voltage | alpha0 | OPERATING main effect >= fault | 0.320/0.510/0.629 | NOT EVALUABLE | NE/NE/NE | pattern differs / incomplete |
| output_voltage | delta_h | FAULT > each operating main effect | 0.565/0.105/0.037 | NOT EVALUABLE | NE/NE/NE | pattern differs / incomplete |
| input_voltage | delta_alpha | FAULT > each operating main effect | 0.289/0.046/0.089 | NOT EVALUABLE | NE/NE/NE | pattern differs / incomplete |
| input_voltage | alpha0 | OPERATING main effect >= fault | 0.242/0.557/0.325 | NOT EVALUABLE | NE/NE/NE | pattern differs / incomplete |
| input_voltage | delta_h | FAULT > each operating main effect | 0.286/0.044/0.045 | NOT EVALUABLE | NE/NE/NE | pattern differs / incomplete |

| Topology | Channel | Pipeline | Feature | Complete cells | Disjoint/evaluable/expected conditions | Disjoint fraction |
| --- | --- | --- | --- | --- | --- | --- |
| Helical | output_voltage | RAW | delta_alpha | 53/60 | 111/122/150 | 91.0% |
| Helical | output_voltage | RAW | alpha0 | 53/60 | 115/122/150 | 94.3% |
| Helical | output_voltage | RAW | delta_h | 53/60 | 107/122/150 | 87.7% |
| Helical | output_voltage | VMD | delta_alpha | 54/60 | 108/126/150 | 85.7% |
| Helical | output_voltage | VMD | alpha0 | 54/60 | 117/126/150 | 92.9% |
| Helical | output_voltage | VMD | delta_h | 54/60 | 105/126/150 | 83.3% |
| Helical | input_voltage | RAW | delta_alpha | 57/60 | 113/135/150 | 83.7% |
| Helical | input_voltage | RAW | alpha0 | 57/60 | 111/135/150 | 82.2% |
| Helical | input_voltage | RAW | delta_h | 57/60 | 113/135/150 | 83.7% |
| Helical | input_voltage | VMD | delta_alpha | 57/60 | 109/135/150 | 80.7% |
| Helical | input_voltage | VMD | alpha0 | 57/60 | 111/135/150 | 82.2% |
| Helical | input_voltage | VMD | delta_h | 57/60 | 111/135/150 | 82.2% |
| Spur | output_voltage | RAW | delta_alpha | 24/80 | 27/31/280 | 87.1% |
| Spur | output_voltage | RAW | alpha0 | 24/80 | 23/31/280 | 74.2% |
| Spur | output_voltage | RAW | delta_h | 24/80 | 30/31/280 | 96.8% |
| Spur | output_voltage | VMD | delta_alpha | 22/80 | 28/30/280 | 93.3% |
| Spur | output_voltage | VMD | alpha0 | 22/80 | 24/30/280 | 80.0% |
| Spur | output_voltage | VMD | delta_h | 22/80 | 29/30/280 | 96.7% |
| Spur | input_voltage | RAW | delta_alpha | 36/80 | 41/55/280 | 74.5% |
| Spur | input_voltage | RAW | alpha0 | 36/80 | 52/55/280 | 94.5% |
| Spur | input_voltage | RAW | delta_h | 36/80 | 45/55/280 | 81.8% |
| Spur | input_voltage | VMD | delta_alpha | 33/80 | 33/49/280 | 67.3% |
| Spur | input_voltage | VMD | alpha0 | 33/80 | 45/49/280 | 91.8% |
| Spur | input_voltage | VMD | delta_h | 33/80 | 37/49/280 | 75.5% |

Feature operating sensitivity và VMD preservation/gains/losses cho từng topology ở comparison/helical_vs_spur_summary.csv. Không đánh giá topology bằng fault pair counts thô (15 so với 28); đọc fractions và missingness.

| Topology | Channel | Pipeline/feature | Median speed normalized range (valid trajectories) | Median load relative difference (valid comparisons) |
| --- | --- | --- | --- | --- |
| Helical | output_voltage | RAW/delta_alpha | 56.5% (6) | 54.4% (23) |
| Helical | output_voltage | RAW/alpha0 | 22.4% (6) | 23.0% (23) |
| Helical | output_voltage | RAW/delta_h | 61.7% (6) | 56.8% (23) |
| Helical | output_voltage | VMD/delta_alpha | 57.0% (6) | 53.9% (24) |
| Helical | output_voltage | VMD/alpha0 | 22.7% (6) | 27.8% (24) |
| Helical | output_voltage | VMD/delta_h | 61.5% (6) | 55.2% (24) |
| Helical | input_voltage | RAW/delta_alpha | 64.0% (10) | 35.4% (27) |
| Helical | input_voltage | RAW/alpha0 | 21.4% (10) | 14.0% (27) |
| Helical | input_voltage | RAW/delta_h | 61.2% (10) | 41.8% (27) |
| Helical | input_voltage | VMD/delta_alpha | 60.4% (10) | 34.5% (27) |
| Helical | input_voltage | VMD/alpha0 | 21.2% (10) | 13.9% (27) |
| Helical | input_voltage | VMD/delta_h | 59.3% (10) | 38.4% (27) |
| Spur | output_voltage | RAW/delta_alpha | 73.4% (1) | 28.8% (6) |
| Spur | output_voltage | RAW/alpha0 | 42.7% (1) | 8.8% (6) |
| Spur | output_voltage | RAW/delta_h | 97.4% (1) | 32.8% (6) |
| Spur | output_voltage | VMD/delta_alpha | 80.7% (1) | 26.8% (4) |
| Spur | output_voltage | VMD/alpha0 | 44.6% (1) | 21.5% (4) |
| Spur | output_voltage | VMD/delta_h | 104.5% (1) | 37.7% (4) |
| Spur | input_voltage | RAW/delta_alpha | 32.4% (3) | 59.8% (13) |
| Spur | input_voltage | RAW/alpha0 | 35.4% (3) | 57.4% (13) |
| Spur | input_voltage | RAW/delta_h | 41.9% (3) | 46.3% (13) |
| Spur | input_voltage | VMD/delta_alpha | 33.2% (2) | 34.1% (10) |
| Spur | input_voltage | VMD/alpha0 | 28.4% (2) | 57.7% (10) |
| Spur | input_voltage | VMD/delta_h | 32.7% (2) | 43.0% (10) |

Association pattern giữ được không có nghĩa feature operating-condition invariant. Raw feature distributions và QC khác nhau có thể phụ thuộc topology/fault set/transfer path; không cô lập causal topology effect. Các normalized ranges/load differences và coverage phía trên đánh giá độ ổn định theo điều kiện, không dùng raw magnitudes giữa gearbox types làm same-fault evidence.

output_voltage/delta_alpha: không giữ nguyên qualitative association pattern Helical trên Spur, hoặc chưa đủ dữ liệu valid để đánh giá.

output_voltage/alpha0: không giữ nguyên qualitative association pattern Helical trên Spur, hoặc chưa đủ dữ liệu valid để đánh giá.

output_voltage/delta_h: không giữ nguyên qualitative association pattern Helical trên Spur, hoặc chưa đủ dữ liệu valid để đánh giá.

input_voltage/delta_alpha: không giữ nguyên qualitative association pattern Helical trên Spur, hoặc chưa đủ dữ liệu valid để đánh giá.

input_voltage/alpha0: không giữ nguyên qualitative association pattern Helical trên Spur, hoặc chưa đủ dữ liệu valid để đánh giá.

input_voltage/delta_h: không giữ nguyên qualitative association pattern Helical trên Spur, hoặc chưa đủ dữ liệu valid để đánh giá.

output_voltage: Helical VMD_SUM preserves RAW; Spur VMD_SUM gần như giữ nguyên RAW; không có cải thiện nhất quán. RAW là reference trực tiếp; VMD có consistency trong preservation chỉ khi cả hai topology đáp ứng rule và difference/separation tables phù hợp.

input_voltage: Helical VMD_SUM preserves RAW; Spur VMD_SUM gần như giữ nguyên RAW; không có cải thiện nhất quán. RAW là reference trực tiếp; VMD có consistency trong preservation chỉ khi cả hai topology đáp ứng rule và difference/separation tables phù hợp.

**RAW hay VMD có kết luận nhất quán hơn?** Không ưu tiên VMD chỉ vì có decomposition. Đọc paired differences và common-condition gains/losses ở cả hai datasets: khi VMD chỉ preserves RAW thì preservation có thể consistent giữa topology, nhưng không tạo bằng chứng fault separation generalize tốt hơn RAW. Nếu Spur main factorial result NE, cả RAW và VMD đều chưa hỗ trợ kết luận toàn bộ fault set; không xếp hạng generalization bằng sparse valid-subset separation.

## 11. Limitations

- Hai acquisitions/cell: disjoint ranges và direction consistency chỉ descriptive, không classification accuracy, không generalization estimate của classifier.
- QC missingness có thể phụ thuộc faults/conditions; partial-grid inference không cứu invalid cells. Additive effects không loại trừ interactions; robust p không cứu confounding hoặc sparse coverage.
- Scaling range <1 decade, same sample scales ứng với số vòng quay khác nhau khi speed đổi; QC pass không chứng minh physical multifractality.
- S2/S4/S5/S6/S7/S8 giữ multi-component definitions. Không causal gear/bearing/shaft localization từ pairwise feature differences.
- Fault sets, topology, tooth counts và transfer paths khác nhau; chưa cô lập topology effect. Không map Helical/Spur bằng class index hoặc feature magnitude.
- Channels cùng recording tương quan, chỉ phân tích riêng mỗi channel; independence giữa acquisitions là thiết kế giả định, chưa chứng minh empirically.
- Individual-mode evidence dựa fixed frequency bands và centers, không xác định cùng nguồn cơ học. Repeated same-dataset extras vẫn cần validation độc lập.
- EMD là practical approximation; strict IMF validity và main-feature QC phải đọc riêng.

## 12. Final conclusion

**Liệu kết luận MF-DFA trên Helical còn giữ được trên Spur khi áp dụng nguyên phương pháp?**

Chưa thể xác nhận toàn bộ kết luận Helical trên Spur: chỉ 0/6 feature/channel association patterns evaluable, trong đó 0 giữ pattern Helical; 6 chưa đánh giá được. NOT EVALUABLE không có nghĩa fault effect bằng zero. Ít nhất một channel không đạt 75% complete RAW cells theo rule Step 6; coverage không hỗ trợ full-dataset transfer.

Coverage failure ở pipeline/range/QC đã khóa không chứng minh MF-DFA nói chung không thể dùng cho Spur. Step 7 không tìm range hoặc tham số khác để cứu coverage, nên kết luận giới hạn đúng phương pháp được chuyển nguyên từ Helical.

output_voltage: VMD_SUM **gần như giữ nguyên RAW; không có cải thiện nhất quán**.

input_voltage: VMD_SUM **gần như giữ nguyên RAW; không có cải thiện nhất quán**.

Giữ robustness rule Step 6: ≥75% complete-cell coverage; ROBUST nếu ≥2 features có disjoint fraction≥75% và full-grid fault/operating SD ratio≥1; NOT ROBUST nếu ≥2 features có fraction≤25% và ratio<1; còn lại CONDITION-DEPENDENT. Thiếu full grid không dùng partial ratio thay verdict.

| Channel | Pipeline | Complete cells/80 | Full-grid evaluable features/3 | Verdict |
| --- | --- | --- | --- | --- |
| output_voltage | RAW | 24/80 | 0 | NOT EVALUABLE |
| output_voltage | VMD | 22/80 | 0 | NOT EVALUABLE |
| output_voltage | EMD | 20/80 | 0 | NOT EVALUABLE |
| input_voltage | RAW | 36/80 | 0 | NOT EVALUABLE |
| input_voltage | VMD | 33/80 | 0 | NOT EVALUABLE |
| input_voltage | EMD | 35/80 | 0 | NOT EVALUABLE |

Các fault-pair/feature cases khó tách nhất ở RAW (chỉ observed evaluable conditions):

| Channel | Pair | Feature | Disjoint/evaluable |
| --- | --- | --- | --- |
| output_voltage | S1-S2 | alpha0 | 0/1 |
| output_voltage | S2-S6 | alpha0 | 0/1 |
| output_voltage | S2-S8 | alpha0 | 0/1 |
| output_voltage | S1-S8 | alpha0 | 2/4 |
| output_voltage | S1-S6 | alpha0 | 2/3 |
| input_voltage | S1-S3 | delta_alpha | 0/1 |
| input_voltage | S2-S8 | delta_alpha | 0/1 |
| input_voltage | S3-S6 | delta_alpha | 0/1 |
| input_voltage | S3-S7 | alpha0 | 0/1 |
| input_voltage | S3-S7 | delta_alpha | 0/1 |

Kết luận chỉ cho fault configurations trong dataset và conditions valid. Không classifier, không tuning, không localization hoặc equivalent-fault claim giữa topology.

Reproduce từ repo root với Python env có NumPy/SciPy/Matplotlib: `python -B result/test/step7/scripts/pipeline.py`, `python -B result/test/step7/scripts/tests.py`, `python -B result/test/step7/scripts/finish.py`. Default output là result/test/step7; PHM_OUTPUT_DIR có thể chỉ định một thư mục mới. Source algorithms/configurations Step 6 được giữ nguyên.

![feature_heatmaps output_voltage](figures/feature_heatmaps_output_voltage.png)

![paired_raw_vmd output_voltage](figures/paired_raw_vmd_output_voltage.png)

![fault_pair_separation output_voltage](figures/fault_pair_separation_output_voltage.png)

![qc_coverage output_voltage](figures/qc_coverage_output_voltage.png)

![feature_vs_load output_voltage](figures/feature_vs_load_output_voltage.png)

![vmd_frequency_modes output_voltage](figures/vmd_frequency_modes_output_voltage.png)

![vmd_frequency_band_results output_voltage](figures/vmd_frequency_band_results_output_voltage.png)

![raw_scaling_qc_audit output_voltage](figures/raw_scaling_qc_audit_output_voltage.png)

![Speed raw output_voltage](figures/feature_vs_speed_raw_output_voltage.png)

![Speed vmd output_voltage](figures/feature_vs_speed_vmd_output_voltage.png)

![feature_heatmaps input_voltage](figures/feature_heatmaps_input_voltage.png)

![paired_raw_vmd input_voltage](figures/paired_raw_vmd_input_voltage.png)

![fault_pair_separation input_voltage](figures/fault_pair_separation_input_voltage.png)

![qc_coverage input_voltage](figures/qc_coverage_input_voltage.png)

![feature_vs_load input_voltage](figures/feature_vs_load_input_voltage.png)

![vmd_frequency_modes input_voltage](figures/vmd_frequency_modes_input_voltage.png)

![vmd_frequency_band_results input_voltage](figures/vmd_frequency_band_results_input_voltage.png)

![raw_scaling_qc_audit input_voltage](figures/raw_scaling_qc_audit_input_voltage.png)

![Speed raw input_voltage](figures/feature_vs_speed_raw_input_voltage.png)

![Speed vmd input_voltage](figures/feature_vs_speed_vmd_input_voltage.png)

![Feature distributions](figures/feature_distributions.png)

![Residual diagnostics](figures/factorial_residual_diagnostics.png)
