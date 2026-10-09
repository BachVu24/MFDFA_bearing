# STEP 10 — FINAL ABLATION / PLAN 1

**FINAL PIPELINE: Full RAW recording → locked MF-DFA → unchanged QC → RAW `[delta_alpha, alpha0]` → fixed Random Forest reference classifier.** Đây là pipeline nghiên cứu cuối của Plan1, chưa được xác nhận cho deployment hoặc full cross-domain generalization. Feature lock Step8 giữ nguyên; Random Forest được chọn làm reference từ MacroF1 Step9, không fit/tune thêm ở Step10.

## 1. Scope and evidence lineage: Step 4–9

Step10 chỉ tổng hợp evidence đã có và phép đối chiếu mô tả trên cùng observations; không signal processing mới, không classifier fit, không parameter/feature search. Step4 dùng **step4_corrected** làm nguồn định lượng cuối, vì bản Step4 đầu đã được audit/correct scaling/QC. Pilot H1/H2/H5/H6 chỉ50Hz/High; Step5 mở5speeds/2loads cho4states; Step6 mở6 Helical states; Step7 xử lý8 Spur states độc lập; Step8 quyết định features; Step9 đánh giá classifiers.

Không cộng recordings Step4/5 vào Step6 như samples mới: chúng là subsets trùng recordings. Step6 đã rerun đầy đủ do historical MF-DFA source hash mismatch; không coi snapshot cũ chứng minh core chưa đổi. Step6/7 numeric configs giữ nguyên và Step8 guard xác minh upstream. Inputs Step10 kiểm SHA256 theo upstream output manifests. Các text artifacts Step4/5 từ Windows đổi CRLF→LF khi checkout: historical digest được xác nhận đúng sau duy nhất việc khôi phục CRLF trong memory, không sửa files. Step6–9 kiểm exact bytes. Provenance giữ current hash/historical hash và verification method trong diagnostics/upstream_manifest.csv.

| stage | dataset | channel | representation | valid_signals | expected_signals | evaluable_cells | expected_cells | disjoint_feature_pair_conditions | evaluable_feature_pair_conditions | expected_feature_pair_conditions |
|---|---|---|---|---|---|---|---|---|---|---|
| Step4_corrected | Helical pilot H1/H2/H5/H6; 50Hz/High | input_voltage | EMD_SUM | 8 | 8 | 4 | 4 | 7 | 9 | 9 |
| Step4_corrected | Helical pilot H1/H2/H5/H6; 50Hz/High | output_voltage | EMD_SUM | 7 | 8 | 3 | 4 | 4 | 6 | 9 |
| Step4_corrected | Helical pilot H1/H2/H5/H6; 50Hz/High | input_voltage | RAW | 8 | 8 | 4 | 4 | 6 | 9 | 9 |
| Step4_corrected | Helical pilot H1/H2/H5/H6; 50Hz/High | output_voltage | RAW | 8 | 8 | 4 | 4 | 7 | 9 | 9 |
| Step4_corrected | Helical pilot H1/H2/H5/H6; 50Hz/High | input_voltage | VMD_SUM | 8 | 8 | 4 | 4 | 7 | 9 | 9 |
| Step4_corrected | Helical pilot H1/H2/H5/H6; 50Hz/High | output_voltage | VMD_SUM | 8 | 8 | 4 | 4 | 6 | 9 | 9 |
| Step5 | Helical subset | input_voltage | EMD_SUM | 74 | 80 | 37 | 40 | 135 | 153 | 180 |
| Step5 | Helical subset | output_voltage | EMD_SUM | 73 | 80 | 36 | 40 | 131 | 147 | 180 |
| Step5 | Helical subset | input_voltage | RAW | 76 | 80 | 38 | 40 | 130 | 162 | 180 |
| Step5 | Helical subset | output_voltage | RAW | 75 | 80 | 37 | 40 | 145 | 156 | 180 |
| Step5 | Helical subset | input_voltage | VMD_SUM | 76 | 80 | 38 | 40 | 129 | 162 | 180 |
| Step5 | Helical subset | output_voltage | VMD_SUM | 76 | 80 | 37 | 40 | 139 | 156 | 180 |
| Step6 | Helical | input_voltage | EMD_SUM | 110 | 120 | 54 | 60 | 324 | 369 | 450 |
| Step6 | Helical | output_voltage | EMD_SUM | 103 | 120 | 51 | 60 | 293 | 339 | 450 |
| Step6 | Helical | input_voltage | RAW | 115 | 120 | 57 | 60 | 337 | 405 | 450 |
| Step6 | Helical | output_voltage | RAW | 109 | 120 | 53 | 60 | 333 | 366 | 450 |
| Step6 | Helical | input_voltage | VMD_SUM | 115 | 120 | 57 | 60 | 331 | 405 | 450 |
| Step6 | Helical | output_voltage | VMD_SUM | 111 | 120 | 54 | 60 | 330 | 378 | 450 |
| Step7 | Spur | input_voltage | EMD_SUM | 75 | 160 | 35 | 80 | 150 | 177 | 840 |
| Step7 | Spur | output_voltage | EMD_SUM | 43 | 160 | 20 | 80 | 45 | 54 | 840 |
| Step7 | Spur | input_voltage | RAW | 76 | 160 | 36 | 80 | 138 | 165 | 840 |
| Step7 | Spur | output_voltage | RAW | 48 | 160 | 24 | 80 | 80 | 93 | 840 |
| Step7 | Spur | input_voltage | VMD_SUM | 69 | 160 | 33 | 80 | 115 | 147 | 840 |
| Step7 | Spur | output_voltage | VMD_SUM | 45 | 160 | 22 | 80 | 81 | 90 | 840 |

Các feature–pair rows ở bảng stage là tổng ba features, khác mẫu số pair-condition của từng feature/set. Pilot chỉ3 cặp chọn trước, Step5 có6pairs, Step6 có15pairs và Step7 có28pairs. Không so trực tiếp rates khác fault sets như một learning curve/statistical improvement; không coi disjoint rates là accuracy.

## 2. Locked configuration and sample unit

q=-5…5 bước0.5; DFA order2;40 scales nguyên bản, fit64–512 (17scales64–468). QC:≥8 scales, minR²≥0.9,≥80%q R²≥0.95,≥80%q local-slope CV≤0.5, min h(q)>0.05. Valid= scaling_valid AND decomposition_valid; shape warnings chỉ audit. Window scaling rất ngắn (~0.96–7.02ms,0.864decade): scaling pass chưa chứng minh asymptotic multifractality hay cơ chế vật lý.

VMD K7/alpha500/tau0/DCFalse/init1/tol1e-7/max_iter2000; practical EMD max6 modes/max200siftings/SD0.2/envelope0.05/extrema mismatch0.005, strict IMF check riêng. Không đổi các tham số này; loại phương pháp khỏi operational path không xóa hoặc tune comparator. Band representative=max energy trước QC, fixed bands0–1/1–4/4–10/10–20/20–33.334kHz, center ratio≤1.5; same mode index không là physical matching.

Mỗi recording toàn chiều dài là unit; hai channels phân tích riêng, cùng recording group khi CV. Output primary, input secondary; fault labels compound nguyên bản. Không gộp topology hoặc suy localization.

## 3. QC coverage and fault separability

| dataset | channel | representation | valid_signals | expected_signals | qc_coverage | evaluable_cells | expected_cells | cell_coverage |
|---|---|---|---|---|---|---|---|---|
| Helical | input_voltage | EMD_SUM | 110 | 120 | 0.9167 | 54 | 60 | 0.9000 |
| Helical | output_voltage | EMD_SUM | 103 | 120 | 0.8583 | 51 | 60 | 0.8500 |
| Helical | input_voltage | RAW | 115 | 120 | 0.9583 | 57 | 60 | 0.9500 |
| Helical | output_voltage | RAW | 109 | 120 | 0.9083 | 53 | 60 | 0.8833 |
| Helical | input_voltage | VMD_SUM | 115 | 120 | 0.9583 | 57 | 60 | 0.9500 |
| Helical | output_voltage | VMD_SUM | 111 | 120 | 0.9250 | 54 | 60 | 0.9000 |
| Spur | input_voltage | EMD_SUM | 75 | 160 | 0.4688 | 35 | 80 | 0.4375 |
| Spur | output_voltage | EMD_SUM | 43 | 160 | 0.2687 | 20 | 80 | 0.2500 |
| Spur | input_voltage | RAW | 76 | 160 | 0.4750 | 36 | 80 | 0.4500 |
| Spur | output_voltage | RAW | 48 | 160 | 0.3000 | 24 | 80 | 0.3000 |
| Spur | input_voltage | VMD_SUM | 69 | 160 | 0.4313 | 33 | 80 | 0.4125 |
| Spur | output_voltage | VMD_SUM | 45 | 160 | 0.2812 | 22 | 80 | 0.2750 |

Coverage shared bởi cả ba feature trong một representation. Spur thiếu S4 ở cả channels, S3 output; full-state effects/validation vẫn NOT EVALUABLE. Missing spectra không phải zero feature hoặc overlap. Main feature evidence:

| dataset | channel | representation | feature | disjoint_pair_conditions | evaluable_pair_conditions | expected_pair_conditions | disjoint_fraction | direction_consistency | direction_evaluable |
|---|---|---|---|---|---|---|---|---|---|
| Helical | input_voltage | EMD_SUM | delta_alpha | 104 | 123 | 150 | 0.8455 | 0.7073 | 123 |
| Helical | input_voltage | EMD_SUM | alpha0 | 110 | 123 | 150 | 0.8943 | 0.7561 | 123 |
| Helical | input_voltage | EMD_SUM | delta_h | 110 | 123 | 150 | 0.8943 | 0.7073 | 123 |
| Helical | output_voltage | EMD_SUM | delta_alpha | 98 | 113 | 150 | 0.8673 | 0.3846 | 52 |
| Helical | output_voltage | EMD_SUM | alpha0 | 97 | 113 | 150 | 0.8584 | 0.5385 | 52 |
| Helical | output_voltage | EMD_SUM | delta_h | 98 | 113 | 150 | 0.8673 | 0.4231 | 52 |
| Helical | input_voltage | RAW | delta_alpha | 113 | 135 | 150 | 0.8370 | 0.6667 | 135 |
| Helical | input_voltage | RAW | alpha0 | 111 | 135 | 150 | 0.8222 | 0.7259 | 135 |
| Helical | input_voltage | RAW | delta_h | 113 | 135 | 150 | 0.8370 | 0.6815 | 135 |
| Helical | output_voltage | RAW | delta_alpha | 111 | 122 | 150 | 0.9098 | 0.6782 | 87 |
| Helical | output_voltage | RAW | alpha0 | 115 | 122 | 150 | 0.9426 | 0.7241 | 87 |
| Helical | output_voltage | RAW | delta_h | 107 | 122 | 150 | 0.8770 | 0.6667 | 87 |
| Helical | input_voltage | VMD_SUM | delta_alpha | 109 | 135 | 150 | 0.8074 | 0.6815 | 135 |
| Helical | input_voltage | VMD_SUM | alpha0 | 111 | 135 | 150 | 0.8222 | 0.6741 | 135 |
| Helical | input_voltage | VMD_SUM | delta_h | 111 | 135 | 150 | 0.8222 | 0.6963 | 135 |
| Helical | output_voltage | VMD_SUM | delta_alpha | 108 | 126 | 150 | 0.8571 | 0.6782 | 87 |
| Helical | output_voltage | VMD_SUM | alpha0 | 117 | 126 | 150 | 0.9286 | 0.6897 | 87 |
| Helical | output_voltage | VMD_SUM | delta_h | 105 | 126 | 150 | 0.8333 | 0.6552 | 87 |
| Spur | input_voltage | EMD_SUM | delta_alpha | 49 | 59 | 280 | 0.8305 | 0.8500 | 20 |
| Spur | input_voltage | EMD_SUM | alpha0 | 55 | 59 | 280 | 0.9322 | 0.8500 | 20 |
| Spur | input_voltage | EMD_SUM | delta_h | 46 | 59 | 280 | 0.7797 | 0.8000 | 20 |
| Spur | output_voltage | EMD_SUM | delta_alpha | 14 | 18 | 280 | 0.7778 | 0.9091 | 11 |
| Spur | output_voltage | EMD_SUM | alpha0 | 14 | 18 | 280 | 0.7778 | 0.9091 | 11 |
| Spur | output_voltage | EMD_SUM | delta_h | 17 | 18 | 280 | 0.9444 | 0.9091 | 11 |
| Spur | input_voltage | RAW | delta_alpha | 41 | 55 | 280 | 0.7455 | 0.8140 | 43 |
| Spur | input_voltage | RAW | alpha0 | 52 | 55 | 280 | 0.9455 | 0.8372 | 43 |
| Spur | input_voltage | RAW | delta_h | 45 | 55 | 280 | 0.8182 | 0.8140 | 43 |
| Spur | output_voltage | RAW | delta_alpha | 27 | 31 | 280 | 0.8710 | 0.7727 | 22 |
| Spur | output_voltage | RAW | alpha0 | 23 | 31 | 280 | 0.7419 | 0.7273 | 22 |
| Spur | output_voltage | RAW | delta_h | 30 | 31 | 280 | 0.9677 | 0.7727 | 22 |
| Spur | input_voltage | VMD_SUM | delta_alpha | 33 | 49 | 280 | 0.6735 | 0.7838 | 37 |
| Spur | input_voltage | VMD_SUM | alpha0 | 45 | 49 | 280 | 0.9184 | 0.8378 | 37 |
| Spur | input_voltage | VMD_SUM | delta_h | 37 | 49 | 280 | 0.7551 | 0.8108 | 37 |
| Spur | output_voltage | VMD_SUM | delta_alpha | 28 | 30 | 280 | 0.9333 | 0.7619 | 21 |
| Spur | output_voltage | VMD_SUM | alpha0 | 24 | 30 | 280 | 0.8000 | 0.7619 | 21 |
| Spur | output_voltage | VMD_SUM | delta_h | 29 | 30 | 280 | 0.9667 | 0.7619 | 21 |

Strict disjoint repeat ranges chỉ kiểm hai acquisitions mỗi fault tại cùng speed/load; reference direction50Hz/High, baseline invalid⇒NE. High within-condition separation không bảo đảm pooled/global class boundary: distributions thay đổi theo operating conditions, nên recording-CV scores thấp hơn nhiều disjoint fractions.

## 4. Feature ablation and delta_h decision

Tái dùng các feature sets **đã có ở Step8**: delta_alpha;alpha0;delta_h;delta_alpha+alpha0;ba features. Không thêm classification ablation mới. Union tiêu chí nghĩa là có ít nhất một marginal feature disjoint, không phải joint-space classification. Không ép metrics thành composite score.

| dataset | channel | feature_set | disjoint_on_at_least_one_feature | evaluable_pair_conditions | expected_pair_conditions |
|---|---|---|---|---|---|
| Helical | input_voltage | delta_alpha | 113 | 135 | 150 |
| Helical | input_voltage | alpha0 | 111 | 135 | 150 |
| Helical | input_voltage | delta_h | 113 | 135 | 150 |
| Helical | input_voltage | delta_alpha+alpha0 | 131 | 135 | 150 |
| Helical | input_voltage | delta_alpha+alpha0+delta_h | 132 | 135 | 150 |
| Helical | output_voltage | delta_alpha | 111 | 122 | 150 |
| Helical | output_voltage | alpha0 | 115 | 122 | 150 |
| Helical | output_voltage | delta_h | 107 | 122 | 150 |
| Helical | output_voltage | delta_alpha+alpha0 | 121 | 122 | 150 |
| Helical | output_voltage | delta_alpha+alpha0+delta_h | 121 | 122 | 150 |
| Spur | input_voltage | delta_alpha | 41 | 55 | 280 |
| Spur | input_voltage | alpha0 | 52 | 55 | 280 |
| Spur | input_voltage | delta_h | 45 | 55 | 280 |
| Spur | input_voltage | delta_alpha+alpha0 | 54 | 55 | 280 |
| Spur | input_voltage | delta_alpha+alpha0+delta_h | 55 | 55 | 280 |
| Spur | output_voltage | delta_alpha | 27 | 31 | 280 |
| Spur | output_voltage | alpha0 | 23 | 31 | 280 |
| Spur | output_voltage | delta_h | 30 | 31 | 280 |
| Spur | output_voltage | delta_alpha+alpha0 | 30 | 31 | 280 |
| Spur | output_voltage | delta_alpha+alpha0+delta_h | 31 | 31 | 280 |

Redundancy trên RAW valid recordings:

| dataset | channel | feature_a | feature_b | n | pearson_r | spearman_r |
|---|---|---|---|---|---|---|
| Helical | input_voltage | delta_alpha | alpha0 | 115 | 0.0752 | -0.0121 |
| Helical | input_voltage | delta_alpha | delta_h | 115 | 0.9749 | 0.9753 |
| Helical | input_voltage | alpha0 | delta_h | 115 | 0.1350 | 0.0714 |
| Helical | output_voltage | delta_alpha | alpha0 | 109 | 0.3178 | 0.2963 |
| Helical | output_voltage | delta_alpha | delta_h | 109 | 0.9885 | 0.9873 |
| Helical | output_voltage | alpha0 | delta_h | 109 | 0.3529 | 0.3209 |
| Spur | input_voltage | delta_alpha | alpha0 | 76 | 0.3475 | 0.4574 |
| Spur | input_voltage | delta_alpha | delta_h | 76 | 0.9553 | 0.9619 |
| Spur | input_voltage | alpha0 | delta_h | 76 | 0.2755 | 0.4747 |
| Spur | output_voltage | delta_alpha | alpha0 | 48 | 0.0505 | 0.0030 |
| Spur | output_voltage | delta_alpha | delta_h | 48 | 0.9636 | 0.9872 |
| Spur | output_voltage | alpha0 | delta_h | 48 | 0.0384 | -0.0100 |

**Loại delta_h ở Step8 là hợp lý theo descriptive evidence hiện có**, không phải chứng minh nó vô ích trong mọi bài toán. Delta_alpha–delta_h Pearson0.955–0.989; sau bộ hai, thêm delta_h chỉ tăng Helical output0/122, input1/135; Spur output1/31, input1/55. Ba cases bổ sung ở ba pairs khác nhau, mỗi case chỉ một condition; không improvement lặp lại. Correlation riêng không đủ để loại, vì vậy quyết định dựa thêm incremental separation. Không có delta_h classifier ablation trong Step9; không tuyên bố đã chứng minh accuracy của hai feature cao hơn ba feature.

Alpha0 ít redundant và có extra within-condition separation lặp nhiều speeds/loads: primary Helical H3–H5 thêm4conditions/3speeds/2loads cùng direction; secondary Spur S1–S6 thêm4conditions/3speeds/2loads cùng direction. Các pairs khác có direction reversals, nên giữ alpha0 như complementary condition-sensitive feature, không invariant fault marker. Giữ nguyên order `[delta_alpha, alpha0]`, không đổi bằng một accuracy gain nhỏ hoặc speculative classifier benefit.

## 5. Speed/load sensitivity and effects

Speed metric=(max−min)/abs(mean) cần5 complete cells; load metric=2abs(High−Low)/(abs(High)+abs(Low)). Denominators của trajectory/load comparisons phải đọc cùng median, không so unmatched populations như superiority.

| dataset | channel | feature | median_speed_normalized_range | speed_trajectories_evaluable | expected_speed_trajectories | median_load_relative_difference | load_comparisons_evaluable | expected_load_comparisons |
|---|---|---|---|---|---|---|---|---|
| Helical | input_voltage | delta_alpha | 0.6402 | 10 | 12 | 0.3535 | 27 | 30 |
| Helical | input_voltage | alpha0 | 0.2141 | 10 | 12 | 0.1399 | 27 | 30 |
| Helical | input_voltage | delta_h | 0.6116 | 10 | 12 | 0.4175 | 27 | 30 |
| Helical | output_voltage | delta_alpha | 0.5655 | 6 | 12 | 0.5440 | 23 | 30 |
| Helical | output_voltage | alpha0 | 0.2244 | 6 | 12 | 0.2300 | 23 | 30 |
| Helical | output_voltage | delta_h | 0.6174 | 6 | 12 | 0.5680 | 23 | 30 |
| Spur | input_voltage | delta_alpha | 0.3244 | 3 | 16 | 0.5977 | 13 | 40 |
| Spur | input_voltage | alpha0 | 0.3544 | 3 | 16 | 0.5741 | 13 | 40 |
| Spur | input_voltage | delta_h | 0.4190 | 3 | 16 | 0.4626 | 13 | 40 |
| Spur | output_voltage | delta_alpha | 0.7336 | 1 | 16 | 0.2884 | 6 | 40 |
| Spur | output_voltage | alpha0 | 0.4270 | 1 | 16 | 0.0875 | 6 | 40 |
| Spur | output_voltage | delta_h | 0.9739 | 1 | 16 | 0.3281 | 6 | 40 |

Giữ nguyên exploratory categorical additive model, partialη², HC3/wild-bootstrap1999/BH procedures Step6/7; không refit fault set nhỏ hơn để cứu Spur rank. Partialη² không phải causal contribution hoặc fractions cộng100%:

| dataset | channel | feature | fault_partial_eta_squared | speed_partial_eta_squared | load_partial_eta_squared | fault_wild_bootstrap_p_BH | speed_wild_bootstrap_p_BH | load_wild_bootstrap_p_BH |
|---|---|---|---|---|---|---|---|---|
| Helical | input_voltage | delta_alpha | 0.2888 | 0.0459 | 0.0888 | 0.0008 | 0.1183 | 0.0071 |
| Helical | input_voltage | alpha0 | 0.2424 | 0.5575 | 0.3245 | 0.0008 | 0.0008 | 0.0008 |
| Helical | input_voltage | delta_h | 0.2857 | 0.0436 | 0.0453 | 0.0008 | 0.1660 | 0.0487 |
| Helical | output_voltage | delta_alpha | 0.5498 | 0.0988 | 0.0165 | 0.0010 | 0.0607 | 0.2322 |
| Helical | output_voltage | alpha0 | 0.3204 | 0.5097 | 0.6287 | 0.0018 | 0.0010 | 0.0010 |
| Helical | output_voltage | delta_h | 0.5650 | 0.1045 | 0.0366 | 0.0010 | 0.0445 | 0.0844 |
| Spur | input_voltage | delta_alpha | NE | NE | NE | NE | NE | NE |
| Spur | input_voltage | alpha0 | NE | NE | NE | NE | NE | NE |
| Spur | input_voltage | delta_h | NE | NE | NE | NE | NE | NE |
| Spur | output_voltage | delta_alpha | NE | NE | NE | NE | NE | NE |
| Spur | output_voltage | alpha0 | NE | NE | NE | NE | NE | NE |
| Spur | output_voltage | delta_h | NE | NE | NE | NE | NE | NE |

Helical delta_alpha/delta_h có fault association mạnh hơn từng speed/load main effect, nhưng sensitivity vẫn lớn; alpha0 main speed/load effects mạnh hơn fault. Full-grid fault/operating ratio vẫn NE do incomplete cells. Không đồng nhất feature normalized range với partialη²: chúng đo khác things/scales. Helical overall verdict CONDITION-DEPENDENT; Spur full-state effects và robustness NOT EVALUABLE vì coverage/rank. Không coi NE là effect0 hay feature vô dụng.

## 6. VMD_SUM: improvement versus preservation

Paired correlations/feature changes:

| dataset | channel | feature | paired_recordings | pearson_r | median_relative_difference |
|---|---|---|---|---|---|
| Helical | output_voltage | delta_alpha | 109 | 0.9891 | 0.0349 |
| Helical | output_voltage | alpha0 | 109 | 0.9903 | 0.0197 |
| Helical | output_voltage | delta_h | 109 | 0.9922 | 0.0328 |
| Helical | input_voltage | delta_alpha | 115 | 0.9980 | 0.0181 |
| Helical | input_voltage | alpha0 | 115 | 0.9992 | 0.0153 |
| Helical | input_voltage | delta_h | 115 | 0.9989 | 0.0150 |
| Spur | output_voltage | delta_alpha | 45 | 0.9990 | 0.0181 |
| Spur | output_voltage | alpha0 | 45 | 0.9979 | 0.0166 |
| Spur | output_voltage | delta_h | 45 | 0.9992 | 0.0161 |
| Spur | input_voltage | delta_alpha | 69 | 0.9895 | 0.0278 |
| Spur | input_voltage | alpha0 | 69 | 0.9900 | 0.0291 |
| Spur | input_voltage | delta_h | 69 | 0.9927 | 0.0208 |

Matched separation/sensitivity — giữ cùng common observations:

| dataset | channel | feature | common_pairs | raw_disjoint_count | vmd_disjoint_count | gained_count | lost_count | matched_speed_trajectories | median_speed_ratio | matched_load_comparisons | median_load_ratio |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Helical | output_voltage | delta_alpha | 122 | 111 | 104 | 0 | 7 | 6 | 0.9925 | 23 | 0.9687 |
| Helical | output_voltage | alpha0 | 122 | 115 | 113 | 1 | 3 | 6 | 1.0126 | 23 | 1.0389 |
| Helical | output_voltage | delta_h | 122 | 107 | 101 | 2 | 8 | 6 | 0.9942 | 23 | 0.9729 |
| Helical | input_voltage | delta_alpha | 135 | 113 | 109 | 0 | 4 | 10 | 0.9678 | 27 | 0.9687 |
| Helical | input_voltage | alpha0 | 135 | 111 | 111 | 5 | 5 | 10 | 1.0109 | 27 | 1.0115 |
| Helical | input_voltage | delta_h | 135 | 113 | 111 | 0 | 2 | 10 | 0.9830 | 27 | 0.9845 |
| Spur | output_voltage | delta_alpha | 30 | 26 | 28 | 2 | 0 | 1 | 1.0999 | 4 | 1.0596 |
| Spur | output_voltage | alpha0 | 30 | 22 | 24 | 2 | 0 | 1 | 1.0443 | 4 | 1.0326 |
| Spur | output_voltage | delta_h | 30 | 29 | 29 | 0 | 0 | 1 | 1.0728 | 4 | 1.0591 |
| Spur | input_voltage | delta_alpha | 49 | 37 | 33 | 0 | 4 | 2 | 1.0437 | 10 | 1.0544 |
| Spur | input_voltage | alpha0 | 49 | 46 | 45 | 1 | 2 | 2 | 1.0731 | 10 | 1.0905 |
| Spur | input_voltage | delta_h | 49 | 40 | 37 | 1 | 4 | 2 | 0.9585 | 10 | 1.0531 |

**VMD_SUM gần như giữ nguyên RAW; chưa có improvement nhất quán.** Pearson~0.989–0.999, median relative changes~1.5–3.5%; correlation cao không là bằng chứng tốt hơn. Primary Helical delta_alpha mất7/gain0, alpha0 mất3/gain1; Spur output gain2 cho mỗi feature trong common30conditions nhưng input net losses và coverage giảm. Sensitivity ratios gần1, đôi khi tốt/xấu, Spur ít trajectories nên chưa evidence robustness gain. Step4/5 đã cùng xu hướng preserve; full validation không đảo kết luận. Không có VMD classifier result ở Step9, không tuyên bố VMD accuracy thấp hơn RAW.

Tau0 không buộc sum(modes)=RAW, nên VMD_SUM thực sự là representation khác; reconstruction/identity không tự chứng minh useful fault information. Loại decomposition khỏi final path theo cost-benefit hiện có, giữ audit artifacts.

## 7. Individual VMD frequency bands

| dataset | channel | representation | qc_coverage | evaluable_cells | expected_cells | cell_coverage |
|---|---|---|---|---|---|---|
| Helical | input_voltage | VMD_band_0_1000_Hz | 0.6667 | 36 | 60 | 0.6000 |
| Helical | input_voltage | VMD_band_10000_20000_Hz | 0.0000 | 0 | 60 | 0.0000 |
| Helical | input_voltage | VMD_band_1000_4000_Hz | 0.0000 | 0 | 60 | 0.0000 |
| Helical | input_voltage | VMD_band_20000_33334_Hz | 0.0000 | 0 | 60 | 0.0000 |
| Helical | input_voltage | VMD_band_4000_10000_Hz | 0.0000 | 0 | 60 | 0.0000 |
| Helical | output_voltage | VMD_band_0_1000_Hz | 0.5917 | 34 | 60 | 0.5667 |
| Helical | output_voltage | VMD_band_10000_20000_Hz | 0.0000 | 0 | 60 | 0.0000 |
| Helical | output_voltage | VMD_band_1000_4000_Hz | 0.1083 | 5 | 60 | 0.0833 |
| Helical | output_voltage | VMD_band_20000_33334_Hz | 0.0000 | 0 | 60 | 0.0000 |
| Helical | output_voltage | VMD_band_4000_10000_Hz | 0.0000 | 0 | 60 | 0.0000 |
| Spur | input_voltage | VMD_band_0_1000_Hz | 0.2125 | 16 | 80 | 0.2000 |
| Spur | input_voltage | VMD_band_10000_20000_Hz | 0.0000 | 0 | 80 | 0.0000 |
| Spur | input_voltage | VMD_band_1000_4000_Hz | 0.0187 | 1 | 80 | 0.0125 |
| Spur | input_voltage | VMD_band_20000_33334_Hz | 0.0000 | 0 | 80 | 0.0000 |
| Spur | input_voltage | VMD_band_4000_10000_Hz | 0.0000 | 0 | 80 | 0.0000 |
| Spur | output_voltage | VMD_band_0_1000_Hz | 0.0938 | 7 | 80 | 0.0875 |
| Spur | output_voltage | VMD_band_10000_20000_Hz | 0.0000 | 0 | 80 | 0.0000 |
| Spur | output_voltage | VMD_band_1000_4000_Hz | 0.0000 | 0 | 80 | 0.0000 |
| Spur | output_voltage | VMD_band_20000_33334_Hz | 0.0000 | 0 | 80 | 0.0000 |
| Spur | output_voltage | VMD_band_4000_10000_Hz | 0.0000 | 0 | 80 | 0.0000 |

Tất cả band coverage dưới gate75% kế thừa; không chọn index/class vì score tốt. Bằng chứng extra separation vs RAW:

| dataset | channel | category | pair | feature | extra_conditions | speeds | loads | frequency_compatible_across_extra_conditions | repeated_compatible_evidence |
|---|---|---|---|---|---|---|---|---|---|
| Helical | input_voltage | band_0_1000_Hz | H1-H2 | alpha0 | 3 | 2 | 2 | True | True |
| Helical | input_voltage | band_0_1000_Hz | H1-H3 | alpha0 | 2 | 2 | 1 | True | True |
| Helical | input_voltage | band_0_1000_Hz | H1-H5 | delta_alpha | 1 | 1 | 1 | True | False |
| Helical | input_voltage | band_0_1000_Hz | H1-H5 | alpha0 | 1 | 1 | 1 | True | False |
| Helical | input_voltage | band_0_1000_Hz | H1-H5 | delta_h | 1 | 1 | 1 | True | False |
| Helical | input_voltage | band_0_1000_Hz | H2-H3 | alpha0 | 1 | 1 | 1 | True | False |
| Helical | input_voltage | band_0_1000_Hz | H4-H5 | delta_alpha | 1 | 1 | 1 | True | False |
| Helical | input_voltage | band_0_1000_Hz | H4-H5 | delta_h | 1 | 1 | 1 | True | False |
| Helical | input_voltage | band_0_1000_Hz | H4-H6 | delta_h | 1 | 1 | 1 | True | False |
| Helical | input_voltage | band_0_1000_Hz | H5-H6 | delta_h | 1 | 1 | 1 | True | False |
| Helical | input_voltage | band_0_1000_Hz | H1-H6 | alpha0 | 1 | 1 | 1 | True | False |
| Helical | input_voltage | band_0_1000_Hz | H2-H5 | alpha0 | 1 | 1 | 1 | True | False |
| Helical | input_voltage | band_0_1000_Hz | H2-H6 | delta_alpha | 1 | 1 | 1 | True | False |
| Helical | input_voltage | band_0_1000_Hz | H2-H6 | delta_h | 1 | 1 | 1 | True | False |
| Helical | input_voltage | band_0_1000_Hz | H3-H5 | delta_alpha | 1 | 1 | 1 | True | False |
| Helical | input_voltage | band_0_1000_Hz | H3-H5 | delta_h | 1 | 1 | 1 | True | False |
| Helical | input_voltage | band_0_1000_Hz | H1-H3 | delta_h | 1 | 1 | 1 | True | False |
| Helical | input_voltage | band_0_1000_Hz | H1-H4 | delta_alpha | 1 | 1 | 1 | True | False |
| Helical | input_voltage | band_0_1000_Hz | H1-H4 | alpha0 | 1 | 1 | 1 | True | False |
| Helical | output_voltage | band_0_1000_Hz | H1-H5 | alpha0 | 1 | 1 | 1 | True | False |
| Helical | output_voltage | band_0_1000_Hz | H4-H6 | delta_alpha | 1 | 1 | 1 | True | False |
| Helical | output_voltage | band_0_1000_Hz | H4-H6 | delta_h | 1 | 1 | 1 | True | False |
| Helical | output_voltage | band_0_1000_Hz | H2-H6 | alpha0 | 1 | 1 | 1 | True | False |
| Helical | output_voltage | band_0_1000_Hz | H3-H5 | delta_alpha | 1 | 1 | 1 | True | False |
| Helical | output_voltage | band_0_1000_Hz | H1-H6 | alpha0 | 1 | 1 | 1 | True | False |
| Helical | output_voltage | band_0_1000_Hz | H2-H4 | delta_h | 1 | 1 | 1 | True | False |
| Helical | output_voltage | band_1000_4000_Hz | H3-H5 | delta_alpha | 1 | 1 | 1 | True | False |
| Helical | output_voltage | band_1000_4000_Hz | H3-H5 | delta_h | 1 | 1 | 1 | True | False |
| Spur | input_voltage | band_0_1000_Hz | S1-S6 | delta_alpha | 1 | 1 | 1 | True | False |
| Spur | input_voltage | band_0_1000_Hz | S1-S6 | delta_h | 1 | 1 | 1 | True | False |
| Spur | output_voltage | band_0_1000_Hz | S6-S8 | alpha0 | 1 | 1 | 1 | True | False |

**Không có lợi ích lặp lại ổn định đủ để giữ modes.** Helical low-frequency input có H1–H2 alpha0 thêm3conditions và H1–H3 thêm2conditions compatible; primary output extras đơn lẻ. Spur không có repeated-compatible extra groups. Đây là tín hiệu exploratory của một subset, không cross-topology mode advantage, không physical component localization. Không chỉnh bands/QC hay chọn representative khác để nâng coverage.

## 8. EMD comparator: keep or remove?

Không so rates trên mẫu số khác rồi kết luận EMD tốt hơn. Matched primary/secondary evidence:

| dataset | channel | feature | common_pair_conditions | raw_disjoint | emd_disjoint | gained | lost |
|---|---|---|---|---|---|---|---|
| Helical | output_voltage | delta_alpha | 113 | 103 | 98 | 3 | 8 |
| Helical | output_voltage | alpha0 | 113 | 106 | 97 | 3 | 12 |
| Helical | output_voltage | delta_h | 113 | 99 | 98 | 4 | 5 |
| Helical | input_voltage | delta_alpha | 123 | 103 | 104 | 10 | 9 |
| Helical | input_voltage | alpha0 | 123 | 100 | 110 | 15 | 5 |
| Helical | input_voltage | delta_h | 123 | 104 | 110 | 15 | 9 |
| Spur | output_voltage | delta_alpha | 18 | 16 | 14 | 0 | 2 |
| Spur | output_voltage | alpha0 | 18 | 12 | 14 | 4 | 2 |
| Spur | output_voltage | delta_h | 18 | 17 | 17 | 0 | 0 |
| Spur | input_voltage | delta_alpha | 48 | 34 | 39 | 6 | 1 |
| Spur | input_voltage | alpha0 | 48 | 46 | 45 | 1 | 2 |
| Spur | input_voltage | delta_h | 48 | 38 | 38 | 5 | 5 |

**EMD không đáng giữ trong operational pipeline của Plan1 theo evidence này; giữ comparator/audit đã có.** Không gọi EMD luôn kém: Helical input alpha0 gain15/loss5, delta_h gain15/loss9; Spur input delta_alpha gain6/loss1 trên48conditions. Tuy nhiên primary Helical output net losses cả ba features, coverage thường thấp hơn, gains không consistent giữa feature/channel/topology; chưa có EMD classifier test hoặc confirmed operating robustness gain. Các EMD modes là practical approximation, strict IMF audit riêng, không biến thành benchmark IMF nghiêm ngặt. Added decomposition cost chưa được bù bằng improvement đủ ổn định. Không sửa EMD/QC để tăng valid samples.

## 9. Computational complexity and practical cost

| dataset | channel | pipeline | count | median_seconds | p90_seconds | max_seconds |
|---|---|---|---|---|---|---|
| Helical | input_voltage | emd | 120 | 5.4024 | 7.0735 | 8.8954 |
| Helical | input_voltage | vmd | 120 | 4.5760 | 10.4643 | 24.5085 |
| Helical | output_voltage | emd | 120 | 4.3922 | 6.7428 | 8.7420 |
| Helical | output_voltage | vmd | 120 | 4.4054 | 8.8049 | 18.7578 |
| Spur | input_voltage | emd | 160 | 4.1268 | 7.0075 | 11.3142 |
| Spur | input_voltage | vmd | 160 | 7.5436 | 16.8264 | 49.9077 |
| Spur | output_voltage | emd | 160 | 5.1946 | 7.8099 | 11.8657 |
| Spur | output_voltage | vmd | 160 | 6.3287 | 15.3927 | 31.3200 |

Các số là wall-clock timers đã ghi từ Step6/7, **decomposition + mode diagnostics**, kết thúc trước MF-DFA component extraction; không phải pure solver-only timing, toàn pipeline runtime hay controlled benchmark. Jobs từng chạy concurrent, hardware/conditions có thể ảnh hưởng; không dùng medians để khẳng định thuật toán nhanh tuyệt đối. RAW extraction/classifier chưa có separate timers, không tạo speedup ratio giả. Removing decomposition loại added work nhưng không lượng hóa total speedup.

| representation | decomposition | mfdfa_calls_operational | mfdfa_calls_upstream_audit | time_structure | memory_structure |
|---|---|---|---|---|---|
| RAW | None | 1 | 1 | O(S*N*(m+1) + Q*sum(N/s)); fixed m,Q,S | O(N + N*(m+1) + Q*S) |
| VMD_SUM | VMD K=7 | 1 | 9 | O(I*K*N + K*N*log(N)) spectral recurrence/FFT + one MF-DFA | O(K*N) plus MF-DFA |
| Individual VMD modes | VMD K=7 | 7 | 9 | Same VMD + up to 7 MF-DFA calls and fixed band matching | O(K*N) plus per-component extraction |
| EMD_SUM | Practical EMD max6 modes | 1 | 8 | Approximately O(N*sum(siftings)) for extrema/spline passes + MF-DFA | O(J*N) modes plus envelopes |

N=signal length, S=40scales, Q=21q, m=2; VMDI≤2000,K7; EMDJ≤6,≤200siftings/mode. VMD spectral recurrence O(IKN), FFT chủ yếu trước/sau O(KNlogN), không giả FFT mỗi ADMM iteration. EMD spline/extrema passes xấp xỉ linear N mỗi sifting với data-dependent constants. Audit từng chạy all modes+sum+residue (9 VMD/8 EMD MF-DFA calls), **không lấy số này làm cost của sum-only pipeline**, vốn chỉ cần1 MF-DFA sau decomposition.

Delta_h được tính cùng MF-DFA spectrum; bỏ khỏi input không bỏ một MF-DFA pass hoặc tiết kiệm thời gian extraction đáng kể. Chi phí classifier giảm dimensionality3→2 nhưng chưa benchmark. Cost advantage chính đến từ bỏ VMD/EMD và mode extraction, không từ giả định feature width đắt hơn alpha0.

## 10. Classification evidence from Step 9

Step9 chỉ classify RAW delta_alpha+alpha0, với cùng recording folds cho4classifiers, scaler training-only khi cần. Không có scores single-feature/triple-feature/VMD/EMD; `ablation_summary.csv` ghi NOT RUN/NaN, không giả0 hoặc thêm inference từ correlation. Không train ablation mới để đổi feature set.

| dataset | channel | classifier | n_samples | n_classes | accuracy | macro_f1 | macro_precision | macro_recall |
|---|---|---|---|---|---|---|---|---|
| Helical | output_voltage | KNN | 109 | 6 | 0.3945 | 0.3790 | 0.3813 | 0.3869 |
| Helical | output_voltage | SVM | 109 | 6 | 0.4495 | 0.4371 | 0.4657 | 0.4410 |
| Helical | output_voltage | LDA | 109 | 6 | 0.2661 | 0.2528 | 0.2546 | 0.2705 |
| Helical | output_voltage | Random Forest | 109 | 6 | 0.5321 | 0.5283 | 0.5377 | 0.5251 |
| Helical | input_voltage | KNN | 115 | 6 | 0.3478 | 0.3421 | 0.3490 | 0.3452 |
| Helical | input_voltage | SVM | 115 | 6 | 0.3826 | 0.3743 | 0.3848 | 0.3751 |
| Helical | input_voltage | LDA | 115 | 6 | 0.3304 | 0.3206 | 0.3236 | 0.3226 |
| Helical | input_voltage | Random Forest | 115 | 6 | 0.4261 | 0.4190 | 0.4194 | 0.4214 |
| Spur | output_voltage | KNN | 48 | 6 | 0.3958 | 0.2238 | 0.2278 | 0.2381 |
| Spur | output_voltage | SVM | 48 | 6 | 0.5833 | 0.2441 | 0.2037 | 0.3110 |
| Spur | output_voltage | LDA | 48 | 6 | 0.4167 | 0.2128 | 0.2378 | 0.2351 |
| Spur | output_voltage | Random Forest | 48 | 6 | 0.5417 | 0.5008 | 0.4889 | 0.5164 |
| Spur | input_voltage | KNN | 76 | 7 | 0.4342 | 0.2448 | 0.2454 | 0.2713 |
| Spur | input_voltage | SVM | 76 | 7 | 0.5658 | 0.4351 | 0.4738 | 0.4342 |
| Spur | input_voltage | LDA | 76 | 7 | 0.6053 | 0.4587 | 0.4893 | 0.4575 |
| Spur | input_voltage | Random Forest | 76 | 7 | 0.6316 | 0.5955 | 0.6155 | 0.6065 |

Random Forest highest MacroF1 ở cả4contexts; SVM accuracy highest Spur output nhưng macroF1 thấp. Reference best không là universal/deployment validated. Condition-transfer của cùng fixed RF:

| dataset | channel | protocol | evaluable_splits | expected_splits | tested_recordings | eligible_recordings | test_coverage | untested_classes | accuracy | macro_f1 |
|---|---|---|---|---|---|---|---|---|---|---|
| Helical | input_voltage | leave_one_load_out | 2 | 2 | 115 | 115 | 1.0000 | NE | 0.1652 | 0.1456 |
| Helical | input_voltage | leave_one_speed_out | 5 | 5 | 115 | 115 | 1.0000 | NE | 0.3478 | 0.3153 |
| Helical | output_voltage | leave_one_load_out | 2 | 2 | 109 | 109 | 1.0000 | NE | 0.0917 | 0.0923 |
| Helical | output_voltage | leave_one_speed_out | 5 | 5 | 109 | 109 | 1.0000 | NE | 0.4128 | 0.4071 |
| Spur | input_voltage | leave_one_load_out | 1 | 2 | 32 | 76 | 0.4211 | S2;S3;S5 | 0.2812 | 0.2000 |
| Spur | input_voltage | leave_one_speed_out | 4 | 5 | 56 | 76 | 0.7368 | S2 | 0.6964 | 0.6341 |
| Spur | output_voltage | leave_one_load_out | 1 | 2 | 12 | 48 | 0.2500 | S1;S2;S5;S6 | 0.8333 | 0.8730 |
| Spur | output_voltage | leave_one_speed_out | 4 | 5 | 34 | 48 | 0.7083 | S2;S5 | 0.3235 | 0.2542 |

Helical output recording-CV MacroF1~0.528→speed transfer0.407→load transfer0.092; input0.419→0.315→0.146. Full load-transfer coverage100%: đây là hạn chế thực tế, không do denominator thiếu của Helical. Spur split/test coverage và class support thiếu; chỉ eligible classes được đánh giá, untestedclasses NE. Step8 selection đã dùng cùng labels, nên Step9 exploratory conditional CV có selection/model-comparison optimism. Cần independent recordings cho final generalization claim.

## 11. Helical versus Spur consistency and generalization

Hai widths redundancy và VMD preservation xuất hiện ở cảdatasets/channels, hỗ trợ consistency của quyết định bỏ redundant/decomposition input. Alpha0 nonredundancy và extra separation tồn tại nhưng strong association conclusions chỉ được xác nhận trên Helical; Spur full-state rank/QC thất bại. Individual band benefit Helical không lặp stable trên Spur.

**Phương pháp chưa generalize tốt qua toàn operating conditions hoặc chứng minh cross-topology transfer.** Helical load-transfer kém; QC rule/scaling locked chuyển sang Spur mất nhiều coverage; classifiers Spur chỉ6 output/7 input states. Phân tích độc lập hai topology là portability evaluation, không phải train-on-Helical/test-on-Spur prediction; class definitions không tương đương nên không thực hiện hay suy transfer classifier như shared physical fault. Không dùng values giữa H/S để causal matching/localization.

## 12. FINAL PIPELINE / KEEP / REMOVE

**FINAL PIPELINE:** full RAW recording → locked MF-DFA(q=-5:0.5:5, order2, fit64–512) → unchanged QC → RAW `[delta_alpha, alpha0]` → fixed Step9 Random Forest reference. Input/output và Helical/Spur riêng; output primary. RF không cần StandardScaler; nếu dùng KNN/SVM/LDA để comparator thì scaler training-only nguyên Step9. Không fit một production model mới trong Step10.

**KEEP:** full recording unit; unchanged MF-DFA/QC; RAW delta_alpha+alpha0 đúng order; original compound labels và operating metadata; topology/channel separation; grouped recording folds; RF reference/config và transparency về NE/coverage.

**REMOVE khỏi operational input/path:** delta_h; VMD_SUM; individual VMD modes; EMD. Giữ mọi features/decomposition outputs và classifier code làm audit/comparator; không delete source artifacts. KNN/SVM/LDA giữ baseline code, không final default.

Final config ở `config/final_pipeline.json`/SHA256, tham chiếu nguyên feature lock Step8 và classifier protocol Step9, cấm retuning/reselection. Không gọi feature-set/parameter lock là guarantee model performance. Không accuracy-driven change hay phương pháp mới. Chín validation checks ở diagnostics/validation.json; upstream SHA256 verified và không sửa code/core hoặc Step4–9.

## 13. Four final answers of Plan 1

1. **MF-DFA có phân biệt fault configurations không?** Có descriptive discrimination trong nhiều matched speed/load cells, especially Helical, và classifier vượt majority baseline; nhưng không global reliable separation. RF MacroF1 chỉ~0.419–0.596 trên valid subsets, một số class recall0; không localization/causal attribution.
2. **Speed/load ảnh hưởng thế nào?** Cả ba features thay đổi theo conditions; Helical widths fault-associated hơn từng main effect nhưng không invariant; alpha0 đặc biệt operating-sensitive. Load transfer failure minh họa hệ quả classification. Spur full-factor effects NE, không kết luận effect0.
3. **VMD/EMD có cải thiện MF-DFA không?** VMD_SUM mostly preserves RAW, chưa consistent gain; modes low coverage và không stable recurrent advantage; EMD có mixed secondary gains nhưng primary losses/coverage/cost và chưa classifier evidence. Không đủ cost-benefit để giữ trong final pipeline.
4. **Generalize tốt qua conditions/Helical–Spur không?** Chưa. Helical load-transfer thấp, Spur QC/class coverage hạn chế; cross-topology classification chưa được đánh giá với compatible targets. Retain locked pipeline như reference nghiên cứu, không deployment/general localization claim.

Kết luận cuối: **chốt RAW MF-DFA + delta_alpha/alpha0 + fixed Random Forest reference**, với unchanged QC và giới hạn generalization được báo cáo. Lựa chọn là giữ pipeline có evidence/cost-benefit tốt nhất trong phạm vi Plan1, không chữa performance bằng retuning hay thêm features.
