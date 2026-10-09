# STEP 8 — FEATURE EVALUATION AND FINAL FEATURE SELECTION

**Khóa bộ B: RAW `delta_alpha` + RAW `alpha0`, theo đúng thứ tự này.** Giữ `delta_alpha` làm feature fault-associated chính; giữ `alpha0` vì thông tin bổ sung thực tế trong cùng speed/load, đồng thời thừa nhận nó nhạy operating condition. Loại `delta_h` khỏi input cuối do redundancy và không có bổ sung lặp lại sau hai feature đã giữ. Không train classifier, không chọn feature theo accuracy, không chạy lại hay tune pipeline.

## 1. Candidate features

Ba features: `delta_alpha`, `alpha0`, `delta_h`; RAW, VMD_SUM, 5 VMD frequency bands cố định; EMD_SUM chỉ comparator. Input/output phân tích riêng; output primary, input secondary. 120 Helical recordings và 160 Spur recordings, mỗi recording có hai channels nhưng không tạo hai acquisitions độc lập. Sử dụng toàn bộ recording, không dùng windows làm samples.

Đọc trực tiếp các bảng Step 6/7 đã xác minh SHA-256 với output manifests và kiểm tra upstream validation. Hai cấu hình xử lý giống nhau ở mọi trường số: q=-5:0.5:5, DFA order=2, fit interval64–512 (17 scales thực tế64–468), QC nguyên bản; VMD K7/alpha500/tau0; EMD nguyên bản. Không sửa metadata compound faults. `diagnostics/upstream_manifest.csv` ghi các đầu vào thực sự dùng. Config khóa giữ đầy đủ cấu hình số và hashes đầu vào. Hash core MF-DFA lịch sử trong snapshot không được dùng để khẳng định core hiện tại chưa từng thay đổi; Step6 đã xử lý việc đó bằng rerun, Step7 dùng cùng code thực tế.

## 2. Coverage

QC là điều kiện chung cho cả ba features của một representation, nên coverage không khác nhau giữa ba feature. Cell evaluable cần đúng hai acquisitions valid và frequency compatibility nguyên bản đối với modes. Pair-condition cần cả hai fault cells evaluable. Không thay missing/invalid bằng zero hoặc overlapping.

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

Spur RAW output không có signals valid ở S3/S4; input không có ở S4. Do đó mô hình full S1–S8 bị thiếu rank và không evaluable. Đây là thiếu bằng chứng/coverage, không phải bằng chứng features hoàn toàn vô dụng. Bảng `tables/qc_by_state.csv` cho từng configuration. Mọi tỷ lệ disjoint bên dưới phải đọc cùng mẫu số evaluable và mẫu số expected.

## 3. Fault separability

Giữ nguyên strict disjoint repeat ranges (gap>0) trong cùng speed/load, reference direction50Hz/High. Không gọi tỷ lệ này là classification accuracy. `hard_pairs_below_50_percent` là nhãn mô tả mới của Step8: <50% disjoint trong các conditions evaluable; không phải threshold tune pipeline và không coi pair chưa evaluable là dễ. Pair thiếu coverage ghi riêng. Direction NE nếu baseline invalid; không chọn direction bằng majority kết quả.

| dataset | channel | feature | disjoint_pair_conditions | evaluable_pair_conditions | expected_pair_conditions | disjoint_fraction | direction_consistency | direction_evaluable | hard_pairs_below_50_percent | completely_unevaluable_pairs |
|---|---|---|---|---|---|---|---|---|---|---|
| Helical | input_voltage | delta_alpha | 113 | 135 | 150 | 0.8370 | 0.6667 | 135 | 0 | 0 |
| Helical | input_voltage | alpha0 | 111 | 135 | 150 | 0.8222 | 0.7259 | 135 | 0 | 0 |
| Helical | input_voltage | delta_h | 113 | 135 | 150 | 0.8370 | 0.6815 | 135 | 1 | 0 |
| Helical | output_voltage | delta_alpha | 111 | 122 | 150 | 0.9098 | 0.6782 | 87 | 1 | 0 |
| Helical | output_voltage | alpha0 | 115 | 122 | 150 | 0.9426 | 0.7241 | 87 | 0 | 0 |
| Helical | output_voltage | delta_h | 107 | 122 | 150 | 0.8770 | 0.6667 | 87 | 1 | 0 |
| Spur | input_voltage | delta_alpha | 41 | 55 | 280 | 0.7455 | 0.8140 | 43 | 4 | 10 |
| Spur | input_voltage | alpha0 | 52 | 55 | 280 | 0.9455 | 0.8372 | 43 | 1 | 10 |
| Spur | input_voltage | delta_h | 45 | 55 | 280 | 0.8182 | 0.8140 | 43 | 0 | 10 |
| Spur | output_voltage | delta_alpha | 27 | 31 | 280 | 0.8710 | 0.7727 | 22 | 0 | 13 |
| Spur | output_voltage | alpha0 | 23 | 31 | 280 | 0.7419 | 0.7273 | 22 | 3 | 13 |
| Spur | output_voltage | delta_h | 30 | 31 | 280 | 0.9677 | 0.7727 | 22 | 0 | 13 |

Các complementary cases dùng cùng observations và cùng pair-condition đã QC, không suy luận geometry phân loại đa biến. Union chỉ nghĩa là ít nhất một marginal feature có repeat ranges disjoint.

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

`alpha0` thêm so với `delta_alpha`: Helical output10 conditions, input18; Spur output3, input13. Những bổ sung lặp lại qua nhiều speeds:

| dataset | channel | pair | extra_conditions | speeds | loads | same_sign_across_extra_conditions | conditions |
|---|---|---|---|---|---|---|---|
| Helical | input_voltage | H1-H2 | 2 | 2 | 1 | True | 35/High;45/High |
| Helical | input_voltage | H2-H6 | 4 | 3 | 2 | False | 35/High;35/Low;45/High;50/High |
| Helical | input_voltage | H3-H5 | 3 | 2 | 2 | True | 35/High;35/Low;45/High |
| Helical | input_voltage | H4-H6 | 2 | 2 | 2 | True | 30/High;45/Low |
| Helical | input_voltage | H5-H6 | 2 | 2 | 1 | True | 30/Low;40/Low |
| Helical | output_voltage | H3-H5 | 4 | 3 | 2 | True | 30/High;35/Low;40/High;40/Low |
| Helical | output_voltage | H4-H6 | 3 | 2 | 2 | False | 35/High;45/High;45/Low |
| Spur | input_voltage | S1-S6 | 4 | 3 | 2 | True | 35/High;40/High;40/Low;45/High |
| Spur | input_voltage | S1-S7 | 2 | 2 | 1 | True | 40/High;50/High |

Trong primary Helical H3–H5, alpha0 bổ sung4 conditions qua3 speeds và2 loads; H4–H6 bổ sung3 conditions qua2 speeds và2 loads. Các cases bổ sung này xuất hiện trong so sánh cùng operating condition, nên không thể bác bỏ toàn bộ alpha0 chỉ vì main speed/load effects lớn. Tuy nhiên direction across extra conditions phải đọc từ bảng; việc tách lặp lại không đồng nghĩa một threshold global ổn định. Spur primary bổ sung alpha0 chỉ đơn lẻ; bằng chứng lặp lại của Spur nằm ở secondary input và subset valid. Không tuyên bố bổ sung generalize toàn S1–S8.

Sau RAW delta_alpha+alpha0, delta_h chỉ thêm Helical input H1–H4 tại50/High, Spur input S3–S7 tại40/High, Spur output S6–S8 tại45/High; mỗi case một condition, Helical primary không thêm case nào. Không đủ bằng chứng giữ feature thứ ba.

## 4. Operating-condition sensitivity

Tái sử dụng mô hình và procedure Step6/7, không refit một fault set nhỏ hơn: additive categorical fault+speed+load, exploratory partial eta squared; HC3 và null-imposed wild bootstrap1999 với BH như upstream. Full interactions/full-grid robustness giữ verdict upstream. Không so partial eta squared như tỷ trọng cộng100%, không giải thích causal. Spur full-factorial effects đều NE do coverage/rank; không điền bằng0 hoặc fit lại trên các states còn lại.

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

Helical widths có fault association lớn hơn từng speed/load main effect; alpha0 bị speed/load chi phối mạnh hơn fault ở cả hai channels. Association không đồng nghĩa invariance. Complete-grid fault/operating ratios vẫn NE; robustness main representations của Helical là CONDITION-DEPENDENT, Spur là NOT EVALUABLE. Sensitivity trên complete subsets:

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

Speed metric=(max−min)/abs(mean) cần5 speeds; load metric=2abs(High−Low)/(abs(High)+abs(Low)) cần hai complete cells. Các median dùng các subset khác nhau, không dùng trực tiếp để khẳng định dataset này robust hơn dataset kia. Alpha0 được giữ như feature thông tin bổ sung có điều kiện, không phải fault-only marker. Step9 cần báo cáo theo speed/load và phép thử chuyển operating condition đã xác định trước.

## 5. Redundancy

Pearson và Spearman chỉ trên observations valid, riêng dataset/channel/representation. Có thêm complete-cell means và within-speed/load centered correlations để kiểm tra pooled association. Centering chỉ là chẩn đoán mô tả trên subset valid, không phải biến đổi input khóa hay fitted Spur full-fault model. Không pool Helical/Spur, không coi hai channels độc lập.

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

Delta_alpha–delta_h Pearson0.955–0.989 trên RAW ở cả hai datasets/channels: hai width features largely redundant. Alpha0–delta_alpha Pearson0.051–0.348, thấp hơn rõ rệt. Không loại chỉ bằng threshold correlation; kết hợp correlation với kiểm tra incremental separability: delta_h không có improvement lặp lại sau delta_alpha+alpha0, alpha0 có. Bảng correlation tất cả candidates/scope trong `tables/feature_correlations.csv`.

## 6. Helical vs Spur consistency

Helical chứng minh width–fault association dưới mô hình exploratory, alpha0 nhạy condition nhưng thêm separation. Spur chỉ cung cấp evidence trong valid subset: width correlation và ít redundancy của alpha0 vẫn thấy; alpha0 có thêm separations lặp lại ở input S1–S6 (4 conditions/3speeds/2loads), S1–S7 (2conditions/2speeds/1load). Full-fault association, overall robustness và full generalization vẫn NOT EVALUABLE. Giữ lựa chọn để Step9 kiểm tra, không coi nó là feature set đã được xác nhận tổng quát trên full Spur. Không so raw magnitudes giữa topology để nhận diện cùng physical fault, không gộp configurations không tương đương.

## 7. RAW vs VMD

So sánh trên paired valid recordings/common pair conditions, không so tỷ lệ có mẫu số khác để gán improvement. Correlation và median relative changes:

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

Matched separation và sensitivity ratios(VMD/RAW; <1 ít nhạy hơn, số trajectories/comparisons phải đọc cùng):

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

VMD_SUM gần như giữ nguyên RAW: correlation cao, changes nhỏ; Helical có net separation losses, Spur output tăng vài cases nhưng input mất và QC giảm. Speed/load changes không nhất quán và Spur denominator rất nhỏ. Không có evidence separation/robustness hoặc thông tin nonredundant đủ rõ để giữ thêm VMD_SUM.

Individual VMD modes vẫn xét frequency bands, không chọn theo index/class. Gate coverage75% kế thừa robustness rule, representative max-energy before QC, center ratio≤1.5; không đổi band hay rescue mode invalid. Coverage:

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

Không band nào đạt gate ở cả datasets/channels. Low band từng có extra Helical input alpha0 H1–H2(3conditions), H1–H3(2conditions); Helical primary extra chỉ đơn lẻ. Spur chỉ extra đơn lẻ input S1–S6 widths tại35/High, output S6–S8 alpha0 tại45/High; không có repeated-compatible extra band groups. Loại tất cả mode candidates vì coverage và recurrence, không kết luận modes vật lý vô ích. EMD_SUM coverage và separability trong feature_summary, ranking và charts chỉ comparator; không sửa EMD/QC để cứu coverage, không xét EMD làm feature input cuối theo scope người dùng.

## 8. Final selected feature set

**B — hai features: RAW `[delta_alpha, alpha0]`.** Cùng danh sách cho primary output và secondary input phân tích riêng. Không chọn lại channel bằng test accuracy. Giữ delta_alpha vì width–fault association và separation, giữ alpha0 vì ít redundant và improvement thực tế lặp lại trong cùng speed/load. Không giữ vì giả định classifier sẽ dùng được. Bộ hai là quyết định trước classification dựa trên descriptive evidence, không phải chứng minh đạt một accuracy nào.

`config/locked_feature_set.json` giữ feature order, representation, QC/processing snapshot, upstream hashes, policy và cấm accuracy-driven reselection. `config/locked_feature_set.sha256` là digest; `scripts/load_locked_features.py` kiểm tra digest/expected feature list/representation/upstream hashes và chỉ trả hai RAW columns trên observations eligible. Đây là integrity guard, không phải cơ chế bất biến chống người cố tình sửa cả config và digest. Step9 phải load config/guard; không silently rewrite.

`ranking_table.csv` gồm mọi metric theo dataset/channel, decision và qualitative selection tier:1 RAW delta_alpha,2 RAW alpha0,3 RAW delta_h(rejected),4 VMD_SUM,5 VMD bands,6 comparator. Không cộng các metric không cùng meaning thành score. Helical/Spur consistency được ghi riêng theo dataset; coverage hạn chế giữ NE. Full evidence nằm trong feature_summary.csv, pair/direction tables và redundancy/complementarity tables.

## 9. Features rejected and reasons

- RAW delta_h: rất correlated với delta_alpha; sau bộ hai chỉ thêm3 single-condition cases trên toàn hai datasets/channels, không repeated benefit. Giữ như diagnostic/comparator, không input classifier.
- VMD_SUM mọi features: gần lặp RAW, matched gains/losses và sensitivity không nhất quán, Spur coverage thấp hơn; không thêm thông tin đủ rõ.
- Individual VMD modes: coverage không đạt gate; lợi ích low-frequency không lặp cross-topology và primary evidence đơn lẻ.
- EMD features: comparator phụ theo scope; QC giữ nguyên, không đưa vào selection final.

Không loại alpha0 vì sensitivity đơn thuần; quyết định giữ có bằng chứng complementary within-condition và kèm giới hạn transfer. Không gọi QC-failed Spur features vô dụng.

## 10. Recommendation for Step 9 classification

Load locked config và exact feature order; không feature selection theo classification accuracy, không thử thêm delta_h/VMD rồi giữ theo test. Không tune MF-DFA/scales/QC/decomposition. Phân tích topology riêng với original compound labels; output primary, input secondary, ghi đủ class/condition coverage, không báo full S1–S8 classifier nếu missing states. Full recording là unit; recording_id grouping giữ input/output chung split. Hai acquisitions mỗi cell rất ít; nếu muốn held-out operating-condition test, giữ toàn conditions cùng fold theo protocol định trước. Không đưa fault label/speed/load vào feature vector khóa; speed/load vẫn metadata để stratified diagnostics và condition-transfer evaluation.

Selection Step8 dùng labels và toàn evidence Step6/7. CV trên đúng recordings này chỉ là exploratory evaluation có điều kiện trên feature set đã chọn; không phải untouched estimate của toàn quy trình selection. Cần recordings mới/held-out chưa dùng selection để có final performance estimate độc lập. Mọi scaler/tuning classifier chỉ fit training fold; giữ test chưa dùng. Không impute QC-failed features để cứu class coverage và không suy luận gear/bearing/shaft localization hoặc causal single-fault effect. Classifier work để Step9; Step8 chưa train gì.

**Trả lời cuối: khóa RAW delta_alpha + RAW alpha0 làm input Step9.** Width signal ổn ở Helical; alpha0 thêm information có điều kiện. Khả năng generalize toàn Spur chưa được xác nhận vì QC coverage, nên Step9 phải giữ giới hạn đó trong báo cáo.
