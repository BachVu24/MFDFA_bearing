# STEP 9 — CLASSIFICATION

## 1. Dataset used

Helical và Spur phân tích riêng, mỗi original recording là một sample; input/output là các analyses riêng, không cộng thành acquisitions độc lập. Chỉ RAW đã pass QC nguyên bản và hai locked features finite. Helical120 recordings nguồn, Spur160; số sample thực dùng theo channel ở bảng bên dưới. Không windows, không imputation/QC rescue, không gộp topology thành cùng class. Metadata compound configurations được lưu nguyên bản từ workbook snapshots đã xác minh; H2–H5 không phải pure bearing contrast.

## 2. Locked features

Input cố định theo Step8: **RAW `[delta_alpha, alpha0]`**, đúng order. Load qua Step8 integrity guard, xác minh upstream/config hashes. Không dùng delta_h, VMD/EMD, speed, load hay fault metadata như feature classifier. Không rerun/tune MF-DFA, scaling, VMD, EMD hoặc QC. Step8 features được chọn từ chính datasets/labels này: kết quả CV ở đây conditional trên bộ feature đã khóa và exploratory, chưa là validation độc lập của toàn quy trình feature selection.

## 3. Class distribution

| dataset | channel | state | valid_recordings | excluded_recordings | qc_coverage | status |
|---|---|---|---|---|---|---|
| Helical | output_voltage | H1 | 20 | 0 | 1.0000 | AVAILABLE |
| Helical | output_voltage | H2 | 20 | 0 | 1.0000 | AVAILABLE |
| Helical | output_voltage | H3 | 16 | 4 | 0.8000 | AVAILABLE |
| Helical | output_voltage | H4 | 18 | 2 | 0.9000 | AVAILABLE |
| Helical | output_voltage | H5 | 17 | 3 | 0.8500 | AVAILABLE |
| Helical | output_voltage | H6 | 18 | 2 | 0.9000 | AVAILABLE |
| Helical | input_voltage | H1 | 16 | 4 | 0.8000 | AVAILABLE |
| Helical | input_voltage | H2 | 20 | 0 | 1.0000 | AVAILABLE |
| Helical | input_voltage | H3 | 20 | 0 | 1.0000 | AVAILABLE |
| Helical | input_voltage | H4 | 19 | 1 | 0.9500 | AVAILABLE |
| Helical | input_voltage | H5 | 20 | 0 | 1.0000 | AVAILABLE |
| Helical | input_voltage | H6 | 20 | 0 | 1.0000 | AVAILABLE |
| Spur | output_voltage | S1 | 8 | 12 | 0.4000 | AVAILABLE |
| Spur | output_voltage | S2 | 2 | 18 | 0.1000 | AVAILABLE |
| Spur | output_voltage | S3 | 0 | 20 | 0.0000 | NOT EVALUABLE - no QC-valid sample |
| Spur | output_voltage | S4 | 0 | 20 | 0.0000 | NOT EVALUABLE - no QC-valid sample |
| Spur | output_voltage | S5 | 2 | 18 | 0.1000 | AVAILABLE |
| Spur | output_voltage | S6 | 6 | 14 | 0.3000 | AVAILABLE |
| Spur | output_voltage | S7 | 16 | 4 | 0.8000 | AVAILABLE |
| Spur | output_voltage | S8 | 14 | 6 | 0.7000 | AVAILABLE |
| Spur | input_voltage | S1 | 17 | 3 | 0.8500 | AVAILABLE |
| Spur | input_voltage | S2 | 2 | 18 | 0.1000 | AVAILABLE |
| Spur | input_voltage | S3 | 3 | 17 | 0.1500 | AVAILABLE |
| Spur | input_voltage | S4 | 0 | 20 | 0.0000 | NOT EVALUABLE - no QC-valid sample |
| Spur | input_voltage | S5 | 3 | 17 | 0.1500 | AVAILABLE |
| Spur | input_voltage | S6 | 18 | 2 | 0.9000 | AVAILABLE |
| Spur | input_voltage | S7 | 18 | 2 | 0.9000 | AVAILABLE |
| Spur | input_voltage | S8 | 15 | 5 | 0.7500 | AVAILABLE |

Helical tương đối cân bằng (16–20 samples/class/channel). Spur input có17/2/3/0/3/18/18/15 cho S1–S8 (max/min nonzero=9); output8/2/0/0/2/6/16/14 (ratio8). S4 không evaluable ở cả channels; S3 không evaluable ở output. **Spur input là bài toán7 configurations còn valid, output6 configurations; không phải full8-class validation.** Macro metrics chỉ tính trên classes còn valid; per-class metrics của missing configurations là NE, không thay bằng recall0. QC selection có thể không ngẫu nhiên và ảnh hưởng độ khó bài toán.

## 4. Classification method

- KNN:5 neighbors, uniform weights, Euclidean distance; StandardScaler.
- SVM:RBF, C=1, gamma=scale; StandardScaler.
- LDA:svd solver, empirical priors; StandardScaler.
- Random Forest:200 trees, default unrestricted depth, min leaf1, max_features=sqrt, bootstrap, seed9609, single thread; không cần scaling.

Không grid search/hyperparameter tuning, không chọn lại features. RF có thể overfit training data; chỉ báo predictions held-out. Không cân bằng class bằng oversampling hoặc dùng thông tin test. Baseline train-only majority báo riêng. Code nằm đúng4 files `code/core/classifier/{knn,svm,lda,random_forest}.py`; factory trả sklearn estimator/Pipeline có thể clone.

Primary protocol: Helical5-fold, Spur2-fold vì rare classes chỉ2 samples. Configuration-stratified partition, balance channel eligibility bằng metadata, seed9609; mỗi recording có một fold toàn dataset dùng chung hai channels. Fold maps và config lưu trước khi fit; tất cả classifiers dùng cùng samples/folds trong mỗi context. Mỗi eligible recording có đúng một prediction out-of-fold cho mỗi classifier/channel. Train/test recording IDs không giao nhau. Scaler nằm trong Pipeline và fit riêng từng training fold; gamma/priors cũng chỉ học training. Các fold đều có train/test observations cho mọi available class. Stratification dùng labels được phép; không dùng features/predictions để phân folds.

Accuracy, macro F1, macro precision, macro recall tính từ pooled OOF predictions; weightedF1 và fold standard deviations báo thêm. Recall macro tương đương balanced accuracy trên available labels. Fold std không phải confidence interval và2-fold Spur rất bất ổn; mỗi rare-class prediction chỉ có một training example cùng class. Không dùng repeated CV như samples độc lập. Best so theo macroF1 trước, accuracy phụ; best là mô tả nội bộ, không triển khai/pick final bằng independent test.

Supplemental fixed stress tests: leave-one-speed-out và leave-one-load-out trên valid records; không chọn features/parameters từ kết quả. Split thiếu bất kỳ available training class ghi NOT EVALUABLE, không drop class để fit. Fold macro metrics tính trên classes xuất hiện ở test condition; aggregate transfer summary cũng chỉ dùng classes có test observations evaluable, các class chưa được test ghi NE trong untested_classes, kèm test coverage (có thể<100%). Split/prediction tables lưu đầy đủ. Không dùng stress-test scores để thay primary protocol sau khi xem kết quả.

## 5. Helical results

| channel | classifier | n_samples | n_classes | accuracy | macro_f1 | macro_precision | macro_recall | fold_macro_f1_std |
|---|---|---|---|---|---|---|---|---|
| output_voltage | KNN | 109 | 6 | 0.3945 | 0.3790 | 0.3813 | 0.3869 | 0.1023 |
| output_voltage | SVM | 109 | 6 | 0.4495 | 0.4371 | 0.4657 | 0.4410 | 0.0564 |
| output_voltage | LDA | 109 | 6 | 0.2661 | 0.2528 | 0.2546 | 0.2705 | 0.1174 |
| output_voltage | Random Forest | 109 | 6 | 0.5321 | 0.5283 | 0.5377 | 0.5251 | 0.0854 |
| input_voltage | KNN | 115 | 6 | 0.3478 | 0.3421 | 0.3490 | 0.3452 | 0.0793 |
| input_voltage | SVM | 115 | 6 | 0.3826 | 0.3743 | 0.3848 | 0.3751 | 0.0746 |
| input_voltage | LDA | 115 | 6 | 0.3304 | 0.3206 | 0.3236 | 0.3226 | 0.1119 |
| input_voltage | Random Forest | 115 | 6 | 0.4261 | 0.4190 | 0.4194 | 0.4214 | 0.0958 |

## 6. Spur results

| channel | classifier | n_samples | n_classes | accuracy | macro_f1 | macro_precision | macro_recall | fold_macro_f1_std |
|---|---|---|---|---|---|---|---|---|
| output_voltage | KNN | 48 | 6 | 0.3958 | 0.2238 | 0.2278 | 0.2381 | 0.1588 |
| output_voltage | SVM | 48 | 6 | 0.5833 | 0.2441 | 0.2037 | 0.3110 | 0.0045 |
| output_voltage | LDA | 48 | 6 | 0.4167 | 0.2128 | 0.2378 | 0.2351 | 0.0224 |
| output_voltage | Random Forest | 48 | 6 | 0.5417 | 0.5008 | 0.4889 | 0.5164 | 0.1657 |
| input_voltage | KNN | 76 | 7 | 0.4342 | 0.2448 | 0.2454 | 0.2713 | 0.0405 |
| input_voltage | SVM | 76 | 7 | 0.5658 | 0.4351 | 0.4738 | 0.4342 | 0.0828 |
| input_voltage | LDA | 76 | 7 | 0.6053 | 0.4587 | 0.4893 | 0.4575 | 0.0531 |
| input_voltage | Random Forest | 76 | 7 | 0.6316 | 0.5955 | 0.6155 | 0.6065 | 0.1567 |

Train-only majority baseline để đọc accuracy trong imbalance:

| dataset | channel | n_samples | accuracy | macro_f1 |
|---|---|---|---|---|
| Helical | output_voltage | 109 | 0.1835 | 0.0517 |
| Helical | input_voltage | 115 | 0.1739 | 0.0494 |
| Spur | output_voltage | 48 | 0.3333 | 0.0833 |
| Spur | input_voltage | 76 | 0.2237 | 0.0874 |

## 7. Input vs output channel

Hai channels dùng cùng recording-to-fold mapping; coverage/class sets khác nhau, vì vậy score cao hơn không chứng minh channel vật lý tốt hơn. Không fuse channels hay cộng sample size. Bảng best theo từng context ở mục8; per-class table và16 confusion matrices cho mọi classifier/channel/dataset. Transfer tests:

| dataset | channel | classifier | protocol | evaluable_splits | expected_splits | tested_recordings | eligible_recordings | test_coverage | untested_classes | accuracy | macro_f1 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Helical | input_voltage | KNN | leave_one_load_out | 2 | 2 | 115 | 115 | 1.0000 |  | 0.1739 | 0.1478 |
| Helical | input_voltage | KNN | leave_one_speed_out | 5 | 5 | 115 | 115 | 1.0000 |  | 0.2609 | 0.2595 |
| Helical | input_voltage | LDA | leave_one_load_out | 2 | 2 | 115 | 115 | 1.0000 |  | 0.1478 | 0.1175 |
| Helical | input_voltage | LDA | leave_one_speed_out | 5 | 5 | 115 | 115 | 1.0000 |  | 0.3217 | 0.3083 |
| Helical | input_voltage | Random Forest | leave_one_load_out | 2 | 2 | 115 | 115 | 1.0000 |  | 0.1652 | 0.1456 |
| Helical | input_voltage | Random Forest | leave_one_speed_out | 5 | 5 | 115 | 115 | 1.0000 |  | 0.3478 | 0.3153 |
| Helical | input_voltage | SVM | leave_one_load_out | 2 | 2 | 115 | 115 | 1.0000 |  | 0.1913 | 0.1473 |
| Helical | input_voltage | SVM | leave_one_speed_out | 5 | 5 | 115 | 115 | 1.0000 |  | 0.3739 | 0.3580 |
| Helical | output_voltage | KNN | leave_one_load_out | 2 | 2 | 109 | 109 | 1.0000 |  | 0.0826 | 0.0556 |
| Helical | output_voltage | KNN | leave_one_speed_out | 5 | 5 | 109 | 109 | 1.0000 |  | 0.3761 | 0.3631 |
| Helical | output_voltage | LDA | leave_one_load_out | 2 | 2 | 109 | 109 | 1.0000 |  | 0.1193 | 0.1062 |
| Helical | output_voltage | LDA | leave_one_speed_out | 5 | 5 | 109 | 109 | 1.0000 |  | 0.3119 | 0.2810 |
| Helical | output_voltage | Random Forest | leave_one_load_out | 2 | 2 | 109 | 109 | 1.0000 |  | 0.0917 | 0.0923 |
| Helical | output_voltage | Random Forest | leave_one_speed_out | 5 | 5 | 109 | 109 | 1.0000 |  | 0.4128 | 0.4071 |
| Helical | output_voltage | SVM | leave_one_load_out | 2 | 2 | 109 | 109 | 1.0000 |  | 0.0826 | 0.0727 |
| Helical | output_voltage | SVM | leave_one_speed_out | 5 | 5 | 109 | 109 | 1.0000 |  | 0.4404 | 0.4225 |
| Spur | input_voltage | KNN | leave_one_load_out | 1 | 2 | 32 | 76 | 0.4211 | S2;S3;S5 | 0.2500 | 0.1905 |
| Spur | input_voltage | KNN | leave_one_speed_out | 4 | 5 | 56 | 76 | 0.7368 | S2 | 0.5536 | 0.3697 |
| Spur | input_voltage | LDA | leave_one_load_out | 1 | 2 | 32 | 76 | 0.4211 | S2;S3;S5 | 0.3125 | 0.2321 |
| Spur | input_voltage | LDA | leave_one_speed_out | 4 | 5 | 56 | 76 | 0.7368 | S2 | 0.6429 | 0.5893 |
| Spur | input_voltage | Random Forest | leave_one_load_out | 1 | 2 | 32 | 76 | 0.4211 | S2;S3;S5 | 0.2812 | 0.2000 |
| Spur | input_voltage | Random Forest | leave_one_speed_out | 4 | 5 | 56 | 76 | 0.7368 | S2 | 0.6964 | 0.6341 |
| Spur | input_voltage | SVM | leave_one_load_out | 1 | 2 | 32 | 76 | 0.4211 | S2;S3;S5 | 0.4062 | 0.3943 |
| Spur | input_voltage | SVM | leave_one_speed_out | 4 | 5 | 56 | 76 | 0.7368 | S2 | 0.5893 | 0.4004 |
| Spur | output_voltage | KNN | leave_one_load_out | 1 | 2 | 12 | 48 | 0.2500 | S1;S2;S5;S6 | 0.4167 | 0.5165 |
| Spur | output_voltage | KNN | leave_one_speed_out | 4 | 5 | 34 | 48 | 0.7083 | S2;S5 | 0.3235 | 0.2438 |
| Spur | output_voltage | LDA | leave_one_load_out | 1 | 2 | 12 | 48 | 0.2500 | S1;S2;S5;S6 | 0.5000 | 0.6250 |
| Spur | output_voltage | LDA | leave_one_speed_out | 4 | 5 | 34 | 48 | 0.7083 | S2;S5 | 0.5294 | 0.3338 |
| Spur | output_voltage | Random Forest | leave_one_load_out | 1 | 2 | 12 | 48 | 0.2500 | S1;S2;S5;S6 | 0.8333 | 0.8730 |
| Spur | output_voltage | Random Forest | leave_one_speed_out | 4 | 5 | 34 | 48 | 0.7083 | S2;S5 | 0.3235 | 0.2542 |
| Spur | output_voltage | SVM | leave_one_load_out | 1 | 2 | 12 | 48 | 0.2500 | S1;S2;S5;S6 | 0.5833 | 0.6061 |
| Spur | output_voltage | SVM | leave_one_speed_out | 4 | 5 | 34 | 48 | 0.7083 | S2;S5 | 0.5882 | 0.3636 |

## 8. Best classifier

Best theo pooled macro F1 trong từng dataset/channel (không tổng hợp các topology khác class definitions thành một accuracy):

| dataset | channel | classifier | accuracy | macro_f1 | macro_precision | macro_recall |
|---|---|---|---|---|---|---|
| Helical | input_voltage | Random Forest | 0.4261 | 0.4190 | 0.4194 | 0.4214 |
| Helical | output_voltage | Random Forest | 0.5321 | 0.5283 | 0.5377 | 0.5251 |
| Spur | input_voltage | Random Forest | 0.6316 | 0.5955 | 0.6155 | 0.6065 |
| Spur | output_voltage | Random Forest | 0.5417 | 0.5008 | 0.4889 | 0.5164 |

Per-class recall của best models, kèm precision/support để tránh gọi class dễ chỉ vì nhiều false positives:

| dataset | channel | classifier | state | support | precision | recall | f1 | status |
|---|---|---|---|---|---|---|---|---|
| Helical | output_voltage | Random Forest | H1 | 20 | 0.6842 | 0.6500 | 0.6667 | EVALUABLE |
| Helical | output_voltage | Random Forest | H2 | 20 | 0.7000 | 0.7000 | 0.7000 | EVALUABLE |
| Helical | output_voltage | Random Forest | H3 | 16 | 0.4706 | 0.5000 | 0.4848 | EVALUABLE |
| Helical | output_voltage | Random Forest | H4 | 18 | 0.5714 | 0.4444 | 0.5000 | EVALUABLE |
| Helical | output_voltage | Random Forest | H5 | 17 | 0.4667 | 0.4118 | 0.4375 | EVALUABLE |
| Helical | output_voltage | Random Forest | H6 | 18 | 0.3333 | 0.4444 | 0.3810 | EVALUABLE |
| Helical | input_voltage | Random Forest | H1 | 16 | 0.3571 | 0.3125 | 0.3333 | EVALUABLE |
| Helical | input_voltage | Random Forest | H2 | 20 | 0.4762 | 0.5000 | 0.4878 | EVALUABLE |
| Helical | input_voltage | Random Forest | H3 | 20 | 0.5000 | 0.5000 | 0.5000 | EVALUABLE |
| Helical | input_voltage | Random Forest | H4 | 19 | 0.3750 | 0.3158 | 0.3429 | EVALUABLE |
| Helical | input_voltage | Random Forest | H5 | 20 | 0.3500 | 0.3500 | 0.3500 | EVALUABLE |
| Helical | input_voltage | Random Forest | H6 | 20 | 0.4583 | 0.5500 | 0.5000 | EVALUABLE |
| Spur | output_voltage | Random Forest | S1 | 8 | 0.1667 | 0.1250 | 0.1429 | EVALUABLE |
| Spur | output_voltage | Random Forest | S2 | 2 | 0.5000 | 0.5000 | 0.5000 | EVALUABLE |
| Spur | output_voltage | Random Forest | S3 | 0 | NE | NE | NE | NOT EVALUABLE - no QC-valid sample |
| Spur | output_voltage | Random Forest | S4 | 0 | NE | NE | NE | NOT EVALUABLE - no QC-valid sample |
| Spur | output_voltage | Random Forest | S5 | 2 | 1.0000 | 1.0000 | 1.0000 | EVALUABLE |
| Spur | output_voltage | Random Forest | S6 | 6 | 0.0000 | 0.0000 | 0.0000 | EVALUABLE |
| Spur | output_voltage | Random Forest | S7 | 16 | 0.5789 | 0.6875 | 0.6286 | EVALUABLE |
| Spur | output_voltage | Random Forest | S8 | 14 | 0.6875 | 0.7857 | 0.7333 | EVALUABLE |
| Spur | input_voltage | Random Forest | S1 | 17 | 0.7000 | 0.8235 | 0.7568 | EVALUABLE |
| Spur | input_voltage | Random Forest | S2 | 2 | 0.6667 | 1.0000 | 0.8000 | EVALUABLE |
| Spur | input_voltage | Random Forest | S3 | 3 | 0.0000 | 0.0000 | 0.0000 | EVALUABLE |
| Spur | input_voltage | Random Forest | S4 | 0 | NE | NE | NE | NOT EVALUABLE - no QC-valid sample |
| Spur | input_voltage | Random Forest | S5 | 3 | 1.0000 | 0.6667 | 0.8000 | EVALUABLE |
| Spur | input_voltage | Random Forest | S6 | 18 | 0.7143 | 0.5556 | 0.6250 | EVALUABLE |
| Spur | input_voltage | Random Forest | S7 | 18 | 0.5000 | 0.6667 | 0.5714 | EVALUABLE |
| Spur | input_voltage | Random Forest | S8 | 15 | 0.7273 | 0.5333 | 0.6154 | EVALUABLE |

Helical/input_voltage: recall cao nhất H6 (0.550); thấp nhất H1 (0.312). Đây là độ dễ/khó trong protocol và subset hiện tại, không properties vật lý cố định.

Helical/output_voltage: recall cao nhất H2 (0.700); thấp nhất H5 (0.412). Đây là độ dễ/khó trong protocol và subset hiện tại, không properties vật lý cố định.

Spur/input_voltage: recall cao nhất S2 (1.000); thấp nhất S3 (0.000). Đây là độ dễ/khó trong protocol và subset hiện tại, không properties vật lý cố định.

Spur/output_voltage: recall cao nhất S5 (1.000); thấp nhất S6 (0.000). Đây là độ dễ/khó trong protocol và subset hiện tại, không properties vật lý cố định.

## 9. Limitations

- Feature-selection labels đã được dùng Step8: primary CV chưa đo unbiased end-to-end generalization. Cần recordings mới chưa tham gia selection để có final independent estimate; Step9 không chọn lại features theo accuracy.
- Chỉ2 acquisitions/condition; recording-level folds có thể cùng fault/speed/load ở train/test. Đây chủ yếu là interpolation trong dataset; supplemental speed/load tests cho giới hạn operating transfer, không giải quyết mọi acquisition/domain dependence.
- Spur imbalance lớn, rare classes2–3 samples, missing classes và QC selection bias. Full S1–S8 performance không evaluable; không suy recall/accuracy cho S3 output hoặc S4. Không coi QC failures là classifier failures hay feature vô dụng.
- So best trong4 classifiers bằng cùng internal OOF có model-comparison optimism; không gọi best là universal classifier. LDA/Gaussian assumptions, simple SVM defaults và KNN neighborhood không tối ưu; scope chỉ basic comparison. Không tăng feature count/tune để cứu scores.
- High accuracy (nếu có) không là bearing/gear/shaft causal localization; targets là original fault configurations và compound faults. Không map H/S như cùng physical faults hay pool datasets.

## 10. Conclusion

**Bộ hai features chứa thông tin phân loại vượt majority baseline, nhưng hiện chưa đủ bằng chứng cho classification đáng tin cậy trên toàn configurations và operating conditions. Random Forest tốt nhất theo Macro F1 ở cả bốn contexts.** Primary MacroF1 chỉ0.419–0.596; một số class recall0 và Helical load-transfer rất kém. Không điều chỉnh feature lock để cải thiện các scores này.

- Helical/input_voltage: best Random Forest, accuracy0.426, macroF10.419; majority baseline accuracy0.174, macroF10.049.
- Helical/output_voltage: best Random Forest, accuracy0.532, macroF10.528; majority baseline accuracy0.183, macroF10.052.
- Spur/input_voltage: best Random Forest, accuracy0.632, macroF10.596; majority baseline accuracy0.224, macroF10.087.
- Spur/output_voltage: best Random Forest, accuracy0.542, macroF10.501; majority baseline accuracy0.333, macroF10.083.

Helical Random Forest leave-one-load-out MacroF1 output0.092, input0.146, giảm mạnh so với recording CV. Leave-one-speed-out output0.407, input0.315. Đây là evidence cụ thể chống diễn giải bộ hai features là operating-condition-invariant. Spur transfer thiếu training classes ở một số splits; đọc test/class coverage thay vì dùng scores subset để xác nhận full transfer.

Với primary output, H2/H1 có recall0.70/0.65, H5 thấp nhất0.412; input Helical H6 cao nhất0.55, H1 thấp nhất0.3125. Spur output S6 recall0, S1 chỉ0.125; S8/S7 có0.786/0.688 trên14/16 samples. S5 output đạt1.0 nhưng chỉ2 samples, chưa đủ gọi class này chắc chắn dễ. Spur input S3 recall0 trên3 samples; S1 đạt0.824 trên17; S2 đạt1.0 nhưng chỉ2 samples. S4 và S3 output không có evidence để đánh giá.

Hai locked features cho phép chạy classification fault configurations trong eligible subset; mức đủ thực dụng phải dựa vào macroF1, từng class và condition-transfer, không chỉ accuracy. Không có target performance threshold đã xác định trước nên không tự gán đạt/không đạt deployment. Kết quả không chứng minh đủ cho robust full Helical/Spur classification hoặc localization; Spur full dataset vẫn chưa evaluable. Giữ nguyên feature lock; bước kế tiếp cần thêm independent recordings và QC coverage thay vì chọn lại feature bằng accuracy của tập này.

Các files chính: classification_results.csv; class_distribution.csv; per_class_metrics.csv; confusion_matrices/*.csv; plots/*.png; tables/out_of_fold_predictions.csv; tables/fold_metrics.csv; tables/fit_leakage_audit.csv; tables/operating_condition_*.csv; config/evaluation_protocol.json. Không commit/push GitHub hay train/deploy final production model.
