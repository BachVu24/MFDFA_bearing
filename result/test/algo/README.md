# Kết quả kiểm thử EMD, VMD, MF-DFA

Ngày thực hiện: 09/10/2026. **24/24 kiểm thử đạt**, không lỗi hoặc kiểm thử thất bại.
Chi tiết từng phép kiểm tra: `test_log.txt`; số liệu và phiên bản môi trường: `summary.json`.

## Phạm vi thực hiện

Chỉ chỉnh sửa ba file `code/core/emd.py`, `code/core/vmd.py`, `code/core/mfdfa.py`.
Script kiểm thử, số liệu, ảnh, bản trích văn bản và ảnh công thức từ PDF đều ở
`result/test/algo`. Không sửa `data_collect.py`, các bài báo, hoặc các thư mục khác.
Không sử dụng thư viện EMD/VMD/MFDFA có sẵn để thay thế giải thuật.

## Cơ sở giải thuật

| File | Bài báo tại `paper` | Các bước triển khai |
|---|---|---|
| `emd.py` | `Huang1996.pdf`: Huang et al., Proc. R. Soc. A 454 (1998), 903–995 | Định nghĩa IMF ở mục 3; sàng lọc bao cực đại/cực tiểu bằng cubic spline ở mục 5; SD theo (5.5); phần dư và tái tạo theo (5.6)–(5.8). |
| `vmd.py` | `dragomiretskiy2014.pdf`: Dragomiretskiy & Zosso, IEEE TSP 62(3) (2014), 531–544 | Algorithm 2; cập nhật mode theo (27), trọng tâm phổ theo (28), dual ascent theo (29); phản chiếu nửa chiều dài ở mỗi đầu theo III.D. |
| `mfdfa.py` | `Kantelhardt2002.pdf`: Kantelhardt et al., Physica A 316 (2002), 87–114 | Profile (1), chia đoạn từ hai đầu (2)–(3), detrending đa thức, moment (4), hồi quy (5), q=0 (6), tau (14), phổ kỳ dị (15). |

Tên `Huang1996.pdf` là tên file người dùng cung cấp. Trang đầu ghi nhận bản thảo
năm 1996, còn năm xuất bản của bài là 1998.

## Kết quả định lượng

| Kiểm tra | Kết quả | Tiêu chí đã đặt trong script |
|---|---|---|
| EMD: tín hiệu tổng ba tone, 2.048 mẫu, fs=1.024 Hz | Ba IMF đầu có đỉnh 160, 64, 8 Hz; tương quan với thành phần gốc ở vùng giữa ≥0,99925 | Đỉnh lệch ≤0,5 Hz; tương quan >0,95 |
| EMD: IMF + phần dư | Sai số tương đối đo được 0 trên tín hiệu kiểm thử | <1e-14 |
| EMD: định nghĩa IMF và tiêu chí dừng | Chênh cực trị/đổi dấu ≤1; SD từ 0,08048 đến 0,18670; tỷ số RMS trung bình bao ≤0,00155 | SD ≤0,2; tỷ số RMS ≤0,05 |
| EMD: chirp điều biên 20–80 Hz | Trung vị sai lệch tần số tức thời khoảng 0,04093 Hz | <1 Hz ở vùng giữa |
| VMD: cùng tín hiệu ba tone, K=3 | Tần số trung tâm 8,00040; 63,99587; 160,00957 Hz | Lệch ≤0,2 Hz |
| VMD: tái tạo, tau=0,5 | Sai số tương đối 0,000315895 = 0,03159%; hội tụ sau 417 vòng | Sai số <0,0004; phải báo hội tụ |
| VMD: tone có nhiễu, K=1, tau=0 | Sai số đối với tone sạch giảm từ 0,43096 xuống 0,09798 | Sai số sau lọc nhỏ hơn trước lọc |
| MF-DFA: đối chiếu vòng lặp polyfit độc lập | Sai số tương đối Fq ≈1,585e-16 | <1e-11 |
| MF-DFA: nhiễu trắng, N=65.536, seed=42 | h(2)=0,51775; max lệch h(q) khỏi 0,5 là 0,01824 | Max lệch <0,07; R² >0,99 |
| MF-DFA: random walk, N=65.536 | h(2)=1,51496 | Lệch khỏi hệ số scaling 1,5 <0,08 |
| MF-DFA: binomial a=0,75, N=65.536 | Max lệch h(q) khỏi (20) là 0,05417 trên q=-5..5 | Max lệch <0,06; h(-5)-h(5)>0,6 |

Chuỗi binomial được tạo đúng (18); dùng DFA1 và scales 32, 64, 128, 256, 512,
1.024, 2.048. Đây là kiểm chứng trên một tập thang hữu hạn, không tái tạo nguyên
bộ thí nghiệm/hình vẽ của bài báo. Kết quả có độ lệch đo được; không áp dụng
bất kỳ hiệu chỉnh nào để ép hệ số khớp lý thuyết.

Các kiểm thử khác bao gồm:

- EMD: một tone, chuỗi hằng/đơn điệu, cực trị plateau, thời gian không đều,
  đổi biên độ, bảo toàn tín hiệu khi sifting không hội tụ.
- VMD: đối chiếu độc lập một và hai bước ADMM (bao gồm dual ascent), số mẫu
  lẻ/chẵn, Nyquist, tín hiệu zero, DC, ngẫu nhiên có seed, đổi biên độ,
  phổ FFT và hàm tương thích `VMD`.
- MF-DFA: q âm/dương/zero/gần zero/±100, xu hướng tuyến tính và bậc hai,
  đổi biên độ/dấu/offset, chọn vùng fit, Legendre transform trên tau tuyến tính
  và bậc hai, chuỗi có phương sai detrending bằng zero.
- Cả ba: từ chối các đầu vào không hợp lệ; báo trạng thái khi đạt giới hạn lặp.

## API và cách chạy lại

Chạy từ thư mục gốc `Project`:

```powershell
python -B result\test\algo\test_algorithms.py
```

Môi trường đã kiểm tra: Python 3.13.4, NumPy 2.2.6, SciPy 1.18.1,
Matplotlib 3.10.8. NumPy dùng cho cả ba; SciPy dùng cho cubic spline/cực trị
trong EMD; Matplotlib chỉ dùng trong script kiểm thử để tạo ảnh.
`-B` ngăn tạo `__pycache__`; cache Matplotlib được đặt trong `result/test/algo/mplconfig`.

Ví dụ sử dụng sau khi thêm `code/core` vào Python import path:

```python
import numpy as np
from emd import emd
from vmd import vmd
from mfdfa import mfdfa, multifractal_spectrum

fs = 1024
t = np.arange(2048) / fs
x = np.cos(2*np.pi*8*t) + 0.7*np.cos(2*np.pi*64*t) + 0.4*np.cos(2*np.pi*160*t)

imfs, residue, emd_info = emd(x, max_imfs=3, return_info=True)
modes, spectra, omega, vmd_info = vmd(x, K=3, fs=fs, return_info=True)

noise = np.random.default_rng(42).normal(size=65536)
q = np.arange(-5., 6.)
scales, Fq, hq, tau = mfdfa(noise, q, m=2)
alpha, f_alpha = multifractal_spectrum(q, tau)
```

- `emd`: IMF có shape `(n_imfs, N)`; phần dư `(N,)`. Bao spline phản chiếu
  tối đa hai cực trị mỗi đầu, dùng điều kiện natural; đây là lựa chọn số học
  cho biên. SD dùng tổng tỷ số theo từng điểm của (5.5), có regularization
  mẫu số gần zero. Không lấy tỷ số hai tổng năng lượng để thay công thức này.
  Có kiểm tra thêm trung bình bao và số đổi dấu. Nếu không hội tụ, giữ phần
  chưa tách trong residue và phát cảnh báo.
- `vmd`: modes `(K,N)`, spectra `(N,K)`, omega `(iterations+1,K)`.
  `omega` tính bằng Hz khi truyền fs thực; fs mặc định 1 cho cycles/sample.
  alpha dùng lưới tần số cycles/sample. Mode giữ thứ tự khởi tạo, không tự
  sắp xếp; dùng `np.argsort(omega[-1])` khi cần. Tín hiệu lẻ giữ nguyên N.
  `tau=0` là lựa chọn denoising trong III.C; không yêu cầu tái tạo chính xác.
  `tau>0` kiểm tra thêm sai số ràng buộc ≤sqrt(tol), ngoài tiêu chí thay đổi
  tương đối của bài báo. Không cộng phần dư vào mode để che sai số.
- `mfdfa`: giữ thứ tự trả về của file cũ. q_list nhận list hoặc ndarray.
  Fq có shape `(len(q), len(scales))`; scale phải >m+2 và ≤N//4.
  Tính moment trong miền log và xử lý q gần zero bằng expm1/log1p.
  Phương sai gần mức lỗi làm tròn được coi là zero; chuỗi không có scaling
  hoặc q≤0 với đoạn phương sai zero bị từ chối, không gán Hurst giả.
  Hệ số h(2) của random walk là hệ số scaling ≈1,5, không diễn giải thành
  Hurst 1,5 của một chuỗi dừng. Công thức tau=q*h(q)-1 giả định support compact.

Các kiểm thử xác nhận các trường hợp nêu trên; chúng không chứng minh mọi tín
hiệu đều tách tốt. Hiệu ứng biên/mode mixing của EMD, lựa chọn K/alpha/init
và hội tụ của VMD, lựa chọn vùng scaling của MF-DFA vẫn phụ thuộc dữ liệu.

## Các file kết quả

- `test_algorithms.py`: bộ kiểm thử có thể chạy lại, tự sinh số liệu/ảnh/log.
- `test_log.txt`: trạng thái từng kiểm thử.
- `summary.json`: chỉ số, trạng thái tổng và phiên bản môi trường.
- `numerical_results.npz`: tín hiệu, các mode, phần dư, lịch sử trung tâm,
  Fq/hq và phổ binomial; đọc bằng `np.load(...)`.
- `mfdfa_binomial.csv`: q, hq ước lượng/lý thuyết, alpha, f(alpha).
- `emd_vmd_decomposition.png`, `mfdfa_validation.png`: hình kiểm chứng.
- `*_paper.txt`, `*_page_*.png`: văn bản và các trang công thức đã đọc từ
  ba PDF gốc. Chỉ dùng đối chiếu, không sửa các PDF.
