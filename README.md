# Countdown Solver bằng GRPO

> **Nghiên cứu khoa học** | Áp dụng Group Relative Policy Optimization (GRPO) để học chính sách giải bài toán Countdown.

## Mục lục

1. [Mục tiêu](#1-mục-tiêu)
2. [Bài toán Countdown](#2-bài-toán-countdown)
3. [Mô hình hóa bài toán](#3-mô-hình-hóa-bài-toán)
4. [GRPO trong dự án](#4-grpo-trong-dự-án)
5. [Cấu trúc các thành phần](#5-cấu-trúc-các-thành-phần)
6. [Cài đặt](#6-cài-đặt)
7. [Huấn luyện, đánh giá và giải bài](#7-huấn-luyện-đánh-giá-và-giải-bài)
8. [Kết quả và giới hạn](#8-kết-quả-và-giới-hạn)
9. [Kiểm thử](#9-kiểm-thử)

---

## 1. Mục tiêu

Dự án nghiên cứu việc huấn luyện một policy neural network bằng **GRPO** để giải các puzzle Countdown. Policy nhận trạng thái các số còn lại cùng Target, sau đó phân phối xác suất trên những phép tính hợp lệ.

Các mục tiêu chính:

- Mô hình hóa Countdown thành môi trường có trạng thái, hành động, chuyển trạng thái và phần thưởng.
- Huấn luyện policy bằng nhóm trajectory cho mỗi puzzle và advantage tương đối trong nhóm.
- Đánh giá trên dữ liệu held-out tách biệt với dữ liệu huấn luyện.
- Kiểm chứng lời giải bằng verifier độc lập trước khi tính là thành công.

## 2. Bài toán Countdown

Cho một tập số nguyên dương và một Target. Mỗi bước chọn hai số hiện có, thực hiện một phép toán hợp lệ trong `+`, `-`, `*`, `/`, rồi thay hai số đó bằng kết quả.

Quy tắc phép toán:

- Phép trừ phải cho kết quả nguyên dương.
- Phép chia phải chia hết và cho kết quả nguyên dương.
- Mỗi số trung gian chỉ được dùng theo đúng số lần nó xuất hiện.
- Mặc định phải dùng hết tất cả các số ban đầu.

Ví dụ:

```text
Các số: 25 50 3 6 7 8
Target: 952

25 - 6 = 19
3 + 7 = 10
19 * 50 = 950
10 - 8 = 2
2 + 950 = 952
```

File đầu vào mặc định là `input.txt`:

```text
25 50 3 6 7 8
Target: 952
```

## 3. Mô hình hóa bài toán

| Thành phần | Mô hình trong dự án |
|---|---|
| Trạng thái `s` | Tuple các số nguyên dương còn lại, được sắp xếp |
| Target `T` | Được đưa trực tiếp vào policy cùng với trạng thái |
| Hành động `a` | Một phép tính hợp lệ `(left, operator, right, result)` |
| Chuyển trạng thái | Xác định: thay hai toán hạng bằng kết quả rồi sắp xếp lại |
| Episode | Chuỗi hành động từ puzzle ban đầu đến kết thúc hoặc hết giới hạn bước |
| Kết quả episode | Được verifier kiểm tra độc lập; reward trajectory dùng để tính advantage |

Môi trường được cài trong `countdown_env.py`. Reward của môi trường theo từng bước và reward trajectory dùng trong GRPO là hai tín hiệu riêng; reward trajectory dựa trên kết quả verifier và độ lệch tới Target.

## 4. GRPO trong dự án

### 4.1 Policy

`PolicyNetwork` trong `grpo_agent.py` dùng kiến trúc DeepSets để biểu diễn tập số có độ dài thay đổi:

1. Mã hóa từng số trong trạng thái bằng MLP.
2. Gộp biểu diễn bằng masked mean và max pooling để không phụ thuộc thứ tự các số.
3. Mã hóa Target và kết hợp với biểu diễn trạng thái.
4. Mã hóa từng hành động hợp lệ, rồi tính một logit cho mỗi hành động.
5. Dùng phân phối categorical trên danh sách hành động hợp lệ.

Vì policy chỉ chấm điểm các hành động môi trường cung cấp, hành động không hợp lệ không được chọn.

### 4.2 Group-relative advantage và mục tiêu tối ưu

Với mỗi puzzle, policy lấy mẫu một nhóm `G` trajectory. Nếu `R_i` là reward của trajectory thứ `i`, advantage được chuẩn hóa tương đối trong nhóm:

```text
A_i = (R_i - mean(R)) / (std(R) + epsilon)
```

Nếu reward trong nhóm không có độ lệch (`std = 0`), advantage được đặt bằng 0. Loss dùng clipped surrogate theo tỷ lệ giữa policy hiện tại và policy cũ; có thể thêm KL penalty so với reference policy bằng `--kl-coeff`.

Các trajectory được lấy mẫu trước khi cập nhật. Mỗi bước hành động trong trajectory dùng chung advantage episode-level của trajectory đó.

### 4.3 Kiểm chứng lời giải

`verify_trajectory` tái mô phỏng các phép tính, kiểm tra số học, số lần sử dụng các số, luật dùng hết số và việc đạt Target. Một trajectory không hợp lệ không được xem là lời giải thành công chỉ vì policy dự đoán như vậy.

### 4.4 Dữ liệu

`make_data_splits` tạo các tập train, validation và test không giao nhau. Checkpoint lưu các split để đánh giá lại trên cùng tập test sau huấn luyện.

## 5. Cấu trúc các thành phần

| Thành phần | Vai trò |
|---|---|
| `main.py` | Entry point; cung cấp các chế độ `train`, `eval`, `solve` và giao diện giải bài |
| `grpo_agent.py` | Policy, sinh trajectory, verifier, reward, advantage, loss, huấn luyện, đánh giá và checkpoint |
| `countdown_env.py` | Luật hành động, chuyển trạng thái và reward theo bước |
| `tree_visualizer.py` | Ghi lại các chuyển trạng thái trong lúc giải và xuất Graphviz DOT |
| `test_grpo.py` | Kiểm thử policy, verifier, loss, cập nhật tham số, checkpoint và tích hợp |
| `input.txt` | Puzzle đầu vào mặc định cho chế độ giải |
| `grpo_checkpoint/` | Checkpoint mặc định của policy GRPO |

Checkpoint GRPO gồm `policy.pt`, `reference.pt`, `optimizer.pt` và `meta.json`. Có thể đổi thư mục checkpoint bằng `--grpo-file`.

## 6. Cài đặt

Yêu cầu Python 3.8 trở lên. Chương trình cần Matplotlib cho entry point và PyTorch cho policy GRPO.

```bash
python -m pip install matplotlib
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
```

Nếu muốn dùng CUDA, hãy cài bản PyTorch tương ứng với phiên bản CUDA trên máy theo hướng dẫn cài đặt chính thức của PyTorch.

## 7. Huấn luyện, đánh giá và giải bài

Chạy các lệnh từ thư mục chứa `main.py`.

### Huấn luyện

```bash
python main.py --mode train \
  --train-instances 150 \
  --epochs 30 \
  --group-size 8 \
  --seed 7
```

Quá trình này tạo split train/validation/test, huấn luyện policy và ghi checkpoint vào `grpo_checkpoint/`. Nếu thư mục đã có checkpoint, chương trình nạp policy hiện có để tiếp tục huấn luyện.

Một số tham số GRPO có thể điều chỉnh:

| Tham số | Mặc định | Ý nghĩa |
|---|---:|---|
| `--lr` | `0.003` | Learning rate |
| `--grpo-batch-size` | `4` | Số puzzle trong một batch cập nhật |
| `--group-size` | `8` | Số trajectory trong mỗi group |
| `--epochs` | `10` | Số epoch |
| `--inner-epochs` | `2` | Số lần cập nhật lại trên batch trajectory đã sinh |
| `--max-steps` | `16` | Giới hạn bước mỗi trajectory |
| `--clip-epsilon` | `0.2` | Hệ số clipping của surrogate objective |
| `--kl-coeff` | `0.0` | Hệ số KL so với reference policy |
| `--device` | `cpu` | Thiết bị PyTorch |

Ví dụ bật KL penalty:

```bash
python main.py --mode train --kl-coeff 0.01 --seed 7
```

### Đánh giá

```bash
python main.py --mode eval --best-of-n 16
```

Mặc định chương trình đánh giá trên tập test đã lưu trong checkpoint. `--best-of-n 1` đánh giá policy greedy; giá trị lớn hơn lấy mẫu nhiều trajectory rồi chọn kết quả tốt nhất. Để chỉ định checkpoint khác:

```bash
python main.py --mode eval \
  --grpo-file grpo_checkpoint_kl \
  --best-of-n 16
```

### Giải puzzle trong `input.txt`

```bash
python main.py --mode solve --no-gui
```

Chế độ này nạp checkpoint GRPO và ghi kết quả vào `ketqua.txt`. Bỏ `--no-gui` để mở giao diện Matplotlib. Mặc định phải dùng hết các số; thêm `--allow-partial` nếu muốn cho phép dừng khi đạt Target trước khi dùng hết số.

## 8. Kết quả và giới hạn

Các benchmark đã ghi nhận cho thấy policy GRPO học được tín hiệu hữu ích: trong thí nghiệm baseline, MAE trên bộ benchmark giảm từ khoảng `200.3` trước huấn luyện xuống `22.6` sau huấn luyện. Tuy nhiên, exact solve rate trên bộ 20 puzzle seed 999 vẫn thấp: greedy `0/20`, best-of-16 `1/20`.

Thí nghiệm dùng KL coefficient `0.01` làm chậm sự suy giảm entropy ở giai đoạn giữa, nhưng chưa cải thiện đáng kể exact solve rate held-out trong các kết quả hiện có. Kết quả phụ thuộc seed, split dữ liệu và cấu hình; các con số trên là kết quả của những lần chạy đã ghi nhận, không phải cam kết cho mọi lần huấn luyện.

Giới hạn đã quan sát:

- Policy có thể mất entropy khi huấn luyện dài, làm các trajectory trong group lặp lại nhiều và giảm lợi ích của best-of-N.
- Việc giảm sai số trung bình chưa đồng nghĩa với tăng tỷ lệ giải chính xác.
- Kết quả cần được đánh giá trên test held-out; không nên dùng chỉ số train để kết luận khả năng tổng quát hóa.

## 9. Kiểm thử

Chạy bộ kiểm thử GRPO:

```bash
python -m unittest test_grpo -v
```

Các kiểm thử bao gồm mã hóa đặc trưng, verifier, advantage, loss và gradient, thay đổi tham số policy sau cập nhật, lưu/nạp checkpoint, chia dữ liệu và các luồng tích hợp liên quan đến GRPO.
