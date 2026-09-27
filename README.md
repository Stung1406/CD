# Countdown Solver

Chương trình tìm cách ghép các số đã cho bằng các phép cộng, trừ, nhân, chia để thu được đúng `Target`. Chương trình in các bước giải, lưu kết quả vào `ketqua.txt` và mở cửa sổ mô phỏng từng bước.

## Cấu trúc thư mục

- `main.py`: đọc đề, tìm lời giải, in và lưu kết quả, hiển thị mô phỏng.
- `input.txt`: danh sách số và giá trị đích.
- `ketqua.txt`: lời giải gần nhất được chương trình tạo ra.

## Định dạng đầu vào

Ghi các số nguyên dương ở trước dòng `Target:`. Có thể cách nhau bằng dấu cách hoặc dấu câu; nên dùng định dạng đơn giản như sau:

```text
25 50 3 6 7 8
Target: 952
```

Chương trình nhận từ 2 đến 8 số. Nó đọc các chữ số xuất hiện trước dòng `Target:` thành danh sách số, vì vậy không nên đặt số thứ tự hay số liệu khác trong phần đó. Giá trị `Target` cũng phải là số nguyên dương.

## Luật tính toán trong chương trình

- Mỗi số đầu vào được sử dụng đúng một lần. Các số trùng nhau vẫn được giữ thành nhiều phần tử riêng.
- Mỗi phép tính lấy hai số hiện có, thay chúng bằng kết quả, rồi tiếp tục với danh sách mới.
- Chỉ dùng `+`, `-`, `*`, `/`.
- Phép trừ chỉ tạo kết quả dương: số lớn trừ số nhỏ.
- Phép chia chỉ được dùng khi chia hết; kết quả là số nguyên dương. Không có phân số trung gian.
- Mọi kết quả trung gian phải lớn hơn 0. Vì vậy không tạo số 0 hoặc số âm.
- Kết thúc khi chỉ còn một số và số đó bằng `Target`.

Các quy tắc trên là quy tắc được cài đặt trong `main.py`. Chúng hẹp hơn một số biến thể Countdown: chương trình không tìm nghiệm gần nhất, không cho phép bỏ số, và không cho phép chia ra phân số.

## Máy tính tìm lời giải như thế nào?

Hàm `countdown()` dùng tìm kiếm đệ quy:

1. Sắp xếp các số hiện có để tạo trạng thái chuẩn. Trạng thái ban đầu là toàn bộ các số trong đề.
2. Chọn lần lượt từng cặp số trong trạng thái.
3. Tạo các phép tính hợp lệ cho cặp đó: cộng, nhân, trừ theo chiều cho kết quả dương, và chia theo chiều chia hết.
4. Với mỗi phép tính, bỏ hai số đã dùng và thêm kết quả để tạo trạng thái tiếp theo; sau đó sắp xếp trạng thái này.
5. Gọi lại chính quy trình trên cho trạng thái mới. Nếu trạng thái chỉ còn một số, so sánh số đó với `Target`.
6. Khi một nhánh tìm thấy đích, trả về chuỗi phép tính của nhánh đó. Nếu tất cả nhánh đều thất bại, kết luận không tìm thấy lời giải chính xác.

Để tránh giải lại cùng một trường hợp, `lru_cache` ghi nhớ kết quả tìm kiếm theo trạng thái đã sắp xếp. Thứ tự các số không ảnh hưởng đến phép tính tiếp theo, nhưng số lần xuất hiện của từng số vẫn được giữ nguyên. Với mỗi cặp, các phép tính tạo cùng một kết quả cũng không cần thử lặp lại, vì chúng dẫn đến cùng trạng thái kế tiếp.

Chương trình trả về lời giải đầu tiên theo thứ tự duyệt, không đảm bảo đó là lời giải ngắn nhất hay đẹp nhất. Do số nhánh tăng nhanh theo số lượng đầu vào, tìm kiếm có thể tốn nhiều thời gian hơn với những bài lớn; giới hạn hiện tại là 8 số.

## Ví dụ với `input.txt` hiện tại

Với các số `25 50 3 6 7 8` và đích `952`, một lời giải là:

```text
3 + 7 = 10
6 * 8 = 48
50 - 10 = 40
25 * 40 = 1000
1000 - 48 = 952
```

Biểu thức đầy đủ tương ứng:

```text
(25 * (50 - (3 + 7))) - (6 * 8) = 952
```

Mỗi bước lấy hai số hiện có và thay bằng kết quả. Chẳng hạn, sau bước đầu danh sách số đang có tương đương với `25, 50, 6, 8, 10`.

## Chạy chương trình

Cần Python và các thư viện `matplotlib`, `networkx` (được import trong `main.py`). Cài thư viện nếu máy chưa có:

```bash
py -m pip install matplotlib networkx
```

Mở terminal tại thư mục `CD`, sau đó chạy:

```bash
py main.py
```

Chương trình sẽ in lời giải và thời gian tìm kiếm, ghi cùng thông tin vào `ketqua.txt`, rồi mở cửa sổ mô phỏng. Dùng nút **Trước** và **Tiếp** để xem từng phép tính. Đóng cửa sổ mô phỏng để kết thúc chương trình.

Nếu không tìm được lời giải chính xác, chương trình ghi thông báo tương ứng vào terminal và `ketqua.txt`; nó không tự tìm kết quả gần `Target`.