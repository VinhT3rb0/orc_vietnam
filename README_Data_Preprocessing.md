# Tài Liệu Giải Thích: Xử Lý Dữ Liệu Cho PaddleOCR

Tài liệu này giải thích chi tiết các bước tiền xử lý đã được thực hiện để chuyển đổi dữ liệu OCR ban đầu (raw data) thành định dạng chuẩn mà PaddleOCR yêu cầu để huấn luyện.

## 1. Phân Tích Dữ Liệu Ban Đầu
Dữ liệu nhãn (`labels/gt_*.txt`) của bạn được lưu dưới định dạng gồm 8 tọa độ và văn bản (kiểu dữ liệu khá giống chuẩn ICDAR2015):
```
x1, y1, x2, y2, x3, y3, x4, y4, noi_dung_van_ban
```
- Số đếm ở tên file nhãn (`gt_1.txt`) tương ứng với tên file ảnh (`im0001.jpg`).
- Biểu tượng `###` trong nhãn được sử dụng để đánh dấu cực mờ hoặc không đọc được. Cần bỏ qua không đưa vào huấn luyện mô hình nhận diện chữ (Recognition) để tránh nhiễu.

Tập ảnh nằm rải rác ở 3 thư mục:
- `train_images`: `im0001` - `im1200`
- `test_image`: `im1201` - `im1500`
- `unseen_test_images`: `im1501` - `im2000`

---

## 2. Bước 1: Chuẩn bị dữ liệu cho Mô Hình Phát Hiện Văn Bản (Text Detection)
**Mục đích:** Mô hình phát hiện chữ chỉ quan tâm đến vị trí xuất hiện của khối chữ trên tấm ảnh ban đầu (Tìm bounding box).
**Script thực hiện:** `preprocess_paddleocr.py`

### Các bước cụ thể script đã làm:
1. **Quét toàn bộ thư mục nhãn:** Đọc toàn bộ file trong `/labels`.
2. **Quy định tỷ lệ phân chia Dataset:** 
   Script sẽ gán các định dạng file tương ứng và trỏ thẳng vào các file nằm trong 3 thư mục có sẵn, bảo đảm chuẩn phân tách của bạn:
   - Các file từ `1-1200` tạo thành tập **Huấn Luyện (Train)**.
   - Các file từ `1201-1500` tạo thành tập **Kiểm Tra trong lúc học (Validation)**.
   - Các file từ `1501-2000` tạo thành tập **Đánh Giá cuối cùng (Test)**.
3. **Định dạng dữ liệu đầu ra:** 
   Khởi tạo các chuỗi `[path_to_image]\t[JSON_data]` và xuất ra file văn bản. File JSON lưu trữ danh sách các hình đa giác khung (points) và chữ (transcription).
4. **Kết quả tạo thành:** `train_det_list.txt`, `val_det_list.txt`, `test_det_list.txt`.

---

## 3. Bước 2: Chuẩn bị dữ liệu cho Mô Hình Nhận Diện Văn Bản (Text Recognition)
**Mục đích:** Mô hình nhận diện chữ lại được học dựa trên một nguyên lý khác: Cho nó 1 ảnh chỉ chứa mỗi dòng chữ thôi (đã bị crop), và một chuỗi từ khóa.
**Script thực hiện:** `crop_recognition_data.py`

Vì các khung chữ trong ảnh của bạn có thể xuất hiện nghiêng, xiên (theo 4 điểm polygon - đa giác), nên không thể dùng lệnh Crop góc vuông đơn giản.
### Các bước cụ thể script đã làm:
1. **Đọc tệp tin Detection:** Đọc file `_det_list.txt` vừa tạo ở bước trên để lấy đường dẫn và tọa độ khung.
2. **Loại bỏ chữ mờ nhiễu:** Bỏ qua những phần tử có `transcription = '###'`.
3. **Căn chỉnh và cắt biến đổi góc (Perspective Warp):**
   Sử dụng hàm `cv2.getPerspectiveTransform` và `cv2.warpPerspective` của bộ lọc OpenCV để cắt xén 4 góc và nắn (xoay thẳng trục) các vùng ảnh bị nghiêng tạo thành những bức ảnh cắt hình chữ nhật nằm thẳng. 
4. **Xuất file Crop:** Lưu tất cả mảnh hình nhận được vào một thư mục mới có tên là `archive\crop_images\`. Ví dụ mảnh cắt số 1 của ảnh `im001` sẽ là `im0001_crop_1.jpg`.
5. **Ghi định dạng đầu ra:** Ghi đường dẫn tới bức ảnh crop đó kèm nhãn. Yêu cầu của PaddleOCR rất đơn giản: `path/to/crop_img \t text`
6. **Kết quả tạo thành:** `train_rec_list.txt`, `val_rec_list.txt`, `test_rec_list.txt`.

---

## Tổng kết
Đến thời điểm này, sự lộn xộn trong chuỗi tọa độ định vị đã hoàn toàn bị loại bỏ. Dữ liệu của bạn được cấu trúc chính xác theo tài liệu chuẩn của thuật toán OCR mới nhất hiện nay. Bước tiếp theo hoàn toàn phụ thuộc vào việc cấu hình mạng Nơ-ron huấn luyện trên những `.txt` vừa sinh này.
