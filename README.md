# Ứng dụng Data Mining và Machine Learning trong Phân tích Rủi ro Tín dụng Cho vay Tiêu dùng

##  Giới thiệu
Dự án này được thực hiện trong khuôn khổ môn học **Khai phá dữ liệu (Data Mining)** tại **Đại học Kinh tế TP. Hồ Chí Minh (UEH)**.

Mục tiêu của dự án là **phân tích động học rủi ro tín dụng** trong cho vay tiêu dùng tín chấp thông qua:

- Phân tích hành vi quá hạn theo thời gian (DPD, Roll Rate, Vintage)
- Khai phá luật kết hợp (Association Rules) bằng thuật toán **ECLAT**
- Xây dựng và đánh giá các mô hình học máy dự báo rủi ro tín dụng

Kết quả nghiên cứu giúp **nhận diện sớm các yếu tố rủi ro**, hỗ trợ **quản trị danh mục tín dụng** và **ra quyết định cho vay**.

---

##  Nội dung chính của dự án
Dự án được triển khai theo **3 hướng phân tích chính**:

### Phân tích động học rủi ro tín dụng
- Days Past Due (DPD)
- Delinquency Moving Matrix (DMM)
- Flow Rate Matrix
- Monthly Roll Rate
- Vintage Analysis (30 DPD, 90 DPD)

 **Mục tiêu:** theo dõi sự dịch chuyển trạng thái nợ của các khoản vay theo thời gian.

---

### 2️.Khai phá luật kết hợp (Association Rule Mining)
- Chuẩn hóa dữ liệu giao dịch theo dạng **transaction**
- Sinh **frequent itemsets** bằng thuật toán **ECLAT**
- Phân tích:
  - Các thuộc tính thường đi kèm với **Tín dụng Tốt**
  - Các thuộc tính thường đi kèm với **Nợ xấu**
- Tính toán các chỉ số:
  - Support
  - Confidence

 **Mục tiêu:** phát hiện các tổ hợp đặc điểm khách hàng có rủi ro cao.

---

###  Mô hình học máy dự báo rủi ro
Các mô hình được huấn luyện và so sánh bao gồm:
- Logistic Regression
- Random Forest
- XGBoost
- Naive Bayes

**Kết quả đầu ra:**
- Dự báo xác suất chuyển từ **B0 → Nợ xấu**
- Phân nhóm khách hàng theo **Roll Score Bands (AAA → C)**

---

## Cấu trúc thư mục
- **Tiền xử lí/**: Thực hiện làm sạch, chuẩn hóa và tiền xử lý dữ liệu đầu vào.
- **SINH LUẬT KẾT HỢP/**: Khai phá luật kết hợp bằng thuật toán ECLAT và lưu trữ kết quả.
  - `SINH_LUAT_KET_HOP.ipynb`: Notebook sinh và phân tích luật kết hợp.
  - `Ket_qua_sinh_luat_ket_hop.csv`: File kết quả các luật kết hợp.
- **Train Models/**: Huấn luyện và đánh giá các mô hình học máy dự báo rủi ro tín dụng.
  - `Train_model.ipynb`: Huấn luyện mô hình nền.
  - `XGBoost.ipynb`: Huấn luyện mô hình XGBoost.
  - `Model_Evaluation.ipynb`: So sánh và đánh giá mô hình.
- **Credit Risk Dynamics Analysis.ipynb**: Notebook tổng hợp toàn bộ quy trình phân tích và trực quan hóa kết quả.
- **README.md**: Mô tả tổng quan dự án và hướng dẫn sử dụng.
  
## Dữ liệu sử dụng
Dự án sử dụng **4 bộ dữ liệu chính**:

| Dataset | Nội dung |
|-------|---------|
| demographic | Thông tin nhân khẩu học khách hàng |
| origin | Thông tin khởi tạo khoản vay |
| repayment | Lịch sử trả nợ & quá hạn |

 **Tổng dữ liệu sau khi gộp:** ~700,000 dòng – 52 thuộc tính.

---

## 🛠️ Công nghệ & Thư viện
- **Ngôn ngữ:** Python  
- **Xử lý dữ liệu:** Pandas, NumPy  
- **Trực quan hóa:** Matplotlib, Seaborn  
- **Data Mining:** ECLAT  
- **Machine Learning:** Scikit-learn, XGBoost  

---

## Kết quả nổi bật
- Xác định rõ các thuộc tính thường đi kèm với **nợ xấu**
- Phát hiện các **quy luật rủi ro có độ tin cậy cao**
- Mô hình học máy cho khả năng **phân biệt rủi ro tốt**
- Cung cấp **góc nhìn động học rủi ro theo thời gian**, không chỉ phân loại tĩnh

---

## 🎓 Nhóm thực hiện
- Nguyễn Vĩnh Sơn Đỉnh  
- Lê Vân Khánh  
- Phạm Minh Sơn  
- Đinh Thị Minh Tâm  
- Ngô Thanh Tâm  

**Giảng viên hướng dẫn:** Nguyễn Thành Huy  
**Trường:** Đại học Kinh tế TP. Hồ Chí Minh (UEH)

