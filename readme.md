HƯỚNG DẪN CÀI ĐẶT VÀ SỬ DỤNG HỆ THỐNG CHATBOT RAG FITHCMUS

Back-end:
+ Bước 1: Mở folder 'back-end' bằng VS Code.
+ Bước 2: Tạo file '.env' trong folder 'back-end' (có thể copy từ file '.env.example').
+ Bước 3: Khai báo các biến bắt buộc: 'GOOGLE_API_KEY', 'QDRANT_URL', 'QDRANT_API_KEY', 'HUGGINGFACE_API_KEY'.
+ Bước 4: Chạy lệnh 'pip install -r requirements.txt' để cài đặt thư viện.
+ Bước 5: Chạy lệnh 'python main.py' để khởi động server FastAPI.
+ Bước 6: Truy cập địa chỉ 'http://127.0.0.1:8000' để kiểm tra API.
+ Ghi chú: File notebook 'ServerRAG_Gemini_flask_1_5.ipynb' vẫn được giữ lại để tham khảo/Colab demo, nhưng luồng vận hành chính đã chuyển sang kiến trúc module trong 'back-end/app'.

Front-end:
+ Bước 1: Mở folder ‘front-end’ bằng VS Code
+ Bước 2: Tạo file '.env' trong folder 'front-end' (có thể copy từ file '.env.example').
+ Bước 3: Thiết lập biến 'VITE_API_BASE_URL' trùng với địa chỉ back-end (ví dụ: 'http://127.0.0.1:8000').
+ Bước 4: Nếu dùng trang góp ý/báo lỗi, thiết lập thêm các biến EmailJS trong '.env'.
+ Bước 5: Chạy lệnh “npm install” để cài đặt toàn bộ package vào máy.
+ Bước 6: Chạy lệnh “npm run dev”, sau đó truy cập địa chỉ 'http://localhost:5173/' để sử dụng hệ thống.

DEMO: [Link](https://www.youtube.com/watch?v=EotYfkb3Oh4&feature=youtu.be)
