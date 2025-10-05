HƯỚNG DẪN CÀI ĐẶT VÀ SỬ DỤNG HỆ THỐNG CHATBOT RAG FITHCMUS

Back-end:
+ Bước 1: Người dùng mở file 'ServerRAG_Gemini_flask_1_5.ipynb' trong folder 'back-end' bằng Google Colab
+ Bước 2: Kiểm tra các biến môi trường, api keys có được khai báo đúng và khả dụng hay không, nếu api keys hết hạn mức trong ngày thì cần đổi sang api keys khác.
+ Bước 3: Bấm 'run all', sau khi cài đặt các thư viện, hệ thống sẽ yêu cầu khởi động lại phiên -> Bấm khởi động lại và tiếp tục 'run all'
+ Bước 4: Khi toàn bộ file được chạy và không gặp lỗi, server sẽ được mở tại địa chỉ: http://127.0.0.1:8000

Front-end:
+ Bước 1: Mở folder ‘front-end’ bằng VS Code
+ Bước 2: Trong file 'front-end\src\components\ChatBot.jsx' tìm đường dẫn có dạng 'https://xxxx-xxx-xxx.ngrok-free.app/rag/' xem có khớp với đường dẫn đã được cài đặt trong server back-end không. Nếu đường dẫn không trùng với back-end thì cập nhật lại đường dẫn.
+ Bước 3: Chạy lệnh “npm install” để cài đặt toàn bộ package vào máy.
+ Bước 4: Sau khi chạy sẽ hiển thị địa chỉ 'http://localhost:5173/', truy cập đường dẫn để mở giao diện hỏi đáp và sử dụng các tính năng của hệ thống.
