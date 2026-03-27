## Front-end Architecture

Ứng dụng front-end được tổ chức theo hướng tách lớp để dễ mở rộng:

- `src/components`: Thành phần UI tái sử dụng (NavBar, ChatBot,...).
- `src/pages`: Các trang theo route (Home, FAQ, Issue).
- `src/services`: Tầng giao tiếp API (ví dụ `chatApi.js`).
- `src/config`: Cấu hình môi trường và biến Vite (`env.js`).
- `src/constants`: Dữ liệu tĩnh, cấu hình nhỏ dùng chung.

## Environment Setup

1. Tạo file `.env` từ `.env.example`.
2. Cập nhật các biến:

- `VITE_API_BASE_URL=http://127.0.0.1:8000`
- `VITE_ENABLE_NGROK_BYPASS_HEADER=true` nếu dùng URL ngrok.
- `VITE_EMAILJS_SERVICE_ID`, `VITE_EMAILJS_TEMPLATE_ID`, `VITE_EMAILJS_PUBLIC_KEY` nếu sử dụng trang gửi góp ý.

## Development

1. Cài dependency:


```bash
npm run dev
```

## Build

```bash
npm run build
```


