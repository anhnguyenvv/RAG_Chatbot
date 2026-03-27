## Front-end Architecture

Ung dung front-end duoc to chuc theo huong tach lop de de mo rong:

- `src/components`: Thanh phan UI tai su dung (NavBar, ChatBot,...).
- `src/pages`: Cac trang theo route (Home, FAQ, Issue).
- `src/services`: Tang giao tiep API (vi du `chatApi.js`).
- `src/config`: Cau hinh moi truong va bien Vite (`env.js`).
- `src/constants`: Du lieu tinh, cau hinh nho dung chung.

## Environment Setup

1. Tao file `.env` tu `.env.example`.
2. Cap nhat cac bien:

- `VITE_API_BASE_URL=http://127.0.0.1:8000`
- `VITE_ENABLE_NGROK_BYPASS_HEADER=true` neu dung URL ngrok.
- `VITE_EMAILJS_SERVICE_ID`, `VITE_EMAILJS_TEMPLATE_ID`, `VITE_EMAILJS_PUBLIC_KEY` neu su dung trang gui gop y.

## Development

1. Cai dependency:

```bash
npm install
```

2. Chay local:

```bash
npm run dev
```

## Build

```bash
npm run build
```


