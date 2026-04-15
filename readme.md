# RAG Chatbot - FIT HCMUS

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?logo=python&logoColor=white)
![React](https://img.shields.io/badge/React-18-61DAFB?logo=react&logoColor=black)
![FastAPI](https://img.shields.io/badge/FastAPI-0.115-009688?logo=fastapi&logoColor=white)
![LangChain](https://img.shields.io/badge/LangChain-1.x-1C3C3C?logo=langchain&logoColor=white)
![Qdrant](https://img.shields.io/badge/Qdrant-Vector%20DB-DC244C)
![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)

Chatbot tư vấn học vụ cho sinh viên Khoa Công Nghệ Thông Tin (FIT), trường Đại học Khoa Học Tự Nhiên - ĐHQG TP.HCM. Sử dụng kiến trúc RAG (Retrieval-Augmented Generation) với ReAct Agent, Memory, và Vector Database.

## System Design

```
┌─────────────────────────────────────────────────────────────────────┐
│                         FRONTEND (React + Vite)                     │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────────────┐   │
│  │ ChatBot  │  │ HomePage │  │ FAQPage  │  │   IssuePage      │   │
│  │  (Chat)  │  │(Landing) │  │  (FAQ)   │  │(Feedback/EmailJS)│   │
│  └────┬─────┘  └──────────┘  └──────────┘  └──────────────────┘   │
│       │ askRag(source, question)                                    │
└───────┼─────────────────────────────────────────────────────────────┘
        │ HTTP GET /rag/{source}?q=...&mode=...&session_id=...
        ▼
┌─────────────────────────────────────────────────────────────────────┐
│                      BACKEND (FastAPI + LangChain)                  │
│                                                                     │
│  ┌────────────────────────────────────────────────────────────────┐ │
│  │                     API Layer (api.py)                         │ │
│  │  GET /              GET /rag/{source}    GET /history          │ │
│  │  GET /sessions      DELETE /sessions/{id}                     │ │
│  └────────┬───────────────────────────────────────────────────────┘ │
│           │                                                         │
│  ┌────────▼───────────────────────────────────────────────────────┐ │
│  │                   LLMServe (llm_service.py)                   │ │
│  │                                                                │ │
│  │  mode=classic ──► Retrieve ──► Rerank ──► Generate            │ │
│  │                                                                │ │
│  │  mode=agentic ──► ReAct Agent (react_agent.py)                │ │
│  │                   ┌──────────────────────────┐                │ │
│  │                   │  THINK ──► ACT ──► OBSERVE│               │ │
│  │                   │      ↻ (loop max 5x)      │               │ │
│  │                   └──────────────────────────┘                │ │
│  └────────────────────────────────────────────────────────────────┘ │
│           │                        │                                │
│  ┌────────▼────────┐   ┌──────────▼─────────┐                     │
│  │  Tools (tools.py)│   │ Memory (memory.py) │                     │
│  │                  │   │                    │                      │
│  │ • qdrant_search  │   │ Buffer+Summary     │                     │
│  │ • fit_website    │   │ Hybrid (per session)│                    │
│  └───────┬──────────┘   └────────────────────┘                     │
│          │                                                          │
│  ┌───────▼──────────────────────────────────────┐                  │
│  │           External Services                   │                  │
│  │  ┌──────────┐  ┌───────────┐  ┌────────────┐ │                  │
│  │  │  Qdrant  │  │  Gemini   │  │ HuggingFace│ │                  │
│  │  │Vector DB │  │  LLM API  │  │ Embeddings │ │                  │
│  │  └──────────┘  └───────────┘  └────────────┘ │                  │
│  └───────────────────────────────────────────────┘                  │
│                                                                     │
│  ┌───────────────────────┐  ┌────────────────────┐                 │
│  │ ChatHistoryStore      │  │ Prometheus Metrics  │                 │
│  │ (SQLite)              │  │ (/metrics endpoint) │                 │
│  └───────────────────────┘  └────────────────────┘                 │
└─────────────────────────────────────────────────────────────────────┘
        │
        ▼
┌─────────────────────────────────────────────────────────────────────┐
│                     MONITORING (Docker)                              │
│  ┌──────────────┐         ┌──────────────────┐                     │
│  │  Prometheus   │────────►│     Grafana       │                    │
│  │  :9090        │         │     :3000         │                    │
│  └──────────────┘         └──────────────────┘                     │
└─────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────┐
│                     DATA PIPELINE                                   │
│  Database/*.txt ──► Loader ──► Splitter ──► Embeddings ──► Qdrant  │
│  (+ optional Web Crawler cho FIT HCMUS website)                    │
└─────────────────────────────────────────────────────────────────────┘
```

## RAG Workflow

### Classic Mode (`mode=classic`)
```
User Query
    │
    ▼
┌─────────────────┐
│ Dense Retrieval  │  Qdrant vector similarity (top_k=20)
│ (Embeddings)     │  Model: paraphrase-multilingual-mpnet-base-v2
└────────┬────────┘
         ▼
┌─────────────────┐
│   Reranking      │  Cosine similarity reranking (top_k=5)
│                  │  Query embedding vs doc embeddings
└────────┬────────┘
         ▼
┌─────────────────┐
│ Context Assembly │  Format top docs into prompt context
└────────┬────────┘
         ▼
┌─────────────────┐
│  LLM Generation  │  Gemini 2.0 Flash
│                  │  "Answer based only on context"
└────────┬────────┘
         ▼
    Final Answer + Source Documents
```

### Agentic Mode (`mode=agentic`) — ReAct Agent
```
User Query + Session Memory
    │
    ▼
┌──────────────────────────────────────────────────┐
│              ReAct Agent (LangGraph)              │
│                                                   │
│  System Prompt: "Trợ lý tư vấn học vụ FIT HCMUS" │
│                                                   │
│  ┌─────────────────────────────────────────────┐  │
│  │ Step 1: THINK                               │  │
│  │ "Cần tìm thông tin về điều kiện tốt nghiệp" │  │
│  └──────────────────┬──────────────────────────┘  │
│                     ▼                              │
│  ┌─────────────────────────────────────────────┐  │
│  │ Step 2: ACT — Call qdrant_search            │  │
│  │ Input: "điều kiện tốt nghiệp ngành CNTT"   │  │
│  └──────────────────┬──────────────────────────┘  │
│                     ▼                              │
│  ┌─────────────────────────────────────────────┐  │
│  │ Step 3: OBSERVE                             │  │
│  │ "[Doc 1] Cần 130 tín chỉ, GPA >= 2.0..."  │  │
│  └──────────────────┬──────────────────────────┘  │
│                     ▼                              │
│  ┌─────────────────────────────────────────────┐  │
│  │ Step 4: THINK                               │  │
│  │ "Đã đủ thông tin, tạo câu trả lời"         │  │
│  └──────────────────┬──────────────────────────┘  │
│                     ▼                              │
│  ┌─────────────────────────────────────────────┐  │
│  │ Step 5: ANSWER                              │  │
│  │ "Để tốt nghiệp ngành CNTT, sinh viên..."   │  │
│  └─────────────────────────────────────────────┘  │
│                                                   │
│  Tools:                                           │
│  • qdrant_search — Vector DB (nội bộ)             │
│  • fit_website_search — FIT HCMUS website         │
│                                                   │
│  Max iterations: 5                                │
└──────────────────────┬───────────────────────────┘
                       ▼
              Memory Update
    ┌──────────────────────────────┐
    │ Buffer+Summary Hybrid       │
    │ • 3 turns gần nhất: giữ     │
    │   nguyên văn                 │
    │ • Turns cũ: LLM tóm tắt    │
    │ • TTL: 30 phút per session  │
    └──────────────────────────────┘
                       ▼
         Final Answer + Sources + Confidence
```

### Data Pipeline
## Tech Stack

| Layer | Technology |
|-------|-----------|
| Frontend | React 18 + Vite + TailwindCSS/DaisyUI |
| Backend | Python, FastAPI, LangChain 1.x, LangGraph |
| Agent | LangGraph ReAct Agent |
| LLM | Google Gemini 2.0 Flash |
| Embeddings | sentence-transformers/paraphrase-multilingual-mpnet-base-v2 |
| Vector DB | Qdrant |
| Memory | Buffer+Summary Hybrid (in-memory, per session) |
| Chat History | SQLite |
| Monitoring | Prometheus + Grafana |

## Project Structure

```
RAG_Chatbot/
├── back-end/
│   ├── app/
│   │   ├── api.py              # FastAPI endpoints
│   │   ├── config.py           # Configuration management
│   │   ├── llm_service.py      # LLM + retrieval orchestration
│   │   ├── react_agent.py      # ReAct agent (LangGraph)
│   │   ├── tools.py            # Agent tools (qdrant, website)
│   │   ├── memory.py           # Session memory store
│   │   └── history_store.py    # SQLite chat history
│   ├── tests/                  # Unit tests (79 tests)
│   ├── main.py                 # Entry point
│   └── requirements.txt
├── front-end/
│   ├── src/
│   │   ├── components/ChatBot.jsx
│   │   ├── services/chatApi.js
│   │   └── ...
│   └── package.json
├── Data/
│   ├── Database/               # Training documents (.txt)
│   ├── pipeline/               # ETL pipeline modules
│   └── run_pipeline.py         # CLI entry point
├── monitoring/
│   ├── docker-compose.monitoring.yml
│   ├── prometheus/
│   └── grafana/
└── eval/                       # Evaluation framework
```

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | `/` | Health check |
| GET | `/rag/{source}` | RAG query (source: qdrant, fit_web, auto) |
| GET | `/history` | Chat history (limit=50) |
| GET | `/history/{id}` | Specific history entry |
| GET | `/sessions` | Active memory sessions |
| DELETE | `/sessions/{id}` | Clear session memory |

**Query Parameters** for `/rag/{source}`:
- `q` (required) — Query text
- `mode` — `classic` or `agentic` (default: classic)
- `session_id` — Session ID for memory continuity
- `debug` — Include thought_process in response

---

## Cài đặt

### Backend
```bash
cd back-end
cp .env.example .env     # Điền GOOGLE_API_KEY, QDRANT_URL, QDRANT_API_KEY, HUGGINGFACE_API_KEY
pip install -r requirements.txt
python main.py           # http://127.0.0.1:8000
```

### Frontend
```bash
cd front-end
cp .env.example .env     # Điền VITE_API_BASE_URL
npm install
npm run dev              # http://localhost:5173
```

### Data Pipeline
```bash
cd Data
cp .env.example .env     # Điền QDRANT_URL, QDRANT_API_KEY, HUGGINGFACE_API_KEY
python run_pipeline.py   # Build vector index
```

### Monitoring
```bash
docker compose -f monitoring/docker-compose.monitoring.yml up -d
# Prometheus: http://localhost:9090
# Grafana: http://localhost:3000 (admin/admin)
```

### Tests
```bash
cd back-end
pip install pytest
python -m pytest tests/ -v    # 79 tests
```

## Demo

[YouTube Demo](https://www.youtube.com/watch?v=EotYfkb3Oh4&feature=youtu.be)

---

## Contributing

1. Fork repo
2. Tao branch moi (`git checkout -b feature/ten-feature`)
3. Commit thay doi (`git commit -m "feat: mo ta thay doi"`)
4. Push branch (`git push origin feature/ten-feature`)
5. Tao Pull Request

Vui long dam bao chay tests truoc khi tao PR:
```bash
cd back-end && python -m pytest tests/ -v
```

## Tai lieu tham khao (References)

### Papers

| Paper | Tac gia | Mo ta |
|-------|---------|-------|
| [Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks](https://arxiv.org/abs/2005.11401) | Lewis et al., 2020 | Kien truc RAG goc — ket hop retrieval voi generation |
| [ReAct: Synergizing Reasoning and Acting in Language Models](https://arxiv.org/abs/2210.03629) | Yao et al., 2022 | ReAct Agent — vong lap Think-Act-Observe |
| [Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks](https://arxiv.org/abs/1908.10084) | Reimers & Gurevych, 2019 | Sentence embeddings cho semantic search |

### Documentation

| Cong nghe | Tai lieu |
|-----------|---------|
| LangChain | [python.langchain.com/docs](https://python.langchain.com/docs/introduction/) |
| LangGraph | [langchain-ai.github.io/langgraph](https://langchain-ai.github.io/langgraph/) |
| FastAPI | [fastapi.tiangolo.com](https://fastapi.tiangolo.com/) |
| Qdrant | [qdrant.tech/documentation](https://qdrant.tech/documentation/) |
| Google Gemini API | [ai.google.dev/docs](https://ai.google.dev/docs) |
| Sentence-Transformers | [sbert.net](https://www.sbert.net/) |
| React | [react.dev](https://react.dev/) |
| Vite | [vite.dev/guide](https://vite.dev/guide/) |
| TailwindCSS | [tailwindcss.com/docs](https://tailwindcss.com/docs/) |
| DaisyUI | [daisyui.com](https://daisyui.com/) |
| Prometheus | [prometheus.io/docs](https://prometheus.io/docs/) |
| Grafana | [grafana.com/docs](https://grafana.com/docs/) |

## Acknowledgments

- **Khoa Cong nghe Thong tin (FIT)** — Truong Dai hoc Khoa Hoc Tu Nhien, DHQG TP.HCM
- Cac du lieu huan luyen duoc thu thap tu [fit.hcmus.edu.vn](https://www.fit.hcmus.edu.vn/)

## License

Du an nay duoc phat hanh theo [MIT License](LICENSE).

```
MIT License - Copyright (c) 2023 phatjk
```
