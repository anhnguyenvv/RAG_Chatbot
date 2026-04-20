# RAG Chatbot - FIT HCMUS

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?logo=python&logoColor=white)
![React](https://img.shields.io/badge/React-18-61DAFB?logo=react&logoColor=black)
![FastAPI](https://img.shields.io/badge/FastAPI-0.115-009688?logo=fastapi&logoColor=white)
![LangChain](https://img.shields.io/badge/LangChain-1.x-1C3C3C?logo=langchain&logoColor=white)
![Qdrant](https://img.shields.io/badge/Qdrant-Vector%20DB-DC244C)
![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)

Chatbot tu van hoc vu cho sinh vien Khoa Cong Nghe Thong Tin (FIT), truong Dai hoc Khoa Hoc Tu Nhien - DHQG TP.HCM. Su dung kien truc RAG (Retrieval-Augmented Generation) voi ReAct Agent, Memory, va Vector Database.

## System Design

```mermaid
graph TB
    subgraph Frontend["FRONTEND (React + Vite)"]
        ChatBot["ChatBot.jsx"]
        HomePage["HomePage.jsx"]
        FAQPage["FAQPage.jsx"]
        IssuePage["IssuePage.jsx"]
    end

    ChatBot -->|"GET /rag/{source}?q=...&mode=...&session_id=..."| API

    subgraph Backend["BACKEND (FastAPI)"]
        API["routes.py<br/><small>Rate Limiting (slowapi)</small>"]
        RAGSvc["RAGService<br/><small>rag_service.py</small>"]
        API --> RAGSvc

        subgraph ClassicPipe["Classic Pipeline"]
            Retriever["RetrieverManager<br/><small>retriever.py</small>"]
            Reranker["Reranker<br/><small>reranker.py</small>"]
            Generator["Generator<br/><small>generator.py</small>"]
            Retriever --> Reranker --> Generator
        end

        subgraph AgenticPipe["Agentic Pipeline"]
            Agent["ReactRAGAgent<br/><small>agent.py (LangGraph)</small>"]
            Tools["Tools<br/><small>qdrant_search | fit_website_search</small>"]
            Memory["MongoSessionMemoryStore<br/><small>memory.py</small>"]
            Agent --> Tools
            Agent --> Memory
        end

        RAGSvc -->|"mode=classic"| Retriever
        RAGSvc -->|"mode=agentic"| Agent

        History["ChatHistoryStore<br/><small>SQLite</small>"]
        Metrics["Prometheus<br/><small>/metrics</small>"]
    end

    subgraph External["EXTERNAL SERVICES"]
        Qdrant["Qdrant<br/>Vector DB"]
        Gemini["Google Gemini<br/>LLM API"]
        HF["HuggingFace<br/>Embeddings"]
        MongoDB["MongoDB<br/>Session Memory"]
    end

    Retriever --> Qdrant
    Tools --> Qdrant
    Generator --> Gemini
    Agent --> Gemini
    Retriever --> HF
    Memory --> MongoDB

    subgraph Monitoring["MONITORING (Docker)"]
        Prom["Prometheus :9090"]
        Graf["Grafana :3000"]
        Prom --> Graf
    end

    Metrics --> Prom

    subgraph Pipeline["DATA PIPELINE"]
        cralw["crawl_fit_pdfs.py<br/><small>FIT PDF crawler</small>"]
        ocr["llm_ocr_pdf.py<br/><small>OCR - legacy</small>"]
        TXT["Database/*.txt"]
        Loader["loaders.py"]
        Splitter["splitters.py"]
        Embed["embeddings.py"]
        cralw-->ocr-->TXT --> Loader --> Splitter --> Embed --> Qdrant
    end
```


## Project Structure

```
RAG_Chatbot/
├── AGENTS.md
├── CLAUDE.md
├── LICENSE
├── readme.md
├── back-end/
│   ├── main.py
│   ├── requirements.txt
│   ├── app/
│   │   ├── api/routes.py
│   │   ├── config/
│   │   │   ├── config.py
│   │   │   └── prompts.py
│   │   ├── core/
│   │   │   ├── dependencies.py
│   │   │   └── memory_agent.py
│   │   ├── rag/
│   │   │   ├── agent.py
│   │   │   ├── retriever.py
│   │   │   ├── reranker.py
│   │   │   ├── generator.py
│   │   │   ├── llm.py
│   │   │   └── tools.py
│   │   ├── services/rag_service.py
│   │   └── storage/
│   │       ├── history.py
│   │       └── memory.py
│   └── tests/
├── front-end/
│   ├── package.json
│   ├── src/
│   │   ├── main.jsx
│   │   ├── components/
│   │   ├── pages/
│   │   └── services/
├── Data/
│   ├── run_pipeline.py
│   ├── requirements.txt
│   ├── Database/
│   └── pipeline/
├── docs/
├── eval/
└── monitoring/
```


**Query Parameters** for `/rag/{source}`:

- `q` (required) — Query text
- `mode` — `classic` or `agentic` (default: classic)
- `session_id` — Session ID for memory continuity (agentic mode)
- `debug` — Include thought_process in response

---

## Cai dat

### Backend
```bash
cd back-end
cp .env.example .env     # Dien GOOGLE_API_KEY, QDRANT_URL, QDRANT_API_KEY
pip install -r requirements.txt
python main.py           # http://127.0.0.1:8000
```

### Frontend
```bash
cd front-end
cp .env.example .env     # Dien VITE_API_BASE_URL
npm install
npm run dev              # http://localhost:5173
```

### Data Pipeline
```bash
cd Data
cp .env.example .env     # Dien QDRANT_URL, QDRANT_API_KEY, HUGGINGFACE_API_KEY
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
python -m pytest tests/ -v    # 78 test cases
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
