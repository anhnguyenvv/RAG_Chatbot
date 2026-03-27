# Back-end Architecture

Back-end da duoc tach khoi notebook thanh ung dung FastAPI module hoa.

## Cau truc

- `app/config.py`: nap env va doc chung config tu `Data/pipeline/config.py`.
- `app/llm_service.py`: quan ly embeddings, retriever, LLM, RAG chain.
- `app/api.py`: dinh nghia endpoint API (`/`, `/rag/{source}`).
- `app/history_store.py`: luu va truy van chat history bang SQLite.
- `main.py`: entrypoint chay uvicorn.
- `ServerRAG_Gemini_flask_1_5.ipynb`: notebook giu lai de tham khao va demo.

## Chay server

1. Cai package:

```bash
pip install -r requirements.txt
```

2. Tao `.env` tu `.env.example`.

3. Chay API:

```bash
python main.py
```

4. Test nhanh:

- `GET /`
- `GET /rag/qdrant?q=...`
- `GET /rag/wiki?q=...`
- `GET /history?limit=50`
- `GET /history/{entry_id}`

## Chat history database

- Back-end tu dong luu lich su hoi dap vao SQLite sau moi request `/rag/{source}`.
- Duong dan mac dinh: `./data/chat_history.db` (co the doi qua env `CHAT_HISTORY_DB_PATH`).

## Retrieval va reranker

- `RETRIEVAL_TOP_K`: so tai lieu lay ban dau tu retriever (mac dinh `20`).
- `RERANK_TOP_K`: so tai lieu giu lai sau rerank (mac dinh `5`).
- `ENABLE_RERANKER`: bat/tat reranker (`true/false`, mac dinh `true`).

Khi reranker bat, he thong embed cau hoi va danh sach tai lieu truy xuat,
sap xep lai theo do tuong dong cosine va chi dua `RERANK_TOP_K` tai lieu vao context sinh cau tra loi.

## Dong bo voi data pipeline

Back-end doc truc tiep `PipelineConfig` tu module `Data/pipeline/config.py`,
vi vay collection name va embedding model se dong bo voi pipeline index.
