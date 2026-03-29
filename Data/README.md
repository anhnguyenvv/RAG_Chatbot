# Datapipeline Architecture

Muc tieu: chuan hoa pipeline theo module ro rang de de bao tri va tai lap ket qua.

## Cau truc

- `pipeline/config.py`: quan ly env va validate input.
- `pipeline/loaders.py`: nap file text va chuan hoa metadata `source`.
- `pipeline/web_crawler.py`: crawl noi dung HTML tu website va tao `Document`.
- `pipeline/splitters.py`: chunking theo `recursive` hoac `outline`.
- `pipeline/embeddings.py`: tao embedding model client.
- `pipeline/vector_store.py`: upsert va retrieval tren Qdrant.
- `pipeline/pipeline.py`: orchestration ingest -> chunk -> embed -> index.
- `run_pipeline.py`: CLI chay end-to-end.

## Data source

Mac dinh pipeline doc du lieu tu:

- `./Database`

Co the bat them crawl web (fit.hcmus.edu.vn va courses.fit.hcmus.edu.vn) de nap du lieu online vao cung collection.

## Cach chay

1. Cai package:

```bash
pip install -r requirements.txt
```

1. Tao file `.env` tu `.env.example`.

1. Dien cac key bat buoc:

- `QDRANT_URL`
- `QDRANT_API_KEY`
- `HUGGINGFACE_API_KEY`

1. Tuy chon env cho crawl web:

- `ENABLE_WEB_CRAWL=false`
- `WEB_START_URLS=https://www.fit.hcmus.edu.vn/vn/Default.aspx?tabid=36,https://www.fit.hcmus.edu.vn/vn/Default.aspx?tabid=289,https://courses.fit.hcmus.edu.vn/q2a/`
- `WEB_ALLOWED_DOMAINS=www.fit.hcmus.edu.vn,courses.fit.hcmus.edu.vn`
- `WEB_MAX_PAGES=30`
- `WEB_TIMEOUT_SECONDS=15`

1. Chay pipeline:

```bash
python run_pipeline.py
```

1. Tuy chinh tham so khi can:

```bash
python run_pipeline.py --collection ITUS_e5_600v3 --chunk-strategy recursive --chunk-size 700 --chunk-overlap 150
```

1. Chay kem crawl web:

```bash
python run_pipeline.py --enable-web-crawl --web-start-urls "https://www.fit.hcmus.edu.vn/vn/Default.aspx?tabid=36,https://www.fit.hcmus.edu.vn/vn/Default.aspx?tabid=289,https://courses.fit.hcmus.edu.vn/q2a/" --web-allowed-domains "www.fit.hcmus.edu.vn,courses.fit.hcmus.edu.vn" --web-max-pages 80
```

## Luu y migration

- Notebook `Vector_Database.ipynb` va `Data_Analysis.ipynb` duoc giu lai de tham khao.
- Luong van hanh chinh nen chay tu `run_pipeline.py` de quan ly phien ban de dang hon.
