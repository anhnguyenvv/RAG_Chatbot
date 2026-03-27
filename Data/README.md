# Datapipeline Architecture

Muc tieu: chuan hoa pipeline theo module ro rang de de bao tri va tai lap ket qua.

## Cau truc

- `pipeline/config.py`: quan ly env va validate input.
- `pipeline/loaders.py`: nap file text va chuan hoa metadata `source`.
- `pipeline/splitters.py`: chunking theo `recursive` hoac `outline`.
- `pipeline/embeddings.py`: tao embedding model client.
- `pipeline/vector_store.py`: upsert va retrieval tren Qdrant.
- `pipeline/pipeline.py`: orchestration ingest -> chunk -> embed -> index.
- `run_pipeline.py`: CLI chay end-to-end.

## Data source

Mac dinh pipeline doc du lieu tu:

- `./Database`

## Cach chay

1. Cai package:

```bash
pip install -r requirements.txt
```

2. Tao file `.env` tu `.env.example`.

3. Dien cac key bat buoc:

- `QDRANT_URL`
- `QDRANT_API_KEY`
- `HUGGINGFACE_API_KEY`

4. Chay pipeline:

```bash
python run_pipeline.py
```

5. Tuy chinh tham so khi can:

```bash
python run_pipeline.py --collection ITUS_e5_600v3 --chunk-strategy recursive --chunk-size 700 --chunk-overlap 150
```

## Luu y migration

- Notebook `Vector_Database.ipynb` va `Data_Analysis.ipynb` duoc giu lai de tham khao.
- Luong van hanh chinh nen chay tu `run_pipeline.py` de quan ly phien ban de dang hon.
