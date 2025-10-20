# CS-GY 6913 Assignment 2: Search System

## Data Requirements

This project uses the **MS MARCO Passage Ranking dataset** provided by Microsoft.

1. Navigate to the official MS MARCO website: [https://microsoft.github.io/msmarco/](https://microsoft.github.io/msmarco/)

2. Agree to the terms and conditions, locate the **Collection dataset download link** within the **"Passage Retrieval"** section.

3. Download the file named **`collection.tar.gz`**.

4. Extract the archive and ensure the file **`collection.tsv`** is available.

5. Place the extracted file in the following directory within your project root:
   ```plaintext
    project_root/
    ├── data/
    │   └── raw/
    │       └── collection.tsv (3.06GB)
    ├── indexer/
    ├── parser/
    ├── query/
    ├── scripts/
    ├── shared/
    ├── requirements.txt
    ├── README.md
    ```

## Setup

### 1. Create and activate a virtual environment

```bash
python3 -m venv .venv
source .venv/bin/activate   # on Windows: .venv\Scripts\activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

## Run

### Parser

```bash
python -m scripts.run_parser
```

### Indexer

```bash
python -m scripts.run_indexer
```

### Query

```bash
python -m scripts.run_query
```
