# OpenAI Dataset Embedding

Scripts để tạo embeddings cho datasets sử dụng OpenAI API.

## Cài đặt

1. **Cài đặt dependencies:**
```bash
pip install -r requirements.txt
```

2. **Cấu hình OpenAI API Key:**
```bash
export OPENAI_API_KEY="sk-your-api-key-here"
```

Hoặc bạn có thể thêm vào file `.bashrc` hoặc `.zshrc` để tự động set mỗi khi mở terminal:
```bash
echo 'export OPENAI_API_KEY="sk-your-api-key-here"' >> ~/.bashrc
source ~/.bashrc
```

## Cấu trúc Dataset

Script này hỗ trợ datasets có cấu trúc:
```
data/
├── dataset_name/
│   ├── corpus.jsonl      # Văn bản corpus
│   ├── queries.jsonl     # Câu hỏi/queries
│   └── qrels.txt        # Relevance judgments (tùy chọn)
```

### Format dữ liệu:

**corpus.jsonl:**
```json
{"doc_id": "0", "text": "Văn bản tài liệu...", "title": "Tiêu đề", "metadata": {...}}
{"doc_id": "1", "text": "Văn bản tài liệu khác...", "title": "Tiêu đề", "metadata": {...}}
```

**queries.jsonl:**
```json
{"query_id": "q1", "text": "Câu hỏi 1?", "metadata": {...}}
{"query_id": "q2", "text": "Câu hỏi 2?", "metadata": {...}}
```

## Sử dụng

### 1. Chạy Demo Đơn Giản

```bash
python demo_embedding.py
```

Script này sẽ:
- Tự động tìm datasets trong thư mục `data/`
- Embedding dataset đầu tiên tìm được
- Lưu kết quả vào thư mục `embeddings/`
- Demo cách load và sử dụng embeddings

### 2. Embedding Dataset Cụ Thể

```bash
python embed_dataset.py --dataset tydiqa_goldp_vietnamese
```

### 3. Embedding Tất Cả Datasets

```bash
python embed_dataset.py
```

### 4. Tùy Chọn Nâng Cao

```bash
python embed_dataset.py \
  --dataset tydiqa_goldp_vietnamese \
  --model text-embedding-3-large \
  --dimensions 1024 \
  --batch-size 100 \
  --force-recompute
```

## Tham Số Command Line

| Tham số | Mô tả | Mặc định |
|---------|-------|----------|
| `--dataset` | Dataset cụ thể để embedding | Tất cả datasets |
| `--data-dir` | Thư mục chứa datasets | `./data` |
| `--output-dir` | Thư mục output | `./embeddings` |
| `--model` | Model embedding OpenAI | `text-embedding-3-small` |
| `--dimensions` | Số chiều embedding | Mặc định của model |
| `--batch-size` | Kích thước batch | 50 |
| `--force-recompute` | Tạo lại embeddings đã có | False |
| `--max-retries` | Số lần retry khi lỗi | 3 |

**Lưu ý:** API key được đọc từ environment variable `OPENAI_API_KEY`

## Models OpenAI

| Model | Dimensions | Performance | Cost |
|-------|------------|-------------|------|
| `text-embedding-3-small` | 1536 | Tốt | Thấp |
| `text-embedding-3-large` | 3072 | Xuất sắc | Cao |

## Cấu trúc Output

Sau khi chạy, embeddings sẽ được lưu theo cấu trúc:

```
embeddings/
├── dataset_name/
│   ├── corpus/
│   │   ├── 0.npy           # Embedding cho document ID 0
│   │   ├── 1.npy           # Embedding cho document ID 1
│   │   └── ...
│   ├── queries/
│   │   ├── q1.npy          # Embedding cho query ID q1
│   │   ├── q2.npy          # Embedding cho query ID q2
│   │   └── ...
│   └── embedding_metadata.json
```

## Sử dụng Embeddings

### Load Embedding:

```python
import numpy as np

# Load corpus embedding
corpus_embedding = np.load("embeddings/dataset_name/corpus/0.npy")

# Load query embedding  
query_embedding = np.load("embeddings/dataset_name/queries/q1.npy")

print(f"Corpus embedding shape: {corpus_embedding.shape}")
print(f"Query embedding shape: {query_embedding.shape}")
```

### Tính Similarity:

```python
import numpy as np

def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

# Tính similarity giữa query và corpus
similarity = cosine_similarity(query_embedding, corpus_embedding)
print(f"Cosine similarity: {similarity:.4f}")
```

### Load Tất Cả Embeddings:

```python
from pathlib import Path
import numpy as np

def load_all_embeddings(embeddings_dir, dataset_name):
    dataset_dir = Path(embeddings_dir) / dataset_name
    
    # Load corpus embeddings
    corpus_embeddings = {}
    corpus_dir = dataset_dir / "corpus"
    for file_path in corpus_dir.glob("*.npy"):
        doc_id = file_path.stem
        embedding = np.load(file_path)
        corpus_embeddings[doc_id] = embedding
    
    # Load query embeddings
    query_embeddings = {}
    queries_dir = dataset_dir / "queries"
    for file_path in queries_dir.glob("*.npy"):
        query_id = file_path.stem
        embedding = np.load(file_path)
        query_embeddings[query_id] = embedding
    
    return corpus_embeddings, query_embeddings

# Sử dụng
corpus_embs, query_embs = load_all_embeddings("embeddings", "tydiqa_goldp_vietnamese")
print(f"Loaded {len(corpus_embs)} corpus embeddings")
print(f"Loaded {len(query_embs)} query embeddings")
```

## Ước Tính Chi Phí

Chi phí OpenAI API (tham khảo 2024):

- **text-embedding-3-small**: $0.00002 / 1K tokens
- **text-embedding-3-large**: $0.00013 / 1K tokens

Ví dụ cho dataset 10K documents, mỗi document 500 tokens:
- Small model: ~$0.10
- Large model: ~$0.65

## Troubleshooting

### 1. Rate Limit Errors
Script tự động retry với exponential backoff. Nếu vẫn gặp lỗi, giảm `--batch-size`.

### 2. Memory Issues
Giảm `--batch-size` hoặc xử lý từng dataset riêng lẻ.

### 3. API Key Issues
Kiểm tra:
- API key đúng format
- Còn credit trong tài khoản OpenAI
- Network connection ổn định

### 4. Dataset Format Issues
Đảm bảo:
- Files `.jsonl` có format đúng
- Có field `doc_id`/`query_id` và `text`
- Encoding UTF-8

## Examples

### Dataset Tiếng Việt Hiện Có:

```bash
# Embedding tydiqa_goldp_vietnamese
python embed_dataset.py --api-key "sk-..." --dataset tydiqa_goldp_vietnamese

# Embedding UIT-ViQuAD2.0  
python embed_dataset.py --api-key "sk-..." --dataset UIT-ViQuAD2.0

# Embedding legal_data
python embed_dataset.py --api-key "sk-..." --dataset legal_data
```

### Kiểm Tra Kết Quả:

```bash
# Xem số lượng files embedding được tạo
ls -la embeddings/tydiqa_goldp_vietnamese/corpus/ | wc -l
ls -la embeddings/tydiqa_goldp_vietnamese/queries/ | wc -l

# Xem metadata
cat embeddings/tydiqa_goldp_vietnamese/embedding_metadata.json
```

## Tích Hợp Với NewAIBench

Sau khi có embeddings, bạn có thể tích hợp với các model retrieval trong NewAIBench để so sánh performance.
