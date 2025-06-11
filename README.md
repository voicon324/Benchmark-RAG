
## 🚀 Tính năng chính

- **Hỗ trợ đa mô hình**: BM25, Dense Embeddings, Multimodal (văn bản + hình ảnh)
- **Đa dạng dataset**: Văn bản tiếng Việt, tài liệu có hình ảnh, và OCR
- **Đánh giá toàn diện**: NDCG, MAP, Recall, Precision, MRR

## 🛠️ Cài đặt

### 1. Clone repository

```bash
git clone https://github.com/voicon324/Benchmark-RAG.git
cd Benchmark-RAG
```

### 2. Tạo virtual environment

```bash
# Sử dụng Python 3.12 (khuyến nghị)
python3.12 -m venv python312_venv
source python312_venv/bin/activate

# Hoặc sử dụng conda
conda create -n Benchmark-RAG python=3.12
conda activate Benchmark-RAG
```

### 3. Cài đặt dependencies

```bash
pip install -r requirements.txt
```

## 📊 Datasets

### 📥 Download Datasets

Tất cả datasets có thể được tải về từ Google Drive:
**[Download All Datasets](https://drive.google.com/drive/folders/1IqBPR17x44kLosQTr54kaJ89yzBUSS7e?usp=sharing)**

Sau khi tải về, giải nén các file vào thư mục `data/`:
```bash
# Tạo thư mục data nếu chưa có
mkdir -p data

# Giải nén các datasets
unzip legal_data.zip -d data/
unzip NewAIBench_VietDocVQAII_with_OCR.zip -d data/
unzip tydiqa_goldp_vietnamese.zip -d data/
unzip UIT-ViQuAD2.0.zip -d data/
```

Framework hỗ trợ các datasets sau:

### 1. Legal Data (BKAI Law)
```
data/legal_data/
├── corpus.jsonl          # Văn bản pháp luật
├── queries.jsonl         # Câu hỏi truy vấn
└── qrels.txt            # Relevance judgments
```

### 2. Vietnamese Document VQA with OCR
```
data/NewAIBench_VietDocVQAII_with_OCR/
├── corpus.jsonl          # Mô tả tài liệu
├── queries.jsonl         # Câu hỏi
├── qrels.jsonl          # Relevance judgments
└── images/              # Hình ảnh tài liệu
    ├── 10041.png
    └── ...
```

### 3. TyDi QA Vietnamese
```
data/tydiqa_goldp_vietnamese/
├── corpus.jsonl
├── queries.jsonl
└── qrels.txt
```

### 4. UIT-ViQuAD 2.0
```
data/UIT-ViQuAD2.0/
├── corpus.jsonl
├── queries.jsonl
└── qrels.txt
```

## 🏃‍♂️ Cách chạy thử nghiệm

## Sử dụng Configuration File (Khuyến nghị)

```bash
# Chạy với config có sẵn
python run_experiment.py --config experiments/example.yaml

# Chạy với ColVintern model cho document images
python run_experiment.py --config experiments/colvintern.yaml
```
## ⚙️ Configuration

### Cấu trúc file config (YAML)

```yaml
description: "Mô tả thử nghiệm"

models:
  - name: "model_name"
    type: "sparse|dense|multimodal"
    model_name_or_path: "path/to/model"
    parameters:
      # Model-specific parameters
    device: "cpu|cuda|auto"
    batch_size: 32

datasets:
  - name: "dataset_name"
    type: "text|image"
    data_dir: "path/to/dataset"
    config_overrides:
      # Dataset-specific configurations

evaluation:
  metrics: ["ndcg", "map", "recall", "precision"]
  k_values: [1, 5, 10, 20, 50]
  top_k: 1000

output:
  output_dir: "./results"
  experiment_name: "experiment_name"
  log_level: "INFO"
```

### Các loại models được hỗ trợ

#### 1. Sparse Models (BM25)
```yaml
- name: "bm25_optimized"
  type: "sparse"
  parameters:
    k1: 1.6
    b: 0.75
    use_parallel_indexing: true
    use_caching: true
```

#### 2. Dense Models
```yaml
- name: "vietnamese_embedding"
  type: "dense"
  model_name_or_path: "dangvantuan/vietnamese-embedding"
  parameters:
    normalize_embeddings: true
    use_ann_index: true
    faiss_index_factory_string: "IVF100,Flat"
```

#### 3. Multimodal Models
```yaml
- name: "colvintern"
  type: "multimodal"
  model_name_or_path: "5CD-AI/ColVintern-1B-v1"
  parameters:
    scoring_method: "multi_vector"
    batch_size_images: 4
```

## 📈 Kết quả và Đánh giá

### Metrics được hỗ trợ:
- **NDCG@k**: Normalized Discounted Cumulative Gain
- **MAP@k**: Mean Average Precision
- **Recall@k**: Recall at rank k
- **Precision@k**: Precision at rank k
- **MRR**: Mean Reciprocal Rank

### Cấu trúc kết quả:
```
results/
├── experiment_name/
│   ├── summary.json          # Tổng quan kết quả
│   ├── detailed_results.json # Chi tiết từng model/dataset
│   ├── run_files/           # TREC run files
│   └── logs/               # Experiment logs
```
