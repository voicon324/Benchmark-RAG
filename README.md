# NewAIBench - Framework Đánh Giá Hệ Thống Truy Vấn AI

<div align="center">
  <img src="https://img.shields.io/badge/Python-3.8%2B-blue.svg" alt="Python Version">
  <img src="https://img.shields.io/badge/License-MIT-green.svg" alt="License">
  <img src="https://img.shields.io/badge/Framework-PyTorch%20%7C%20Transformers-orange.svg" alt="Framework">
  <img src="https://img.shields.io/badge/Status-Active-brightgreen.svg" alt="Status">
</div>

---

## 🎯 Tổng Quan

**NewAIBench** là framework đánh giá hiệu suất toàn diện cho các hệ thống truy vấn thông tin (Information Retrieval) với trọng tâm là dữ liệu tiếng Việt. Framework hỗ trợ đa dạng mô hình retrieval và cung cấp các metrics đánh giá chuẩn cho nghiên cứu và phát triển.

### 🚀 Tính Năng Chính

- **🔍 Đa Dạng Mô Hình**: Sparse (BM25), Dense (BERT, Sentence-BERT), Vision (CLIP), Multimodal
- **📊 Metrics Toàn Diện**: NDCG, MAP, Recall, Precision, MRR với k-values linh hoạt
- **🗂️ Hỗ Trợ Đa Dataset**: Text, Document Images, OCR, Multimodal
- **⚡ Tối Ưu Hóa**: Parallel processing, GPU acceleration, caching
- **🔧 Cấu Hình Linh Hoạt**: YAML configuration, CLI interface
- **📈 Báo Cáo Phong Phú**: CSV, JSON, visualization charts

---

## 🏗️ Kiến Trúc Hệ Thống

```
NewAIBench/
├── src/newaibench/           # Core framework
│   ├── datasets/             # Dataset loaders  
│   ├── models/               # Retrieval models
│   ├── evaluation/           # Metrics & evaluation
│   ├── experiment/           # Experiment runner
│   └── reporting/            # Results storage & reports
├── experiments/              # Configuration files
├── data/                     # Datasets
├── embedding_tools/          # OpenAI embedding utilities
└── results/                  # Experiment results
```

---

## 🛠️ Cài Đặt

### 1. Yêu Cầu Hệ Thống

- **Python**: 3.8+ (khuyến nghị 3.12)
- **GPU**: CUDA-compatible (optional, tăng tốc đáng kể)
- **Memory**: 8GB+ RAM (tùy thuộc dataset size)

### 2. Cài Đặt Môi Trường

```bash
# Clone repository
git clone https://github.com/voicon324/NewAIBench.git
cd NewAIBench

# Tạo virtual environment
python3.12 -m venv python312_venv
source python312_venv/bin/activate

# Cài đặt dependencies
pip install -r requirements.txt
```

### 3. Cấu Hình OpenAI API (Optional)

```bash
# Để sử dụng OpenAI embeddings
export OPENAI_API_KEY="sk-your-api-key-here"
```

---

## 📦 Datasets

### 📥 Tải Dataset

**[🔗 Download All Datasets](https://drive.google.com/drive/folders/1IqBPR17x44kLosQTr54kaJ89yzBUSS7e?usp=sharing)**

```bash
# Tạo thư mục data
mkdir -p data

# Giải nén datasets
unzip legal_data.zip -d data/
unzip NewAIBench_VietDocVQAII_with_OCR.zip -d data/
unzip tydiqa_goldp_vietnamese.zip -d data/
unzip UIT-ViQuAD2.0.zip -d data/
```

### 📊 Datasets Hỗ Trợ

| Dataset | Type | Size | Description |
|---------|------|------|-------------|
| **Legal Data** | Text | ~50K docs | Văn bản pháp luật Việt Nam |
| **VietDocVQA** | Multimodal | ~10K docs | Tài liệu có hình ảnh + OCR |
| **TyDi QA Vietnamese** | Text | ~18K docs | Q&A Wikipedia tiếng Việt |
| **UIT-ViQuAD 2.0** | Text | ~23K docs | Reading comprehension |

---

## 🚀 Sử Dụng Framework

### 1. Chạy Experiment Từ Config File

```bash
# Chạy experiment cơ bản
python run_experiment.py --config experiments/example.yaml

# Chạy với ColVintern model
python run_experiment.py --config experiments/colvintern.yaml

# So sánh OpenAI embeddings
python run_experiment.py --config experiments/openai_embedding_comparison.yaml
```

### 2. Cấu Hình Experiment (YAML)

```yaml
description: "Legal Data Retrieval Benchmark"

models:
  - name: "optimized_bm25"
    type: "optimized_sparse"
    parameters:
      k1: 1.6
      b: 0.75
      use_parallel_indexing: true
      use_caching: true
  
  - name: "vietnamese_sbert"
    type: "dense"
    model_name_or_path: "dangvantuan/vietnamese-embedding"
    parameters:
      normalize_embeddings: true
      use_ann_index: true

datasets:
  - name: "legal_data_small"
    type: "text"
    data_dir: "data/legal_data_small"

evaluation:
  metrics: ["ndcg", "map", "recall", "precision"]
  k_values: [1, 5, 10, 20, 50]
  top_k: 1000

output:
  output_dir: "./results"
  experiment_name: "legal_retrieval_benchmark"
```

### 3. Programmatic Usage

```python
from newaibench import (
    ExperimentRunner, 
    ExperimentConfig,
    TextDatasetLoader,
    OptimizedBM25Model
)

# Load dataset
config = DatasetConfig(data_dir="data/legal_data_small")
dataset = TextDatasetLoader(config)

# Initialize model
model = OptimizedBM25Model({
    'name': 'bm25_legal',
    'parameters': {'k1': 1.6, 'b': 0.75}
})

# Run evaluation
runner = ExperimentRunner(experiment_config)
results = runner.run()
```

---

## 🔧 Tạo Embeddings với OpenAI

```bash
# Chuyển đến embedding tools
cd embedding_tools

# Tạo embeddings cho dataset
python embed_dataset.py --dataset tydiqa_goldp_vietnamese --model text-embedding-3-large

# Sử dụng embeddings trong experiment
python run_experiment.py --config experiments/openai_embedding.yaml
```

---

## 🎯 Các Loại Mô Hình Hỗ Trợ

### 1. Sparse Models
- **BM25**: Chuẩn BM25 với tối ưu hóa
- **Optimized BM25**: Parallel processing, GPU acceleration
- **Custom Sparse**: Tùy chỉnh scoring function

### 2. Dense Models
- **Sentence-BERT**: Multilingual embeddings
- **Vietnamese BERT**: Specialized cho tiếng Việt
- **OpenAI Embeddings**: text-embedding-3-large/small
- **Custom Dense**: Tùy chỉnh encoder architecture

### 3. Vision Models
- **CLIP**: Multimodal text-image retrieval
- **OCR-based**: Text extraction + dense retrieval
- **ColVintern**: Specialized Vietnamese document understanding

### 4. Multimodal Models
- **OCR + Dense**: Combine text extraction with embeddings
- **Vision + Text**: Joint multimodal representations

---

## 📊 Metrics Đánh Giá

### Supported Metrics

| Metric | Description | Best Use Case |
|--------|-------------|---------------|
| **NDCG@k** | Normalized Discounted Cumulative Gain | Ranked retrieval |
| **MAP@k** | Mean Average Precision | Overall precision |
| **Recall@k** | Recall at rank k | Coverage evaluation |
| **Precision@k** | Precision at rank k | Accuracy evaluation |
| **MRR** | Mean Reciprocal Rank | First relevant result |

### Customizable Parameters

```yaml
evaluation:
  metrics: ["ndcg", "map", "recall", "precision", "mrr"]
  k_values: [1, 3, 5, 10, 20, 50, 100]
  top_k: 1000
  relevance_threshold: 1
  include_per_query: true
```

---

## 📈 Kết Quả và Báo Cáo

### 1. Cấu Trúc Kết Quả

```
results/
├── experiment_name/
│   ├── summary.json              # Tổng quan metrics
│   ├── detailed_results.json     # Chi tiết từng model
│   ├── cost_analysis.json        # Phân tích chi phí
│   └── runs/                     # Individual run data
```

### 2. Xuất Báo Cáo CSV

```bash
# Chuyển đổi kết quả thành CSV
python convert_results_to_csv.py --input results/

# Kết quả trong thư mục report_csv/
ls report_csv/
# tydiqa_goldp_vietnamese_recall_report.csv
# legal_data_recall_report.csv
```

### 3. Ví Dụ Kết Quả

| Model | Dataset | Recall@1 | Recall@5 | Recall@10 | Execution Time |
|-------|---------|----------|----------|-----------|----------------|
| BM25 | Legal Data | 0.607 | 0.800 | 0.834 | 3.39s |
| Vietnamese BERT | Legal Data | 0.416 | 0.611 | 0.673 | 28.15s |
| OpenAI Embedding | Legal Data | 0.523 | 0.702 | 0.745 | 45.20s |

---

## 🙏 Acknowledgments

- **OpenAI**: Embedding APIs
- **Hugging Face**: Transformers library
- **Sentence-Transformers**: Dense retrieval models
- **BEIR**: Benchmark framework inspiration
- **Vietnamese NLP Community**: Dataset contributions

---