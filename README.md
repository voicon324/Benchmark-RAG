# NewAIBench - Information Retrieval Benchmark Framework

## Tổng quan

NewAIBench là một framework benchmark chuyên dụng cho việc đánh giá các mô hình Information Retrieval (IR) trong môi trường "zero-shot". Framework hỗ trợ đồng thời hai loại retrieval chính:

- **Text-based Retrieval**: Tìm kiếm văn bản thuần túy
- **Document Image Retrieval**: Tìm kiếm trên tài liệu ảnh với OCR và image embedding

## Tính năng chính

### 🎯 Multi-modal Retrieval Support
- Text retrieval với sparse (BM25, TF-IDF) và dense models (BERT, DPR)
- Document image retrieval với OCR và vision-language models
- Hybrid retrieval kết hợp multiple modalities

### 🔧 Modular Architecture
- Plugin system cho models, datasets, và evaluation metrics
- Easy integration của custom models và datasets
- Standardized API cho consistent workflow

### 📊 Comprehensive Evaluation
- Standard IR metrics: nDCG@k, MAP@k, Recall@k, MRR@k, P@k
- Per-query analysis và statistical significance testing
- Automated report generation với visualization

### ⚡ High Performance
- Batch processing để optimize throughput
- GPU acceleration support
- Distributed computing cho large-scale experiments
- Memory-efficient processing cho large corpora

### 🛠 Developer Friendly
- Type-safe design với Python dataclasses
- Comprehensive configuration system
- Extensive documentation và examples
- CI/CD ready với testing framework

## Project Structure

```
new_bench/
├── src/
│   └── newai_bench/
│       ├── __init__.py
│       ├── config/              # Configuration management
│       │   ├── __init__.py
│       │   ├── experiment.py    # Experiment configuration
│       │   ├── model.py         # Model configuration
│       │   └── dataset.py       # Dataset configuration
│       ├── datasets/            # Dataset loading and processing
│       │   ├── __init__.py
│       │   ├── base.py          # Base dataset classes
│       │   ├── loaders/         # Format-specific loaders
│       │   ├── processors/      # Data preprocessing
│       │   └── validators/      # Data validation
│       ├── models/              # Model interfaces and implementations
│       │   ├── __init__.py
│       │   ├── base.py          # Base model interface
│       │   ├── text/            # Text retrieval models
│       │   ├── vision/          # Vision-based models
│       │   ├── multimodal/      # Multimodal models
│       │   └── registry.py      # Model registry
│       ├── indexing/            # Indexing engines
│       │   ├── __init__.py
│       │   ├── text_index.py    # Text indexing (Elasticsearch, Whoosh)
│       │   ├── vector_index.py  # Dense vector indexing (FAISS)
│       │   └── hybrid_index.py  # Multi-modal indexing
│       ├── retrieval/           # Search and retrieval engines
│       │   ├── __init__.py
│       │   ├── search_engine.py # Main search orchestrator
│       │   ├── query_processor.py # Query preprocessing
│       │   └── rankers.py       # Result ranking and fusion
│       ├── evaluation/          # Evaluation metrics and analysis
│       │   ├── __init__.py
│       │   ├── metrics.py       # IR metrics implementation
│       │   ├── evaluator.py     # Evaluation engine
│       │   └── analysis.py      # Statistical analysis
│       ├── experiments/         # Experiment orchestration
│       │   ├── __init__.py
│       │   ├── runner.py        # Main experiment runner
│       │   ├── scheduler.py     # Experiment scheduling
│       │   └── monitor.py       # Progress monitoring
│       ├── storage/             # Result storage and management
│       │   ├── __init__.py
│       │   ├── database.py      # Database operations
│       │   ├── models.py        # SQLAlchemy models
│       │   └── exports.py       # Export functionality
│       ├── reporting/           # Report generation
│       │   ├── __init__.py
│       │   ├── generators.py    # Report generators
│       │   ├── visualizers.py   # Visualization tools
│       │   └── templates/       # Report templates
│       └── utils/               # Utility functions
│           ├── __init__.py
│           ├── logging.py       # Logging configuration
│           ├── device.py        # Device management
│           └── reproducibility.py # Seed and env management
├── tests/                       # Test suite
│   ├── unit/                    # Unit tests
│   ├── integration/             # Integration tests
│   ├── performance/             # Performance tests
│   └── fixtures/                # Test data and fixtures
├── examples/                    # Usage examples
│   ├── quick_start.py           # Basic usage example
│   ├── text_retrieval.py       # Text retrieval example
│   ├── image_retrieval.py      # Image retrieval example
│   ├── custom_model.py         # Custom model integration
│   └── notebooks/              # Jupyter notebooks
├── docs/                        # Documentation
│   ├── user_guide/             # User documentation
│   ├── developer_guide/        # Developer documentation
│   ├── api_reference/          # API documentation
│   └── tutorials/              # Step-by-step tutorials
├── configs/                     # Example configurations
│   ├── experiments/            # Experiment configs
│   ├── models/                 # Model configs
│   └── datasets/               # Dataset configs
├── docker/                      # Docker configurations
│   ├── Dockerfile
│   ├── docker-compose.yml
│   └── requirements/
├── scripts/                     # Utility scripts
│   ├── setup_env.sh            # Environment setup
│   ├── download_datasets.py    # Dataset download
│   └── benchmark_models.py     # Pre-defined benchmarks
├── requirements/                # Dependencies
│   ├── base.txt               # Core dependencies
│   ├── dev.txt                # Development dependencies
│   ├── gpu.txt                # GPU-specific dependencies
│   └── optional.txt           # Optional dependencies
├── .github/                     # GitHub workflows
│   └── workflows/
├── setup.py                     # Package setup
├── pyproject.toml              # Modern Python packaging
├── README.md                   # Project README
├── LICENSE                     # License file
├── CHANGELOG.md                # Version history
├── CONTRIBUTING.md             # Contribution guidelines
└── .gitignore                  # Git ignore rules
```

## Design Documents

Các tài liệu thiết kế chi tiết được lưu trong project:

1. **[DESIGN_ARCHITECTURE.md](./DESIGN_ARCHITECTURE.md)**: Kiến trúc tổng thể và sơ đồ hệ thống
2. **[IMPLEMENTATION_PLAN.md](./IMPLEMENTATION_PLAN.md)**: Kế hoạch implementation chi tiết theo từng giai đoạn
3. **[DATA_STRUCTURES_API.md](./DATA_STRUCTURES_API.md)**: Cấu trúc dữ liệu và API specifications

## Technology Stack

### Core Technologies
- **Python 3.9+**: Primary programming language
- **PyTorch**: Deep learning framework
- **Transformers**: Hugging Face transformers library
- **FAISS**: Vector similarity search
- **Elasticsearch**: Text search engine

### ML/IR Libraries
- **Sentence-Transformers**: Sentence embedding models
- **PyTerrier**: IR framework integration
- **ir-measures**: IR evaluation metrics
- **scikit-learn**: ML utilities

### Data & Storage
- **Pandas**: Data manipulation
- **SQLAlchemy**: Database ORM
- **HDF5**: Large dataset storage
- **Pillow/OpenCV**: Image processing

### Infrastructure
- **Docker**: Containerization
- **Hydra**: Configuration management
- **Ray**: Distributed computing
- **pytest**: Testing framework

## Development Phases

### Phase 1: Core Infrastructure (2-3 weeks)
- [x] Project setup và foundation
- [x] Configuration system
- [x] Dataset Manager foundation
- [x] Model Interface design
- [ ] **Status**: Ready to implement

### Phase 2: Evaluation & Storage (1-2 weeks)
- [ ] Evaluation Module
- [ ] Result Storage system
- [ ] Basic reporting capabilities

### Phase 3: Image Retrieval (2-3 weeks)
- [ ] Image Document processing
- [ ] Vision model integration
- [ ] Multi-modal indexing
- [ ] Hybrid retrieval pipelines

### Phase 4: Advanced Features (2-3 weeks)
- [ ] Performance optimization
- [ ] Distributed processing
- [ ] Advanced evaluation metrics
- [ ] Plugin architecture

### Phase 5: Testing & Validation (1-2 weeks)
- [ ] Comprehensive testing
- [ ] Validation với published results
- [ ] Documentation và examples

## Quick Start (Preview)

```python
from newai_bench import ExperimentRunner
from newai_bench.config import ExperimentConfig

# Load experiment configuration
config = ExperimentConfig.from_yaml("configs/experiments/msmarco_comparison.yaml")

# Run experiment
runner = ExperimentRunner(config)
results = runner.run()

# Print results summary
for result in results:
    print(f"Model: {result.model_name}")
    print(f"nDCG@10: {result.metrics.ndcg_at_10:.4f}")
    print(f"Recall@1000: {result.metrics.recall_at_1000:.4f}")
```

## Contributing

Hiện tại project đang trong giai đoạn thiết kế. Contribution guidelines sẽ được update khi bắt đầu implementation.

## License

TBD - Sẽ được quyết định khi project ready for public release.

## Contact

- **Project Lead**: [Your Name]
- **Repository**: [GitHub URL when available]
- **Documentation**: [Docs URL when available]

---

**Note**: Project hiện đang trong giai đoạn 1 (Design & Planning). Implementation sẽ bắt đầu sau khi thiết kế được approve.
