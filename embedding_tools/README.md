# Embedding Tools

This directory contains tools for creating embeddings using OpenAI API.

## Quick Start

1. **Set your OpenAI API key:**
```bash
export OPENAI_API_KEY="sk-your-api-key-here"
```

2. **Install dependencies:**
```bash
pip install aiohttp numpy tqdm
```

3. **Run embedding:**
```bash
python embed_dataset.py --dataset tydiqa_goldp_vietnamese
```

## Files

- `embed_dataset.py` - Main embedding script
- `demo_embedding.py` - Demo script showing usage
- `embedding_utils.py` - Utility functions for using embeddings in NewAIBench
- `test_embedding_setup.sh` - Setup validation script
- `EMBEDDING_README.md` - Detailed documentation
- `.env.example` - Environment variable template

## Usage Examples

```bash
# Embed specific dataset
python embed_dataset.py --dataset tydiqa_goldp_vietnamese

# Embed all datasets
python embed_dataset.py

# Run demo
python demo_embedding.py

# Test setup
bash test_embedding_setup.sh
```

For detailed documentation, see `EMBEDDING_README.md`.
