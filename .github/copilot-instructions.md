# ComfyUI Image Scorer - Development Guide

## Project Overview

**Project Name:** ComfyUI Image Scorer  
**Location:** `custom_nodes/comfyui-image-scorer/` (inside ComfyUI installation)  
**Purpose:** Aesthetic and technical scoring system for AI-generated images with comparative analysis and data preparation tools

## Quick Start

### Environment Setup

```bash
# Navigate to project directory
cd e:\ComfyUI\custom_nodes\comfyui-image-scorer

# Virtual environment is at: e:\ComfyUI\.venv
# Activate it:
..\..\.venv\Scripts\activate

# Install/update dependencies
pip install -r requirements.txt

# Run tests
python -m pytest tests/ -v
```

### Project Structure

```
comfyui-image-scorer/
├── __init__.py                    # ComfyUI node registration
├── nodes/                         # ComfyUI node implementations
│   └── aesthetic_score/           # Main scoring node
├── shared/                        # Shared utilities
│   ├── config.py                 # Configuration management (AutoSaveDict)
│   ├── helpers.py                # Image/vector utilities
│   ├── image_analysis.py         # Image feature extraction (PIL-based)
│   ├── io.py                     # File I/O (JSONL, JSON, files)
│   ├── loaders/                  # Data loading modules
│   ├── paths.py                  # Path configuration
│   ├── training/                 # Training & analysis tools
│   │   ├── model_trainer.py      # PyTorch model training
│   │   ├── parameter_analysis.py # 2D scatter plots & correlations
│   │   └── plot.py               # Visualization utilities
│   ├── utils.py                  # General utilities
│   └── vectors/                  # Vector/embedding operations
│       ├── image_vector.py       # Image vector base class
│       ├── embedding_vector.py   # Text embedding vectors
│       ├── terms.py              # Text parsing & term extraction
│       └── vectors.py            # Main vector management
├── external_modules/
│   ├── step01ranking/            # Scoring UI & server
│   │   ├── cache.py              # SQLite cache for scores
│   │   ├── score_server.py       # Flask API server
│   │   ├── utils.py              # Helper functions
│   │   └── frontend/             # Web UI (HTML/CSS/JS)
│   ├── step02prepare/            # Data preparation
│   │   ├── full_data/
│   │   │   └── prepare_data.py   # Main data processing pipeline
│   │   └── text_data/
│   │       ├── prepare_text_data.py
│   │       └── text_processing.py
│   └── step03training/
│       └── full_data/
│           └── run.py            # Training pipeline entrypoint
├── tests/                        # Pytest test suite (50 tests)
├── config/                       # Configuration files
└── README.md                     # Project documentation
```

## Key Concepts

### Type Hints (Python 3.9+ Style)

All type hints follow Python 3.9+ style:
```python
# ✓ Correct (Python 3.9+)
def process(data: dict[str, Any]) -> list[str]:
    return list(data.keys())

# ✗ Old style (Python 3.8 and earlier)
from typing import Dict, List
def process(data: Dict[str, Any]) -> List[str]:
```

### Configuration System

Uses `AutoSaveDict` for persistent auto-saving configuration:
```python
from shared.config import config

# Access nested config
image_root = config["image_root"]
prepare_config = config["prepare"]  # Sub-config auto-saves on change
```

### Image Analysis Pipeline

1. **Image Discovery:** Recursively scan image directories
2. **Feature Extraction:** PIL-based analysis (contrast, sharpness, noise, colorfulness, etc.)
3. **Vector Generation:** Convert features to numerical vectors
4. **Scoring:** Apply trained model or manual scoring via UI

### Scoring System

- **Score Range:** 1-5 (floating point, recorded with precise decimal)
- **Effective Score:** Combines base score + modifier/10
  - Formula: `(score + modifier/10).toFixed(2)`
  - Example: score=3.5, modifier=2 → effective_score=3.52
- **Comparison Tracking:** Records how many times images were compared
- **Volatility:** Measures score stability across comparisons

### Data Preparation

**Full Pipeline (all steps in sequence):**
```bash
python -m external_modules.step02prepare.full_data.prepare_data
```

**Text-Only Mode (reprocess text without re-analyzing images):**
```bash
python -m external_modules.step02prepare.full_data.prepare_data --text-only
```

**Rebuild Scores Only:**
```bash
python -m external_modules.step02prepare.full_data.prepare_data --rebuild-scores
```

**With Limits & Incremental Processing:**
```bash
python -m external_modules.step02prepare.full_data.prepare_data --limit 1000 --steps
```

### File Format Reference

**vectors.jsonl**: Image feature vectors
```json
{"image": "path/to/image.png", "vector": [0.5, 0.3, ...], "metadata": {...}}
```

**scores.jsonl**: Score data with metadata
```json
{"image": "path/to/image.png", "score": 3.5, "modifier": 2, "comparison_count": 7}
```

**text_data.jsonl**: Text prompts and embeddings
```json
{"image": "path/to/image.png", "prompt": "...", "embedding": [...], "terms": }
```

## Web UI - Scoring Interface

**Server:** Starts automatically on `http://localhost:5001`

**Modes:**
- **Single:** Score one image at a time
- **Batch:** Score 30 images at once in a grid
- **Compare:** Direct comparison scoring between pairs
- **Gallery:** View scored images with filters and stats

**Filter Features:**
- Score range (1-5, default unfiltered)  
- Comparison count grouping
- Volatility ranges
- Advanced filters (hidden by default, toggle to show)

**Image Preview:**
- Thumbnail lazy-loading with IntersectionObserver
- Click image to expand in modal with full metadata
- Dark theme matching application design

## Running Tests

```bash
# All tests (50 total)
python -m pytest tests/ -v

# Specific test file
python -m pytest tests/test_text_parsing.py -v

# Specific test
python -m pytest tests/test_text_parsing.py::TestExtractTerms::test_plain_text -v
```

**Test Coverage:**
- Text parsing & term extraction (30+ tests)
- Weight calculation and validation (10+ tests)
- Integration scenarios with complex prompts (10+ tests)

## Common Tasks

### Adding a New ComfyUI Node

1. Create node class in `nodes/your_node/node.py`
2. Register in `__init__.py`:
```python
from .your_node.node import YourNodeClass

NODE_CLASS_MAPPINGS = {
    "YourNode": YourNodeClass,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "YourNode": "Your Node Display Name",
}
```

### Processing New Images

1. Place images in configured `image_root` directory
2. API automatically discovers and caches them
3. They appear in scoring UI as unscored

### Regenerating Training Data

```bash
# Full rebuild (remove old vectors first)
python external_modules/step02prepare/full_data/prepare_data.py --rebuild

# Incremental update (default)
python external_modules/step02prepare/full_data/prepare_data.py
```

### Analyzing Parameter Correlations

```bash
python shared/training/parameter_analysis.py \
  --vectors path/to/vectors.jsonl \
  --scores path/to/scores.jsonl \
  --output parameter_analysis
```

Generates:
- `analysis_report.json`: Correlation statistics
- `plot_dim_X_vs_score.png`: Scatter plots (top 4 correlated dimensions)

## Performance Notes

- **Lazy-Loading:** Images use IntersectionObserver for efficient DOM rendering
- **Pagination:** Gallery loads 20 items per page server-side
- **Thumbnail Caching:** Generated on-first-access, cached in temp directory
- **Database:** SQLite cache stores image metadata for fast queries
- **Type Hints:** All Python 3.9+ style for better IDE support

## Troubleshooting

### Tests Not Running
- Ensure `__init__.py` error handling is present (handle relative import errors during test collection)
- Tests run from project root: `cd custom_nodes/comfyui-image-scorer`

### Images Not Appearing in Gallery
- Check `image_root` configuration points to correct directory
- Verify cache database is updated: Check `/status` endpoint
- Force rescan by toggling to different mode and back

### New Images Don't Show After Scoring All
- Polling triggers auto-rescan when transitioning back to scoring modes
- Or manually trigger with `/status` endpoint call

## Entry Points

| Script | Purpose |
|--------|---------|
| `nodes/aesthetic_score/node.py` | ComfyUI node execution |
| `external_modules/step01ranking/score_server.py` | Flask web server (localhost:5001) |
| `external_modules/step02prepare/full_data/prepare_data.py` | Data pipeline |
| `external_modules/step03training/full_data/run.py` | PyTorch training |
| `shared/training/parameter_analysis.py` | 2D analysis & visualization |

## Additional Resources

- See [IMPLEMENTATION_GUIDE.md](IMPLEMENTATION_GUIDE.md) for detailed feature documentation
- See [SESSION_SUMMARY.md](SESSION_SUMMARY.md) for recent development history
- Run `pytest tests/ -v` for executable examples via test suite
