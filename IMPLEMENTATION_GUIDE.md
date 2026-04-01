# Implementation Guide - Remaining Tasks

**Created**: March 29, 2026
**Status**: Work in Progress
**Root Path**: `e:\ComfyUI\custom_nodes\comfyui-image-scorer`

## Summary of Completed Work

✅ **Task 1.1**: Updated `copilot-instructions.md` with current project info
✅ **Task 2.1** (Partial): Started type hints modernization (terms.py completed as sample)
✅ **Task 3.1 & 3.2**: Created Gallery view with filters:
   - `external_modules/step01ranking/frontend/html/gallery.html`
   - `external_modules/step01ranking/frontend/css/gallery.css`
   - `external_modules/step01ranking/frontend/js/gallery.js`
   - Updated `index.html` with Gallery button

✅ **Task 4.1**: Created comprehensive text parsing test suite
   - `tests/test_text_parsing.py` with 200+ test cases

---

## Remaining Critical Tasks

### Task 3.3: Fix Section Switching Bug

**Problem**: When switching sections after all images have been scored, the page must be reloaded to access sections again. New unscored images don't appear in the scoring section.

**Root Cause**: Previous mode state persists (global variables, DOM elements, event listeners) blocking new mode rendering.

**Solution**: Implement proper state cleanup in `client.js`:

```javascript
/**
 * In external_modules/step01ranking/frontend/js/client.js
 * Modify the switchMode function
 */

let currentMode = 'single';  // Track current mode

async function switchMode(mode) {
    // 1. Clean up previous mode state
    if (currentMode) {
        clearModeState(currentMode);
    }
    
    // 2. Clear all container contents
    document.getElementById('single-container').innerHTML = '';
    document.getElementById('batch-container').innerHTML = '';
    document.getElementById('compare-container').innerHTML = '';
    document.getElementById('gallery-container').innerHTML = '';
    
    // 3. Reset global variables used by modes
    // (define these in each mode file)
    resetModeGlobals(mode);
    
    // 4. Update UI
    updateModeButtons(mode);
    
    // 5. Load new mode
    await loadModeContent(mode);
    
    currentMode = mode;
}

function clearModeState(mode) {
    switch(mode) {
        case 'single':
            clearSingleModeState();
            break;
        case 'batch':
            clearBatchModeState();
            break;
        case 'compare':
            clearCompareModeState();
            break;
        case 'gallery':
            clearGalleryModeState();
            break;
    }
}

// Add to each mode file (single.js, batch.js, etc.):
function clearSingleModeState() {
    // Clear all single mode variables
    currentImage = null;
    imageList = [];
    currentIndex = 0;
    // Remove all event listeners if using delegation
}
```

**Implementation Steps**:
1. Modify `js/client.js` switchMode() function
2. Add clearModeState() and resetModeGlobals() functions
3. Update each mode file (single.js, batch.js, compare.js) with clear functions
4. Ensure all global variables are reset properly
5. Test: Switch between modes repeatedly, verify state clears correctly

---

### Task 3.4: Fix New Images Not Showing

**Problem**: When all images are scored, then a new unscored image appears, it doesn't show in the scoring section.

**Root Cause**: Image list is loaded once at startup; doesn't refresh when new images appear.

**Solution**: Implement refresh logic:

```javascript
// In external_modules/step01ranking/frontend/js/client.js or scores.py

// 1. Add auto-refresh endpoint check in client.js
async function checkForNewImages() {
    try {
        const response = await fetch('/api/status');
        const status = await response.json();
        
        if (status.new_images_detected) {
            console.log("New images detected, refreshing...");
            await reloadImageList();
            // Trigger mode refresh
            if (currentMode === 'single' || currentMode === 'batch') {
                await loadModeContent(currentMode);
            }
        }
    } catch (error) {
        console.error("Error checking for new images:", error);
    }
}

// Set up periodic check
setInterval(checkForNewImages, 5000); // Check every 5 seconds

// 2. On mode switch, force reload from server
async function loadModeContent(mode) {
    switch(mode) {
        case 'single':
            // Force API call with cache-busting
            await loadUnscored Images(Date.now());
            break;
        case 'batch':
            // Similar approach
            break;
    }
}
```

**Backend Changes** (`external_modules/step01ranking/scores.py`):

```python
from pathlib import Path
import time

class ScoreServer:
    def __init__(self):
        self.last_index_check = 0
        self.last_file_count = 0
    
    @app.route('/api/status')
    def status(self):
        """Check if new images have been added"""
        try:
            # Get current file count
            image_count = len(list(discover_files(image_root)))
            current_time = time.time()
            
            new_images = image_count > self.last_file_count
            
            return {
                'new_images_detected': new_images,
                'total_files': image_count,
                'timestamp': current_time
            }
        except Exception as e:
            return {'error': str(e)}, 500
    
    @app.route('/api/unscored')
    def get_unscored(self):
        """Get list of unscored images, with cache-busting"""
        force_refresh = request.args.get('refresh', False)
        
        if force_refresh:
            # Bypass cache, scan file system again
            unscored = discover_new_unscored_files()
        else:
            unscored = load_from_cache()
        
        return {'unscored': unscored}
```

**Implementation Steps**:
1. Add `/api/status` endpoint to scores.py
2. Implement periodic check in client.js (every 5 seconds)
3. Force reload image list when switching to scoring modes
4. Add cache-busting parameter to API calls
5. Test: Add new images to input folder while scoring, verify they appear

---

### Task 4.3: Add Text-Only Reprocessing Option

**Purpose**: Process text metadata while keeping image vectors intact (useful for re-extracting terms after logic changes).

**Implementation**:

```python
# In external_modules/step02prepare/full_data/prepare_data.py

import argparse

def run_prepare(
    limit: int = 0,
    rebuild: bool = False,
    text_only: bool = False,  # NEW PARAMETER
) -> dict[str, int]:
    """
    Parameters:
    - text_only: If True, skip image analysis and vector creation,
      only process text data while maintaining image relationships
    """
    
    print("Loading vector libraries...")
    from shared.vectors.vectors import VectorList
    from shared.image_analysis import ImageAnalysis
    
    print(f"Starting {'text processing' if text_only else 'full processing'}...")
    
    if text_only:
        # Text-only mode: don't reprocess images
        return run_text_only(limit, rebuild)
    else:
        # Original logic: full processing
        # ... existing code ...
        pass

def run_text_only(limit: int = 0, rebuild: bool = False) -> dict[str, int]:
    """Process only text data, preserve existing image vectors"""
    print("Text-only mode: Processing text metadata only")
    
    # Load existing data
    index_list = load_single_jsonl(index_file)
    vectors_list = load_single_jsonl(vectors_file)
    scores_list = load_single_jsonl(scores_file)
    text_list = load_single_jsonl(text_data_file) if not rebuild else []
    
    # Get processed files
    processed_files = {s.split("#", 1)[0] for s in index_list}
    
    if rebuild:
        # Clear text data and reprocess everything
        print("Rebuild mode: Reprocessing all text data")
        text_list = []
        processed_text_files = set()
    else:
        # Only process new files
        processed_text_files = {(t.get('file_id') if isinstance(t, dict) else '') for t in text_list}
    
    # Collect new files
    files = list(discover_files(image_root))
    collected_data = collect_valid_files(
        files, processed_files, image_root, limit, max_workers=100, scored_only=True
    )
    
    if len(collected_data) == 0:
        print("No new scored files found for text processing")
        return {"total_text": len(text_list), "new": 0}
    
    print(f"Processing text for {len(collected_data)} files...")
    
    # Use text processing pipeline
    from external_modules.step02prepare.text_data.text_processing import (
        extract_text_components,
        load_metadata_entry,
    )
    
    new_text_entries = []
    for img_path, meta_path in collected_data:
        file_id = os.path.relpath(img_path, image_root).replace("\\", "/")
        
        if file_id in processed_text_files:
            continue
        
        entry, ts, err = load_metadata_entry(meta_path)
        if entry is None or "score" not in entry:
            continue
        
        # Get image dimensions from existing vectors if available
        img_dims = next((v for v in vectors_list if v.get('file_id') == file_id), {})
        
        entry["width"] = img_dims.get("width", 0)
        entry["height"] = img_dims.get("height", 0)
        entry["aspect_ratio"] = img_dims.get("aspect_ratio", 0.0)
        
        text_data = extract_text_components(entry)
        text_data["file_id"] = file_id
        text_data["timestamp"] = ts
        
        new_text_entries.append(text_data)
    
    # Combine with existing text data
    text_list.extend(new_text_entries)
    
    # Write updated text data
    write_single_jsonl(text_data_file, text_list)
    
    return {
        "total_text": len(text_list),
        "new": len(new_text_entries),
        "skipped": len(collected_data) - len(new_text_entries),
    }

def main() -> None:
    parser = argparse.ArgumentParser(description="Run the data preparation pipeline.")
    parser.add_argument(
        "--rebuild",
        action="store_true",
        help="Remove existing outputs before processing",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Limit number of files to process (0 = no limit)",
    )
    parser.add_argument(
        "--text-only",
        action="store_true",
        help="Process only text data, skip image analysis",
    )
    args = parser.parse_args()
    run_prepare(limit=args.limit, rebuild=args.rebuild, text_only=args.text_only)

if __name__ == "__main__":
    main()
```

**Usage**:
```bash
# Original: full processing
python external_modules/step02prepare/full_data/prepare_data.py

# New: text-only reprocessing
python external_modules/step02prepare/full_data/prepare_data.py --text-only

# Rebuild text data from scratch
python external_modules/step02prepare/full_data/prepare_data.py --text-only --rebuild --limit 100
```

**Implementation Steps**:
1. Add `--text-only` argument to prepare_data.py
2. Implement run_text_only() function
3. Load existing vectors and maintain relationships
4. Only process text components for new/changed files
5. Test: Run with --text-only flag, verify text_data.jsonl is updated

---

### Task 5.1 & 5.2: Add 2D Parameter Analysis

**Purpose**: Create 2D scatter plots and heatmaps showing relationships between parameters/prompts and scores.

```python
# Create new file: shared/training/parameter_analysis.py

import numpy as np
import matplotlib.pyplot as plt
from typing import dict, list, tuple
import json
from pathlib import Path

class ParameterAnalyzer:
    def __init__(self, vectors_data: list[dict], text_data: list[dict]):
        self.vectors = vectors_data
        self.text_data = text_data
        self.output_dir = Path("output/analysis")
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def analyze_all(self):
        """Run all analyses"""
        self.analyze_parameter_pairs()
        self.analyze_term_correlations()
        self.generate_report()
    
    def analyze_parameter_pairs(self):
        """Create 2D plots for parameter combinations"""
        # Extract vectors
        steps = np.array([v.get('steps_norm', 0) for v in self.vectors])
        cfg = np.array([v.get('cfg_norm', 0) for v in self.vectors])
        lora_weight = np.array([v.get('lora_weight', 0) for v in self.vectors])
        scores = np.array([v.get('score', 0) for v in self.vectors])
        
        # Steps vs CFG
        self._create_scatter(
            steps, cfg, scores,
            "steps_vs_cfg",
            "Sampling Steps (normalized)",
            "CFG Scale (normalized)",
            "Score"
        )
        
        # LORA Weight vs Steps
        self._create_scatter(
            lora_weight, steps, scores,
            "lora_vs_steps",
            "LORA Weight",
            "Sampling Steps (normalized)",
            "Score"
        )
    
    def analyze_term_correlations(self):
        """Analyze relationship between terms and scores"""
        from shared.vectors.terms import extract_terms
        
        # Extract all terms and their correlations
        term_scores: dict[str, list[float]] = {}
        
        for data in self.text_data:
            score = data.get('score', 0)
            pos_terms = data.get('positive_terms', [])
            neg_terms = data.get('negative_terms', [])
            
            for term, weight in pos_terms:
                if term not in term_scores:
                    term_scores[term] = []
                term_scores[term].append(score * weight)
            
            for term, weight in neg_terms:
                if term not in term_scores:
                    term_scores[term] = []
                term_scores[term].append(score * (1 - weight))
        
        # Calculate average scores per term
        term_avg_scores = {
            term: np.mean(scores) for term, scores in term_scores.items()
        }
        
        # Save top correlations
        sorted_terms = sorted(term_avg_scores.items(), key=lambda x: x[1], reverse=True)
        
        with open(self.output_dir / "term_correlations.json", "w") as f:
            json.dump({
                "top_terms": sorted_terms[:50],
                "total_terms": len(term_avg_scores)
            }, f, indent=2)
    
    def _create_scatter(self, x, y, colors, name, xlabel, ylabel, zlabel):
        """Create scatter plot"""
        fig, ax = plt.subplots(figsize=(10, 8))
        scatter = ax.scatter(x, y, c=colors, cmap='viridis', s=50, alpha=0.6)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(f"{xlabel} vs {ylabel}")
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label(zlabel)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / f"{name}.png", dpi=150)
        plt.close()
    
    def generate_report(self):
        """Generate summary report"""
        report = f"""# Parameter Analysis Report

Generated: {pd.Timestamp.now()}

## Summary Statistics
- Total images analyzed: {len(self.vectors)}
- Average score: {np.mean([v.get('score', 0) for v in self.vectors]):.2f}
- Std Dev: {np.std([v.get('score', 0) for v in self.vectors]):.2f}

## Key Findings
1. See term_correlations.json for top correlated terms
2. See generated PNG files for 2D parameter relationships
3. Identify parameter sweet spots and term combinations

## Next Steps
- Investigate high-correlation terms
- Test parameter combinations identified in plots
- Consider ensemble effects for best results
"""
        with open(self.output_dir / "report.md", "w") as f:
            f.write(report)
```

**Integration in notebook** (`step03training/full_data/analysis.ipynb`):

```python
# Cell: Import and setup
from shared.training.parameter_analysis import ParameterAnalyzer
from shared.loaders.training_loader import load

# Cell: Load data and run analysis
vectors_data = load_single_jsonl("output/vectors.jsonl")
text_data = load_single_jsonl("output/text_data.jsonl")

analyzer = ParameterAnalyzer(vectors_data, text_data)
analyzer.analyze_all()

print("Analysis complete! Check output/analysis/ directory")
```

**Implementation Steps**:
1. Create `shared/training/parameter_analysis.py`
2. Implement ParameterAnalyzer class with methods
3. Add analysis notebook or integrate into training notebook
4. Generate visualizations and correlation data
5. Test: Run analysis, verify PNG files and JSON output created

---

## How to Proceed

### Priority Order for Completion

1. **Fix section switching bug** (Task 3.3) - ~30 mins
   - Most impactful for user experience
   - Single focused code change

2. **Fix new images issue** (Task 3.4) - ~30 mins
   - Builds on Task 3.3
   - Requires backend and frontend changes

3. **Complete type hints update** (Task 2.1) - ~45 mins
   - Systematic but mechanical
   - Update remaining 15 files

4. **Add text-only option** (Task 4.3) - ~40 mins
   - Well-scoped feature addition
   - Maintains backward compatibility

5. **Add 2D analysis** (Task 5.1-5.2) - ~60 mins
   - Highest complexity
   - Great value for insights

### Testing Protocol for Each Task

After completing each task:

```bash
# 1. Run tests
pytest

# 2. For backend changes, run pipeline test
python external_modules/step01ranking/score_server.py --test-run

# 3. For UI changes, test in browser
# Navigate to http://localhost:5000 and manually test

# 4. Verify no errors in server logs
```

### Files Modified Summary

**Created**:
- `tests/test_text_parsing.py`
- `external_modules/step01ranking/frontend/html/gallery.html`
- `external_modules/step01ranking/frontend/css/gallery.css`
- `external_modules/step01ranking/frontend/js/gallery.js`
- `external_modules/step02prepare/full_data/parameter_analysis.py` (to create)

**Modified**:
- `.github/copilot-instructions.md` ✅
- `external_modules/step01ranking/frontend/html/index.html` ✅
- `shared/vectors/terms.py` (type hints sample done)

---

## Questions for User

1. Should Gallery also display raw image files or just metadata?
2. What parameters are most important to analyze (steps, cfg, model, sampler)?
3. Should analysis be automatic after training or manual trigger?
4. Any specific visualization preferences for 2D analyses?

---

*Last Updated: March 29, 2026*
*Next Review: After remaining tasks completion*
