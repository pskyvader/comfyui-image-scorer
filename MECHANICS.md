# Image Scorer: Ranking Mechanics & Logic

This document explains how the current system calculates image rankings and chooses which images to compare.

## 1. Score and Confidence (0.0 - 1.0)

The system uses two primary values to track an image's standing in your library:

### **Score (The "Rank")**
- **Range:** `0.0` (Lowest) to `1.0` (Highest).
- **Function:** This is a continuous value representing the image's relative quality compared to the rest of the library.
- **Adjustment:** When Image A wins against Image B, the scores are updated using a **Fast Jump Exponential Decay** formula.
- **Formula:** 
    `delta = 0.5 * exp(-0.23 * comparison_count) * impact_factor`
    - `Winner Score = old_score + delta`
    - `Loser Score = old_score - delta`
- **Why this formula?**
    - **Initial Phase (0 comps):** `delta` starts at **0.5**. An image at the middle (`0.5`) will jump immediately to `1.0` (or `0.0`) in a single win or loss. This ensures new images reach their approximate tier instantly.
    - **Stable Phase (10 comps):** After 10 comparisons, the movement is exactly **0.05**, allowing for meaningful fine-tuning even for established images.
- **Recorded Weight:** The `impact_factor` (1.0 for direct, <1.0 for indirect) is recorded as the **"weight"** in both the SQLite database and the image's companion JSON file.

### **Confidence (The "Stability")**
- **Range:** `0.0` to `1.0`.
- **Function:** This represents the statistical weight and history of the image.
- **Threshold:** It takes **5 DIRECT comparisons** (configurable as `target_comparisons_for_max_confidence`) for an image to reach `1.0` confidence.
- **Formula:** Confidence is the ratio: `effective_count / target_divisor`.
    - **Target Divisor:** This is the sum of weights for `N` direct comparisons (default 5). We use direct comparisons as the "gold standard" to define what 100% confidence looks like.
    - **Effective Count:** A weighted sum of ALL comparisons (Direct and Indirect).
        - **Temporal Weight:** Most recent = `1.0`. 10th most recent = `0.5`. Formula: `0.5 ^ (rank / 9.0)`.
        - **Impact Weight:** Direct = `1.0`. Indirect = `0.6 ^ depth`.
- **Indirect vs. Old Direct:** 
    - A **fresh indirect** comparison (Index 0, weight 0.6) has more impact on confidence than a **direct comparison that is 7th in the list** (Index 7, temporal weight ~0.58).
- **Confidence Levels:**
    - **Low Confidence (<0.3):** "Volatile." The image is new and its score can change drastically.
    - **Medium Confidence (0.3 - 0.7):** "Refining." The image has a general tier, but is still being fine-tuned against neighbors.
    - **High Confidence (>0.7):** "Benchmark." The score is stable and the image is used to test others.

---

## 2. Pair Selection Algorithm

The system uses a **Density-Based Merge Sort Ranker** to prioritize images that need the most attention:

### **1. Tier Density & Priority**
- **Priority Formula:** `priority = comparison_count / log10(tier_size + 9)`
- **How it works:**
    - **Primary Priority:** Images with **0 comparisons** (unranked) always get `priority = 0` and are picked first.
    - **Secondary Priority:** Images in the **most crowded folders** get a higher priority (lower score) to force "sorting pressure" and clear the backlog.
- **Comparison Range (Adaptive):** 
    - **Unranked/Low Confidence:** The system allows comparisons within a wide `0.2` score range to let images jump tiers quickly.
    - **High Confidence:** As images stabilize, the system automatically narrows the search range (down to `±0.05`) to perform "fine-tuning" between very similar images.

### **2. LRU (Least Recently Used) Safety**
- **Backend Enforcement:** The LRU logic is handled entirely on the **Backend**. The server maintains a list of the last 100 images shown and ensures they aren't picked again for a while, regardless of which client is connected.

### **3. Newest File Preservation**
- **Definition of "Recent":** The scanner looks at the `os.path.getmtime` (Last Modified Time) of all files in your input folder.
- **Reserve Logic:** It sorts them by time (newest first) and skips the first `reserve_count` (default 10) images. 
- **Benefit:** This ensures that while you are generating new images in ComfyUI, the ranker isn't moving the 10 most recent files, which prevents breakage of ComfyUI's internal filename numbering sequence.

---

## 3. Indirect (Transitive) Comparisons

The system uses Graph Theory to reduce the number of clicks you have to make.

### **How it works (BFS Search)**
- **Formula:** `Indirect Weight = Direct Weight (1.0) * (0.6 ^ depth)`
- **Conflict Handling:** If a loop is found (A > B and B > A indirectly), the system **ignores the indirect result** and forces a manual comparison to break the cycle.
- **Efficiency:** The search is limited to a depth of 3, making it extremely fast.
