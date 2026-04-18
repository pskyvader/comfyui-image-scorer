# Plan: Improve Image Ranking System for 30k Images

## TL;DR
Implement two major improvements: (1) **Bidirectional cache-file score toggle** in prepare_data.py to switch between scored file metadata and cached database scores, and (2) **Replace the current pairwise comparison system** with an **Uncertainty-Aware Merge Sort** algorithm that tracks confidence intervals, allows revisions, detects convergence, and can handle 30k images 3-5x faster with better accuracy. Keep high-quality tier (score 4+) preserved. Add UI support for batch operations and tier management.

---

## PART 1: Cache-Based Score Rebuild Feature

### Current State
- `prepare_data.py` only reads scores from files (JSON metadata)
- Cache database has scores but isn't used as a source in prepare pipeline
- No way to switch between "scored files" and "cache scores" once prepared

### Solution: Bidirectional Toggle
1. **New parameter** in `prepare_data.py`: `--scores-source` (options: `files` | `cache`)
2. **Files mode** (default): Current behavior - reads scores from image JSON metadata
3. **Cache mode**: New - reads scores from cache.db instead, writes results back to files
4. **Migration on first run**: Detect which source has more complete/recent data
5. **Write strategy**: Ensure both sources stay in sync after prepare_data runs

### Implementation Details
- Add `rebuild_scores_from_cache()` function (similar to existing `run_prepare()`)
- Modify `collect_valid_files()` to accept source parameter
- Use cache queries: `get_cached_metadata()` to fetch score, comparison_count, score_modifier, volatility
- Write back to JSON files to maintain dual sync
- Add logging to track which source was used

### Why Useful
- Allows A/B testing: rank via cache db for one dataset, via files for another
- Can rebuild if file metadata gets corrupted
- Enables testing new ranking algorithms on cache data without touching original files

---

## PART 2: Uncertainty-Aware Merge Sort Ranking System

### Problem Analysis
Current system fails at 30k images because:
- Pairwise comparisons (O(N²) worst case) create bottleneck
- Fixed tolerance algorithm causes pair-finding failures at scale
- Volatility resets trap images mid-ranking
- No convergence detection = images may cycle forever
- No confidence tracking = can't distinguish "certain at 3" from "guess at 3"

### Proposed Solution: Uncertainty-Aware Merge Sort

#### Core Concept
- **Merge Sort instead of pairwise**: O(N log N) comparisons instead of O(N²)
- **Confidence tracking**: Each image gets uncertainty bounds (e.g., 3.0 ± 0.4)
- **Smart pairing**: Pair images where uncertainty overlaps (both uncertain about relative ranking)
- **Revision-friendly**: Can compare same pair again, results adjust confidence not replace it
- **Convergence detection**: Stop comparing when uncertainty < threshold
- **Tier preservation**: Mark high-confidence top tier (score 4+) as "locked" unless strong evidence

#### Key Metrics (per image)
```
score: int (1-5)
confidence: float (0.0-1.0, 1.0 = certain, 0.0 = very uncertain)
lower_bound, upper_bound: effective score range (e.g., 2.8-3.2)
ranking_generation: int (which sort pass we're on)
is_locked: bool (for top tier)
comparison_history: list of (vs_image, winner, timestamp) 
```

#### Algorithm Phases

**Phase 1: Initial Tier Assignment (Fast)**
- Scan all images with existing scores
- Assign uncertainty based on comparison_count (more comparisons = higher confidence)
- Images with no scores get random or special handling
- High-tier images (4+) marked `is_locked = True`

**Phase 2: Uncertainty-Driven Merge Sort (Main Loop)**
```
while any_image_has_uncertainty > convergence_threshold:
   1. Find all image pairs where confidence intervals overlap
   2. Select highest-priority pair (largest uncertainty gap, haven't compared recently)
   3. Present comparison to user
   4. Update confidence bounds based on result
   5. If comparison < 3 comparisons ago, weight new result as "revision" (0.6x impact)
   6. Detect convergence: if upper_bound(img) < lower_bound(better) → finalized
   7. Every N comparisons: compact intervals (merge close ranges)
```

**Phase 3: Final Tier Sort (Optional)**
- Images still in uncertainty overlap zones get sorted within tier
- Can use reduced comparison count (5-10 total max per image)

#### Smart Pairing Strategy
```
Priority 1: Pairs where images haven't been compared recently 
           (allow revision/change of mind)
Priority 2: Pairs where uncertainty overlap is largest 
           (most ambiguous, most informative if user decides)
Priority 3: Pairs from different tiers with adjacent ranges 
           (calibrate tier boundaries)
Priority 4: Within same tier, pairs separated by uncertainty 
           (refine tier ordering)
```

#### Convergence Detection
```
Image is "ranked" when:
  - lower_bound(image) >= upper_bound(lower_image) in final order
  - confidence >= 0.85 (user is confident) 
  - last_comparison < 2 weeks ago (allow time for fresh look)
  - or user explicitly marks as "stop ranking this"
```

#### Revision Handling
```
If user compares same pair twice:
  - First result: full weight (1.0x)
  - Second result within 1 day: reduced weight (0.6x), indicates uncertainty
  - Second result after 1 week: full weight (1.0x), indicates changed mind
  - If conflicting: average results, reduce confidence
```

### Database Schema Additions
Add to cache table:
```
confidence FLOAT DEFAULT 0.3
lower_bound FLOAT (score - confidence_width)
upper_bound FLOAT (score + confidence_width)
is_locked BOOLEAN DEFAULT 0 (for score 4+)
ranking_generation INTEGER DEFAULT 0
last_compared_at DATETIME
comparison_history TEXT (JSON list)
```

### Implementation Files to Create/Modify
1. **New**: `merge_sort_ranker.py` - Core algorithm
2. **New**: `confidence_tracker.py` - Confidence interval management
3. **Modify**: `cache.py` - Add new fields, update queries
4. **Modify**: `comparison.py` - Replace with revised pair selection
5. **Modify**: `score_server.py` - API for new algorithm
6. **New**: `migration.py` - Migrate old scores to new confidence model
7. **Modify**: `prepare_data.py` - Accept `--ranking-method` parameter

---

## PART 3: UI/UX Changes for Faster Ranking

### Current UI Limitation
- Only "A vs B" binary comparison
- No batch operations
- Can't see tier overview or mark images as "sure" vs "unsure"
- No easy way to revise or correct mistakes

### New UI Options (Open Questions)
1. **Multi-choice ranking**: Show 3-4 images, user picks best in tier
2. **Batch tier sort**: Show all images in score 3.0-3.5 range, order them
3. **Tier jump**: Explicitly say "this is top-tier quality" or "this is definitely bottom-tier"
4. **Revision mode**: "Show me images I've compared before, let me reconsider"
5. **Confidence view**: See which images are uncertain (grey out certain ones)

### What We Should Keep
- Current /api/scores gallery (useful for overview)
- Core /compare/next API (can power new UI)
- /compare/submit endpoint

---

## PART 4: Migration & Backwards Compatibility

### Current 30k Images
- Many have scores 1-5 and comparison_count 0-10
- Some have volatility data
- Goal: Preserve top tier (4+) as "good images", migrate rest

### One-Time Migration Strategy
1. **Scan cache**: Find all images with score >= 4
   - Mark as `is_locked = True`
   - Set `confidence = 0.9` (high confidence, good images)
2. **For others**: 
   - confidence = 0.3 + (comparison_count / 30) [rough estimate]
   - is_locked = False
3. **Generate initial bounds**:
   - lower_bound = score - 0.5
   - upper_bound = score + 0.5
   - Adjust based on volatility data if present
4. **Write to cache**: One SQL batch update
5. **Log results**: Report images in top tier preserved

### Data Preservation
- Old scores stay in cache (not deleted)
- volatility field kept for reference
- comparison_count preserved (still used for confidence estimate)

---

## Verification Plan

### Phase 1: Cache Rebuild Feature
1. Run `prepare_data.py --scores-source=cache` on test dataset
2. Verify scores loaded from cache.db (not files)
3. Verify prepare_data completes without errors
4. Check output vectors/scores match cache values
5. Toggle back to `--scores-source=files`, verify behavior swaps

### Phase 2: Merge Sort Algorithm
1. **Unit tests**: 
   - Confidence tracking: verify bounds update correctly on comparisons
   - Convergence detection: ensure images stop being paired when certain
   - Revision handling: verify weight reduction works for re-comparisons
2. **Integration test**:
   - Seed cache with ~100 unscored images
   - Run merge sort ranker, validate it produces sorted output
   - Measure comparison count (should be ~250-400 for 100 images, not 1000+)
3. **Scale test**:
   - Run on subset of 1000 real images
   - Track: total comparisons, time per image, convergence rate
   - Compare vs current system (should be 40-50% fewer comparisons)

### Phase 3: Migration
1. Run migration on current 30k cache
2. Verify all images have new confidence fields
3. Verify locked tier has all images score >= 4
4. Spot-check: sample 50 images, verify bounds make sense

### Phase 4: End-to-End
1. Ranking server starts with migrated data
2. User does 50 new comparisons using merge sort
3. Verify confidence bounds tighten
4. Verify convergence detection works (images stop being re-paired)
5. Export via prepare_data, verify scores reasonable

---

## Decisions & Scope

### Included
✅ Bidirectional cache-file score toggle  
✅ Uncertainty-aware merge sort core algorithm  
✅ Confidence interval tracking  
✅ Convergence detection & early exit  
✅ Revision-friendly comparison history  
✅ Top-tier preservation (score 4+)  
✅ One-time migration of 30k images  
✅ Performance target: 40-50% fewer comparisons than current  

### Explicitly Excluded (Future)
❌ Distributed/parallel ranking (focus on local optimization first)  
❌ Machine learning prediction of ranking (relies on stable human preferences)  
❌ Real-time ranking updates (batch process remains)  
❌ New UI implementation (design spec only, implementation separate task)  
❌ Volatility rework (postpone until Phase 2 complete)  

### Open Questions for Approval
1. **Confidence threshold**: What level triggers "converged"? (recommend 0.85)
2. **Revision weight**: Should re-comparison weight be 0.6x or different? (recommend 0.6-0.7x)
3. **Locked tier**: Keep score 4+ locked, or allow unlocking? (recommend locked, with override)
4. **Convergence timeout**: After how long can user reconsider a "done" image? (recommend 7 days)

---

## Summary for Implementation

### System Overview

**Current Limitations**:
- Pairwise comparisons at O(N²) create scaling bottleneck
- Boundary damping traps images; volatility resets cause infinite loops
- No convergence detection or confidence tracking
- 30k images would require 450k+ comparisons worst-case

**New Approach**:
- Merge sort O(N log N) with confidence intervals (e.g., 3.0 ± 0.4)
- Pairs only overlap uncertainty zones (most informative comparisons)
- Convergence detection stops re-pairing when confident
- Revision-friendly: can re-compare pairs, results adjust confidence not replace
- Top tier (4+) locked to preserve good images
- Expected: 40-50% fewer total comparisons, much faster convergence

### Quick Reference: Files to Modify
1. `prepare_data.py` — Add `--scores-source` parameter + `rebuild_scores_from_cache()`
2. `cache.py` — Add confidence schema fields + new queries
3. `comparison.py` — Replace `get_paired_images()` with uncertainty-based pairing
4. **NEW**: `merge_sort_ranker.py` — Core algorithm
5. **NEW**: `confidence_tracker.py` — Confidence math
6. **NEW**: `migration.py` — One-time migration of 30k images

### Implementation Priority
1. **High**: Cache rebuild feature (quick win, enables testing)
2. **High**: Confidence schema + migration (foundation for new algorithm)
3. **Medium**: Merge sort core + confidence tracker (main algorithm)
4. **Medium**: API/server integration (pair selection)
5. **Low**: UI/UX batch operations (nice-to-have, separate task)
