import os
import sys
import math
from pathlib import Path
from typing import List, Dict, Any
from tqdm import tqdm

# Add workspace root to sys.path
def find_workspace_root() -> Path:
    p = Path(__file__).resolve()
    for ancestor in p.parents:
        if (ancestor / "main.py").exists() or (ancestor / "custom_nodes").is_dir():
            return ancestor
    return p.parents[-1]

workspace_root = find_workspace_root()
sys.path.insert(0, str(workspace_root))

# Import new modules
try:
    from external_modules.step01ranking_new.database.images_table import get_all_images, update_image_score_confidence
    from external_modules.step01ranking_new.file_management.path_handler import get_ranked_root, compute_path_from_filename
    from external_modules.step01ranking_new.image_processor import ImageProcessor
except ImportError as e:
    print(f"Error importing modules: {e}")
    print("Ensure you are running this from within the ComfyUI workspace.")
    sys.exit(1)

# --- CONFIGURATION ---
UPDATE_DATABASE = False  # Set to True to actually save new scores to DB
MOVE_FILES = False      # Set to True to move files to new score folders
DRY_RUN = True          # If True, only print what would happen
# ---------------------

def redistribute_scores():
    """
    Redistributes image scores to follow a uniform 0.0-1.0 distribution.
    This helps fill gaps and flattens the histogram.
    """
    print("Fetching images from database...")
    all_images = get_all_images()
    if not all_images:
        print("No images found in database.")
        return

    total_images = len(all_images)
    print(f"Found {total_images} images.")

    # Sort by current score
    all_images.sort(key=lambda x: x["score"])

    print(f"Calculating new uniform scores (0.0 to 1.0)...")
    
    # We'll use a mapping of filename -> new_score
    updates = []
    for i, img in enumerate(all_images):
        # Map rank to 0.0-1.0 range
        # Use (i + 0.5) / total to avoid pinning exactly to 0 and 1 if desired,
        # or i / (total - 1) for full range.
        if total_images > 1:
            new_score = i / (total_images - 1)
        else:
            new_score = 0.5
            
        updates.append({
            "filename": img["filename"],
            "old_score": img["score"],
            "new_score": new_score,
            "confidence": img["confidence"],
            "count": img["comparison_count"]
        })

    # --- Phase 1: Database Updates ---
    if UPDATE_DATABASE:
        if DRY_RUN:
            print("[DRY RUN] Would update database scores for all images.")
        else:
            print(f"Updating database for {total_images} images...")
            for up in tqdm(updates, desc="DB Updates"):
                update_image_score_confidence(
                    up["filename"], 
                    up["new_score"], 
                    up["confidence"], 
                    up["count"]
                )
    else:
        print("Database update skipped (UPDATE_DATABASE = False).")

    # --- Phase 2: File Movements ---
    if MOVE_FILES:
        processor = ImageProcessor()
        moved_count = 0
        
        # We need the ranked root to find files
        ranked_root = get_ranked_root()
        
        print(f"Checking file positions for {total_images} images...")
        for up in tqdm(updates, desc="Relocating files"):
            filename = up["filename"]
            new_score = up["new_score"]
            
            # Find current file location (using path_handler logic)
            from external_modules.step01ranking_new.file_management.path_handler import find_image_path
            current_path = find_image_path(filename)
            
            if not current_path:
                # logger or print error? Skip for now.
                continue
                
            # Compute where it SHOULD be now
            target_path = compute_path_from_filename(filename, new_score)
            
            if current_path != target_path:
                if DRY_RUN:
                    # print(f"[DRY RUN] Move {filename} from {current_path.parent.name} to {target_path.parent.name}")
                    moved_count += 1
                else:
                    try:
                        import shutil
                        target_path.parent.mkdir(parents=True, exist_ok=True)
                        shutil.move(str(current_path), str(target_path))
                        
                        # Also move JSON if it exists
                        json_src = current_path.with_suffix(".json")
                        if json_src.exists():
                            shutil.move(str(json_src), str(target_path.with_suffix(".json")))
                        
                        moved_count += 1
                    except Exception as e:
                        print(f"Failed to move {filename}: {e}")
        
        status = "Would move" if DRY_RUN else "Moved"
        print(f"{status} {moved_count} files to new score folders.")
    else:
        print("File movement skipped (MOVE_FILES = False).")

    print("\nRedistribution Summary:")
    print(f"Total Images: {total_images}")
    if total_images > 0:
        print(f"Score Range: {updates[0]['new_score']:.4f} to {updates[-1]['new_score']:.4f}")

if __name__ == "__main__":
    redistribute_scores()
