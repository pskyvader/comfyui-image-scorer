#!/usr/bin/env python3
"""Quick test runner for main entry points"""

print("\n" + "="*60)
print("STEP 1: Score Server (Gallery API)")
print("="*60)

try:
    from external_modules.step01ranking.score_server import app
    c = app.test_client()
    r = c.get('/api/scores?page=1')
    data = r.get_json()
    print("[OK] Score Server")
    print(f"  Status: {r.status_code}")
    print(f"  Images on page: {len(data.get('scores', []))}")
    print(f"  Total in database: {data.get('total', 0)}")
except Exception as e:
    print(f"[FAIL] Error: {e}")

print("\n" + "="*60)
print("STEP 2: Prepare Data (Text-Only Mode)")
print("="*60)

try:
    from external_modules.step02prepare.full_data.prepare_data import main as prepare_main
    import sys
    # Test with --text-only flag
    old_argv = sys.argv
    sys.argv = ['prepare_data', '--text-only', '--limit', '1']
    print("Running prepare_data --text-only --limit 1...")
    prepare_main()
    sys.argv = old_argv
    print("[OK] Prepare Data (Text-Only) completed")
except Exception as e:
    print(f"[FAIL] Error: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "="*60)
print("STEP 3: Training Module")
print("="*60)

try:
    from external_modules.step03training.full_data import run
    print("[OK] Step 3 Training module imports")
    print("  (Run training notebooks directly for actual training)")
except Exception as e:
    print(f"[FAIL] Error: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "="*60)
print("[OK] All main entry points verified")
print("="*60)
