import numpy as np
def check_for_leakage(vectors_list: list[list[float]], scores_list: list[int]) -> None:
    if not vectors_list or not scores_list:
        raise RuntimeError(f"check leakage: Empty vectors or scores list, vectors list: {len(vectors_list)}, scores list: {len(scores_list)}")
    if len(vectors_list) != len(scores_list):
        msg = f"check leakage: lengths don't match, vectors list: {len(vectors_list)}, scores list: {len(scores_list)}"
        raise RuntimeError(msg)

    # for i,v in enumerate(vectors_list):
    #     print(f"vector list {i} : {len(v)}")

    arr = np.array(vectors_list)
    y_arr = np.array(scores_list)
    corrs: list[float] = []
    for i in range(arr.shape[1]):
        col = arr[:, i]
        if np.allclose(col, col[0]):
            corrs.append(0.0)
            continue
        c = np.corrcoef(col, y_arr)[0, 1]
        corrs.append(float(c) if not np.isnan(c) else 0.0)
    corrs_arr = np.array(corrs)
    leak_cols = np.where(np.abs(corrs_arr) >= 0.9999)[0].tolist()
    eq_cols = [i for i in range(arr.shape[1]) if np.allclose(arr[:, i], y_arr)]
    leak_cols = sorted(set(leak_cols + eq_cols))
    if leak_cols:
        msg = (
            f"Detected feature columns that strongly match the target (possible leakage): {leak_cols}. "
            "Fix the feature assembly to avoid including target values as features."
        )
        print(msg)
        raise RuntimeError(msg)
