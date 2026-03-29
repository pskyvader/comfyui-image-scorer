import sqlite3
from time import time
from shared.paths import cache_file


# ───────────────────────────
# SQLite helpers
# ───────────────────────────
def _get_conn() -> sqlite3.Connection:
    conn = sqlite3.connect(cache_file)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    with _get_conn() as conn:
        # main cache table
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS cache (
                path TEXT PRIMARY KEY,
                score INTEGER DEFAULT NULL,
                comparison_count INTEGER DEFAULT 0,
                score_modifier FLOAT DEFAULT 0,
                volatility FLOAT DEFAULT 0
            )
            """
        )
        # meta table for scan flags and absolute total
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS meta (
                key TEXT PRIMARY KEY,
                value TEXT
            )
            """
        )

        conn.commit()


def _set_meta(key: str, value: str):
    with _get_conn() as conn:
        conn.execute(
            "INSERT INTO meta(key,value) VALUES (?,?) ON CONFLICT(key) DO UPDATE SET value=excluded.value",
            (key, value),
        )
        conn.commit()


def _get_meta(key: str) -> str | None:
    with _get_conn() as conn:
        row = conn.execute("SELECT value FROM meta WHERE key=?", (key,)).fetchone()
        return row["value"] if row else None


# ───────────────────────────
# lifecycle / scan flags
# ───────────────────────────


# 1️⃣ Get total elements per level, grouped by score
def get_total_per_level(score: int = 0) -> dict[int, dict[int, int]]:
    query = "SELECT score, comparison_count, COUNT(*) FROM cache "
    params: list[int] = []

    if 1 <= score <= 5:
        query += "WHERE score = ? "
        params.append(score)
    else:
        query += "WHERE score IS NOT NULL "

    query += "GROUP BY score, comparison_count"

    with _get_conn() as connection:
        cursor = connection.cursor()
        cursor.execute(query, params)
        rows = cursor.fetchall()

    result: dict[int, dict[int, int]] = {}
    for s, lvl, count in rows:
        if s not in result:
            result[s] = {}
        result[s][lvl] = count

    return result


# 2️⃣ Fetch paths for a specific tier
def get_images_by_level(score: int, level: int):
    query = "SELECT path FROM cache WHERE score = ? AND comparison_count = ?"
    with _get_conn() as conn:
        cursor = conn.cursor()
        cursor.execute(query, [score, level])
        return [row[0] for row in cursor.fetchall()]


Meta = dict[str, int | float | str]
Pair = tuple[str, str, Meta, Meta]


def get_image_pair(
    max_comparison_count: int = 10,
    tolerance: float = 1.0,
    min_tolerance:float=0,
    score: int = 0,
    safety_limit: int = 20,
    _cached_pairs: list[str] = [],
) -> Pair | None:
    base_filter = "FROM cache WHERE comparison_count < ?"
    excluded_paths: tuple[str, ...] = tuple(_cached_pairs)
    exclusion_clause: str = (
        f" AND path NOT IN ({','.join(['?'] * len(excluded_paths))})"
        if excluded_paths
        else ""
    )

    with _get_conn() as conn:
        conn.row_factory = sqlite3.Row

        # Safety check
        totals: dict[int, dict[int, int]] = get_total_per_level(score)
        print(f"Score groups: {totals}")
        
        if score > 0:
            score_levels: dict[int, int] = totals.get(score, {})
            available_count = sum(
                count
                for lvl, count in score_levels.items()
                if lvl < max_comparison_count
            )
        else:
            available_count = sum(
                count
                for score_levels in totals.values()
                for lvl, count in score_levels.items()
                if lvl < max_comparison_count
            )

        if available_count < safety_limit:
            return None

        if score > 0:
            query_first = f"{base_filter} AND score = ?{exclusion_clause}"
            params_first = [max_comparison_count, score, *excluded_paths]
        else:
            query_first = f"{base_filter} AND score IS NOT NULL{exclusion_clause}"
            params_first = [max_comparison_count, *excluded_paths]

        first_row = conn.execute(
            f"SELECT * {query_first} ORDER BY RANDOM() LIMIT 1", params_first
        ).fetchone()

        if not first_row:
            return None

        first_image: Meta = dict(first_row)
        first_effective_score: float = (
            float(first_image["score"]) + float(first_image["score_modifier"]) / 10.0
        )
        
        #print(f"excluded paths:{excluded_paths}")

        # Pick second image directly in SQL
        second_row: Meta = conn.execute(
            f"""
            SELECT * {base_filter}
            AND score IS NOT NULL
            AND path != ?
            AND ABS((score + score_modifier / 10.0) - ?) BETWEEN ? AND ?
            {exclusion_clause}
            ORDER BY RANDOM()
            LIMIT 1
            """,
            (
                max_comparison_count,
                first_image["path"],
                first_effective_score,
                min_tolerance,
                tolerance,
                *excluded_paths,
            ),
        ).fetchone()

        if not second_row:
            return None

        return (
            str(first_image["path"]),
            str(second_row["path"]),
            first_image,
            dict(second_row),
        )


# ───────────────────────────
# cache operations
# ───────────────────────────
def add(
    path: str,
    score: int | None = None,
    comparison_count: int = 0,
    score_modifier: int = 0,
    volatility: int = 0,
) -> None:
    """Add image path to cache with optional scoring metadata."""
    with _get_conn() as conn:
        conn.execute(
            "INSERT OR REPLACE INTO cache(path, score, comparison_count, score_modifier,volatility) VALUES (?, ?, ?, ?,?)",
            (path, score, comparison_count, score_modifier, volatility),
        )
        conn.commit()


def get_all(unscored_only: bool = True) -> list[str]:
    """Return cached image paths."""
    global _last_served
    _last_served = time()
    with _get_conn() as conn:
        if unscored_only:
            rows = conn.execute("SELECT path FROM cache WHERE score IS NULL").fetchall()
        else:
            rows = conn.execute("SELECT path FROM cache").fetchall()
        return [r["path"] for r in rows]


def total_cached_unscored() -> int:
    """Count how many unscored images are in cache."""
    with _get_conn() as conn:
        row = conn.execute(
            "SELECT COUNT(*) as cnt FROM cache WHERE score IS NULL"
        ).fetchone()
        return row["cnt"]


def get_comparison_stats() -> dict[str, int]:
    """Get stats for comparison mode."""
    with _get_conn() as conn:
        scored = conn.execute(
            "SELECT COUNT(*) as cnt FROM cache WHERE score IS NOT NULL"
        ).fetchone()["cnt"]
        not_compared = conn.execute(
            "SELECT COUNT(*) as cnt FROM cache WHERE score IS NOT NULL AND comparison_count<10"
        ).fetchone()["cnt"]
        fully_compared = conn.execute(
            "SELECT COUNT(*) as cnt FROM cache WHERE comparison_count>=10"
        ).fetchone()["cnt"]
        partially_compared = conn.execute(
            "SELECT COUNT(*) as cnt FROM cache WHERE comparison_count>0 AND comparison_count<10"
        ).fetchone()["cnt"]
        total = conn.execute("SELECT COUNT(*) as cnt FROM cache").fetchone()["cnt"]
        return {
            "scored": scored,
            "not_compared": not_compared,
            "fully_compared": fully_compared,
            "partially_compared": partially_compared,
            "total": total,
        }


def get_cached_metadata(path: str) -> dict[str, int | float | str | None] | None:
    """Get cached score/comparison_count/score_modifier from database."""
    with _get_conn() as conn:
        row = conn.execute(
            "SELECT score, comparison_count, score_modifier, volatility FROM cache WHERE path=?",
            (path,),
        ).fetchone()
        if row:
            return {
                "score": int(row["score"]) if row["score"] else None,
                "comparison_count": int(row["comparison_count"]),
                "score_modifier": float(row["score_modifier"]),
                "volatility": float(row["volatility"]),
            }
        return None


_last_served: float = 0.0


def total_cached() -> int:
    with _get_conn() as conn:
        row = conn.execute("SELECT COUNT(*) as cnt FROM cache").fetchone()
        return row["cnt"]


# ───────────────────────────
# totals
# ───────────────────────────
def set_absolute_total(n: int) -> None:
    _set_meta("absolute_total", str(n))


def get_absolute_total() -> int:
    val = _get_meta("absolute_total")
    return int(val) if val else 0


# ───────────────────────────
# Initialize DB on import
# ───────────────────────────
init_db()
