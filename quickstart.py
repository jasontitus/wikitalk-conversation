#!/usr/bin/env python3
"""
WikiTalk Quick Start
====================
Downloads ~100 popular Wikipedia articles, chunks them, builds a search
index, and launches the conversational interface.

Usage:
    python quickstart.py              # Download articles + build index + launch
    python quickstart.py --download   # Download articles only
    python quickstart.py --build      # Build index only (articles must exist)
    python quickstart.py --launch     # Launch only (index must exist)
    python quickstart.py --topic TOPIC  # Download articles about a specific topic
"""

import argparse
import json
import logging
import os
import pickle
import re
import sqlite3
import subprocess
import sys
import time
import urllib.request
import urllib.parse
from pathlib import Path

# ---------------------------------------------------------------------------
# Ensure project root is on the path so we can import config
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Curated list of ~100 popular, diverse Wikipedia articles
# Covers: history, science, geography, culture, technology, people, events
# ---------------------------------------------------------------------------
TOP_ARTICLES = [
    # History & Civilizations
    "Ancient Rome", "Ancient Egypt", "Ancient Greece", "Roman Empire",
    "Byzantine Empire", "Ottoman Empire", "Mongol Empire", "British Empire",
    "Han dynasty", "Inca Empire",
    # Major wars & conflicts
    "World War I", "World War II", "Cold War", "American Civil War",
    "French Revolution", "Russian Revolution",
    # Science & Nature
    "Albert Einstein", "Isaac Newton", "Charles Darwin", "Nikola Tesla",
    "Theory of relativity", "Quantum mechanics", "Evolution",
    "DNA", "Photosynthesis", "Solar System", "Black hole",
    "Climate change", "Plate tectonics",
    # Technology
    "Internet", "Artificial intelligence", "Computer",
    "Printing press", "Telephone", "Space exploration",
    "Nuclear energy", "Transistor",
    # Geography & Places
    "Earth", "Moon", "Mars", "Amazon rainforest",
    "Sahara", "Mount Everest", "Pacific Ocean", "Great Barrier Reef",
    "Grand Canyon", "Antarctica",
    # Countries
    "United States", "United Kingdom", "China", "India", "Japan",
    "Brazil", "Australia", "France", "Germany", "Russia",
    # Cities
    "New York City", "London", "Tokyo", "Rome", "Paris",
    # People
    "Leonardo da Vinci", "William Shakespeare", "Marie Curie",
    "Mahatma Gandhi", "Martin Luther King Jr.", "Nelson Mandela",
    "Cleopatra", "Alexander the Great", "Napoleon", "Genghis Khan",
    # Culture & Arts
    "Renaissance", "Classical music", "Olympic Games",
    "Architecture", "Philosophy", "Democracy",
    # Biology & Medicine
    "Human body", "Cell (biology)", "Virus", "Bacteria",
    "Vaccine", "Antibiotic", "Heart",
    # Mathematics & Physics
    "Mathematics", "Pi", "Speed of light", "Gravity",
    # Modern topics
    "Cryptocurrency", "Machine learning", "Renewable energy",
    "Hubble Space Telescope", "International Space Station",
    "Human trafficking", "United Nations",
]


# ---------------------------------------------------------------------------
# Wikipedia API helpers
# ---------------------------------------------------------------------------
def fetch_article(title: str) -> dict | None:
    """Fetch a single Wikipedia article's full plain-text extract via the API."""
    params = urllib.parse.urlencode({
        "action": "query",
        "titles": title,
        "prop": "extracts|info",
        "explaintext": "1",
        "inprop": "url",
        "format": "json",
        "redirects": "1",
    })
    url = f"https://en.wikipedia.org/w/api.php?{params}"

    try:
        req = urllib.request.Request(url, headers={"User-Agent": "WikiTalk-QuickStart/1.0"})
        with urllib.request.urlopen(req, timeout=15) as resp:
            data = json.loads(resp.read().decode())

        pages = data.get("query", {}).get("pages", {})
        for page_id, page in pages.items():
            if page_id == "-1" or "extract" not in page:
                return None
            return {
                "title": page.get("title", title),
                "text": page["extract"],
                "page_id": int(page_id),
                "url": page.get("fullurl", f"https://en.wikipedia.org/wiki/{urllib.parse.quote(title)}"),
            }
    except Exception as e:
        logger.warning(f"  Failed to fetch '{title}': {e}")
    return None


def search_articles_by_topic(topic: str, limit: int = 100) -> list[str]:
    """Search Wikipedia for articles matching a topic."""
    params = urllib.parse.urlencode({
        "action": "query",
        "list": "search",
        "srsearch": topic,
        "srlimit": str(limit),
        "format": "json",
    })
    url = f"https://en.wikipedia.org/w/api.php?{params}"

    try:
        req = urllib.request.Request(url, headers={"User-Agent": "WikiTalk-QuickStart/1.0"})
        with urllib.request.urlopen(req, timeout=15) as resp:
            data = json.loads(resp.read().decode())
        results = data.get("query", {}).get("search", [])
        return [r["title"] for r in results]
    except Exception as e:
        logger.warning(f"  Topic search failed: {e}")
        return []


# ---------------------------------------------------------------------------
# Text chunking (mirrors data_processor.py logic)
# ---------------------------------------------------------------------------
def chunk_article(article: dict, chunk_size: int = 1000, overlap: int = 200) -> list[dict]:
    """Split article text into overlapping chunks."""
    text = re.sub(r"\s+", " ", article["text"].strip())
    if len(text) < 100:
        return []

    chunks = []
    start = 0
    chunk_id = 0

    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunk_text = text[start:end]

        # Try to break at sentence boundaries
        if end < len(text):
            last_period = chunk_text.rfind(".")
            last_newline = chunk_text.rfind("\n")
            break_point = max(last_period, last_newline)
            if break_point > chunk_size // 2:
                chunk_text = chunk_text[: break_point + 1]
                end = start + break_point + 1

        unique_id = f"{article['page_id']}_{chunk_id}_{int(time.time() * 1_000_000)}"

        chunks.append({
            "id": unique_id,
            "text": chunk_text,
            "title": article["title"],
            "page_id": article["page_id"],
            "url": article["url"],
            "date_modified": "",
            "wikidata_id": "",
            "infoboxes": "",
            "has_math": False,
            "start_pos": start,
            "end_pos": end,
        })

        start = max(start + chunk_size - overlap, end)
        chunk_id += 1

    return chunks


# ---------------------------------------------------------------------------
# Database + FAISS index creation
# ---------------------------------------------------------------------------
def create_database(all_chunks: list[dict], db_path: Path):
    """Create the SQLite database with chunks table."""
    conn = sqlite3.connect(str(db_path))
    cur = conn.cursor()

    cur.execute("""
        CREATE TABLE IF NOT EXISTS chunks (
            id TEXT PRIMARY KEY,
            text TEXT,
            title TEXT,
            page_id INTEGER,
            url TEXT,
            date_modified TEXT,
            wikidata_id TEXT,
            infoboxes TEXT,
            has_math BOOLEAN,
            start_pos INTEGER,
            end_pos INTEGER
        )
    """)

    cur.execute("""
        CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
            text, title,
            content='chunks', content_rowid='rowid'
        )
    """)

    inserted = 0
    for chunk in all_chunks:
        try:
            cur.execute(
                "INSERT INTO chunks (id,text,title,page_id,url,date_modified,"
                "wikidata_id,infoboxes,has_math,start_pos,end_pos) "
                "VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                (
                    chunk["id"], chunk["text"], chunk["title"], chunk["page_id"],
                    chunk["url"], chunk["date_modified"], chunk["wikidata_id"],
                    chunk["infoboxes"], chunk["has_math"], chunk["start_pos"],
                    chunk["end_pos"],
                ),
            )
            cur.execute(
                "INSERT INTO chunks_fts (text, title) VALUES (?,?)",
                (chunk["text"], chunk["title"]),
            )
            inserted += 1
        except sqlite3.IntegrityError:
            pass

    # Create indexes for faster lookups
    cur.execute("CREATE INDEX IF NOT EXISTS idx_chunks_page_id ON chunks(page_id)")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_chunks_title ON chunks(title)")

    conn.commit()
    conn.close()
    return inserted


def build_faiss_index(db_path: Path, index_path: Path, ids_path: Path):
    """Build a FAISS index from the SQLite chunks."""
    # Import here so config picks up current settings
    from config import EMBEDDING_MODEL, EMBEDDING_DIM, EMBEDDING_PREFIXES

    import faiss
    import numpy as np
    from sentence_transformers import SentenceTransformer

    logger.info(f"  Loading embedding model: {EMBEDDING_MODEL}")
    model = SentenceTransformer(EMBEDDING_MODEL)

    conn = sqlite3.connect(str(db_path))
    cur = conn.cursor()
    cur.execute("SELECT id, text, title FROM chunks ORDER BY rowid")
    rows = cur.fetchall()
    conn.close()

    if not rows:
        logger.error("  No chunks found in database!")
        return False

    passage_prefix = EMBEDDING_PREFIXES["passage"]
    texts = [f"{passage_prefix}{row[2]}: {row[1][:500]}" for row in rows]
    chunk_ids = [row[0] for row in rows]

    logger.info(f"  Generating embeddings for {len(texts)} chunks...")
    embeddings = model.encode(
        texts,
        batch_size=256,
        show_progress_bar=True,
        convert_to_numpy=True,
    ).astype("float32")

    faiss.normalize_L2(embeddings)

    index = faiss.IndexFlatL2(EMBEDDING_DIM)
    index.add(embeddings)

    id_mapping = {i: cid for i, cid in enumerate(chunk_ids)}

    faiss.write_index(index, str(index_path))
    with open(ids_path, "wb") as f:
        pickle.dump(id_mapping, f)

    logger.info(f"  FAISS index saved: {len(id_mapping)} vectors, {EMBEDDING_DIM}d")
    return True


# ---------------------------------------------------------------------------
# Dependency checking
# ---------------------------------------------------------------------------
def check_dependencies():
    """Verify required packages are installed, offer to install if missing."""
    missing = []
    for pkg, import_name in [
        ("sentence-transformers", "sentence_transformers"),
        ("faiss-cpu", "faiss"),
        ("torch", "torch"),
        ("rapidfuzz", "rapidfuzz"),
        ("moonshine", "moonshine"),
        ("sounddevice", "sounddevice"),
    ]:
        try:
            __import__(import_name)
        except ImportError:
            missing.append(pkg)

    if not missing:
        return True

    logger.info(f"\nMissing packages: {', '.join(missing)}")
    logger.info("Installing from requirements.txt ...")
    try:
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "-r",
             str(PROJECT_ROOT / "requirements.txt"), "-q"],
        )
        logger.info("  Packages installed successfully.\n")
        return True
    except subprocess.CalledProcessError:
        logger.error("  Failed to install packages. Please run:")
        logger.error(f"    pip install -r {PROJECT_ROOT / 'requirements.txt'}")
        return False


# ---------------------------------------------------------------------------
# Main entry points
# ---------------------------------------------------------------------------
def download_articles(titles: list[str]) -> list[dict]:
    """Download articles from Wikipedia and return raw article dicts."""
    articles = []
    logger.info(f"\nDownloading {len(titles)} Wikipedia articles...")
    for i, title in enumerate(titles, 1):
        article = fetch_article(title)
        if article and len(article["text"]) >= 100:
            articles.append(article)
            chars = len(article["text"])
            logger.info(f"  [{i:3d}/{len(titles)}] {article['title']} ({chars:,} chars)")
        else:
            logger.info(f"  [{i:3d}/{len(titles)}] {title} -- skipped")
        # Be polite to the API
        time.sleep(0.1)

    logger.info(f"\nDownloaded {len(articles)} articles.")
    return articles


def process_and_build(articles: list[dict]):
    """Chunk articles, create DB, and build FAISS index."""
    from config import SQLITE_DB_PATH, FAISS_INDEX_PATH, IDS_MAPPING_PATH, DATA_DIR

    DATA_DIR.mkdir(exist_ok=True)

    # --- Chunk ---
    logger.info("\nChunking articles...")
    all_chunks = []
    for article in articles:
        chunks = chunk_article(article)
        all_chunks.extend(chunks)
    logger.info(f"  Created {len(all_chunks)} chunks from {len(articles)} articles.")

    # --- SQLite ---
    if SQLITE_DB_PATH.exists():
        logger.info(f"  Removing old database: {SQLITE_DB_PATH}")
        SQLITE_DB_PATH.unlink()

    logger.info("  Writing SQLite database...")
    inserted = create_database(all_chunks, SQLITE_DB_PATH)
    db_size_mb = SQLITE_DB_PATH.stat().st_size / (1024 * 1024)
    logger.info(f"  Database: {inserted} chunks, {db_size_mb:.1f} MB")

    # --- FAISS ---
    logger.info("\nBuilding search index (this may take a minute on first run)...")
    # Remove old index files if they exist
    for p in [FAISS_INDEX_PATH, IDS_MAPPING_PATH]:
        if p.exists():
            p.unlink()

    ok = build_faiss_index(SQLITE_DB_PATH, FAISS_INDEX_PATH, IDS_MAPPING_PATH)
    if not ok:
        logger.error("Failed to build search index.")
        sys.exit(1)

    return len(articles), len(all_chunks)


def collect_topic_names(articles: list[dict]) -> str:
    """Build a human-friendly summary of what was downloaded."""
    titles = [a["title"] for a in articles]
    if len(titles) <= 5:
        return ", ".join(titles)
    return ", ".join(titles[:5]) + f", and {len(titles) - 5} more"


def launch_wikitalk():
    """Launch the interactive WikiTalk session."""
    logger.info("\n" + "=" * 60)
    logger.info("Starting WikiTalk...")
    logger.info("=" * 60 + "\n")

    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    # Import and run in-process so the user gets the interactive loop
    from wikitalk import WikiTalk
    wt = WikiTalk()
    if not wt.initialize():
        logger.error("Failed to initialize WikiTalk.")
        sys.exit(1)
    try:
        wt.interactive_mode()
    finally:
        wt.close()


def main():
    parser = argparse.ArgumentParser(
        description="WikiTalk Quick Start - download articles, build index, and chat",
    )
    parser.add_argument(
        "--download", action="store_true",
        help="Download articles only (don't build index or launch)",
    )
    parser.add_argument(
        "--build", action="store_true",
        help="Build index only (articles must already be in the database)",
    )
    parser.add_argument(
        "--launch", action="store_true",
        help="Launch WikiTalk only (index must already exist)",
    )
    parser.add_argument(
        "--topic", type=str, default=None,
        help="Search Wikipedia for articles on a specific topic instead of using the default list",
    )
    parser.add_argument(
        "--count", type=int, default=100,
        help="Number of articles to download when using --topic (default: 100)",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("  WikiTalk Quick Start")
    print("=" * 60)

    # If only launching, skip everything else
    if args.launch:
        launch_wikitalk()
        return

    # Check deps
    if not check_dependencies():
        sys.exit(1)

    # If only building index from existing DB
    if args.build:
        from config import SQLITE_DB_PATH, FAISS_INDEX_PATH, IDS_MAPPING_PATH
        if not SQLITE_DB_PATH.exists():
            logger.error(f"Database not found at {SQLITE_DB_PATH}. Run without --build first.")
            sys.exit(1)
        logger.info("Building search index from existing database...")
        for p in [FAISS_INDEX_PATH, IDS_MAPPING_PATH]:
            if p.exists():
                p.unlink()
        build_faiss_index(SQLITE_DB_PATH, FAISS_INDEX_PATH, IDS_MAPPING_PATH)
        logger.info("Done! Run: python quickstart.py --launch")
        return

    # Determine article list
    if args.topic:
        logger.info(f"\nSearching Wikipedia for: \"{args.topic}\"")
        titles = search_articles_by_topic(args.topic, limit=args.count)
        if not titles:
            logger.error("No articles found for that topic. Try a broader search.")
            sys.exit(1)
        logger.info(f"  Found {len(titles)} articles.")
    else:
        titles = TOP_ARTICLES

    # Download
    articles = download_articles(titles)
    if not articles:
        logger.error("No articles were downloaded. Check your internet connection.")
        sys.exit(1)

    if args.download:
        # Save articles as JSON for later processing
        cache_path = PROJECT_ROOT / "data" / "quickstart_articles.json"
        cache_path.parent.mkdir(exist_ok=True)
        with open(cache_path, "w") as f:
            json.dump(articles, f)
        logger.info(f"\nArticles saved to {cache_path}")
        logger.info("Run: python quickstart.py --build && python quickstart.py --launch")
        return

    # Process + build
    num_articles, num_chunks = process_and_build(articles)

    topic_label = args.topic if args.topic else "Wikipedia's greatest hits"
    summary = collect_topic_names(articles)

    print()
    print("=" * 60)
    print(f"  Ready! {num_articles} articles, {num_chunks} searchable chunks.")
    print(f"  Topic: {topic_label}")
    print(f"  Articles: {summary}")
    print("=" * 60)
    print()
    print(f"  Let's start talking about {topic_label}!")
    print()

    launch_wikitalk()


if __name__ == "__main__":
    main()
