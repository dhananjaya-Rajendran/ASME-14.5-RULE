"""Build and query local OCR and Chroma HNSW knowledge bases for ASME Y14.5M-1994."""

import re
import sqlite3
from pathlib import Path
from typing import Dict, List, Optional

import pymupdf
from rapidocr_onnxruntime import RapidOCR


DEFAULT_KB_PATH = Path(__file__).resolve().parent / "data" / "asme_y14_5m_1994.sqlite3"
DEFAULT_CHROMA_PATH = Path(__file__).resolve().parent / "data" / "chroma_asme_y14_5m_1994"
CHROMA_COLLECTION_NAME = "asme_y145m_1994"
HNSW_METADATA = {
    "hnsw:space": "cosine",
    "hnsw:construction_ef": 128,
    "hnsw:search_ef": 64,
    "hnsw:M": 32,
}
# ASME Y14.5 in the page header must not be mistaken for a clause number.
# This edition's numbered main sections are 1 through 6.
SECTION_RE = re.compile(r"(?i)\b(?:section|sec\.?|para\.?|paragraph)\s*([1-6]\.\d{1,2}(?:\.\d{1,2})?)")
SECTION_HEADING_RE = re.compile(
    r"(?<![\d.])([1-6]\.\d{1,2}(?:\.\d{1,2})?)(?:\s*)([A-Z][A-Z]{2,})")
WORD_RE = re.compile(r"[\w.]+", re.UNICODE)
STOP_WORDS = {
    "asme", "y14", "y14.5m", "1994", "section", "sections", "para", "paragraph",
    "rule", "rules", "finding", "findings", "review", "confirm", "control", "controls",
    "with", "from", "that", "this", "into", "only", "where", "which", "their",
    "shall", "should", "must", "have", "been", "being", "does", "not", "than",
}


def _detect_section(text: str) -> Optional[str]:
    match = SECTION_HEADING_RE.search(text)
    return match.group(1) if match else None


def _ocr_page(page, ocr, dpi: int) -> str:
    pix = page.get_pixmap(dpi=dpi, colorspace=pymupdf.csGRAY, alpha=False)
    result, _ = ocr(pix.tobytes("png"))
    if not result:
        return ""

    # Reading-order approximation: group OCR detections into horizontal lines.
    boxes = []
    for polygon, text, _score in result:
        if not text or not text.strip():
            continue
        xs = [point[0] for point in polygon]
        ys = [point[1] for point in polygon]
        boxes.append((sum(ys) / len(ys), min(xs), max(ys) - min(ys), text.strip()))
    boxes.sort(key=lambda box: (box[0], box[1]))

    lines = []
    for center_y, x0, height, text in boxes:
        tolerance = max(5.0, height * 0.45)
        if not lines or abs(center_y - lines[-1]["center_y"]) > tolerance:
            lines.append({"center_y": center_y, "words": [(x0, text)]})
        else:
            line = lines[-1]
            line["words"].append((x0, text))
            line["center_y"] = sum(y for y, _ in line["words"]) / len(line["words"])
    return "\n".join(" ".join(word for _, word in sorted(line["words"])) for line in lines)


def _split_chunks(text: str, max_chars: int = 1800, overlap: int = 250) -> List[str]:
    paragraphs = [re.sub(r"\s+", " ", part).strip() for part in re.split(r"\n+", text)]
    paragraphs = [part for part in paragraphs if part]
    chunks: List[str] = []
    current = ""
    for paragraph in paragraphs:
        while len(paragraph) > max_chars:
            if current:
                chunks.append(current)
                current = current[-overlap:] if overlap else ""
            cut = paragraph.rfind(" ", 0, max_chars)
            if cut < max_chars // 2:
                cut = max_chars
            piece, paragraph = paragraph[:cut].strip(), paragraph[cut:].strip()
            combined = f"{current} {piece}".strip()
            if combined:
                chunks.append(combined)
            current = paragraph[:overlap] if overlap else ""
        candidate = f"{current} {paragraph}".strip()
        if len(candidate) > max_chars and current:
            chunks.append(current)
            current = paragraph
        else:
            current = candidate
    if current:
        chunks.append(current)
    return chunks


def build_knowledge_base(pdf_path: str, database_path: str = str(DEFAULT_KB_PATH),
                         dpi: int = 120, chroma_path: str = str(DEFAULT_CHROMA_PATH),
                         progress_callback=None) -> Dict[str, object]:
    """OCR every PDF page, finalize its text archive, then build the HNSW index."""
    source = Path(pdf_path).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(f"Knowledge source PDF not found: {source}")
    target = Path(database_path).expanduser().resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(target.stem + ".building.sqlite3")
    if temporary.exists():
        temporary.unlink()

    ocr = RapidOCR()
    page_count = 0
    chunk_count = 0
    error_pages = []
    try:
        with sqlite3.connect(temporary) as connection, pymupdf.open(source) as document:
            connection.execute("PRAGMA journal_mode=WAL")
            connection.execute("CREATE TABLE metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL)")
            connection.execute("""CREATE TABLE chunks (
                id INTEGER PRIMARY KEY, page INTEGER NOT NULL, chunk_no INTEGER NOT NULL,
                section TEXT, text TEXT NOT NULL)""")
            connection.execute("CREATE INDEX chunks_section_idx ON chunks(section)")
            connection.execute("""CREATE TABLE chunk_sections (
                section TEXT NOT NULL, chunk_id INTEGER NOT NULL,
                PRIMARY KEY(section, chunk_id))""")
            connection.execute("CREATE INDEX chunk_sections_chunk_idx ON chunk_sections(chunk_id)")
            connection.execute("CREATE VIRTUAL TABLE chunks_fts USING fts5(text, section, content='chunks', content_rowid='id', tokenize='unicode61')")
            connection.execute("INSERT INTO metadata VALUES ('edition', 'ASME Y14.5M-1994')")
            connection.execute("INSERT INTO metadata VALUES ('source_name', ?)", (source.name,))
            connection.execute("INSERT INTO metadata VALUES ('source_pages', ?)", (str(len(document)),))
            connection.execute("INSERT INTO metadata VALUES ('ocr_dpi', ?)", (str(dpi),))

            for page_index, page in enumerate(document):
                page_num = page_index + 1
                try:
                    text = _ocr_page(page, ocr, dpi)
                    chunks = _split_chunks(text)
                    for chunk_index, chunk in enumerate(chunks):
                        section_matches = sorted({match[0] for match in SECTION_HEADING_RE.findall(chunk)})
                        section = section_matches[0] if section_matches else None
                        cursor = connection.execute(
                            "INSERT INTO chunks(page, chunk_no, section, text) VALUES (?, ?, ?, ?)",
                            (page_num, chunk_index, section, chunk))
                        connection.execute(
                            "INSERT INTO chunks_fts(rowid, text, section) VALUES (?, ?, ?)",
                            (cursor.lastrowid, chunk, section or ""))
                        connection.executemany(
                            "INSERT INTO chunk_sections(section, chunk_id) VALUES (?, ?)",
                            [(value, cursor.lastrowid) for value in section_matches])
                        chunk_count += 1
                except Exception as exc:
                    error_pages.append({"page": page_num, "error": str(exc)[:300]})
                page_count = page_num
                connection.commit()
                if progress_callback:
                    progress_callback("ocr", page_num, len(document), chunk_count)
            connection.execute("INSERT INTO metadata VALUES ('indexed_pages', ?)", (str(page_count),))
            connection.execute("INSERT INTO metadata VALUES ('chunk_count', ?)", (str(chunk_count),))
            connection.execute("INSERT INTO metadata VALUES ('build_complete', 'true')")
            connection.commit()

        # SQLite may have a WAL sidecar; checkpoint before swapping into place.
        # sqlite3.Connection's context manager commits but does not close it.
        connection.close()
        checkpoint_connection = sqlite3.connect(temporary)
        checkpoint_connection.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        checkpoint_connection.close()
        temporary.replace(target)
    finally:
        for suffix in ("-wal", "-shm"):
            sidecar = Path(str(temporary) + suffix)
            if sidecar.exists():
                sidecar.unlink()
    result = {"database_path": str(target), "indexed_pages": page_count,
              "chunk_count": chunk_count, "failed_pages": error_pages}
    result.update(build_chroma_index(
        database_path=str(target), chroma_path=chroma_path,
        progress_callback=progress_callback))
    return result


def _chroma_embedding_function():
    """Use Chroma's ONNX build of all-MiniLM-L6-v2 without a PyTorch install."""
    from chromadb.utils.embedding_functions import ONNXMiniLM_L6_V2

    return ONNXMiniLM_L6_V2(preferred_providers=["CPUExecutionProvider"])


def _split_for_embedding(text: str, tokenizer, max_tokens: int = 220,
                         overlap_words: int = 25) -> List[str]:
    def token_count(value: str) -> int:
        # Chroma's ONNX tokenizer pads to 256 positions; count the attention mask
        # so padding is not confused with real tokens.
        return int(sum(tokenizer.encode(value).attention_mask))

    words = []
    for word in text.split():
        if token_count(word) <= max_tokens:
            words.append(word)
            continue
        # Extremely long OCR strings without spaces are split into manageable pieces.
        width = max(16, len(word) // max(2, token_count(word) // max_tokens + 1))
        words.extend(word[start:start + width] for start in range(0, len(word), width))
    segments: List[str] = []
    current: List[str] = []
    for word in words:
        candidate = current + [word]
        if token_count(" ".join(candidate)) > max_tokens and current:
            segments.append(" ".join(current))
            carry = current[-overlap_words:]
            while carry and token_count(" ".join(carry + [word])) > max_tokens:
                carry.pop(0)
            current = carry + [word]
        else:
            current = candidate
    if current:
        segments.append(" ".join(current))
    return segments or [text]


def build_chroma_index(database_path: str = str(DEFAULT_KB_PATH),
                       chroma_path: str = str(DEFAULT_CHROMA_PATH),
                       progress_callback=None, batch_size: int = 64) -> Dict[str, object]:
    """Embed OCR chunks into a persistent Chroma collection using HNSW."""
    import chromadb

    archive = Path(database_path).expanduser().resolve()
    if not archive.is_file():
        raise FileNotFoundError(f"OCR archive not found: {archive}")
    with sqlite3.connect(archive) as connection:
        metadata = dict(connection.execute("SELECT key, value FROM metadata"))
        if metadata.get("build_complete") != "true":
            raise RuntimeError("OCR knowledge archive is incomplete; rebuild it before indexing.")
        rows = connection.execute(
            "SELECT id, page, chunk_no, section, text, "
            "(SELECT group_concat(section, ',') FROM chunk_sections WHERE chunk_id=chunks.id) "
            "FROM chunks ORDER BY id").fetchall()
    if not rows:
        raise RuntimeError("OCR archive contains no searchable text chunks.")

    store = Path(chroma_path).expanduser().resolve()
    store.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(store))
    embedding_fn = _chroma_embedding_function()
    build_name = f"{CHROMA_COLLECTION_NAME}_building"
    backup_name = f"{CHROMA_COLLECTION_NAME}_previous"

    # Build a replacement collection while any current index remains queryable.
    for stale_name in (build_name, backup_name):
        try:
            client.delete_collection(stale_name)
        except Exception:
            pass
    collection = client.create_collection(
        name=build_name,
        embedding_function=embedding_fn,
        metadata={**HNSW_METADATA, "edition": "ASME Y14.5M-1994"},
    )
    # Load the compact ONNX model once and keep each passage within its 256-token limit.
    embedding_fn._download_model_if_not_exists()
    tokenizer = embedding_fn.tokenizer
    vector_rows = []
    for row in rows:
        for segment_index, text in enumerate(_split_for_embedding(row[4], tokenizer)):
            vector_rows.append((row, segment_index, text))
    total_batches = (len(vector_rows) + batch_size - 1) // batch_size
    if progress_callback:
        progress_callback("vectors", 0, total_batches, 0)
    for batch_index, start in enumerate(range(0, len(vector_rows), batch_size), start=1):
        batch = vector_rows[start:start + batch_size]
        collection.add(
            ids=[f"asme1994-p{row[1]:03d}-c{row[2]:03d}-s{segment_index:02d}"
                 for row, segment_index, _ in batch],
            documents=[text for _, _, text in batch],
            metadatas=[{"page": row[1], "chunk_no": row[2], "segment": segment_index,
                        "section": row[3] or "", "sections": row[5] or "",
                        "edition": "ASME Y14.5M-1994"}
                       for row, segment_index, _ in batch],
        )
        if progress_callback:
            progress_callback("vectors", batch_index, total_batches,
                              min(start + len(batch), len(vector_rows)))

    # Swap collection names after the new HNSW index has been fully populated.
    try:
        client.get_collection(CHROMA_COLLECTION_NAME)
        had_previous = True
    except Exception:
        had_previous = False
    try:
        if had_previous:
            client.get_collection(CHROMA_COLLECTION_NAME).modify(name=backup_name)
        collection.modify(name=CHROMA_COLLECTION_NAME)
        if had_previous:
            client.delete_collection(backup_name)
    except Exception:
        try:
            client.delete_collection(CHROMA_COLLECTION_NAME)
        except Exception:
            pass
        if had_previous:
            try:
                client.get_collection(backup_name).modify(name=CHROMA_COLLECTION_NAME)
            except Exception:
                pass
        raise

    return {"vector_store_path": str(store), "collection_name": CHROMA_COLLECTION_NAME,
            "vector_count": len(vector_rows), "hnsw_metadata": HNSW_METADATA}


def retrieve_context(query: str, database_path: str = str(DEFAULT_KB_PATH),
                     limit: int = 4, max_chars: int = 6500,
                     collection=None) -> List[Dict[str, object]]:
    """Retrieve page-addressable passages with Chroma vector search or FTS fallback."""
    if collection is not None and collection.count():
        section_match = SECTION_RE.search(query)
        archive = Path(database_path)
        if section_match and archive.is_file():
            try:
                with sqlite3.connect(archive) as connection:
                    matches = connection.execute(
                        "SELECT c.page, c.chunk_no, cs.section FROM chunk_sections cs "
                        "JOIN chunks c ON c.id=cs.chunk_id WHERE cs.section=? "
                        "ORDER BY c.page, c.chunk_no",
                        (section_match.group(1),)).fetchall()
                exact = []
                for page, chunk_no, section in matches:
                    docs = collection.get(
                        where={"$and": [{"page": page}, {"chunk_no": chunk_no}]},
                        include=["documents", "metadatas"],
                    )
                    pairs = list(zip(docs.get("documents") or [], docs.get("metadatas") or []))
                    pairs.sort(key=lambda pair: pair[1].get("segment", 0))
                    if not pairs:
                        continue
                    text = " ".join(part for part, _ in pairs if part)
                    exact.append({"page": page, "section": section,
                                  "text": text, "distance": 0.0})
                    if len(exact) >= limit:
                        break
                if exact:
                    output = []
                    remaining = max_chars
                    for item in exact:
                        text = item["text"][:remaining]
                        if not text:
                            break
                        output.append({**item, "text": text})
                        remaining -= len(text)
                    return output
            except (sqlite3.Error, ValueError):
                pass

        result = collection.query(
            query_texts=[query],
            n_results=min(limit, collection.count()),
            include=["documents", "metadatas", "distances"],
        )
        passages = []
        remaining = max_chars
        documents = (result.get("documents") or [[]])[0]
        metadatas = (result.get("metadatas") or [[]])[0]
        distances = (result.get("distances") or [[]])[0]
        query_section = SECTION_RE.search(query)
        for text, meta, distance in zip(documents, metadatas, distances):
            text = (text or "")[:remaining]
            if not text:
                break
            sections = (meta.get("sections", "") or "").split(",")
            section = (query_section.group(1) if query_section and query_section.group(1) in sections
                       else meta.get("section"))
            passages.append({"page": meta.get("page"), "section": section or None,
                             "text": text, "distance": distance})
            remaining -= len(text)
        return passages

    # Keyword fallback supports retrieval before a Chroma index has been built.
    database = Path(database_path)
    if not database.is_file():
        return []
    terms = []
    for term in WORD_RE.findall(query.lower()):
        normalized = term.strip(".")
        if (len(normalized) > 2 and normalized not in STOP_WORDS
                and not re.fullmatch(r"\d+(?:\.\d+)?", normalized)
                and normalized not in terms):
            terms.append(normalized)
    if not terms:
        return []
    # Require matching terms when possible. If OCR noise prevents a match,
    # progressively relax the tail instead of matching common page headers.
    fts_queries = []
    useful_terms = terms[-10:]
    while len(useful_terms) >= 2:
        fts_queries.append(" AND ".join('"' + term.replace('"', '""') + '"'
                                        for term in useful_terms))
        useful_terms = useful_terms[:-1]
    results = []
    with sqlite3.connect(database) as connection:
        connection.row_factory = sqlite3.Row
        section_match = SECTION_RE.search(query)
        if section_match:
            try:
                results = connection.execute(
                    "SELECT c.page, cs.section, c.text FROM chunk_sections cs "
                    "JOIN chunks c ON c.id=cs.chunk_id WHERE cs.section=? "
                    "ORDER BY c.page, c.chunk_no LIMIT ?",
                    (section_match.group(1), limit)).fetchall()
            except sqlite3.OperationalError:
                results = []
        for fts_query in fts_queries:
            if results:
                break
            try:
                results = connection.execute(
                    "SELECT c.page, c.section, c.text FROM chunks_fts f "
                    "JOIN chunks c ON c.id=f.rowid WHERE chunks_fts MATCH ? "
                    "ORDER BY bm25(chunks_fts), c.page LIMIT ?",
                    (fts_query, limit)).fetchall()
                if results:
                    break
            except sqlite3.OperationalError:
                continue
    output = []
    remaining = max_chars
    for row in results:
        text = row["text"][:remaining]
        if not text:
            break
        output.append({"page": row["page"], "section": row["section"], "text": text})
        remaining -= len(text)
    return output


def knowledge_base_info(database_path: str = str(DEFAULT_KB_PATH)) -> Optional[Dict[str, str]]:
    database = Path(database_path)
    if not database.is_file():
        return None
    with sqlite3.connect(database) as connection:
        info = dict(connection.execute("SELECT key, value FROM metadata"))
    if info.get("build_complete") != "true":
        return None
    info["database_path"] = str(database)
    return info
