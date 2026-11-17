"""Answer natural-language questions from Defter: curated notes first, conversations after.

Notes are ranked by reciprocal-rank fusion of FTS5 keyword search and bge-m3 similarity over overlapping word
windows; conversations by bge-m3 similarity over pieces cut between exchanges (`chat_passages`), with keyword
search over the pieces standing in where vectors are missing. Passage vectors are a
cache keyed by passage hash and model. `embed` fills it; refresh never calls a model. Without vectors or a
reachable embedding server, recall uses keyword search alone and says so.
"""
from __future__ import annotations

import fcntl
import hashlib
import json
import os
import re
import sqlite3
import urllib.request
from collections.abc import Callable
from contextlib import closing
from datetime import date
from pathlib import Path
from typing import NotRequired, TypedDict

import numpy as np
from cfv.db import fold_tr

NOTE_KIND = "brain_note"
CHAT_KINDS = ("brain_conversation", "brain_session")
KEYWORD_KINDS = ("calendar",)  # matched by keywords alone, after the conversations
PASSAGE_WORDS, PASSAGE_STEP = 220, 110
CHAT_WORDS, QUESTION_WORDS = 400, 30
RRF_K = 60
CANDIDATES = 30
CHAT_POOL = 300  # keyword-matched pieces read to find the best CANDIDATES conversations
EXCERPT_LIMIT = 600
PREFIX = 5  # Turkish suffixes: match a word's first five letters (fixed-prefix stemming)

SCHEMA = """
CREATE TABLE IF NOT EXISTS passage (
    event_id INTEGER NOT NULL REFERENCES event(id) ON DELETE CASCADE,
    seq INTEGER NOT NULL, hash TEXT NOT NULL, PRIMARY KEY (event_id, seq));
CREATE INDEX IF NOT EXISTS passage_hash ON passage (hash);
CREATE TABLE IF NOT EXISTS embedding (
    hash TEXT NOT NULL, model TEXT NOT NULL, vector BLOB NOT NULL, PRIMARY KEY (hash, model));
CREATE VIRTUAL TABLE IF NOT EXISTS passage_fts USING fts5(
    text, content='', tokenize='unicode61 remove_diacritics 2');
"""

Embedder = Callable[[list[str]], list[list[float]]]
_TOKEN = re.compile(r"\w+", re.UNICODE)
# Turn headers: the extractor's `**user** · 12:01Z` line and the older exports' inline `**User** (date): `.
_TURN = re.compile(r"^\*\*(user|assistant)\*\*(?: ·[^\n]*(?:\n|$)| \([^)\n]*\):? ?|: ?)", re.M | re.I)
_TOOL_LINE = re.compile(r"^- `[^`\n]+` — [^\n]*$", re.M)  # the extractor's one-line tool-call summaries
_PARAGRAPH = re.compile(r"\S[^\n]*(?:\n[ \t]*\S[^\n]*)*")
_PLAIN = str.maketrans("ıİçğöşüâîûÇĞÖŞÜÂÎÛ", "iicgosuaiucgosuaiu")  # one character each, so offsets survive
_VECTORS: dict[tuple, tuple] = {}


class RecallResult(TypedDict):
    evidence: list[dict]
    semantic: bool
    warning: NotRequired[str]


def settings() -> tuple[str, str]:
    """Embedding endpoint (OpenAI-compatible /v1/embeddings) and model, from the environment."""
    return (os.environ.get("DEFTER_EMBED_URL", "http://localhost:11434/v1/embeddings"),
            os.environ.get("DEFTER_EMBED_MODEL", "bge-m3"))


def http_embedder(url: str, model: str, timeout: float = 120) -> Embedder:
    def embed(texts):
        request = urllib.request.Request(url, data=json.dumps({"model": model, "input": texts}).encode(),
                                         headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(request, timeout=timeout) as response:
            data = json.loads(response.read())["data"]
        return [item["embedding"] for item in sorted(data, key=lambda item: item["index"])]
    return embed


def passages(body: str) -> list[str]:
    """Overlapping windows of 220 words, every word covered, no window contained in the previous one."""
    words = body.split()
    return [" ".join(words[i:i + PASSAGE_WORDS])
            for i in range(0, max(len(words) - PASSAGE_STEP, 1), PASSAGE_STEP)]


def chat_passages(title: str | None, body: str) -> list[tuple[str, int]]:
    """A conversation in pieces of at most CHAT_WORDS words, with each piece's character offset in the body.

    An exchange is one user turn and the replies to it. Consecutive exchanges share a piece while they fit; a longer
    one is cut between paragraphs, and its later pieces repeat the start of its question. Tool-call lines are left
    out. Each piece starts with the conversation's title. Text appended to a conversation changes only its last
    pieces, so earlier pieces keep their cached vectors.
    """
    marks = list(_TURN.finditer(body))
    turns = [("", 0, 0, marks[0].start() if marks else len(body))] if not marks or marks[0].start() else []
    turns += [(mark[1].lower(), mark.start(), mark.end(), marks[i + 1].start() if i + 1 < len(marks) else len(body))
              for i, mark in enumerate(marks)]
    exchanges: list[list[tuple[int, list[str]]]] = []
    preamble = False  # text before the first turn (a title heading) belongs to the first exchange
    for role, header, start, end in turns:
        paragraphs = [(match.start(), words) for match in _PARAGRAPH.finditer(body, start, end)
                      if (words := _TOOL_LINE.sub("", match.group()).split())]
        if not paragraphs:
            continue
        if role:  # a piece that starts with this turn starts at its header
            paragraphs[0] = (header, [f"{role.capitalize()}:", *paragraphs[0][1]])
        if not exchanges or (role == "user" and not preamble):
            exchanges.append([])
        preamble = not role and len(exchanges) == 1
        exchanges[-1].extend(paragraphs)
    pieces: list[tuple[int, list[str] | None, list[str]]] = []
    current: list[str] = []
    lead: list[str] | None = None
    at = 0

    def close(next_lead=None):
        nonlocal current, lead
        if current:
            pieces.append((at, lead, current))
        current, lead = [], next_lead

    for exchange in exchanges:
        size = sum(len(words) for _, words in exchange)
        if size <= CHAT_WORDS < len(current) + size:  # fits a fresh piece, not this one; a longer one is cut anyway
            close()
        question = next((words for _, words in exchange if words[0] == "User:"), exchange[0][1])[:QUESTION_WORDS]
        begun = False  # once part of this exchange is in a piece, later pieces repeat its question
        for offset, words in exchange:
            if len(words) <= CHAT_WORDS < len(current) + len(words):  # cut between paragraphs where one fits
                close(question if begun else None)
            position = 0
            while position < len(words):  # a paragraph longer than a piece fills this one, then the next
                if len(current) >= CHAT_WORDS:
                    close(question if begun else None)
                if not current:
                    at = offset
                part = words[position:position + CHAT_WORDS - len(current)]
                current.extend(part)
                position += len(part)
                begun = True
    close()
    head = (title or "").strip()
    return [("\n".join(line for line in (head, f"[{' '.join(lead)} …]" if lead else "", " ".join(words)) if line), at)
            for at, lead, words in pieces]


def passage_texts(kind: str, title: str | None, body: str | None) -> list[str]:
    """The passages a record is embedded and searched by: note windows or conversation pieces."""
    if kind == NOTE_KIND:
        return passages(body or "")
    return [text for text, _ in chat_passages(title, body or "")]


def passage_hash(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def index_passages(conn: sqlite3.Connection):
    """Rebuild the passage map and the conversation pieces' keyword index. Called by refresh inside its staging
    transaction."""
    conn.execute("DELETE FROM passage")
    conn.execute("INSERT INTO passage_fts(passage_fts) VALUES ('delete-all')")
    rows = conn.execute(f"SELECT id, kind, title, body FROM event WHERE kind IN ({','.join('?' * (1 + len(CHAT_KINDS)))})",
                        (NOTE_KIND, *CHAT_KINDS))
    passage_rows, keyword_rows = [], []
    for event_id, kind, title, body in rows:
        for seq, text in enumerate(passage_texts(kind, title, body)):
            passage_rows.append((len(passage_rows) + 1, event_id, seq, passage_hash(text)))
            if kind != NOTE_KIND:
                keyword_rows.append((len(passage_rows), fold_tr(text)))
    conn.executemany("INSERT INTO passage (rowid, event_id, seq, hash) VALUES (?, ?, ?, ?)", passage_rows)
    conn.executemany("INSERT INTO passage_fts (rowid, text) VALUES (?, ?)", keyword_rows)


def _unit(vectors) -> np.ndarray:
    matrix = np.asarray(vectors, dtype=np.float32)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    return matrix / np.where(norms == 0, 1, norms)


def embed(path: Path, embedder: Embedder | None = None, model: str | None = None, batch: int = 16,
          progress: Callable[[int, int], None] | None = None, limit: int | None = None) -> dict:
    """Embed passages that lack a vector for the model, notes first, then conversations newest first, committing per
    batch; drop vectors no passage uses. With `limit`, stop after that many and report how many remain."""
    url, default_model = settings()
    model = model or default_model
    embedder = embedder or http_embedder(url, model)
    with open(path.with_suffix(".lock"), "a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        with closing(sqlite3.connect(path)) as conn:
            if not conn.execute("SELECT name FROM sqlite_master WHERE name='passage'").fetchone():
                raise ValueError("No passage table. Run refresh first.")
            rows = conn.execute(
                "SELECT p.event_id, p.seq, p.hash FROM passage p JOIN event e ON e.id=p.event_id "
                "WHERE NOT EXISTS (SELECT 1 FROM embedding m WHERE m.hash=p.hash AND m.model=?) "
                "ORDER BY e.kind != ?, e.start_utc DESC, p.event_id, p.seq", (model, NOTE_KIND)).fetchall()
            todo, owner, texts = {}, None, []
            for event_id, seq, digest in rows:
                if digest in todo or (limit is not None and len(todo) >= limit):
                    continue
                if event_id != owner:
                    owner = event_id
                    texts = passage_texts(*conn.execute("SELECT kind, title, body FROM event WHERE id=?",
                                                        (event_id,)).fetchone())
                todo[digest] = texts[seq]
            remaining = len({digest for _, _, digest in rows}) - len(todo)
            items = list(todo.items())
            for start in range(0, len(items), batch):
                chunk = items[start:start + batch]
                vectors = _unit(embedder([text for _, text in chunk]))
                with conn:
                    conn.executemany("INSERT OR REPLACE INTO embedding VALUES (?, ?, ?)",
                                     [(digest, model, vector.tobytes()) for (digest, _), vector in zip(chunk, vectors)])
                if progress:
                    progress(min(start + batch, len(items)), len(items))
            with conn:
                pruned = conn.execute("DELETE FROM embedding WHERE hash NOT IN (SELECT hash FROM passage)").rowcount
            total = conn.execute("SELECT count(*) FROM passage").fetchone()[0]
    return {"model": model, "embedded": len(items), "remaining": remaining, "pruned": pruned, "passages": total}


def copy_vectors(source: Path, path: Path) -> int:
    """Add the cached vectors of another Defter database that this one's passages use and lack, and say how many.

    Vectors are keyed by passage text and model, so passages embedded on another machine (a GPU laptop) serve here
    unchanged; only the database file travels.
    """
    with open(path.with_suffix(".lock"), "a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        with closing(sqlite3.connect(path.resolve().as_uri(), uri=True)) as conn:
            if not conn.execute("SELECT name FROM sqlite_master WHERE name='passage'").fetchone():
                raise ValueError("No passage table. Run refresh first.")
            conn.execute("ATTACH DATABASE ? AS other", (source.resolve().as_uri() + "?mode=ro",))
            if not conn.execute("SELECT name FROM other.sqlite_master WHERE name='embedding'").fetchone():
                raise ValueError(f"No vectors in {source}")
            with conn:
                copied = conn.execute(
                    "INSERT OR IGNORE INTO embedding SELECT o.hash, o.model, o.vector FROM other.embedding o "
                    "WHERE o.hash IN (SELECT hash FROM passage)").rowcount
            conn.execute("DETACH DATABASE other")
    return copied


def export_vectors(path: Path, out: Path) -> int:
    """Write only the embedding cache to a new database file and say how many vectors it holds.

    The laptop database can hold personal life data (defter.life); this file holds none, so it is what to copy to the
    VPS before `copy-vectors` there.
    """
    if out.exists():
        raise ValueError(f"{out} exists; remove it first")
    with closing(sqlite3.connect(out)) as conn:
        conn.execute("ATTACH DATABASE ? AS src", (path.resolve().as_uri() + "?mode=ro",))
        conn.execute("CREATE TABLE embedding (hash TEXT NOT NULL, model TEXT NOT NULL, vector BLOB NOT NULL, "
                     "PRIMARY KEY (hash, model))")
        with conn:
            conn.execute("INSERT INTO embedding SELECT hash, model, vector FROM src.embedding")
        return conn.execute("SELECT count(*) FROM embedding").fetchone()[0]


def keyword_query(text: str, every: bool = False) -> str:
    """The question's words, Turkish-folded; words longer than five letters match by their first five.

    Any word may match, or with every=True each word of four or more letters must appear. Whole conversations
    are long enough to contain most common words, so they need the stricter form.
    """
    tokens = dict.fromkeys(t for t in _TOKEN.findall(fold_tr(text).lower()) if len(t) > (3 if every else 1))
    return (" AND " if every else " OR ").join(f'"{t[:PREFIX]}"*' if len(t) > PREFIX else f'"{t}"' for t in tokens)


def excerpt(text: str, query: str) -> str:
    """A window of the text around its first occurrence of the question's most specific word."""
    plain = text.translate(_PLAIN).lower()
    words = sorted({t[:PREFIX + 3] for t in _TOKEN.findall(query.translate(_PLAIN).lower()) if len(t) > 3}, key=len,
                   reverse=True)
    position = next((p for p in (plain.find(w) for w in words) if p >= 0), 0)
    begin = max(0, min(position - EXCERPT_LIMIT // 3, len(text) - EXCERPT_LIMIT))
    return ("…" if begin else "") + text[begin:begin + EXCERPT_LIMIT]


def _dates(start, end) -> tuple[list[str], list[str]]:
    where, args = [], []
    if start:
        where.append("e.local_date >= ?")
        args.append(start)
    if end:
        where.append("e.local_date <= ?")
        args.append(end)
    return where, args


def _keyword(conn, query: str, kinds: tuple[str, ...], start, end, limit: int, every: bool = False) -> list[int]:
    match = keyword_query(query, every)
    if not match:
        return []
    where, args = _dates(start, end)
    rows = conn.execute(
        "SELECT e.id FROM event_fts JOIN event e ON e.id=event_fts.rowid "
        f"WHERE event_fts MATCH ? AND e.kind IN ({','.join('?' * len(kinds))}) {''.join(' AND ' + w for w in where)} "
        "ORDER BY bm25(event_fts) LIMIT ?", (match, *kinds, *args, limit))
    return [row[0] for row in rows]


def _chat_keyword(conn, query: str, start, end) -> tuple[list[int], dict[int, int]]:
    """Conversations ranked by their best keyword-matching piece, and that piece per conversation."""
    match = keyword_query(query)
    if not match:
        return [], {}
    where, args = _dates(start, end)
    rows = conn.execute(
        "SELECT p.event_id, p.seq FROM passage_fts JOIN passage p ON p.rowid=passage_fts.rowid "
        f"JOIN event e ON e.id=p.event_id WHERE passage_fts MATCH ? {''.join(' AND ' + w for w in where)} "
        "ORDER BY bm25(passage_fts) LIMIT ?", (match, *args, CHAT_POOL))
    best: dict[int, int] = {}
    for event_id, seq in rows:
        best.setdefault(event_id, seq)
        if len(best) >= CANDIDATES:
            break
    return list(best), best


def _vectors(conn, path: Path, model: str):
    """The model's vectors, one row per distinct passage text, and for each passage its owner, sequence number,
    lane, day and row; cached per database file version.

    The vectors are read in one pass in table order. Looking each passage's vector up by hash instead reads the file
    in random order: 9.1 s against 3.4 s for 37k vectors from a cold cache on the VPS (2026-11-17).
    """
    stat = path.stat()
    key = (str(path.resolve()), stat.st_mtime_ns, stat.st_size, model)
    if key not in _VECTORS:
        _VECTORS.clear()
        count, size = conn.execute("SELECT count(*), max(length(vector)) FROM embedding WHERE model=?",
                                   (model,)).fetchone()
        if not count:
            _VECTORS[key] = None
            return None
        matrix = np.empty((count, size // 4), dtype=np.float32)  # filled row by row: one copy in memory
        row_of = {}
        for i, (digest, vector) in enumerate(conn.execute("SELECT hash, vector FROM embedding WHERE model=?", (model,))):
            matrix[i] = np.frombuffer(vector, dtype=np.float32)
            row_of[digest] = i
        owners, seqs, notes, days, rows = [], [], [], [], []
        for owner, seq, is_note, day, digest in conn.execute(
                "SELECT p.event_id, p.seq, e.kind=?, e.local_date, p.hash FROM passage p JOIN event e ON e.id=p.event_id "
                "ORDER BY p.event_id, p.seq", (NOTE_KIND,)):
            if digest in row_of:
                owners.append(owner), seqs.append(seq), notes.append(is_note), days.append(day)
                rows.append(row_of[digest])
        if not rows:
            _VECTORS[key] = None
            return None
        _VECTORS[key] = (np.array(owners), np.array(seqs), np.array(notes, dtype=bool), np.array(days),
                         np.array(rows), matrix)
    return _VECTORS[key]


def _semantic(conn, path, query, start, end, embedder) -> tuple[dict[bool, tuple[list[int], dict[int, int]]], str | None]:
    """For notes (True) and conversations (False): records ranked by their best passage's cosine similarity, with that
    passage per record; and any reason it failed."""
    empty = {True: ([], {}), False: ([], {})}
    url, model = settings()
    if not conn.execute("SELECT name FROM sqlite_master WHERE name='embedding'").fetchone():
        return empty, "no passage vectors: run refresh, then embed"
    index = _vectors(conn, path, model)
    if index is None:
        return empty, f"no passage vectors for {model}: run embed"
    owners, seqs, is_note, days, rows, matrix = index
    try:
        vector = _unit((embedder or http_embedder(url, model, timeout=30))([query]))[0]
    except Exception as exc:  # an unreachable or failing server must not break keyword recall
        return empty, f"embedding server failed ({type(exc).__name__}): keyword results only"
    if vector.shape[0] != matrix.shape[1]:
        return empty, f"query vector has {vector.shape[0]} dimensions, stored vectors {matrix.shape[1]}: rerun embed"
    sims = (matrix @ vector)[rows]  # one score per distinct text, then one per passage
    keep = np.ones(len(sims), dtype=bool)
    if start:
        keep &= days >= start
    if end:
        keep &= days <= end
    rankings = {}
    for lane in (True, False):
        best: dict[int, int] = {}
        candidates = np.flatnonzero(keep & (is_note == lane))
        for i in candidates[np.argsort(-sims[candidates])]:
            best.setdefault(int(owners[i]), int(seqs[i]))
            if len(best) >= CANDIDATES:
                break
        rankings[lane] = (list(best), best)
    return rankings, None


def _unembedded(conn, event_ids: list[int], model: str) -> set[int]:
    """The given conversations that have a piece without a vector for the model."""
    if not event_ids:
        return set()
    rows = conn.execute(
        f"SELECT DISTINCT p.event_id FROM passage p WHERE p.event_id IN ({','.join('?' * len(event_ids))}) "
        "AND NOT EXISTS (SELECT 1 FROM embedding m WHERE m.hash=p.hash AND m.model=?)", (*event_ids, model))
    return {row[0] for row in rows}


def _fuse(*rankings: list[int]) -> list[int]:
    scores: dict[int, float] = {}
    for ranking in rankings:
        for rank, event_id in enumerate(ranking):
            scores[event_id] = scores.get(event_id, 0.0) + 1 / (RRF_K + rank + 1)
    return sorted(scores, key=lambda event_id: (-scores[event_id], event_id))


def recall(path: Path, query: str, start: str | None = None, end: str | None = None, limit: int = 8,
           conversations: int = 4, embedder: Embedder | None = None) -> RecallResult:
    """Notes ranked by keyword and passage similarity together, then conversations ranked by their best piece.

    Conversations are ranked by meaning: on 2026-11-15 probes, adding keyword ranks at any weight made both reworded
    and exact-term questions worse. Keyword matches stand in where meaning is blind: conversations with pieces not
    yet embedded, or every conversation when there are no vectors or the embedding server fails. A conversation's
    excerpt comes from its best piece, and `offset` is where that piece starts in the text `read` returns.
    Conversations whose best piece is word-for-word the same as an earlier result's are left out.
    """
    if not query.strip() or len(query) > 500:
        raise ValueError("Use a question of 1 to 500 characters")
    for value in (start, end):
        if value:
            date.fromisoformat(value)
    if start and end and start > end:
        raise ValueError("The start date must be on or before the end date")
    limit, conversations = max(1, min(limit, 30)), max(0, min(conversations, 20))
    with closing(sqlite3.connect(path.resolve().as_uri() + "?mode=ro", uri=True)) as conn:
        conn.execute("BEGIN")
        semantic, warning = _semantic(conn, path, query, start, end, embedder)
        keyword = {True: (_keyword(conn, query, (NOTE_KIND,), start, end, CANDIDATES), {}),
                   False: _chat_keyword(conn, query, start, end)}
        calendar = _keyword(conn, query, KEYWORD_KINDS, start, end, conversations, every=True)
        notes = _fuse(keyword[True][0], semantic[True][0])[:limit]
        if warning is None:  # meaning ranks conversations; keywords stand in where it is blind (pieces not yet embedded)
            blind = _unembedded(conn, keyword[False][0], settings()[1])
            keyword[False] = ([event_id for event_id in keyword[False][0] if event_id in blind], keyword[False][1])
        chats = _fuse(keyword[False][0], semantic[False][0], calendar) if conversations else []
        conn.row_factory = sqlite3.Row
        evidence, seen = [], set()
        for event_id in notes + chats:
            if event_id not in notes and len(evidence) >= len(notes) + conversations:
                break
            row = conn.execute(
                "SELECT e.kind, e.start_utc, e.title, e.body, e.payload, s.raw_path, r.ref FROM event e "
                "JOIN source s ON s.id=e.source_id LEFT JOIN brain_reference r ON r.event_id=e.id WHERE e.id=?",
                (event_id,)).fetchone()
            payload, body, kind = json.loads(row["payload"] or "{}"), row["body"] or "", row["kind"]
            lane = kind == NOTE_KIND
            seq = semantic[lane][1].get(event_id, keyword[lane][1].get(event_id))
            found = {"reference": row["ref"], "kind": kind, "timestamp": row["start_utc"], "title": row["title"],
                     "path": payload.get("note_path"), "source": payload.get("source_file") or row["raw_path"]}
            if lane:
                text = passages(body)[seq] if seq is not None else body
            elif kind in CHAT_KINDS and seq is not None:
                text, found["offset"] = chat_passages(row["title"], body)[seq]
                if passage_hash(text) in seen:
                    continue
                seen.add(passage_hash(text))
            else:
                text = body
            found.update(excerpt=excerpt(text, query), supports=payload.get("evidence_refs", []),
                         matched=list(dict.fromkeys(name for name, ranking in (
                             ("keyword", keyword[lane][0]), ("semantic", semantic[lane][0]), ("keyword", calendar))
                             if event_id in ranking)))
            evidence.append(found)
    result = {"evidence": evidence, "semantic": warning is None}
    if warning:
        result["warning"] = warning
    return result
