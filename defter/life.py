"""Personal sources for the CFV site, from files an Android user can export: WhatsApp chats, Google Takeout.

Layout of the life folder (default data/private/life, outside git):
    calendar/*.ics      Takeout > Calendar
    mail/*.mbox         Takeout > Mail. Only headers and a 200-character snippet are kept, never whole bodies.
    contacts/*.vcf      Takeout > Contacts, or any vCard export
    whatsapp/*.txt|zip  WhatsApp "Export chat" files (without media)
    me.txt              optional, one per line: your WhatsApp display names and e-mail addresses
`refresh --life` reads the folder on the machine that runs it. The VPS refresh never gets this folder, and the
change log hides these kinds from `delta`, so agents served from the VPS see none of it.
"""
from __future__ import annotations

import datetime as dt
import hashlib
import mailbox
import quopri
import re
import shutil
import unicodedata
import zipfile
from collections import Counter
from email import message_from_bytes, policy
from email.utils import getaddresses, parsedate_to_datetime
from pathlib import Path

from cfv.ingest import Batch, drop_other_snapshots
from cfv.sources import calendar_ics
from cfv.sources.contacts import e164, whatsapp_jid
from cfv.timeutil import DEFAULT_TZ

IMPORTER = "defter.life@1"
SNIPPET = 200
SKIP_LABELS = {"spam", "trash", "drafts", "draft"}

# ------------------------------------------------------------------ WhatsApp
_ANDROID = re.compile(r"^(\d{1,2})[./](\d{1,2})[./](\d{2,4}),?\s(\d{1,2}):(\d{2})(?::(\d{2}))?\s?([AaPp][Mm])?\s-\s(.*)$")
_IOS = re.compile(r"^\[(\d{1,2})[./](\d{1,2})[./](\d{2,4}),?\s(\d{1,2}):(\d{2})(?::(\d{2}))?\s?([AaPp][Mm])?\]\s(.*)$")
_CHAT_NAME = re.compile(r"^WhatsApp (?:Chat with|Chat -|Sohbeti -|Sohbeti:?)\s*(.+)$", re.I)
_INVISIBLE = dict.fromkeys(map(ord, "‎‏‪‫‬‭‮﻿"), None)
_MEDIA = ("<media omitted>", "<medya dahil edilmedi>", "(file attached)", "(dosya ekli)", "image omitted",
          "video omitted", "audio omitted", "sticker omitted", "gif omitted", "document omitted")
_DELETED = ("this message was deleted", "you deleted this message", "bu mesaj silindi", "bu mesajı sildiniz")


def norm(text: str) -> str:
    """Comparison form of a name: Turkish-folded, no accents, lower case, single spaces."""
    text = text.replace("İ", "i").replace("I", "ı").lower().replace("ı", "i")
    text = "".join(c for c in unicodedata.normalize("NFKD", text) if not unicodedata.combining(c))
    return " ".join(re.sub(r"[^\w\s]", " ", text).split())


def parse_whatsapp(text: str) -> list[dict]:
    """Messages of one exported chat as {t, sender, text}; system lines have sender None, continuations are joined.

    The day/month order is read from the file: a first number above 12 means day first, a second one above 12 month
    first, and a file with neither is taken as day first (Turkey).
    """
    text = text.translate(_INVISIBLE).replace(" ", " ").replace(" ", " ")
    rows, tail = [], None
    for line in text.splitlines():
        m = _ANDROID.match(line) or _IOS.match(line)
        if m:
            rows.append([m.groups(), []])
            rows[-1][1].append(m.group(8))
        elif rows:
            rows[-1][1].append(line)
    firsts = [int(g[0]) for g, _ in rows]
    seconds = [int(g[1]) for g, _ in rows]
    day_first = any(n > 12 for n in firsts) or not any(n > 12 for n in seconds)
    out = []
    for g, lines in rows:
        a, b, year, hour, minute, second, ampm, _ = g
        day, month = (int(a), int(b)) if day_first else (int(b), int(a))
        year = int(year) + (2000 if int(year) < 100 else 0)
        hour = int(hour)
        if ampm:
            hour = hour % 12 + (12 if ampm.lower() == "pm" else 0)
        try:
            t = dt.datetime(year, month, day, hour, int(minute), int(second or 0), tzinfo=DEFAULT_TZ)
        except ValueError:
            if out:  # not a real timestamp: part of the previous message
                out[-1]["text"] += "\n" + "\n".join(lines)
            continue
        body = "\n".join(lines)
        who = re.match(r"^([^:\n]{1,60}?): (.*)$", body, re.S)
        out.append({"t": t, "sender": who.group(1).strip() if who else None, "text": (who.group(2) if who else body).strip()})
    return out


def chat_name(path_name: str) -> str:
    stem = path_name[:-4] if path_name.lower().endswith((".txt", ".zip")) else path_name
    m = _CHAT_NAME.match(stem)
    return (m.group(1) if m else stem).strip()


def whatsapp_chats(folder: Path) -> dict[str, list[dict]]:
    """Every export in the folder, by chat name; two exports of one chat (a zip and a text file) are merged."""
    merged: dict[str, dict[tuple, list[dict]]] = {}
    for path in sorted(folder.glob("*")):
        texts = []
        if path.suffix.lower() == ".txt":
            texts.append((path.name, path.read_text(encoding="utf-8-sig", errors="replace")))
        elif path.suffix.lower() == ".zip":
            with zipfile.ZipFile(path) as z:
                for info in z.infolist():
                    if info.filename.lower().endswith(".txt") and "__MACOSX" not in info.filename:
                        texts.append((path.name, z.read(info).decode("utf-8-sig", errors="replace")))
        for name, body in texts:
            per_file: dict[tuple, list[dict]] = {}
            for m in parse_whatsapp(body):
                per_file.setdefault((m["t"], m["sender"], m["text"]), []).append(m)
            chat = merged.setdefault(chat_name(name), {})
            for key, rows in per_file.items():  # a message in two exports is one; the same line twice in one export is two
                if len(rows) > len(chat.get(key, [])):
                    chat[key] = rows
    return {chat: [m for rows in keyed.values() for m in rows] for chat, keyed in merged.items()}


def ingest_whatsapp(conn, folder: Path, me: set[str] = frozenset()) -> dict:
    batch = Batch(conn, "whatsapp-export", folder, IMPORTER)
    drop_other_snapshots(conn, "whatsapp-export", batch.source_id)
    stats = {"chats": 0, "messages": 0, "skipped_system": 0}
    mine = {norm(n) for n in me}
    for chat, messages in sorted(whatsapp_chats(folder).items()):
        senders = Counter(m["sender"] for m in messages if m["sender"])
        kind = "group" if len(senders) > 2 else "dm"
        own = {s for s in senders if norm(s) in mine}
        if kind == "dm" and not own:  # the other side's name is the chat's; the remaining sender is you
            theirs = {s for s in senders if norm(s) == norm(chat) or norm(chat) in norm(s) or norm(s) in norm(chat)}
            rest = set(senders) - theirs
            own = rest if theirs and len(rest) == 1 else set()
        stats["chats"] += 1
        seen: dict[str, int] = {}
        for m in sorted(messages, key=lambda x: x["t"]):
            if m["sender"] is None:
                stats["skipped_system"] += 1
                continue
            digest = hashlib.sha1(f"{m['sender']}\0{m['text']}".encode()).hexdigest()[:10]
            ext = f"{chat}/{m['t'].isoformat()}/{digest}"
            seen[ext] = seen.get(ext, 0) + 1
            if seen[ext] > 1:
                ext += f"#{seen[ext]}"
            low = m["text"].lower()
            media = low.startswith("<") and low in _MEDIA or any(low.endswith(x) for x in _MEDIA)
            deleted = low in _DELETED
            from_me = m["sender"] in own if own else None
            eid = batch.event(
                ext, "whatsapp", m["t"], title=chat, tz_offset_min=int(m["t"].utcoffset().total_seconds() // 60),
                body="[medya]" if media and low in _MEDIA else ("[silinmiş mesaj]" if deleted else m["text"]),
                payload={"chat": chat, "chat_type": kind, "chat_jid": f"export:{chat}", "from_me": from_me,
                         "sender": m["sender"], "type": "media" if media else "deleted" if deleted else "text"})
            if not from_me:
                person = chat if kind == "dm" else m["sender"]
                batch.link_person(eid, batch.person(f"export:{person}", person, "whatsapp"),
                                  "counterpart" if kind == "dm" else "sender")
            stats["messages"] += 1
    return stats


# ------------------------------------------------------------------ mail
def _snippet(msg) -> str:
    part = msg.get_body(preferencelist=("plain", "html"))
    if part is None:
        return ""
    try:
        text = part.get_content()
    except (LookupError, ValueError, AssertionError):
        return ""
    if part.get_content_type() == "text/html":
        text = re.sub(r"<[^>]+>", " ", re.sub(r"(?is)<(script|style).*?</\1>", " ", text))
    return " ".join(text.split())[:SNIPPET]


def _addresses(*headers) -> list[tuple[str, str]]:
    return [(n.strip(), a.strip().lower()) for n, a in getaddresses([str(h) for h in headers if h]) if "@" in a]


def ingest_mail(conn, folder: Path, me: set[str] = frozenset()) -> dict:
    stats = {"emails": 0, "skipped": 0}
    for path in sorted(folder.glob("*.mbox")):
        batch = Batch(conn, "gmail-takeout", path, IMPORTER)
        records, sent_from = [], Counter()
        box = mailbox.mbox(path)
        for key in box.iterkeys():
            try:
                msg = message_from_bytes(box.get_bytes(key), policy=policy.default)
                when = parsedate_to_datetime(str(msg["Date"]))
            except (TypeError, ValueError, IndexError):
                stats["skipped"] += 1
                continue
            if when.tzinfo is None:
                when = when.replace(tzinfo=dt.timezone.utc)
            labels = [x.strip() for x in str(msg["X-Gmail-Labels"] or "").split(",") if x.strip()]
            if SKIP_LABELS & {x.lower() for x in labels}:
                stats["skipped"] += 1
                continue
            sender = _addresses(msg["From"])
            if "sent" in {x.lower() for x in labels} and sender:
                sent_from[sender[0][1]] += 1
            ident = str(msg["Message-ID"] or "").strip("<> ") or hashlib.sha1(
                f"{msg['From']}{msg['Date']}{msg['Subject']}".encode()).hexdigest()
            records.append((ident, when, labels, msg, _snippet(msg)))
        mine = {a.lower() for a in me if "@" in a} | set(sent_from)
        seen = set()
        for ident, when, labels, msg, snippet in records:
            if ident in seen:
                continue
            seen.add(ident)
            out = "sent" in {x.lower() for x in labels}
            to = ", ".join(str(x) for x in (msg["To"], msg["Cc"]) if x)
            body = "\n".join(x for x in (msg["From"] and f"Kimden: {msg['From']}", to and f"Kime: {to}", snippet) if x)
            eid = batch.event(ident, "email", when, title=str(msg["Subject"] or "") or None, body=body or None,
                              payload={"account": path.stem, "from": str(msg["From"] or "") or None, "to": to or None,
                                       "labels": labels, "thread": str(msg["X-GM-THRID"] or "") or None,
                                       "direction": "out" if out else "in"})
            for name, addr in (_addresses(msg["To"], msg["Cc"]) if out else _addresses(msg["From"])):
                if addr not in mine:
                    batch.link_person(eid, batch.person(addr, name or addr, "email"), "recipient" if out else "sender")
            stats["emails"] += 1
    return stats


# ------------------------------------------------------------------ contacts
def parse_vcards(text: str) -> list[dict]:
    """vCard 2.1/3.0/4.0 cards as {uid, name, phones, emails}."""
    text = re.sub(r"\r?\n[ \t]", "", text)
    cards = []
    for block in re.findall(r"BEGIN:VCARD(.*?)END:VCARD", text, re.S | re.I):
        card = {"uid": None, "name": None, "org": None, "phones": [], "emails": []}
        for line in block.strip().splitlines():
            head, sep, value = line.partition(":")
            if not sep:
                continue
            name, *params = head.split(";")
            name = name.split(".")[-1].upper()
            if any("QUOTED-PRINTABLE" in p.upper() for p in params):
                value = quopri.decodestring(value.encode()).decode("utf-8", errors="replace")
            value = value.strip()
            if name == "FN" and value:
                card["name"] = value
            elif name == "N" and not card["name"]:
                card["name"] = " ".join(p for p in value.split(";")[1::-1] if p).strip() or None
            elif name == "ORG" and value:
                card["org"] = value.replace(";", " ").strip()
            elif name == "UID":
                card["uid"] = value
            elif name == "TEL":
                n = e164(value)
                if n and n not in card["phones"]:
                    card["phones"].append(n)
            elif name == "EMAIL" and "@" in value and value.lower() not in card["emails"]:
                card["emails"].append(value.lower())
        cards.append(card)
    return cards


def ingest_contacts(conn, folder: Path) -> dict:
    """Same aliases as CFV's iPhone contacts importer: the card itself, E.164 phone, WhatsApp JID, e-mail."""
    cards = [c for p in sorted(folder.glob("*.vcf")) for c in parse_vcards(p.read_text(encoding="utf-8-sig", errors="replace"))]
    owners = Counter(n for c in cards for n in c["phones"])
    shared = {n for n, k in owners.items() if k > 1}
    stats = {"contacts": 0, "new_persons": 0, "matched_existing": 0}

    def owner(kind, alias):
        row = conn.execute("SELECT person_id FROM person_alias WHERE source_kind=? AND alias=?", (kind, alias)).fetchone()
        return row[0] if row else None

    for card in cards:
        phones = [n for n in card["phones"] if n not in shared]
        name = card["name"] or card["org"] or (phones[0] if phones else card["emails"][0] if card["emails"] else None)
        if not name:
            continue
        key = card["uid"] or "vcf:" + hashlib.sha1(f"{name}{phones}{card['emails']}".encode()).hexdigest()[:12]
        aliases = [("contacts", key)] + [a for n in phones for a in (("whatsapp", whatsapp_jid(n)), ("phone", n))]
        aliases += [("email", e) for e in card["emails"]]
        found = {a: owner(*a) for a in aliases}
        existing = [p for p in found.values() if p is not None]
        stats["contacts"] += 1
        if found[("contacts", key)] is not None:
            pid = found[("contacts", key)]
        elif existing:
            pid = existing[0]
            stats["matched_existing"] += 1
        else:
            pid = conn.execute("INSERT INTO person (display_name) VALUES (?)", (name,)).lastrowid
            stats["new_persons"] += 1
        for (kind, alias), p in found.items():
            if p is None:
                conn.execute("INSERT INTO person_alias (person_id, source_kind, alias, name) VALUES (?,?,?,?)",
                             (pid, kind, alias, name))
    return stats


# ------------------------------------------------------------------ entry points
def read_me(folder: Path) -> set[str]:
    path = folder / "me.txt"
    return {x.strip() for x in path.read_text(encoding="utf-8").splitlines() if x.strip() and not x.startswith("#")} \
        if path.is_file() else set()


def ingest_life(conn, folder: Path) -> dict:
    """Ingest whatever the life folder holds. Mail goes before contacts so a card joins the person its address made."""
    me, stats = read_me(folder), {}
    if (folder / "whatsapp").is_dir() and any((folder / "whatsapp").glob("*")):
        stats["whatsapp"] = ingest_whatsapp(conn, folder / "whatsapp", me)
    if (folder / "mail").is_dir() and any((folder / "mail").glob("*.mbox")):
        stats["mail"] = ingest_mail(conn, folder / "mail", me)
    if (folder / "contacts").is_dir() and any((folder / "contacts").glob("*.vcf")):
        stats["contacts"] = ingest_contacts(conn, folder / "contacts")
    files = sorted((folder / "calendar").glob("*.ics")) if (folder / "calendar").is_dir() else []
    if files:
        stats["calendar"] = calendar_ics.ingest(conn, files)
    return stats


_TAKEOUT = ((re.compile(r"(^|/)Calendar/[^/]+\.ics$", re.I), "calendar"),
            (re.compile(r"(^|/)Mail/[^/]+\.mbox$", re.I), "mail"),
            (re.compile(r"(^|/)Contacts/.+\.vcf$", re.I), "contacts"))


def unpack_takeout(zips: list[Path], folder: Path) -> dict[str, int]:
    """Copy the calendar, mail and contact files out of Takeout zips into the life folder; skip everything else."""
    counts = Counter()
    for number, path in enumerate(zips):
        with zipfile.ZipFile(path) as z:
            for info in z.infolist():
                for pattern, target in _TAKEOUT:
                    if pattern.search(info.filename):
                        dest = folder / target / (Path(info.filename).name if number == 0 else f"{number}-{Path(info.filename).name}")
                        dest.parent.mkdir(parents=True, exist_ok=True)
                        with z.open(info) as src, open(dest, "wb") as out:
                            shutil.copyfileobj(src, out)
                        counts[target] += 1
                        break
    return dict(counts)
