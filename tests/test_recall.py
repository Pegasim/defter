import hashlib
import re
import shutil
import sqlite3
from contextlib import closing
import tempfile
import unittest
from pathlib import Path

from defter.recall import CHAT_WORDS, QUESTION_WORDS, chat_passages, copy_vectors, embed, export_vectors, recall
from defter.store import ROOT, read, refresh

SYNONYMS = {"port": "harbour"}


def fake_embedder(calls=None):
    """Bag of words hashed into 64 dimensions; 'port' and 'harbour' share one, standing in for meaning."""
    def run(texts):
        if calls is not None:
            calls.append(len(texts))
        vectors = []
        for text in texts:
            vector = [0.0] * 64
            for word in re.findall(r"\w+", text.lower()):
                word = SYNONYMS.get(word, word)
                vector[int(hashlib.md5(word.encode()).hexdigest(), 16) % 64] += 1
            vectors.append(vector)
        return vectors
    return run


def failing_embedder(texts):
    raise OSError("connection refused")


class RecallTest(unittest.TestCase):
    def setUp(self):
        self.temporary = Path(tempfile.mkdtemp())
        self.vault, self.db = self.temporary / "vault", self.temporary / "cfv.sqlite"
        notes = self.vault / "notes"
        (notes / "topics").mkdir(parents=True)
        (notes / "log").mkdir()
        (notes / "topics" / "harbour.md").write_text("# Harbour pilot\n\nThe harbour pilot runs weekly audits at the dock.\n")
        (notes / "topics" / "canteen.md").write_text("# Canteen\n\nLunch menus rotate every Monday.\n")
        (notes / "topics" / "izmir.md").write_text(
            "---\ntitle: İzmir ofisi\nupdated: 2026-11-02\n---\nİzmir ofisinin kira sözleşmesi Aralık'ta yenilenir.\n")
        (notes / "log" / "old-audit.md").write_text(
            "---\nid: abc123\ndate: 2026-10-01\ntitle: Old audit\n---\n# Old audit\n\nThe audit schedule moved to Fridays.\n")
        filler = " ".join(f"filler{i}" for i in range(300))
        (notes / "topics" / "long.md").write_text(f"# Long page\n\n{filler} the secret dock timetable is posted.\n")
        refresh(self.db, self.vault)

    def tearDown(self):
        shutil.rmtree(self.temporary)

    def test_keyword_recall_works_before_embedding(self):
        result = recall(self.db, "Which weekly audits does the pilot run?")
        self.assertFalse(result["semantic"])
        self.assertIn("run embed", result["warning"])
        top = result["evidence"][0]
        self.assertEqual((top["path"], top["reference"], top["matched"]), ("topics/harbour", "note:topics/harbour", ["keyword"]))

    def test_turkish_words_match_across_dotless_i_diacritics_and_suffixes(self):
        top = recall(self.db, "ızmir kira sozlesmesi ne zaman?")["evidence"][0]
        self.assertEqual((top["path"], top["title"]), ("topics/izmir", "İzmir ofisi"))

    def test_embedding_is_cached_by_passage_across_refreshes(self):
        calls = []
        first = embed(self.db, fake_embedder(calls), model="fake")
        self.assertEqual(first["embedded"], first["passages"])
        self.assertEqual(first["passages"], 6)  # five notes, the 300-word page split into two windows
        calls.clear()
        refresh(self.db, self.vault)  # notes get new event ids; vectors are keyed by passage text
        self.assertEqual(embed(self.db, fake_embedder(calls), model="fake")["embedded"], 0)
        self.assertEqual(calls, [])
        (self.vault / "notes" / "topics" / "canteen.md").unlink()
        refresh(self.db, self.vault)
        self.assertEqual(embed(self.db, fake_embedder(calls), model="fake")["pruned"], 1)

    def test_meaning_finds_a_note_without_shared_words_and_cites_its_best_passage(self):
        embed(self.db, fake_embedder(), model="bge-m3")
        result = recall(self.db, "port", embedder=fake_embedder())
        self.assertTrue(result["semantic"])
        top = result["evidence"][0]
        self.assertEqual((top["path"], top["matched"]), ("topics/harbour", ["semantic"]))
        long = recall(self.db, "secret dock timetable", embedder=fake_embedder())["evidence"][0]
        self.assertEqual((long["path"], long["matched"]), ("topics/long", ["keyword", "semantic"]))
        self.assertIn("secret dock timetable", long["excerpt"])
        self.assertNotIn("filler0 ", long["excerpt"])

    def test_failed_embedding_server_falls_back_to_keywords(self):
        embed(self.db, fake_embedder(), model="bge-m3")
        result = recall(self.db, "weekly audits", embedder=failing_embedder)
        self.assertFalse(result["semantic"])
        self.assertIn("OSError", result["warning"])
        self.assertEqual(result["evidence"][0]["path"], "topics/harbour")

    def test_dates_filter_both_rankings(self):
        embed(self.db, fake_embedder(), model="bge-m3")
        result = recall(self.db, "audit schedule", start="2026-10-01", end="2026-10-01", embedder=fake_embedder())
        self.assertEqual([r["reference"] for r in result["evidence"]], ["note:abc123"])

    def test_notes_come_before_conversations(self):
        trace = self.temporary / "traces"
        trace.mkdir()
        shutil.copy(ROOT / "fixtures/demo-session.jsonl", trace)
        (self.vault / "notes" / "topics" / "review.md").write_text("# Reviews\n\nThe review moved.\n")
        refresh(self.db, self.vault, sessions=trace)
        kinds = [r["kind"] for r in recall(self.db, "demo review moved to Monday")["evidence"]]
        self.assertIn("brain_session", kinds)
        self.assertEqual(kinds, sorted(kinds, key=lambda kind: kind != "brain_note"))
        with self.assertRaises(ValueError):
            recall(self.db, "x", start="2026-11-02", end="2026-11-01")


def words(prefix: str, count: int) -> str:
    return " ".join(f"{prefix}{i}" for i in range(count))


def chat(*turns: tuple[str, str]) -> str:
    """A conversation body in the extractor's format."""
    return "# Title\n\n" + "".join(f"**{role}** · 12:0{i % 10}Z\n\n{text}\n\n" for i, (role, text) in enumerate(turns))


class ChatPassageTest(unittest.TestCase):
    def test_short_exchanges_share_a_piece_and_tool_lines_are_left_out(self):
        body = chat(("user", "Where is the harbour?"), ("assistant", "- `Bash` — ls /srv/harbour\nAt the dock."),
                    ("user", "And the canteen?"), ("assistant", "Next to it."))
        (text, offset), = chat_passages("Places", body)
        self.assertEqual(text, "Places\n# Title User: Where is the harbour? Assistant: At the dock. "
                               "User: And the canteen? Assistant: Next to it.")
        self.assertEqual(offset, 0)

    def test_a_long_exchange_is_cut_between_paragraphs_and_later_pieces_repeat_the_question(self):
        reply = "\n\n".join(words(f"p{n}w", 150) for n in range(4))  # 600 words in four paragraphs
        body = chat(("user", "Which audit finding matters most for the harbour pilot?"), ("assistant", reply))
        pieces = chat_passages("Audit", body)
        self.assertEqual(len(pieces), 2)
        first, second = (text for text, _ in pieces)
        self.assertTrue(first.startswith("Audit\n# Title User: Which audit finding"))
        self.assertIn("p1w149", first)
        self.assertNotIn("p2w0", first)
        self.assertTrue(second.startswith("Audit\n[User: Which audit finding matters most for the harbour pilot? …]\np2w0"))
        self.assertTrue(body[pieces[1][1]:].startswith("p2w0"))
        self.assertTrue(all(len(text.split()) <= CHAT_WORDS + QUESTION_WORDS + 3 for text, _ in pieces))

    def test_a_long_exchange_that_starts_a_piece_does_not_repeat_its_own_question(self):
        body = chat(("user", words("a", 396)), ("user", "Second question here?"),  # 399 words, then 4 that do not fit
                    ("assistant", "\n\n".join(words(f"r{n}w", 200) for n in range(3))))
        second = chat_passages("Two", body)[1][0]
        self.assertTrue(second.startswith("Two\nUser: Second question here?"), second[:60])

    def test_a_paragraph_longer_than_a_piece_is_cut_by_words(self):
        pieces = chat_passages("Long", chat(("user", words("w", CHAT_WORDS * 2 + 10))))
        self.assertEqual([len(text.split("\n")[-1].split()) for text, _ in pieces], [CHAT_WORDS, CHAT_WORDS, 13])

    def test_appending_to_a_conversation_keeps_the_earlier_pieces(self):
        turns = [("user", f"question {n} " + words(f"q{n}w", 60)) for n in range(20)]
        before = chat_passages("Grow", chat(*turns))
        after = chat_passages("Grow", chat(*turns, ("assistant", words("late", 500))))
        self.assertGreater(len(before), 2)
        self.assertEqual(after[:len(before) - 1], before[:-1])

    def test_older_exports_inline_markers_and_unmarked_text(self):
        legacy = "**User** (2026-06-24 10:43 PM): plan the harbour audit\n\n**Assistant** (2026-06-24 10:44 PM): Fridays.\n"
        self.assertEqual(chat_passages("Plan", legacy), [("Plan\nUser: plan the harbour audit Assistant: Fridays.", 0)])
        self.assertEqual(chat_passages(None, "first line\n\nsecond line"), [("first line second line", 0)])
        self.assertEqual(chat_passages("Empty", "   \n"), [])


class ChatRecallTest(unittest.TestCase):
    def setUp(self):
        self.temporary = Path(tempfile.mkdtemp())
        self.vault, self.db, self.archive = self.temporary / "vault", self.temporary / "cfv.sqlite", self.temporary / "archive"
        (self.vault / "notes" / "topics").mkdir(parents=True)
        (self.vault / "notes" / "topics" / "canteen.md").write_text("# Canteen\n\nLunch menus rotate every Monday.\n")
        self.archive.mkdir()
        filler = "\n\n".join(words(f"f{n}w", 195) for n in range(4))  # two full pieces before the harbour exchange
        for name, day in (("long", "01"), ("copy", "02")):  # the same chat exported twice
            self.conversation(name, "Weekly planning", chat(
                ("user", "Let us plan the week."), ("assistant", filler),
                ("user", "Also: the harbour inspection moved to Thursday."), ("assistant", "Noted, Thursday.")), day=day)
        self.conversation("other", "Canteen chat", chat(("user", "What is for lunch?"), ("assistant", "Soup.")), day="03")
        refresh(self.db, self.vault, archive=self.archive, source_name="archive")

    def tearDown(self):
        shutil.rmtree(self.temporary)

    def conversation(self, name, title, body, day):
        (self.archive / f"{name}.md").write_text(
            f'---\ntype: "conversation"\nharness: "codex"\nsession_id: "{name}"\ntitle: "{title}"\n'
            f'started: "2026-10-{day}T10:00:00+00:00"\n---\n{body}')

    def test_keyword_match_points_at_the_matching_piece(self):
        found = [r for r in recall(self.db, "harbour inspection Thursday")["evidence"] if r["kind"] == "brain_conversation"]
        top = found[0]
        self.assertEqual(top["matched"], ["keyword"])
        self.assertIn("harbour inspection moved to Thursday", top["excerpt"])
        self.assertNotIn("f0w0 ", top["excerpt"])
        text = read(self.db, top["reference"], offset=top["offset"])["text"]
        self.assertTrue(text.startswith("**user** · 12:02Z\n\nAlso: the harbour inspection"), text[:60])

    def test_meaning_finds_a_conversation_without_shared_words_and_copies_are_shown_once(self):
        embed(self.db, fake_embedder(), model="bge-m3")
        result = recall(self.db, "port", embedder=fake_embedder())
        chats = [r for r in result["evidence"] if r["kind"] == "brain_conversation"]
        self.assertIn("semantic", chats[0]["matched"])
        self.assertIn("harbour inspection", chats[0]["excerpt"])
        self.assertEqual(sum(r["title"] == "Weekly planning" for r in chats), 1)
        dated = recall(self.db, "port", start="2026-10-02", end="2026-10-02", embedder=fake_embedder())["evidence"]
        self.assertEqual([r["reference"] for r in dated], ["session:archive:codex:copy"])

    def test_keywords_stand_in_for_a_conversation_not_yet_embedded(self):
        embed(self.db, fake_embedder(), model="bge-m3")
        self.conversation("new", "Fresh chat", chat(("user", "The crane schedule slipped."), ("assistant", "Noted.")),
                          day="04")
        refresh(self.db, self.vault, archive=self.archive, source_name="archive")
        result = recall(self.db, "crane schedule slipped", embedder=fake_embedder())
        self.assertTrue(result["semantic"])
        fresh = [r for r in result["evidence"] if r["title"] == "Fresh chat"]
        self.assertEqual(fresh[0]["matched"], ["keyword"])
        embed(self.db, fake_embedder(), model="bge-m3")
        fresh = [r for r in recall(self.db, "crane schedule slipped", embedder=fake_embedder())["evidence"]
                 if r["title"] == "Fresh chat"]
        self.assertEqual(fresh[0]["matched"], ["semantic"])

    def test_embed_takes_notes_first_then_newest_conversations_and_stops_at_the_limit(self):
        texts = []
        report = embed(self.db, lambda batch: texts.extend(batch) or fake_embedder()(batch), model="fake", limit=2)
        self.assertEqual((report["embedded"], report["remaining"], report["passages"]), (2, 3, 8))  # copies share 3
        self.assertEqual(texts, ["# Canteen Lunch menus rotate every Monday.",
                                 "Canteen chat\n# Title User: What is for lunch? Assistant: Soup."])
        self.assertEqual(embed(self.db, fake_embedder(), model="fake")["remaining"], 0)

    def test_vectors_copied_from_another_database_need_no_new_embedding(self):
        other = self.temporary / "laptop.sqlite"
        refresh(other, self.vault, archive=self.archive, source_name="archive")
        embed(other, fake_embedder(), model="bge-m3")
        self.assertGreater(copy_vectors(other, self.db), 0)
        calls = []
        self.assertEqual(embed(self.db, fake_embedder(calls), model="bge-m3")["embedded"], 0)
        self.assertEqual(calls, [])
        self.assertEqual(copy_vectors(other, self.db), 0)


    def test_a_vectors_only_export_carries_no_records_and_works_with_copy_vectors(self):
        embed(self.db, fake_embedder(), model="bge-m3")
        out = self.temporary / "vectors.sqlite"
        count = export_vectors(self.db, out)
        self.assertGreater(count, 0)
        with closing(sqlite3.connect(out)) as conn:
            self.assertEqual({row[0] for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}, {"embedding"})
        other = self.temporary / "elsewhere.sqlite"
        refresh(other, self.vault, archive=self.archive, source_name="archive")
        self.assertEqual(copy_vectors(out, other), count)
        with self.assertRaises(ValueError):
            export_vectors(self.db, out)  # never overwrites


if __name__ == "__main__":
    unittest.main()
