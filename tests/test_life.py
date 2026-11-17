"""Personal life sources: WhatsApp exports, Takeout mail, contacts, calendar. All data here is invented."""
import mailbox
import shutil
import sqlite3
import tempfile
import unittest
import zipfile
from contextlib import closing
from email.message import EmailMessage
from pathlib import Path

from defter.changes import delta
from defter.life import chat_name, parse_vcards, parse_whatsapp, unpack_takeout, whatsapp_chats
from defter.store import refresh

DM = """02/11/26, 09:15 - Messages and calls are end-to-end encrypted. No one outside of this chat can read them.
02/11/26, 09:16 - Mira Test: Are we still on for Thursday?
02/11/26, 09:20 - Ben: Yes, harbour audit at 10.
Bring the checklist.
02/11/26, 09:21 - Ben: <Media omitted>
27/11/26, 18:02 - Mira Test: This message was deleted
27/11/26, 18:03 - Ben: ok
"""
GROUP = """03/11/26, 10:00 - Ana: morning
03/11/26, 10:01 - Cem: hi
03/11/26, 10:02 - Ben: hello all
"""
MONTH_FIRST = "11/27/26, 3:05 PM - Dee: hello\n11/28/26, 11:59 PM - Eli: late\n"


def vcard(name, *lines):
    return "BEGIN:VCARD\nVERSION:3.0\nFN:" + name + "\n" + "\n".join(lines) + "\nEND:VCARD\n"


class LifeTest(unittest.TestCase):
    def setUp(self):
        self.temporary = Path(tempfile.mkdtemp())
        self.life = self.temporary / "life"
        for sub in ("whatsapp", "mail", "contacts", "calendar"):
            (self.life / sub).mkdir(parents=True)
        self.db = self.temporary / "cfv.sqlite"

    def tearDown(self):
        shutil.rmtree(self.temporary)

    def refresh(self):
        refresh(self.db, self.temporary / "vault", life=self.life)
        return closing(sqlite3.connect(self.db))

    def test_whatsapp_dates_continuations_and_system_lines(self):
        rows = parse_whatsapp(DM)
        self.assertEqual([r["sender"] for r in rows], [None, "Mira Test", "Ben", "Ben", "Mira Test", "Ben"])
        self.assertEqual(rows[2]["text"], "Yes, harbour audit at 10.\nBring the checklist.")
        self.assertEqual((rows[1]["t"].month, rows[1]["t"].day), (11, 2))  # day first: 27/11 proves it
        month_first = parse_whatsapp(MONTH_FIRST)
        self.assertEqual([(r["t"].month, r["t"].day, r["t"].hour) for r in month_first], [(11, 27, 15), (11, 28, 23)])
        self.assertEqual(chat_name("WhatsApp Chat with Mira Test.txt"), "Mira Test")
        self.assertEqual(chat_name("WhatsApp Sohbeti - Aile.zip"), "Aile")

    def test_chats_become_events_with_people_and_a_reexport_is_not_counted_twice(self):
        (self.life / "whatsapp" / "WhatsApp Chat with Mira Test.txt").write_text(DM)
        with zipfile.ZipFile(self.life / "whatsapp" / "WhatsApp Chat with Mira Test.zip", "w") as z:
            z.writestr("WhatsApp Chat with Mira Test.txt", DM + "28/11/26, 08:00 - Mira Test: later\n")
        (self.life / "whatsapp" / "WhatsApp Chat with Team.txt").write_text(GROUP)
        self.assertEqual(len(whatsapp_chats(self.life / "whatsapp")["Mira Test"]), 7)
        with self.refresh() as conn:
            events = conn.execute("SELECT title, body, json_extract(payload,'$.from_me'), json_extract(payload,'$.chat_type'), "
                                  "json_extract(payload,'$.type') FROM event WHERE kind='whatsapp' ORDER BY start_utc, id").fetchall()
            self.assertEqual(len(events), 6 + 1 + 3 - 1)  # the system line is skipped; one extra message only the zip has
            by_body = {e[1]: e for e in events}
            self.assertEqual(by_body["Are we still on for Thursday?"][2:4], (0, "dm"))  # Mira is the chat's person
            self.assertEqual(by_body["ok"][2], 1)  # the other sender in a two-person chat is you
            self.assertEqual(by_body["[medya]"][4], "media")
            self.assertEqual(by_body["[silinmiş mesaj]"][4], "deleted")
            self.assertEqual(by_body["morning"][3], "group")
            names = {r[0] for r in conn.execute("SELECT name FROM person_alias WHERE source_kind='whatsapp'")}
            self.assertEqual(names, {"Mira Test", "Ana", "Cem", "Ben"})  # Ben is not you in the group: me.txt is empty

    def test_me_txt_marks_your_group_messages(self):
        (self.life / "whatsapp" / "WhatsApp Chat with Team.txt").write_text(GROUP)
        (self.life / "me.txt").write_text("# my names\nBen\n")
        with self.refresh() as conn:
            self.assertEqual(conn.execute("SELECT count(*) FROM event WHERE json_extract(payload,'$.from_me')=1").fetchone()[0], 1)
            self.assertEqual({r[0] for r in conn.execute("SELECT name FROM person_alias WHERE source_kind='whatsapp'")},
                             {"Ana", "Cem"})

    def test_mail_keeps_headers_and_a_short_snippet_and_skips_spam(self):
        box = mailbox.mbox(self.life / "mail" / "All mail.mbox")
        for subject, frm, to, labels, body, ident in (
                ("Harbour plan", "Mira Test <mira@example.invalid>", "me@example.invalid", "Inbox,Important", "x" * 500, "m1"),
                ("Re: Harbour plan", "Me <me@example.invalid>", "Mira Test <mira@example.invalid>", "Sent", "On it.", "m2"),
                ("Win money", "Spam <s@example.invalid>", "me@example.invalid", "Spam", "buy", "m3")):
            msg = EmailMessage()
            msg["Subject"], msg["From"], msg["To"], msg["Message-ID"] = subject, frm, to, f"<{ident}@example.invalid>"
            msg["Date"], msg["X-Gmail-Labels"] = "Tue, 03 Nov 2026 10:00:00 +0300", labels
            msg.set_content(body)
            box.add(msg)
        box.flush()
        with self.refresh() as conn:
            rows = conn.execute("SELECT title, body, json_extract(payload,'$.direction') FROM event WHERE kind='email' "
                                "ORDER BY title").fetchall()
            self.assertEqual([r[0] for r in rows], ["Harbour plan", "Re: Harbour plan"])
            self.assertEqual([r[2] for r in rows], ["in", "out"])
            self.assertLess(len(rows[0][1]), 330)  # sender line, recipient line and 200 characters at most
            people = conn.execute("SELECT alias FROM person_alias WHERE source_kind='email'").fetchall()
            self.assertEqual(people, [("mira@example.invalid",)])  # you (the sender of Sent mail) are not a person

    def test_contacts_join_people_made_by_mail_and_calendar_loads(self):
        box = mailbox.mbox(self.life / "mail" / "a.mbox")
        msg = EmailMessage()
        msg["Subject"], msg["From"], msg["To"], msg["Date"] = "Hi", "Mira <mira@example.invalid>", "me@x.invalid", "Tue, 03 Nov 2026 10:00:00 +0300"
        msg.set_content("hi")
        box.add(msg)
        box.flush()
        (self.life / "contacts" / "c.vcf").write_text(
            vcard("Mira Test", "TEL;TYPE=CELL:0532 000 11 22", "EMAIL:Mira@example.invalid", "UID:u1") +
            vcard("Solo Person", "TEL:+90 533 000 00 01"))
        (self.life / "calendar" / "Work.ics").write_text(
            "BEGIN:VCALENDAR\nVERSION:2.0\nPRODID:-//t//\nX-WR-CALNAME:Work\nBEGIN:VEVENT\nUID:a1\n"
            "DTSTART:20261103T090000Z\nDTEND:20261103T100000Z\nSUMMARY:Harbour audit\nEND:VEVENT\nEND:VCALENDAR\n")
        with self.refresh() as conn:
            persons = conn.execute("SELECT count(*) FROM person").fetchone()[0]
            self.assertEqual(persons, 2)  # Mira (mail address + card) and Solo Person
            kinds = {r[0] for r in conn.execute("SELECT source_kind FROM person_alias WHERE person_id="
                                                "(SELECT person_id FROM person_alias WHERE alias='mira@example.invalid')")}
            self.assertEqual(kinds, {"email", "contacts", "phone", "whatsapp"})
            self.assertEqual(conn.execute("SELECT title FROM event WHERE kind='calendar'").fetchone()[0], "Harbour audit")

    def test_vcards_unfold_lines_and_decode_quoted_printable(self):
        card, = parse_vcards("BEGIN:VCARD\nVERSION:2.1\nFN;ENCODING=QUOTED-PRINTABLE;CHARSET=UTF-8:=C5=9Eeyma\nitem1.EMAIL:A@Example.invalid\n"
                             "TEL:+905320001122\n  \nNOTE:long\n  folded\nEND:VCARD")
        self.assertEqual((card["name"], card["emails"], card["phones"]), ("Şeyma", ["a@example.invalid"], ["+905320001122"]))

    def test_delta_never_lists_personal_life_kinds(self):
        (self.life / "whatsapp" / "WhatsApp Chat with Mira Test.txt").write_text(DM)
        (self.temporary / "vault" / "notes").mkdir(parents=True)
        (self.temporary / "vault" / "notes" / "a.md").write_text("# A\n\nplan\n")
        with self.refresh():
            pass
        kinds = {c["kind"] for c in delta(self.db, since="2000-01-01")["changes"]}
        self.assertEqual(kinds, {"brain_note"})

    def test_takeout_zip_gives_up_only_calendar_mail_and_contacts(self):
        zip_path = self.temporary / "takeout-1.zip"
        with zipfile.ZipFile(zip_path, "w") as z:
            z.writestr("Takeout/Calendar/Work.ics", "ics")
            z.writestr("Takeout/Mail/All mail Including Spam and Trash.mbox", "From a\n")
            z.writestr("Takeout/Contacts/All Contacts/All Contacts.vcf", "vcf")
            z.writestr("Takeout/Google Photos/IMG_1.jpg", "jpg")
            z.writestr("Takeout/Drive/secret.pdf", "pdf")
        self.assertEqual(unpack_takeout([zip_path], self.temporary / "out"), {"calendar": 1, "mail": 1, "contacts": 1})
        self.assertEqual(sorted(p.relative_to(self.temporary / "out").as_posix() for p in (self.temporary / "out").rglob("*") if p.is_file()),
                         ["calendar/Work.ics", "contacts/All Contacts.vcf", "mail/All mail Including Spam and Trash.mbox"])


if __name__ == "__main__":
    unittest.main()
