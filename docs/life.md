# Personal life sources (WhatsApp, mail, calendar, contacts)

The CFV site draws people, groups, calendar events and a WhatsApp and mail density from these. They are read by
`defter refresh --life DIR` on **the laptop only**. The VPS refresh never gets this folder, the VPS database holds none of
it, and the change log hides `whatsapp` and `email` from `delta`, so agents served from the VPS see none of it.

## The life folder

Default `data/private/life` (outside git, mode 700):

| Path | Content | Where it comes from |
|---|---|---|
| `whatsapp/*.txt`, `*.zip` | one file per chat | WhatsApp > chat menu > More > Export chat > Without media |
| `mail/*.mbox` | mail | Google Takeout > Mail |
| `calendar/*.ics` | calendars | Google Takeout > Calendar |
| `contacts/*.vcf` | contacts | Google Takeout > Contacts |
| `me.txt` | optional: your WhatsApp display names and e-mail addresses, one per line | you |

What is stored: every WhatsApp message (text, and `[medya]` for media), for mail only the headers and a 200-character
snippet, never a whole body, for calendar the events, for contacts name, phone numbers and addresses. Media files and
attachments are not read.

## Steps

1. **Google Takeout** (takeout.google.com): select only Calendar, Mail and Contacts, export once, download the zip.
   `defter life-unpack ~/Downloads/takeout-*.zip` copies just those three kinds into the life folder.
2. **WhatsApp**: export each chat without media and put the files in `whatsapp/`. A chat exported twice (a zip and a
   text file, or an older and a newer export) counts once. Put your own WhatsApp name in `me.txt` so your messages are
   marked; two-person chats work out the other side from the file name.
3. `defter refresh --db data/private/processed/cfv.sqlite --vault ~/second-brain --archive ~/second-brain-archive/conversations
   --source-name archive --legacy ~/second-brain-archive/legacy --life data/private/life`, then reload the site.

## Moving vectors to the VPS

The laptop database now holds personal data, so never copy it to another machine. Export only the embedding cache:
`defter export-vectors /tmp/vectors.sqlite`, copy that file, then on the VPS `defter copy-vectors /path/vectors.sqlite`.

## Not done yet

- **Whole WhatsApp history from an Android phone.** Per-chat export is the simple route. The full history lives in the
  phone's `msgstore.db.crypt15` backup, which can only be decrypted with the key of an end-to-end encrypted backup
  (WhatsApp > Settings > Chats > Chat backup > End-to-end encrypted backup, choose the 64-digit key and keep it in a
  password manager). There is no importer for the decrypted `msgstore.db` yet; it needs a real file to test against.
- **Photos.** CFV's photo importer reads an iPhone library. An Android folder reader (date and place from the image
  files) is not written.
- **Gmail without Takeout.** CFV's own sync uses the `gog` tool of its first owner and is not set up here.
