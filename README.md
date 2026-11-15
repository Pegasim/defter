# Defter (working title)

Defter combines CFV's event timeline with the second brain's conversation capture and memory maintenance. An existing agent queries both through a local, read-only MCP server.

The current integration imports calendar exports, Codex traces and redacted conversation archives. It indexes decisions filed through the maintenance inbox and follows their evidence references. Curated notes without log frontmatter, such as indexes, timelines and topic pages, are indexed by their vault path and cited as `note:<path>`; their date comes from a `date` or `updated` field, otherwise from the file's last git commit when the vault is a git checkout, otherwise from its modification time. The bundled demo uses synthetic data.

## Run the demo

Use Python 3.12+, SQLite 3.43+ with FTS5, and uv. From the repository root:

```sh
uv sync --locked
uv run python -m defter load
```

The demo runs CFV's ICS importer, parses the conversation with the second-brain extractor, files a decision through its inbox and consolidator, and builds a CFV database at `data/processed/cfv.sqlite`.

Start Codex with Defter for this session:

```sh
codex -c 'mcp_servers.defter.command="uv"' \
  -c 'mcp_servers.defter.args=["run","--locked","--offline","python","-m","defter","serve"]'
```

Ask “Was the October 30 demo review held as planned, and what did we decide about future demo reviews? Use lookup and cite its references.”

The returned records describe the original plan, the owner's reported rescheduling and the standing Monday decision. Actual attendance is unknown. Ask about the November 6 release review to exercise the calendar-only case.

Any stdio MCP host can launch `uv run --locked --offline python -m defter serve` with this repo as its working directory. The available tools are `recall`, `read`, `delta`, `lookup`, `search`, `when`, `timeline`, `digest`, `people`, `photos` and `week_page`. All are read-only. Photo results and weekly pages require the corresponding CFV imports and generated views.

## Use a private vault

Choose separate paths for personal data. These commands assume an explicitly selected local calendar file and directory of redacted conversation Markdown produced by the imported extractor:

```sh
uv run python -m defter refresh \
  --db data/private/processed/cfv.sqlite \
  --vault data/private/vault \
  --calendar /absolute/path/calendar.ics \
  --archive /absolute/path/redacted-conversations \
  --source-name archive
```

Use `--sessions /absolute/path/codex-traces --source-name laptop` for raw Codex traces in the extractor's supported format. Use `--legacy /absolute/path/legacy` for older conversation exports (Claude web, ChatGPT, Hermes, Cursor) whose YAML frontmatter has a `title` and a `date` or `updated_at`; they are cited as `session:legacy:<path>`. Each source name identifies one capture directory. Reimport the same directory under the same name. Calendar ingestion uses CFV's existing ICS recurrence and timezone handling.

Write a durable entry with the imported maintenance CLI. Replace the example evidence reference with one returned by lookup:

```sh
uv run python -m defter memory --vault data/private/vault add \
  --kind decision --topic reviews --title "Monday review schedule" \
  --context "The owner changed the review schedule." \
  --result "Future reviews are scheduled on Mondays." \
  --evidence session:archive:codex:SESSION_ID
uv run python -m defter memory --vault data/private/vault consolidate
uv run python -m defter refresh \
  --db data/private/processed/cfv.sqlite --vault data/private/vault
```

Pending inbox entries become searchable after consolidation and refresh. Rebuilding selected sources retains the other sources in the database. Failed staged imports leave the published snapshot intact. A deleted supporting record appears in lookup's `missing_references`. A private refreshed database is protected from replacement by the synthetic loader.

Serve that store with `uv run python -m defter serve --db data/private/processed/cfv.sqlite`. The server reads the selected database. The host agent and its model receive the retrieved excerpts.

To review how agents use the server, set `DEFTER_QUERY_LOG` to a file path. Each `recall`, `read`, `delta` or `lookup` call then appends one JSON line with the time, `DEFTER_CLIENT` (a label you set per agent), the arguments, the number of results, the top references, the latency and any error. The log contains the questions asked, so keep it with the private data.

## Ask in plain language

`recall` answers a question in any wording, English or Turkish. It ranks curated notes first, fusing FTS5 keyword matches with bge-m3 passage similarity by reciprocal rank, then conversations by bge-m3 similarity. Dotted and dotless i and Turkish diacritics are folded, and words longer than five letters match by their first five, so suffixed forms still match.

Notes are searched in overlapping windows of 220 words. Conversations are searched in pieces of up to 400 words cut between exchanges, an exchange being one user turn and the replies to it. Short exchanges share a piece; a longer one is cut between paragraphs, and its later pieces repeat the start of its question. Every piece starts with the conversation's title, and the extractor's one-line tool-call summaries are left out. A conversation is ranked by its best piece. Its excerpt comes from that piece, and `offset` gives where the piece starts, so `read` can continue from there. A conversation whose best piece is word for word the same as an earlier result's, such as a second export of one chat, is shown once.

Conversations are ranked by meaning alone because, on 32 reworded and 7 exact-term questions aimed at known chats, adding keyword ranks at any weight made both sets worse. Keyword search over the pieces stands in where meaning is blind: for conversations with pieces not yet embedded, and for all of them when there are no vectors or the embedding server fails.

Passage vectors come from a separate step, so refresh never calls a model:

```sh
uv run python -m defter embed --db data/private/processed/cfv.sqlite
uv run python -m defter recall "When did the review schedule change?" --db data/private/processed/cfv.sqlite
```

`embed` sends passages to an OpenAI-compatible `/v1/embeddings` endpoint: by default a local Ollama at `http://localhost:11434/v1/embeddings` with the `bge-m3` model. Set `DEFTER_EMBED_URL` and `DEFTER_EMBED_MODEL` to change them. The endpoint receives note and conversation text, so keep it local ([Privacy](docs/privacy.md)). Notes go first, then conversations, newest first. `--limit N` stops after N passages and reports how many remain for a later run, which keeps a CPU-only machine's scheduled run short. Vectors are cached in the database by passage text and model. A later run embeds only new or changed passages and removes vectors no passage uses. Appending to a conversation changes only its last pieces, so the rest keep their vectors. Without vectors, or with the endpoint unreachable, recall returns keyword results with `semantic: false` and a `warning`.

Embedding the whole archive is much faster on a GPU. Vectors depend only on passage text, so another machine's database can supply them:

```sh
uv run python -m defter copy-vectors /path/to/gpu-embedded/cfv.sqlite --db data/private/processed/cfv.sqlite
```

This copies the vectors that this database's passages use and lack. Both databases must be refreshed from the same notes and archive content; the files may sit at different paths.

`read` returns a record's full stored text by a reference from recall, lookup or delta, or a note by its vault path such as `topics/bilun`, in windows of up to 50,000 characters. It works when the agent cannot open the server's files, for example over ssh.

`delta` lists records added, changed or removed since an ISO date or time, oldest first, with a cursor to continue or to ask again later. Each refresh compares every record with the previous one, so change times are refresh times. The first refresh has nothing to compare with and marks every record `initial`, dated by a note's last git commit or otherwise the record's own timestamp. The first import of a new source does the same for that source's records, so adding an old export does not list it as new.

```sh
uv run python -m defter delta --since 2026-11-13 --db data/private/processed/cfv.sqlite
uv run python -m defter read topics/harbour --db data/private/processed/cfv.sqlite
```

The distillation and context compiler tools are imported under `third_party/second-brain/tools/`. The distiller writes through the bundled maintenance CLI into the vault selected by `BRAIN_REPO`. Its archive and state paths use `BRAIN_ARCHIVE` and `BRAIN_DISTILL_STATE`, with defaults under this repo's ignored `data/` directory. Running it requires a configured model backend. Scheduled jobs and model calls are operator actions.

## Verify

```sh
uv run python -m unittest discover -s tests -v
uv run pytest -q third_party/cfv/tests
uv run python third_party/second-brain/tools/brain/brain.py selftest
uv run python third_party/second-brain/tools/extract/brain_extract.py --selftest
```

## Project docs

- [Architecture](docs/architecture.md)
- [Project thesis](docs/thesis.md)
- [Second-brain architecture](docs/second-brain-architecture.md)
- [CFV architecture](docs/cfv-architecture.md)
- [Evaluation](docs/evaluation.md)
- [Privacy](docs/privacy.md)
- [Landscape](docs/landscape.md)
- [Source inventory and local adaptations](docs/source-inventory.md)

Keep personal exports, archives, vaults, media, databases, credentials and generated logs outside Git. All committed fixtures are synthetic.
