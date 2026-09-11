# servicenow-graphrag

Turn a real ServiceNow instance into a Neo4j graph, then measure whether GraphRAG
actually beats plain keyword search on it.

This is the companion code for the freeCodeCamp article. Everything here runs against a
free ServiceNow developer instance and a free Neo4j Aura instance, so you can reproduce
every number in the article without paying for anything.

**The part that matters is the read path.** The graph is built by reading ServiceNow
through [`snowloader`](https://pypi.org/project/snowloader/), not by loading local files
into Neo4j. That distinction is the whole point: anything the platform does to your data
on the way in and out is invisible if you skip it, and three of the article's teaching
moments live in exactly that gap.

---

## What is in here

```
dataset/     the records you load INTO ServiceNow
generator/   the loaders, for ServiceNow and for Neo4j
questions/   the frozen question set and the gold answers
retrieval/   chunking, the retrieval arms, the scoring
gpu/         renting a GPU and serving both models on it
results/     the numbers the article quotes, so they can be checked
tests/       the tests that prove the above
```

### `dataset/` — what goes into ServiceNow

Seven files of generated records. Generated, not scraped: a real CMDB is somebody's
confidential estate, and an article you cannot reproduce is an article you have to take on
trust. The generator is in `generator/estate.py` and the shape of the data is measured in
`dataset/manifest.json`.

| File | Rows | Becomes |
|---|---|---|
| `configuration_items.jsonl` | 11,891 | `cmdb_ci_*` records |
| `relationships.jsonl` | 28,694 | `cmdb_rel_ci` rows |
| `incidents.jsonl` | 60,000 | `incident` records, with work notes |
| `changes.jsonl` | 8,000 | `change_request` records |
| `problems.jsonl` | 900 | `problem` records |
| `knowledge.jsonl` | 301 | `kb_knowledge` articles |
| `manifest.json` | — | the measured rates the article publishes |

### `generator/` — getting data in, and back out

| File | What it does |
|---|---|
| `estate.py` | Builds the estate: items, and the dependency edges between them. |
| `incidents.py` | Builds tickets against that estate, with the messiness real tickets have. |
| `records.py` | Changes, problems and knowledge articles. |
| `build.py` | Runs the three above and writes `dataset/`. |
| `load_servicenow.py` | Pushes `dataset/` into your ServiceNow instance. Idempotent. |
| **`graph_from_servicenow.py`** | **Reads ServiceNow back through snowloader and builds the Neo4j graph. This is the one that matters.** |
| `load_neo4j.py` | Builds the graph from the local files instead. Faster, and the route to take if you would rather not stand up a ServiceNow instance at all. Article section 66b says when that is the right call. |
| `load_chunks.py` | Cuts every record into chunks, embeds them, and writes 82,296 `:Chunk` nodes with a vector index. Run it after the graph exists, or every retriever queries an empty index without complaining. |
| `repair_relationships.py` | Fixes the direction of `cmdb_rel_ci` rows that were written backwards. |
| `inspect_rel_type.py` | Prints every column ServiceNow defines on `cmdb_rel_type`, read from a live instance. Part 6 rests a decision on there being no impact column there, and this is the evidence. It exits non-zero if a future release adds one. |
| `verify_relationships.py` | Asks the instance what is actually there, rather than trusting the files. |
| `env.py` | Finds `.env.local` wherever you put it, and says where it looked if it fails. |
| `ask.py` | Ask a question in English and get an answer out of the graph. |

### `questions/` — the measurement's honest half

| File | What it does |
|---|---|
| `questions.py` | 39 questions, frozen before any retriever existed, each with the answer shape it expects and a written prediction of which retriever will win. |
| `gold.py` | The rules that decide, mechanically, which records a correct answer must rest on. |

Questions are frozen first on purpose. A question set written after you have seen the
results is a question set that flatters them.

### `retrieval/` — the arms, and the scoring

| File | What it does |
|---|---|
| `chunking.py` | Turns records into the documents every arm searches. |
| `embed.py` | Embeds them through an OpenAI-compatible endpoint, cached on a fingerprint of the text. Point `EMBED_BASE_URL` at the vLLM server from `gpu/`. |
| `arms.py` | The eight arms: six retrieval strategies, plus two controls that let the comparison fail. |
| `evaluate.py` | recall, MRR, precision, an exact sign test, and a refusal to name a winner the data cannot support. |
| `run.py` | Runs every arm against every question and writes `results/scores.json`. |
| `degraded.py` | Removes dependency edges on purpose and re-asks, to measure what a stale CMDB costs. |
| `scaling.py` | Repeats the comparison at 2,000 / 5,000 / 20,000 / 82,296 documents. |
| `ablation.py` | Puts the graph's facts into the text and checks whether the graph still adds anything. |
| `judge.py` | Grades the answers, not just the retrieval, and checks the judge three ways before printing a grade. Article section 108b. |
| `damage_sweep.py` | Deletes a share of the dependency edges, re-runs the graph arms, puts the edges back and verifies the count returned. Article section 114b. |
| `stemming.py` | Runs the keyword arm with and without a stemmer. The article claimed stemming changed nothing and had no code behind the claim. It does now, and the interesting half is that stemming fixes the vocabulary miss and still does not move the score. |

### `tests/` — 132 of them

Run them before you trust any number:

```bash
python3 -m pytest tests/ -q
```

---

## Setup

```bash
git clone https://github.com/ronidas39/servicenow-graphrag.git
cd servicenow-graphrag
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Create `.env.local` in the project root. It is gitignored and it never leaves your machine:

```bash
SERVICENOW_INSTANCE=devXXXXXX.service-now.com
SERVICENOW_USER=admin
SERVICENOW_PASSWORD=...

NEO4J_URI=neo4j+s://XXXXXXXX.databases.neo4j.io
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=...
NEO4J_DATABASE=
```

**Leave `NEO4J_DATABASE` empty unless you know otherwise.** On Aura the database is not
always called `neo4j`. Some instances name it after the instance id, and asking for
`neo4j` gets you `DatabaseNotFound` on an instance that is running perfectly. Leaving it
empty lets the driver use whatever the instance says its default is. If you want to see
the real name, run `SHOW DATABASES` against the `system` database.

## Running it

```bash
# 1. put the records into ServiceNow
python3 generator/load_servicenow.py

# 2. read them back out and build the graph  <- the one that matters
python3 generator/graph_from_servicenow.py --wipe

# 3. chunk, embed and index. Needs an embedding server, see gpu/ below
python3 generator/load_chunks.py

# 4. measure
python3 retrieval/run.py
```

### The GPU, for steps 3 and 4

Two of the eight arms need a model of your own: the embedding server for anything
similarity-based, and a chat server for the arm that writes its own Cypher. Both run on
one rented card. `gpu/` is the runbook, in the order you run it:

```bash
bash gpu/01-launch.sh     # rent it, with a spending ceiling and a self destruct
bash gpu/02-setup.sh      # driver, CUDA, a venv, vLLM. Runs ON the server
bash gpu/03-serve.sh      # both models on one card, ports 8000 and 8001
python3 gpu/04-measure.py --price-per-hour 0.9776
bash gpu/05-teardown.sh   # and it proves zero remaining rather than claiming it
```

Then point the code at it:

```bash
export EMBED_BASE_URL=http://<the address 01-launch.sh printed>:8001/v1
export CHAT_BASE_URL=http://<the same address>:8000/v1
```

**Embed before you tear the GPU down.** `load_chunks.py` caches the vectors under a
fingerprint of the corpus, so it is a one-off, but the cache is 482MB and is not in this
repo. Without it, and without a server, step 3 has nothing to embed against.

Step 2 is the article's premise, and section 74c is the section about it: reading
through the platform is what keeps every trap in Part 5 in play, and it is where the
`cmdb_ci` join trap lives, the one that loads zero incident edges while every count in
between looks right.

`load_neo4j.py` builds the same graph from the local files in a fraction of the time.
The article teaches that route too, in sections 72 to 74b, and section 66b says to take
it if you would rather not stand up a ServiceNow instance at all. Both routes end on the
same corpus and Part 10 scores the same numbers either way. What the file route gives up
is everything the platform does to your data between the table and the traversal.

## Licence

MIT.
