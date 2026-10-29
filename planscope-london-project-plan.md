# PlanScope London — Capstone Project Plan (Project #3 / Roadmap Project #5)

**Repo name:** `planscope-london`
**Tagline:** An LLM-powered intelligence system over London's planning applications — extraction, prediction, and grounded Q&A, with a real evaluation harness.
**One-line pitch for your CV/README:** "Reads thousands of raw London planning applications, extracts structured data with fine-tuned transformers and LLMs, predicts approval outcomes, and answers natural-language questions about development activity — with measured accuracy at every stage."

---

## 1. Why this project (the story you tell recruiters)

- Planning applications are a **leading indicator**: they precede price changes (your Project 1) and shape transit demand (your Project 2). Your portfolio becomes one coherent London urban-intelligence stack.
- The data is **genuinely messy and underused** — free-text descriptions, inconsistent categories, 33+ boroughs. Nobody's bootcamp portfolio has this. Companies (LandTech, Searchland, PlanningAPI.uk) sell exactly this analysis commercially, so the value is self-evident.
- It exercises the **exact skills your roadmap's Phase 5 targets**: PyTorch fluency, Hugging Face fine-tuning, LangChain RAG, FAISS, LLM evaluation — plus reuses your Phase 3/4 deployment muscle (FastAPI, Docker, MLflow, GitHub Actions, Evidently).
- The differentiator vs. every other LLM portfolio project: **evaluation**. Anyone can wire up RAG. Almost nobody measures it. You will.

**What each project in your portfolio now proves (no overlap):**

| Project | Core proof |
|---|---|
| 1. Property Intelligence | Supervised ML, SHAP, deployment, drift/retraining, RAG chatbot |
| 2. Transit Archetypes | Unsupervised ML, geospatial, clean shipping (NO RAG — cut it) |
| 3. PlanScope London | Transformers & fine-tuning, LLM-as-pipeline-component, RAG **evaluation**, LLM cost/latency trade-offs |

---

## 2. Data sources (verified July 2026)

**Primary: Planning London Datahub (PLD) guest API** — live database aggregating planning applications from all London Planning Authorities, updated daily.
- Guest endpoint (read-only, no account needed): `https://planningdata.london.gov.uk/api-guest/`
- Auth: none, but requests need header `X-API-AllowRequest` (value published in the GLA's technical doc — verify the current value there, keys can rotate).
- It's an **Elasticsearch 7.9 API** — you query it with Elasticsearch JSON query syntax, results come back as JSON. Key index: `applications` (includes nested data like residential units).
- Technical docs: https://www.london.gov.uk/programmes-strategies/planning/digital-planning/planning-london-datahub — links to the API connection PDF and the field schema. Read both before writing any code.

**Backup: London Datastore bulk dataset** — https://data.london.gov.uk/dataset/planning-london-datahub-applications/ — requires a free account + access request (checked: the download is gated behind login). Request access on day 1 anyway so it's granted before you need it.

**Enrichment: national Planning Data platform** — https://www.planning.data.gov.uk/ (free API, no key). Point-in-polygon lookups for conservation areas, listed buildings, flood-risk zones, green belt, Article 4 directions. These become **features for your outcome model** ("is the site in a conservation area?" is a real predictor of refusal) — and using a second API for feature enrichment is a strong engineering signal.

**Optional cross-link:** your Project 1's Land Registry price data by postcode/borough — one joined chart ("boroughs with high approval rates vs. price growth") for the README. One sentence, big systems-thinking signal.

---

## 3. Architecture

```
PLD API ──> ingest.py ──> raw parquet ──> clean.py ──> SQLite/parquet store
                                              │
              ┌───────────────────────────────┼───────────────────────────┐
              ▼                               ▼                           ▼
   NLP layer (Phase B)              Outcome model (Phase C)      RAG layer (Phase D)
   - LLM structured extraction     - approval/refusal prob       - chunk + embed docs
   - fine-tuned DistilBERT          - features: extracted +       - FAISS index
     classifier                       enrichment + metadata       - LangChain retrieval QA
   - TF-IDF + LogReg baseline      - baseline first, always      - EVAL HARNESS (golden set)
              │                               │                           │
              └───────────────┬───────────────┴───────────┬───────────────┘
                              ▼                           ▼
                    FastAPI (/extract, /predict,   Streamlit app (borough map
                     /ask) + Docker + tests         hero image + Q&A tab)
```

---

## 4. Phase-by-phase plan

### Phase A — Data foundation (NOW: July–August, ~15 hrs, alongside your Phase 3 courses)
Uses only skills you already have. Its job is to **de-risk everything else**.

1. **Day 1 smoke test:** hit the guest endpoint with `requests`, pull 10 applications, print the JSON. If the endpoint or header has changed, fall back to the Datastore access request and planning.data.gov.uk. Do not build anything else until this works.
2. Read the PLD schema doc. Write down the ~15 fields you actually need: application id, borough, dates (received/valid/decision), application type, decision, decision route (delegated vs committee), description text, site address/postcode, units proposed. Ignore the rest.
3. Build `src/ingest.py`: paginated Elasticsearch queries (`search_after` or scroll), polite rate limiting, retry with backoff, save raw JSON to parquet. Target: 3–5 years of data, all boroughs if feasible; if volume is unwieldy, start with 6–8 boroughs mixing inner/outer London.
4. Build `src/clean.py`: types, dedupe, normalise decision values (boroughs will encode "Approved" a dozen ways — map them to APPROVED / REFUSED / WITHDRAWN / PENDING), handle missing decisions.
5. **Data profiling notebook (`notebooks/01_profile.ipynb`):** row counts per borough per year, % missing per field, decision distribution, description length distribution. This notebook decides Phase B/C feasibility.
6. Write **tests** for the cleaning logic (decision mapping, date parsing) — pytest from day one, CI workflow from day one.

**Definition of done:** one command (`make ingest`) rebuilds the dataset; profiling notebook committed; CI green.
**Known risk:** decision fields may be incomplete for some boroughs/years. Mitigation: the outcome model (Phase C) trains only on decided applications; note completeness honestly in the README.

### Phase B — Extraction & classification (September, ~25 hrs, alongside HuggingFace NLP course + PyTorch tutorials)
The description field is free text ("Demolition of existing garage and erection of two-storey side extension..."). Build three competing approaches to structure it, then benchmark them:

1. **Hand-label a golden set:** 200–300 applications. Label development type (extension / new-build residential / change of use / commercial / other) and, where stated, unit count and storeys. ~3 hours of tedium that makes every claim in this project measurable. Store as `data/golden/extraction_labels.csv`.
2. **Baseline:** TF-IDF + logistic regression for development type. 30 minutes of work, and the yardstick everything must beat.
3. **Fine-tuned transformer:** DistilBERT (or similar small model) fine-tuned for development-type classification with the HF `Trainer` API. Track runs in MLflow. This is your PyTorch/HF learning vehicle — train locally or on free Colab GPU; your RTX 5060 laptop handles DistilBERT fine-tuning comfortably.
4. **LLM structured extraction:** prompt an LLM to return a Pydantic-validated JSON schema (type, units, storeys). Run a small local model via Ollama for bulk work; optionally compare one API model. **Cache every call** (hash input → store output in SQLite) and develop on 100 rows before scaling.
5. **The benchmark table** (this is the interview goldmine): accuracy/F1 on the golden set, cost per 1,000 documents, latency per document — for all three approaches. Write up which you'd choose in production and why.

**Definition of done:** benchmark table in README; extracted fields written back to the main store for Phase C.
**Pitfall to avoid:** don't fine-tune on labels produced by the same LLM you benchmark against without saying so — that's circular. Golden set labels are yours, by hand.

### Phase C — Outcome model (late September, ~10 hrs — deliberately lean)
1. Predict APPROVED vs REFUSED on decided applications. Features: extracted fields (Phase B) + metadata (borough, application type, decision route) + **enrichment flags from planning.data.gov.uk** (conservation area, listed building, flood zone) for a sample of geocodable sites.
2. Majority-class + logistic regression baselines first; then one gradient-boosted model. Metrics: PR-AUC and calibration — approvals dominate, so plain accuracy is meaningless (say this in the README; it's a classic interviewer probe).
3. NO SMOTE unless you can articulate why the imbalance is actually hurting a probability model (it usually isn't — you want calibrated probabilities, not resampled classes).
4. Brief SHAP pass + an **error analysis section**: which application types / boroughs does it fail on?

**Definition of done:** model beats baseline meaningfully, calibration plot exists, error analysis written. Then stop — this phase's point is "LLM-extracted features feeding a downstream model," not another Project 1.

### Phase D — RAG with an evaluation harness (October, ~25 hrs, alongside LangChain course)
1. Corpus: application descriptions + decision info, plus a small set of planning-policy PDFs (e.g., the London Plan chapters — public GLA documents) so retrieval spans two document types.
2. Standard pipeline: chunking → embeddings → FAISS → LangChain retrieval QA with citations. Every answer must cite the application ids or policy sections it used.
3. **The harness (the actual point):**
   - 50 golden Q&A pairs you write, mixing lookups ("how many major applications did Camden decide in 2025?"), aggregations, and policy questions — including 5 unanswerable ones (correct behaviour: "I don't know").
   - Retrieval metrics: hit-rate@k, MRR against annotated relevant documents.
   - Generation metrics: faithfulness/groundedness (LLM-as-judge — RAGAS or a hand-rolled judge prompt; hand-rolled teaches you more) + exact-match on the numeric lookups.
   - An experiment log: change chunk size, k, embedding model → measure → keep the winner. Three or four logged experiments is plenty.
4. README section: **"Where the system fails"** — the 5–10 question types it gets wrong and why.

**Definition of done:** eval script runs with one command and prints the scorecard; failure-modes section written.

### Phase E — Ship & tell (late October, ~15 hrs)
1. FastAPI: `/extract`, `/predict`, `/ask` endpoints, Pydantic schemas, pytest coverage, Docker, deploy to Render (all muscle memory from Projects 1–2 by then).
2. Streamlit: **hero image = borough-level map of development activity** (approval rates or units proposed, colour-graded — visually rhymes with your Project 1 price map), plus a Q&A tab hitting `/ask`, plus the benchmark and eval scorecards as visible tables.
3. README per your instruction template (problem → value → data → decisions with rejected alternatives → results vs baselines → demo link → limitations). Loom walkthrough (≤4 min). LinkedIn post: problem, one technical challenge (the three-way extraction benchmark is the natural pick), one number, demo link.
4. Add the one cross-project sentence + chart linking planning activity to your Project 1 price data.

---

## 5. Repo structure

```
planscope-london/
├── src/planscope/
│   ├── ingest.py          # PLD API client, pagination, retries
│   ├── clean.py           # normalisation, decision mapping
│   ├── enrich.py          # planning.data.gov.uk lookups
│   ├── extraction/        # llm_extractor.py, bert_classifier.py, baseline.py
│   ├── outcome/           # features.py, train.py
│   ├── rag/               # index.py, chain.py, evaluate.py
│   └── api/               # FastAPI app
├── notebooks/             # 01_profile, 02_extraction_benchmark, 03_outcome, 04_rag_eval
├── data/golden/           # hand labels + golden Q&A (committed; raw data is not)
├── tests/
├── app/                   # Streamlit
├── configs/  Dockerfile  Makefile  pyproject.toml  README.md
└── .github/workflows/ci.yml
```

---

## 6. Budget & compute

- PLD + planning.data.gov.uk: **free**.
- Fine-tuning DistilBERT: free (your laptop's RTX 5060 or Colab free tier).
- LLM calls: bulk extraction on a local Ollama model = £0; if you add an API model for comparison, cached + capped development means **$5–15 total** is realistic. Hard rule: never run the full corpus through a paid API without a cost estimate first (rows × avg tokens × price).
- Hosting: Render free tier (as Projects 1–2), embeddings computed offline and shipped with the image.

## 7. Risks & mitigations

| Risk | Likelihood | Mitigation |
|---|---|---|
| Guest API header/endpoint changed | Low | Day-1 smoke test; Datastore access request submitted as backup; planning.data.gov.uk fallback |
| Decision data patchy for some boroughs | Medium | Train on decided apps only; report completeness per borough honestly |
| Scope creep (objection letters, all 33 boroughs, agents) | High | Objection letters = stretch goal only; borough subset is fine; no agents — RAG + tools is enough |
| LLM costs runaway | Medium | Local model for bulk, caching, 100-row dev subsets, cost estimate before any full run |
| Phase 5 courses slip | Medium | Phase A is skills-independent — do it now; B–D flex with your course pace |

## 8. Interview-readiness checklist (what "good command" means)

By the end you should be able to, unaided on a whiteboard:
- Explain tokenisation, attention at a high level, and what fine-tuning DistilBERT actually updates
- Defend when you'd pick TF-IDF vs a fine-tuned transformer vs a prompted LLM (with YOUR numbers)
- Explain your chunking/embedding/retrieval choices and what happened when you varied them
- Define faithfulness, hit-rate@k, MRR, calibration, PR-AUC — and why accuracy was wrong here
- Explain why you did NOT use SMOTE, and why agents were out of scope
- Walk through the pipeline end-to-end: API → parquet → extraction → model → RAG → FastAPI → Streamlit
- State the system's three biggest limitations without being asked

If any of these fails, that's the gap to close before applying — the project exists to make these answers yours.

---

## 9. Instructions for Opus (paste into the project's custom instructions)

> I am building "PlanScope London", the capstone of my ML portfolio: an LLM-powered system over London planning applications (Planning London Datahub) with three layers — (1) structured extraction from free-text descriptions comparing TF-IDF baseline vs fine-tuned DistilBERT vs prompted LLM, (2) an approval-outcome model using the extracted features, (3) a RAG Q&A system with a proper evaluation harness (golden Q&A set, hit-rate@k, faithfulness). Deployed with FastAPI + Docker + Streamlit, tracked with MLflow, tested with pytest + GitHub Actions. I am following a phased plan (ingest → extraction benchmark → outcome model → RAG eval → ship) and I will tell you which phase I'm in.
>
> My goal is not just a finished project — it is that I can defend every decision in a technical interview. Work with me like this:
>
> 1. **Teach before code.** Before implementing anything new (Elasticsearch pagination, tokenisers, the HF Trainer, FAISS, LLM-as-judge), explain the concept and the design options in plain language first, ask me which option I'd choose and why, and only then write code. If I just say "write it", give me a one-question quiz on the previous component first.
> 2. **One component at a time.** Never generate multi-file scaffolds in one response. Build the smallest working piece, have me run it, then extend. I learn from the run-fail-fix loop.
> 3. **Make me write some of it.** For core logic (the decision-mapping function, the eval metrics, the judge prompt), give me the function signature and a description of the behaviour, let me attempt it, then review my attempt harshly before showing yours.
> 4. **Enforce the plan's guardrails:** golden labels are hand-made by me, never LLM-generated silently; baselines before fancy models; no SMOTE without a written justification; every LLM call cached; cost estimate before any full-corpus run; tests and CI stay green; no agents, no scope creep into objection letters unless core phases are done.
> 5. **Interrogate my results.** When I show you metrics, ask about leakage, class balance, and whether the comparison is fair before congratulating me. If a result looks too good, assume a bug.
> 6. **Interview lens always on.** After each phase, generate 10 interview questions an ML engineer would ask about what I just built, and grill me on them until my answers are solid.
> 7. Explanations in plain language with intuition first, code second — that's how I learn best.

---

## 10. Timeline summary

| When | What | Hours |
|---|---|---|
| July–Aug (now, alongside Phase 3) | Phase A: ingest, clean, profile, tests, CI | ~15 |
| September (with HF/PyTorch study) | Phase B: golden set, baseline, fine-tune, LLM extraction, benchmark | ~25 |
| Late September | Phase C: outcome model + error analysis | ~10 |
| October (with LangChain study) | Phase D: RAG + evaluation harness | ~25 |
| Late October | Phase E: API, Docker, Streamlit, README, Loom, LinkedIn | ~15 |
| **Total** | | **~90 hrs** |

This lands the finished capstone right at the start of your Phase 6 application window (November), exactly as your roadmap intends.
