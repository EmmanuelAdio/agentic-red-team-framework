# Claude Code Prompt — Generate Appendix Chapters

Copy everything below the line into Claude Code, working from the root of your thesis repository (the one containing both your LaTeX source and your `src/redteam/` code).

---

## Task

I am submitting an MSci dissertation titled *"Agentic Red-Team Framework for Evaluating LLM-Augmented Search Systems"*. The thesis is written in LaTeX. I need you to generate a set of **appendix chapters** as LaTeX, and to tell me **exactly where in the main chapters to insert cross-references** to those appendices.

You have access to:
1. The full thesis LaTeX source (the `.tex` files for Design, Methodology, Experimentation, Results, Discussion, Conclusion).
2. The actual project codebase (`src/redteam/`, `scripts/`, `tests/`, `results/`, `data/`).

**Critical rule: every value you put in the appendices must be extracted from the real codebase or the real result files. Do not invent, approximate, or hallucinate hashes, payloads, test names, file paths, or statistics.** If you cannot find a real value, leave a clearly marked `% TODO: <what is missing and where to find it>` placeholder rather than fabricating. This is a reproducibility-focused thesis; a fabricated hash or payload would be an integrity problem.

## Before you start

1. Read the existing `.tex` files to learn my conventions: how I name `\label{}`s (e.g. `sec:`, `tab:`, `fig:`, `ch:`), how I structure `\chapter`/`\section`, which packages I load in the preamble (`listings`? `booktabs`? `csvsimple`?), and my citation style.
2. Locate the appendix-relevant source files: the prompt templates, the attack payload generators, the bundle Pydantic schema, the test suite, and the four committed result CSVs under `results/tables/`.
3. Match my existing formatting exactly. Do not introduce a new table style or code-listing style if I already have one.

## Deliverable

Produce a file `appendices.tex` (or one file per appendix if that fits my project structure better — check how my `main.tex` includes chapters and follow the same pattern). It must begin with `\appendix` and contain the six appendices below, each as `\chapter{...}` with a stable `\label{}`.

### Appendix A — Prompt templates with SHA-256 hashes
Extract the full text of each prompt template from the code:
- The RAG generator system prompt (the "answer from context only" instruction).
- The IPI instruction-override template.
- The corpus-poisoning answer-replacement template.
- The jamming template.
- The LLM exploit-generator system prompt.

For each, present the verbatim template text in a `lstlisting` (or my existing code-listing environment) and print its real SHA-256 hash beneath it. **Compute the hash the same way the code does** — find the hashing function in the source (it populates `prompt_template_hash` and `exploit_prompt_template_hash` in the bundle JSON) and reuse its exact normalisation so the printed hash matches what appears in the bundles. Verify at least one hash against a real bundle file in `results/runs/`.

### Appendix B — Concrete attack payload examples
Pick ONE representative query (use `"where did huntington's disease get its name"` if it exists in `data/queries.json`, otherwise pick another query that has bundles for all four cells). For that single query, show the actual payload produced by each of the four cells, for BOTH the template path (iteration 0) and the LLM path (iteration ≥ 1):
- Cell 1 IPI — the injected document body.
- Cell 2 corpus poisoning (answer replacement) — the injected document body.
- Cell 3 corpus poisoning (jamming) — the injected document body.
- Cell 4 query injection — the `modified_query` string.

Pull these from real bundle JSON files (the `attack.payload` and `attack.modified_query` fields), not by re-running the generators. Caption each with the `run_id` it came from so it is traceable.

### Appendix C — Complete statistical tables
Reproduce the four committed CSV files in full as LaTeX tables: `summary_by_cell.csv`, `ragas_by_cell.csv`, `paired_differences_vs_ipi.csv`, `baseline_summary.csv`. Read the actual files in `results/tables/`. Use `booktabs` (or my existing table style). If a table is wide, use `\small` or `\resizebox` rather than dropping columns. Caption each with its source filename.

### Appendix D — Unit test suite reference table
Parse the `tests/` directory. Produce a table grouped by module (orchestration, metrics, dashboard smoke, corpus/query pipeline, schema) listing each test function name and a one-line description of the invariant it pins (infer the description from the test name and body). Count the tests and confirm the total matches the "76/76" claim in the Experimentation chapter — if the real count differs, report the real number and flag the discrepancy with a `% TODO` note so I can correct the main text.

### Appendix E — Ethics and responsible disclosure statement
This is the one appendix written as prose, not extracted from code. Draft a half-page statement covering: experiments ran only on a locally hosted testbed; no third-party or production systems were probed; all poisoned documents were generated and inserted by the framework itself and removed after each run; no real user data or PII was involved; the threat model studied is one defenders need in order to protect against EchoLeak-class attacks; and the artefact is released under its licence (check the repo `LICENSE` file for the actual licence — do not assume MIT) with documentation for responsible use. Keep the tone factual and measured.

### Appendix F — Annotated project file structure
Generate a directory tree of the repository (respect `.gitignore`; exclude `.venv`, `__pycache__`, `.cache.sqlite` contents, and the bulk `results/runs/` bundle files — show the folder but not all 600 files). Annotate each directory and key file with a one-line comment. Present it in a `verbatim` or listing environment.

## Cross-reference task

After generating the appendices, scan the main chapters and produce a **separate list** (not edits yet — a list I can review first) of every place where a `see Appendix~\ref{...}` cross-reference should be added. For each, give me:
- The file and approximate line or the surrounding sentence.
- The exact LaTeX to insert (e.g. `\Cref{app:payloads}` or `(see Appendix~\ref{app:payloads})`).

At minimum, link:
- §2.5 (attack taxonomy) and Chapter 4 attack discussions → Appendix B (payloads).
- The `prompt_template_hash` / reproducibility discussion in §1.5.3 and §2.8 → Appendix A (templates + hashes).
- Every table in Chapter 4 sourced from a CSV → Appendix C (full tables).
- The "76/76 green" claim in §3.7 → Appendix D (test table).
- The threat-model section §1.2 and the introduction → Appendix E (ethics).
- The reproduction recipe §3.7 → Appendix F (file structure).

## Output format

1. First, a short summary of what you found in the codebase (which files held the templates, how the hash is computed, the real test count, the actual licence).
2. The generated `appendices.tex` (or per-appendix files).
3. The cross-reference insertion list.
4. Any `% TODO` placeholders collected into one list at the end so I can resolve them.

Do not modify my main chapter `.tex` files yet — I want to review the cross-reference list first.
