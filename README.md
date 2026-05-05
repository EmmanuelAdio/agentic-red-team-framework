# Agentic Red-Team Framework for LLM-Augmented Search Systems

**Author:** Emmanuel Adio (F229639) &nbsp;·&nbsp; **Supervisor:** Dr Georgina Cosma &nbsp;·&nbsp; **Module:** 25COD290 (MSci, Loughborough University)

An open-source agentic framework that autonomously plans, executes, and evaluates adversarial attacks
(prompt injection + corpus poisoning) against a Retrieval-Augmented Generation pipeline.
Each run produces a reproducible exploit-bundle JSON scored by RAGAS and custom attack-success metrics.

## Setup

```powershell
python -m venv .venv ; .venv\Scripts\activate
pip install -r requirements.txt
Copy-Item .env.example .env   # then edit .env and set OPENAI_API_KEY
python scripts/01_build_corpus.py ; python scripts/02_run_baseline.py
```
