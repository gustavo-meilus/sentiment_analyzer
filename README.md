# Sentiment Analyzer — Aspect-Based Sentiment Analysis of YouTube Live Chat

> Aspect-based sentiment analysis of Brazilian Portuguese YouTube live stream chat using open-source LLMs. MBA thesis research project comparing three prompting strategies (zero-shot, ICL, CoT) across nine model architectures. **Experimental design: 9 models × 3 strategies × 5 repetitions = 135 classification runs.**

## What this project does

The pipeline extracts all chat messages from a YouTube live stream replay, classifies each message on two dimensions (sentiment + aspect) using nine local open-source LLMs running on Google Colab GPUs, and produces statistically rigorous metrics comparing model performance across prompting strategies.

**Data source:** CazéTV — RD Congo vs Jamaica, FIFA 2026 World Cup Playoff Final (Brazilian Portuguese live stream chat).

**Classification dimensions:**

| Dimension | Labels |
|---|---|
| Sentiment | Positive / Negative / Neutral |
| Aspect | content / presenter / community / audio\_video / meta / unknown |

## Research hypotheses

The 9×3 factorial design tests whether prompting strategy and model architecture significantly affect classification quality:

- **H1:** Does ICL improve F1 over zero-shot?
- **H2:** Does CoT improve F1 over zero-shot?
- **H3:** Is there a model × strategy interaction?
- **H4:** Do reasoning-distilled models (DeepSeek R1) benefit more from CoT than instruction-tuned models?
- **H5:** Does ICL reduce classification variance?
- **H6:** Does multi-model consensus approximate human judgment (Pearson r >= 0.85)?

Statistical analysis uses two-way ANOVA with Shapiro-Wilk and Levene's assumption checks, with Kruskal-Wallis as a non-parametric fallback.

## Architecture

```
LOCAL MACHINE                          GOOGLE COLAB (T4/A100 GPU)
==============                         ==========================

Phase 1-2: Extract                     Phase 3: Setup + Ingest
  uv run extract --url URL               pip install vllm polars ...
  -> output/{video_id}/                  Mount Drive, detect GPU
     chat_messages.csv                   Configure vLLM dtype/memory
     chat_messages.json
     metadata.json                     Phase 4: Clean + Sample
          |                              Polars LazyFrame pipeline
          | upload to Drive              Volatility-peak sampling
          +------------------------>     Export gt_sample.csv

Phase 4 (local): Annotate              Phase 4.6: Ingest human GT
  uv run annotate                        Load ground_truth_human.csv
  -> ground_truth_human.csv
          |                             Phase 5: Classify (135 runs)
          | upload to Drive               9 models × 3 strategies × 5 reps
          +------------------------>      vLLM + GuidedDecodingParams
                                          Checkpoint per model

                                        Phase 6: Evaluate
                                          Dual-GT metrics, ANOVA,
                                          Fleiss' kappa, variance analysis

                                        Phase 7: Visualize
                                          Heatmaps, scatter plots, timeline
                                          Thesis-ready PNG/SVG/HTML
```

## Models

Nine open-source models run via vLLM on Colab GPU (T4/A100). No API cost.

| Model | Parameters |
|---|---|
| `deepseek-r1-1.5b` | 1.5B — reasoning-distilled |
| `deepseek-r1-7b` | 7B — reasoning-distilled |
| `gemma2-2b` | 2B |
| `gemma2-9b` | 9B |
| `llama3.2-3b` | 3B |
| `phi4-mini` | ~3.8B |
| `qwen2.5-3b` | 3B |
| `qwen2.5-7b` | 7B |
| `qwen3.5-4b` | 4B |

## Prompting strategies

| Strategy | Description |
|---|---|
| **Zero-shot** | No examples. Model relies on pretrained understanding of sentiment and aspect. |
| **ICL** | In-context learning with labeled examples drawn from a holdout pool. |
| **CoT** | Chain-of-thought reasoning before classification. |

## Ground truth

Dual-tier evaluation:

- **Tier 1 (Gold):** Human-annotated messages. Primary accuracy reference for all metrics.
- **Tier 2 (Silver):** Consensus proxy messages where ≥ 2/3 models agree. Validated via Pearson correlation with Tier 1.

## Project structure

```
sentiment_analyzer/
  sentiment_analyzer.ipynb         # Colab notebook (Phases 3-7)
  pyproject.toml                   # uv project — CLI tools only (extract + annotate)
  ground_truth_human.csv           # Human-annotated ground truth
  gt_sample.csv                    # Sample exported for annotation
  src/
    extractor/
      cli.py                       # CLI: uv run extract --url URL
      chat_extractor.py            # YouTube chat extraction + SHA-256[:8] author anonymization
    annotator/
      cli.py                       # CLI: uv run annotate --input FILE
      tui.py                       # Curses TUI — 2-keypress annotation (sentiment then aspect)
    sentiment_analyzer/
      __init__.py
      py.typed
  docker/
    Dockerfile                     # Python 3.11-slim + Jupyter + full data science stack
    docker-compose.yml             # Jupyter on port 8888 for local analysis
    data/                          # Docker data volume
  pipeline/                        # gitignored — generated outputs
    classifier_output/             # 9 model subdirs × 3 scenarios × 5 runs = 135 parquets
    metrics_output/                # CSVs + variance_report.json
    visualizations/                # PNGs, SVGs, HTML
```

## Quick start

**Prerequisites:** Python >= 3.11, [uv](https://docs.astral.sh/uv/) package manager, Google Colab account.

```bash
# Step 1: Install and extract chat data (local)
uv sync
uv run extract --url "https://www.youtube.com/watch?v=VIDEO_ID" --output-dir ./output
```

```bash
# Step 2: Annotate ground truth sample (local, after downloading gt_sample.csv from Drive)
uv run annotate --input ./output/gt_sample.csv --output ./output/ground_truth_human.csv
```

```bash
# Step 3: Local analysis and visualization (Docker)
cd docker && docker compose up
# Jupyter available at http://localhost:8888 (no token or password)
```

**Full workflow:**

1. `uv run extract` → upload `output/` to Google Drive
2. Run Colab notebook Phases 3–4 → exports `gt_sample.csv` to Drive
3. Download `gt_sample.csv` → `uv run annotate` → upload `ground_truth_human.csv` to Drive
4. Continue Colab notebook Phases 4.6–7 (vLLM inference on T4/A100)
5. For local analysis: `cd docker && docker compose up` → Jupyter at `http://localhost:8888`

## Extractor details

`src/extractor/chat_extractor.py` uses `yt-chat-downloader` — no browser, no authentication. The library calls the same continuation-token API as the YouTube web player.

Author anonymization runs before any disk write:

```python
# src/extractor/chat_extractor.py
import hashlib
author_hash = hashlib.sha256(author.encode("utf-8")).hexdigest()[:8]
```

**Output schema:** `message_id`, `timestamp`, `time_in_seconds`, `author_hash`, `message`, `author_type`, `message_type`, `datetime`, `captured_at`

**Output directory:** `./output/{video_id}/chat_messages.csv` + `chat_messages.json` + `metadata.json`

## Annotator details

`src/annotator/tui.py` implements a curses TUI with immutable state (`AnnotationState` frozen dataclass).

| Step | Keys | Labels |
|---|---|---|
| 1 — Sentiment | `p`/`P` = Positive, `n`/`N` = Negative, `u`/`U` = Neutral | — |
| 2 — Aspect | `c` = content, `r` = presenter, `m` = community, `a` = audio\_video, `t` = meta, `x` = unknown | — |

Navigation: `←` / Backspace to go back, `↑`/`↓` to scroll context pane (±4 messages), `s` = save, `q` = save and quit. Auto-saves every 25 annotations.

## Pipeline outputs (gitignored)

### `pipeline/classifier_output/`

Per-model files: `classified_{model}_{scenario}_run_{N}.parquet`, `latency_log_{model}.json`, `checkpoint.parquet`

Aggregate files: `all_classified.parquet`, `full_classified.parquet`

### `pipeline/metrics_output/`

| File | Content |
|---|---|
| `metrics_vs_human_gt.csv` | Accuracy, precision, recall, F1, kappa per (model, scenario) |
| `metrics_vs_consensus_gt.csv` | Same metrics against consensus proxy |
| `statistical_tests.csv` | Two-way ANOVA p-values (model, scenario, interaction) |
| `aggregated_run_metrics.csv` | Per-run F1 for variance analysis (135 rows) |
| `variance_report.json` | Per-model σ², ICL vs zero-shot variance comparison |
| `cross_model_agreement.csv` | Fleiss' kappa and pairwise Cohen's kappa |
| `per_message_entropy.csv` | Shannon entropy per message across models |
| `proxy_validation.csv` | Pearson r between dual-GT F1 vectors |
| `confusion_matrix_{m}_{s}.csv` | 27 confusion matrices (one per model × scenario) |
| `aspect_distribution.csv` | Aspect frequency and per-aspect F1 |
| `latency_table.csv` | Model × scenario latency and F1 |
| `consistency_report.csv` | Intra-model consistency across repetitions |
| `coverage_report.csv` | Message coverage per run |

### `pipeline/visualizations/`

| File | Type | Description |
|---|---|---|
| `confusion_matrix_*.png/svg` | Static | 27 confusion matrix heatmaps (one per model × scenario) |
| `model_scenario_heatmap.png/svg` | Static | F1 scores: models (rows) × scenarios (columns) |
| `metrics_comparison.png/svg` | Static | Grouped bar chart with error bars |
| `variance_scatter.png/svg` | Static | Per-message entropy colored by label |
| `latency_f1_scatter.png/svg` | Static | Efficiency frontier: latency vs quality |
| `consistency_heatmap.png/svg` | Static | Intra-model consistency across runs |
| `sentiment_timeline.html` | Interactive | Full stream timeline with range slider |
| `figure_catalog.csv` | Index | Catalog of all generated figures |

## Technology stack

| Category | Packages |
|---|---|
| **Inference** | vLLM, Pydantic 2.12.3 |
| **Data** | Polars 1.39.3, Pandas 3.0.2, PyArrow 23.0.1, NumPy 2.4.4 |
| **Statistics** | SciPy 1.17.1, scikit-learn 1.6.1, pingouin 0.6.1, statsmodels 0.14.4 |
| **Visualization** | Plotly 6.6.0, Seaborn 0.13.2, Matplotlib 3.10.8, Kaleido 1.2.0 |
| **Extraction** | yt-chat-downloader 0.0.3, chat-downloader 0.2.8, python-docx 1.2.0 |

Docker (`docker/Dockerfile`): Python 3.11-slim + Jupyter notebook on port 8888. Purpose: local analysis and visualization only — vLLM inference runs on Colab GPU.

## Privacy and legal

All author display names are SHA-256[:8] hashed before any disk write. No raw usernames are stored in any file. The hash cannot be reversed to recover the original display name.

LGPD Article 4 §II-a (academic research exemption) and Article 7 §IV (manifestly public data) apply. Chat messages are voluntarily posted in a public forum. Only anonymized, aggregated results (metrics and visualizations) are published from this research.
