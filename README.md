# TechSight

**Technology Trend Aggregation and Forecasting Platform**

A scalable data pipeline and analytics platform designed to monitor, cluster, and forecast emerging technology trends by synthesizing public research literature (OpenAlex, arXiv) and patent filing records (USPTO).

---

## Overview

### Problem Statement
Identifying technological breakthroughs before widespread commercialization is critical for R&D strategists, investors, and research institutions. However, technology indicators are siloed across disconnected datasets—academic preprints, peer-reviewed journals, and patent registries—each with disparate schemas, indexing latencies, and metadata structures. TechSight unifies these streams into an integrated analytical pipeline featuring unsupervised topic discovery, multi-factor momentum scoring, and probabilistic activity forecasting.

### Core Capabilities
- **Multi-Source Data Synthesis**: Automated ingestion and schema normalization across scientific literature and patent databases.
- **Unsupervised Topic Discovery**: TF-IDF vectorization and KMeans clustering to detect evolving terminology and emerging sub-fields without manual taxonomy maintenance.
- **Explainable Trend Scoring**: A transparent, 5-factor composite index (0–100) combining CAGR, growth consistency, acceleration, volume, and citation momentum.
- **Time-Series Forecasting**: Dual-model projection framework (Linear Regression & Moving Average) with 80% confidence intervals and historical accuracy tracking.
- **High-Performance Interface**: Interactive analytics dashboard with dynamic filtering, comparative factor breakdowns, and tabular exploration.

---

## System Architecture

```
                       [ Data Sources ]
                OpenAlex / arXiv  |  USPTO Archives
                       │                 │
                       ▼                 ▼
             ┌─────────────────────────────────────┐
             │         1. INGESTION LAYER          │
             │   Connectors, Pagination, Caching   │
             └──────────────────┬──────────────────┘
                                │
                                ▼
             ┌─────────────────────────────────────┐
             │         2. CLEANING & NLP           │
             │ Text Normalization, TF-IDF, KMeans  │
             └──────────────────┬──────────────────┘
                                │
                                ▼
             ┌─────────────────────────────────────┐
             │       3. ANALYTICS & FORECAST       │
             │  Trend Score Indexing, Projections  │
             └──────────────────┬──────────────────┘
                                │
               ┌────────────────┴────────────────┐
               ▼                                 ▼
    ┌─────────────────────┐           ┌─────────────────────┐
    │   FastAPI Backend   │           │ Streamlit Dashboard │
    │    REST Endpoints   │           │ Plotly Visualizer   │
    └─────────────────────┘           └─────────────────────┘
```

### Architectural Artifacts
- **Architecture Diagram**:
  ![TechSight Architecture](docs/architecture_diagram.jpg)
- **Dashboard Overview**:
  ![Dashboard Overview](docs/dashboard_screenshot.jpg)

---

## Technology Stack

| Layer | Component | Description |
|---|---|---|
| Runtime | Python 3.11+ | Core execution environment |
| Processing | pandas, numpy | Vectorized transformations & indicator aggregation |
| Machine Learning | scikit-learn | TF-IDF extraction, KMeans clustering, linear models |
| API Service | FastAPI, Uvicorn | Asynchronous RESTful service layer with OpenAPI |
| Data Layer | PostgreSQL, Psycopg2 | Relational schema with indexed temporal querying |
| Visual Analytics | Streamlit, Plotly | Clean dark-mode dashboard with interactive charts |
| Testing | pytest | Unit, integration, and mathematical correctness tests |

---

## Analyzed Technology Domains

| Domain | Focus Areas | Signal Status | Momentum Score |
|---|---|---|:---:|
| **Solid-State Batteries** | Solid electrolyte, lithium-metal, garnet structures | Growing | 72.4 |
| **Quantum Computing** | Qubit coherence, error mitigation, topological circuits | Emerging | 82.1 |
| **Generative AI** | Foundation models, attention mechanisms, diffusion | Emerging | 96.8 |

---

## Quickstart

### 1. Environment Setup

```bash
# Clone the repository
git clone https://github.com/yashvanthkanna-s/TECHSIGHT.git
cd TECHSIGHT

# Create and activate virtual environment
python -m venv venv
venv\Scripts\activate        # Windows
source venv/bin/activate     # macOS/Linux

# Install dependencies
pip install -r requirements.txt
```

### 2. Configure Environment Variables

Copy `.env.example` to `.env` and provide your credentials (the pipeline supports offline calibrated mode if keys are omitted):

```bash
cp .env.example .env
```

```env
OPENALEX_API_KEY=your_key_here
OPENALEX_EMAIL=your_email@domain.com
USPTO_API_KEY=
DB_HOST=localhost
DB_NAME=techsight
```

### 3. Run Pipeline Execution

Execute the full end-to-end data pipeline:

```bash
python run_pipeline.py
```

This ingests raw records, performs text cleaning, computes KMeans topic clusters, calculates composite Trend Scores, and fits forecasting models.

### 4. Launch the Applications

**Launch Dashboard:**
```bash
streamlit run dashboard/app.py
```
*Access interface at `http://localhost:8501`*

**Launch REST API:**
```bash
uvicorn api.main:app --reload --port 8000
```
*Interactive Swagger docs available at `http://localhost:8000/docs`*

### 5. Execute Test Suite

```bash
pytest tests/ -v
```

---

## Repository Structure

```
TECHSIGHT/
├── config.py                  # Global application configuration
├── run_pipeline.py            # End-to-end pipeline entrypoint
├── requirements.txt           # Environment dependencies
├── schema.sql                 # PostgreSQL relational schema
├── ingestion/                 # Source data connectors
│   ├── openalex_connector.py  #   OpenAlex API connector
│   ├── uspto_connector.py     #   USPTO data connector
│   └── arxiv_connector.py     #   arXiv preprint connector
├── processing/
│   └── cleaner.py             # Normalization and entity deduplication
├── nlp/
│   └── topic_model.py         # TF-IDF vectorization & KMeans clustering
├── analytics/
│   └── trend_score.py         # 5-factor Trend Score calculator
├── forecasting/
│   └── models.py              # Time-series regression & evaluation
├── database/
│   └── connection.py          # Relational database interface
├── api/
│   └── main.py                # FastAPI REST endpoints
├── dashboard/
│   └── app.py                 # Streamlit analytical dashboard
├── tests/
│   └── test_pipeline.py       # Comprehensive pytest validation suite
├── mock/
│   ├── mock_data.py           # Calibrated synthetic data generator
│   └── README.md
└── docs/                      # Architectural and algorithmic documentation
    ├── design_decisions.md
    └── trend_score.md
```

---

## Mathematical Formulation: Trend Score

The composite Momentum Index $S \in [0, 100]$ evaluates five quantitative dimensions:

$$S = 0.35 \cdot G + 0.25 \cdot C + 0.20 \cdot A + 0.10 \cdot V + 0.10 \cdot Cit$$

| Parameter | Weight | Description |
|---|:---:|---|
| **Growth Rate ($G$)** | 35% | Compound Annual Growth Rate (CAGR) over the lookback window |
| **Consistency ($C$)** | 25% | Proportion of periods exhibiting positive year-over-year expansion |
| **Acceleration ($A$)** | 20% | Second derivative of trajectory (inflection detection) |
| **Volume ($V$)** | 10% | Log-scaled total corpus volume |
| **Citation Velocity ($Cit$)** | 10% | Growth rate of citation impact over observed window |

**Classification Thresholds:**
- **Emerging**: $75 \le S \le 100$
- **Growing**: $55 \le S < 75$
- **Mature / Stable**: $40 \le S < 55$
- **Declining**: $0 \le S < 40$

---

## Evaluation & Error Metrics

Forecast reliability is quantified using standard time-series validation measures:
- **Mean Absolute Error (MAE)**: Average absolute magnitude of projection deviations.
- **Root Mean Squared Error (RMSE)**: Penalizes severe outlier variances.
- **Mean Absolute Percentage Error (MAPE)**: Relative accuracy percentage across varying scale magnitudes.

---

## License

MIT License. Open for academic, non-commercial, and research use.
