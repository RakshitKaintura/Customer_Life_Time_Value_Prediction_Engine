# Customer Lifetime Value (LTV) Prediction Engine

A high-performance pipeline for predicting Customer Lifetime Value using a hybrid approach of traditional probabilistic models (BG/NBD, Gamma-Gamma) and modern Transformer ensembles.

## Project Structure
- `backend/`: Core logic and Python packages.
- `supabase/`: Database schema and migrations.
- `notebooks/`: Exploratory Data Analysis (EDA) and experimental models.
- `tests/`: Automated test suite.

## Tech Stack
- **Data:** Polars, DuckDB, Supabase (PostgreSQL).
- **ML:** BG/NBD, Lifetimes, PyTorch (Transformers), XGBoost.
- **Orchestration:** UV, Python 3.11.
- **Monitoring:** Weights & Biases.

## Setup
1. Create environment: `uv venv --python 3.11`
2. Activate environment: `.venv\Scripts\activate`
3. Install dependencies: `uv sync --all-extras`
