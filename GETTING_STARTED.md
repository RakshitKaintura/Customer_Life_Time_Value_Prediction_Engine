# 🎯 Project Overview & Quick Start Guide

## What Is This Project?

**Customer Lifetime Value Prediction Engine** — A production-grade machine learning system that predicts how much value each customer will bring to your business over time.

Think of it as a crystal ball for customer analytics: given a customer's purchase history, we predict their future spending with high accuracy.

---

## 🎯 The Big Picture

### Problem We Solve
- **Identify high-value customers** before they churn
- **Allocate marketing budget** to customers with highest ROI potential
- **Prevent wasteful spending** on low-LTV segments
- **Personalize retention** based on predicted future value

### How We Solve It
We use a **hybrid ensemble approach** combining 3 different ML strategies:

1. **Probabilistic Models** (BG/NBD + Gamma-Gamma)
   - Classic statistical approach with uncertainty quantification
   - Best for stable, mature customer bases
   
2. **Deep Learning** (Transformer Models)
   - Neural networks learning from purchase sequence patterns
   - Excellent for capturing non-linear behaviors
   
3. **Causal Analysis** (DAGs & Treatment Effects)
   - Understand what marketing actions actually drive value
   - Not just predictions, but insights

These three approaches are **combined via learned weights** to produce final predictions.

---

## 📊 System Components

### Data Pipeline
```
Raw Transactions → Clean & Validate → RFM Analysis → Calibration/Holdout Split
```

### ML Models
```
BG/NBD              Transformer         Causal ML
(Probabilistic)     (Deep Learning)     (Inference)
      ↓                   ↓                   ↓
                  XGBoost Fusion
                       ↓
                  Final LTV Score
```

### Serving Layer
```
FastAPI Server → Real-time Predictions, Batch Scoring, Explainability
```

### Storage
```
PostgreSQL (Supabase) → Features, Predictions, Customers
```

---

## 🗂️ Where to Find Things

| What | Where |
|------|-------|
| **API** | `backend/api/main.py` |
| **ML Models** | `backend/ml/` |
| **Feature Engineering** | `backend/features/` |
| **Data Pipeline** | `orchestration/assets/data_assets.py` |
| **Database Schema** | `supabase/schema.sql` |
| **Tests** | `tests/` |
| **Experiments** | `notebooks/` |
| **Docker** | `docker/docker-compose.yml` |

---

## ⚡ Quick Start (5 Minutes)

### 1. Setup Environment
```bash
# Create Python virtual environment (3.11+)
uv venv --python 3.11
source .venv/bin/activate  # Linux/Mac
.venv\Scripts\Activate.ps1  # Windows

# Install dependencies
uv pip install -e ".[ml]"
```

### 2. Create `.env` File
```bash
SUPABASE_URL=your_url_here
SUPABASE_SERVICE_ROLE_KEY=your_key_here
DATABASE_URL=postgresql://...
API_SECRET_KEY=any_secret_here
```

### 3. Start API Server
```bash
uvicorn backend.api.main:app --reload
```

### 4. Visit API Docs
```
http://localhost:8000/docs
```

**Done!** You now have the full system running locally.

---

## 🔍 Key Files to Understand

### Backend Structure

**`backend/api/main.py`** — FastAPI app setup
- Request handlers
- Middleware
- Dependency injection

**`backend/ml/bgnbd_model.py`** — BG/NBD Model
- Probabilistic customer lifetime value
- Based on Fader & Hardie's research

**`backend/ml/transformer_model.py`** — Deep Learning Model
- Sequence-to-value neural network
- ONNX-exportable

**`backend/ml/fusion.py`** — Ensemble Learning
- Combines multiple models
- Learns optimal weighting strategy

**`backend/features/rfm.py`** — Feature Engineering
- RFM (Recency, Frequency, Monetary)
- Purchase frequency & monetary value

**`backend/data/load_data.py`** — Data Loading
- UCI Online Retail dataset support
- Extensible to custom data sources

---

## 🧪 Running Tests

```bash
# All tests
pytest tests/ -v

# Specific test file
pytest tests/test_bgnbd_model.py -v

# With coverage
pytest --cov=backend tests/

# Quick tests only
pytest -m "not integration" -v
```

---

## 📓 Notebooks (For Exploration)

Each notebook focuses on a specific aspect:

| Notebook | Purpose |
|----------|---------|
| `01_eda.ipynb` | Exploratory data analysis |
| `02_rfm_cohort_analysis.ipynb` | Customer segmentation |
| `03_bgnbd_gamma_gamma.ipynb` | Probabilistic model training |
| `04_transformer_training.ipynb` | Deep learning model |
| `05_causal_ml.ipynb` | Causal analysis |
| `06_fusion_evaluation.ipynb` | Model comparison & ensemble |

Run any notebook:
```bash
jupyter lab notebooks/
```

---

## 🚀 API Examples

### Get LTV Prediction for One Customer
```bash
curl -X GET "http://localhost:8000/api/v1/ltv/customer_123" \
  -H "Authorization: Bearer YOUR_TOKEN"
```

**Response:**
```json
{
  "customer_id": "customer_123",
  "ltv_12m": 4250.50,
  "ltv_36m": 12150.75,
  "confidence": 0.92,
  "model_used": "fusion_v1"
}
```

### Batch Score Multiple Customers
```bash
curl -X POST "http://localhost:8000/api/v1/batch/score" \
  -H "Content-Type: application/json" \
  -d '{
    "customer_ids": ["C001", "C002", "C003"],
    "horizon_months": 12
  }'
```

### Get High-Value Customer Segment
```bash
curl -X GET "http://localhost:8000/api/v1/cohorts/high-value?limit=100"
```

### Explain a Prediction (SHAP)
```bash
curl -X GET "http://localhost:8000/api/v1/ltv/customer_123/explain"
```

All endpoints documented at `http://localhost:8000/docs` (Swagger UI).

---

## 🐳 Docker (Optional)

Run everything in Docker:
```bash
docker-compose -f docker/docker-compose.yml up -d
```

This starts:
- API server (port 8000)
- PostgreSQL database (port 5432)
- Redis cache (port 6379)

---

## 📈 Common Tasks

### Train a New Model
```bash
python -m backend.ml.bgnbd_model --train --data-path ./data/raw/
```

### Run Data Pipeline
```bash
dagster dev -f orchestration/dagster_assets.py
```

### Export Model to ONNX
```bash
python backend/ml/transformer_model.py --export-onnx --output-path ./models/
```

### Check for Data Drift
```bash
python backend/monitoring/drift.py --reference-data ./data/reference.parquet
```

### Batch Score Customers to Database
```bash
python -m backend.workers.batch_scorer --customer-ids-file ./customers.csv
```

---

## 🔧 Configuration

### Environment Variables
All settings in `.env`:
```bash
# Required
SUPABASE_URL              # Your Supabase project URL
SUPABASE_SERVICE_ROLE_KEY # Service role key for server-side access
DATABASE_URL              # PostgreSQL connection string
API_SECRET_KEY            # For API authentication

# Optional
WANDB_API_KEY             # For experiment tracking
HUBSPOT_API_KEY           # For CRM integration
MODEL_CACHE_DIR           # Where to store models
```

### Backend Settings
`backend/config.py` for model parameters:
```python
OBSERVATION_WINDOW_MONTHS = 6      # Calibration period
HOLDOUT_WINDOW_MONTHS = 6          # Holdout period
MAX_SEQUENCE_LENGTH = 50           # Transformer input
```

---

## 🐛 Debugging

### Check API Health
```bash
curl http://localhost:8000/health
```

### View Server Logs
```bash
# If running with uvicorn
# Logs are printed to terminal

# If running with Docker
docker-compose logs -f api
```

### Run Tests with Verbose Output
```bash
pytest tests/ -vv --tb=long
```

### Debug a Specific Model
```python
from backend.ml.bgnbd_model import BgnbdModel
model = BgnbdModel()
model.fit(calibration_data)
model.predict(customer_features)  # Add breakpoints here
```

---

## 📚 Learning Resources

### Key Concepts
- **BG/NBD**: [Fader & Hardie's CLV Tutorials](https://www.brucehardie.com/lectures/index.html)
- **Causal Inference**: [Microsoft DoWhy Documentation](https://py-why.github.io/dowhy/)
- **Transformers**: [Hugging Face NLP Course](https://huggingface.co/course/)

### Project Structure
1. Read [README_ENHANCED.md](./README_ENHANCED.md) for full system overview
2. Browse [orchestration/dagster_assets.py](./orchestration/dagster_assets.py) for data pipeline
3. Check [backend/api/main.py](./backend/api/main.py) for API endpoints
4. Explore [notebooks/](./notebooks/) for experimental code

---

## 🆘 Troubleshooting

| Issue | Solution |
|-------|----------|
| Import errors | Reinstall: `uv pip install -e ".[ml]"` |
| Database connection fails | Check `.env` variables and Supabase status |
| Model training hangs | Check RAM, reduce batch size in config |
| API slow | Check database query logs, consider indexing |
| Docker fails to start | `docker-compose down -v` then retry |

---

## 🤝 Contributing

1. Create a feature branch: `git checkout -b feature/my-feature`
2. Make changes
3. Run tests: `pytest tests/ -v`
4. Ensure coverage: `pytest --cov=backend`
5. Push and open PR

All PRs must pass:
- Linting (`black`, `isort`)
- Tests (>90% coverage)
- Documentation updates

---

## 📞 Next Steps

1. **Try the API**: Visit `http://localhost:8000/docs`
2. **Run a notebook**: Open `notebooks/01_eda.ipynb`
3. **Read the docs**: See [README_ENHANCED.md](./README_ENHANCED.md)
4. **Explore the code**: Start with `backend/api/main.py`
5. **Run tests**: Execute `pytest tests/ -v`

---

## 🎓 Key Takeaways

✅ This is a **production-ready** LTV prediction system  
✅ It uses **hybrid modeling** for robust predictions  
✅ **Well-tested** with 45+ test suites  
✅ **Fully documented** with API, notebooks, and guides  
✅ **Enterprise-ready** with monitoring & integrations  

Start predicting customer value today! 🚀

---

**Built with ❤️ | Python 3.11 | FastAPI • Polars • PyTorch • Scikit-learn**
