# Patch for `notebooks/04_transformer_training.ipynb`

Paste these snippets into the matching notebook code cells.

## Cell 2
```python
# -- Config ---------------------------------------------------
MAX_SEQ_LEN         = 50
OBSERVATION_MONTHS  = 6
HOLDOUT_MONTHS      = 6  # kept for compatibility in split diagnostics
RUN_OPTUNA          = False   # Set True to tune (slow -- ~20 trials)
N_OPTUNA_TRIALS     = 20
TRAIN_EPOCHS        = 80
PATIENCE            = 15
SAVE_TO_DB          = True
USE_WANDB           = True
ONNX_PATH           = settings.MODELS_DIR / 'transformer.onnx'
MODELS_DIR          = settings.MODELS_DIR

print(f'ONNX path: {ONNX_PATH}')
```

## Cell 4
```python
# Load and clean
raw_df  = load_uci_csv(settings.UCI_CSV_PATH)
cleaned = clean_transactions(raw_df)
cleaned = assign_product_categories(cleaned)
cleaned = assign_amount_buckets(cleaned)

# Split (used for observation-end anchoring + diagnostics)
calibration, holdout, obs_end, holdout_end = make_calibration_holdout_split(
    cleaned, OBSERVATION_MONTHS, HOLDOUT_MONTHS
)

# RFM + labels
# Important: compute labels from full cleaned data with horizon filters in RFMPipeline,
# not only from the pre-cut holdout slice.
rfm_pipe = RFMPipeline(calibration, observation_end_date=obs_end)
rfm_df   = rfm_pipe.compute()
rfm_df   = rfm_pipe.compute_ltv_labels(cleaned, rfm_df, horizon_months=12)
rfm_df   = rfm_pipe.compute_ltv_labels(cleaned, rfm_df, horizon_months=24)
rfm_df   = rfm_pipe.compute_ltv_labels(cleaned, rfm_df, horizon_months=36)

print(f'Calibration: {len(calibration):,} rows | Holdout: {len(holdout):,} rows')
print(f'RFM customers: {len(rfm_df):,}')
print(f'Non-zero 12m LTV: {(rfm_df["actual_ltv_12m"] > 0).sum():,}')
```

## Cell 5
```python
# Build purchase sequences from calibration window only
# (features available at prediction time)
builder_cal  = SequenceBuilder(calibration, max_length=MAX_SEQ_LEN, observation_end_date=str(obs_end))
sequences_df = builder_cal.build()

print(f'Calibration sequences: {len(sequences_df):,}')
print(f'Avg sequence length:   {sequences_df["sequence_length"].mean():.1f}')
```

## Cell 6
```python
# Build datasets -- customer-level train/val/holdout split from calibration sequences
from sklearn.model_selection import train_test_split

all_ids = sequences_df['customer_id'].to_list()
label_map = {
    row['customer_id']: float(row['actual_ltv_12m'])
    for row in rfm_df.select(['customer_id', 'actual_ltv_12m']).iter_rows(named=True)
}
all_y = [1 if label_map.get(cid, 0.0) > 0 else 0 for cid in all_ids]

train_ids, temp_ids, y_train, y_temp = train_test_split(
    all_ids, all_y,
    test_size=0.30,
    random_state=42,
    stratify=all_y,
)
val_ids, hold_ids, _, _ = train_test_split(
    temp_ids, y_temp,
    test_size=0.50,
    random_state=42,
    stratify=y_temp,
)

train_seq = sequences_df.filter(pl.col('customer_id').is_in(train_ids))
val_seq   = sequences_df.filter(pl.col('customer_id').is_in(val_ids))
hold_seq  = sequences_df.filter(pl.col('customer_id').is_in(hold_ids))

train_dataset = PurchaseSequenceDataset(
    train_seq, rfm_df, max_length=MAX_SEQ_LEN,
    ltv_12m_col='actual_ltv_12m', ltv_24m_col='actual_ltv_24m', ltv_36m_col='actual_ltv_36m'
)
val_dataset = PurchaseSequenceDataset(
    val_seq, rfm_df, max_length=MAX_SEQ_LEN,
    ltv_12m_col='actual_ltv_12m', ltv_24m_col='actual_ltv_24m', ltv_36m_col='actual_ltv_36m'
)
hold_dataset = PurchaseSequenceDataset(
    hold_seq, rfm_df, max_length=MAX_SEQ_LEN,
    ltv_12m_col='actual_ltv_12m', ltv_24m_col='actual_ltv_24m', ltv_36m_col='actual_ltv_36m'
)

def _pos_rate(ds):
    return float((ds.ltv_12m > 0).float().mean().item())

print(f'Train: {len(train_dataset):,}  Val: {len(val_dataset):,}  Hold: {len(hold_dataset):,}')
print(f'Positive 12m rate -- train: {_pos_rate(train_dataset):.2%}, val: {_pos_rate(val_dataset):.2%}, hold: {_pos_rate(hold_dataset):.2%}')
```

## Cell 8
```python
if RUN_OPTUNA:
    rprint('[bold blue]Running Optuna tuning...[/bold blue]')
    best_params, study = run_optuna_study(
        train_dataset, val_dataset,
        n_trials=N_OPTUNA_TRIALS, device=DEVICE, quick_epochs=15
    )
    rprint(f'Best params: {best_params}')
else:
    # Stronger baseline for sparse/zero-inflated LTV targets
    best_params = {
        'n_layers':      6,
        'n_heads':       8,
        'ffn_dim':       384,
        'dropout':       0.2,
        'learning_rate': 3e-4,
        'weight_decay':  5e-4,
        'batch_size':    64,
    }
    rprint(f'Using tuned default config: {best_params}')
```

## Cell 10
```python
config = {
    'model_dim':             96,
    'max_seq_len':           MAX_SEQ_LEN,
    'grad_clip':             1.0,
    'huber_delta':           1.0,
    'loss_weights':          [1.0, 0.0, 0.0],  # optimise directly for 12m targets
    'positive_ltv_weight':   4.0,              # up-weight non-zero LTV rows
    'train_pos_oversample':  4.0,              # weighted sampler multiplier
    **best_params,
}

model = build_model(config)
rprint(f'Model parameters: {count_parameters(model):,}')

from torch.utils.data import WeightedRandomSampler

is_pos = (train_dataset.ltv_12m > 0).numpy().astype(np.float32)
sample_weights = np.where(is_pos > 0, config['train_pos_oversample'], 1.0).astype(np.float32)
sampler = WeightedRandomSampler(
    weights=torch.DoubleTensor(sample_weights),
    num_samples=len(sample_weights),
    replacement=True,
)

train_loader = DataLoader(
    train_dataset,
    batch_size=config['batch_size'],
    sampler=sampler,
    collate_fn=collate_fn,
    num_workers=0,
    pin_memory=torch.cuda.is_available(),
    drop_last=True,
)
val_loader = DataLoader(
    val_dataset,
    batch_size=config['batch_size'],
    shuffle=False,
    collate_fn=collate_fn,
    num_workers=0,
    pin_memory=torch.cuda.is_available(),
)
```
