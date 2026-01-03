# Decision-Engine Evaluation (System-Level)

Inputs:
- test_csv: `data/processed/test.csv`
- xgb_model: `artifacts/models/xgb_model.pkl`
- ae_errors: `artifacts/ae_errors/ae_test_errors.npy`

Config:
- xgb_t_low=0.05, xgb_t_high=0.8
- ae_block=4.895553

## Triage rates
- APPROVE: 42529 (0.9955)
- REVIEW : 44 (0.0010)
- BLOCK  : 149 (0.0035)

## Fraud distribution by bucket
- Fraud total: 52
- Fraud in APPROVE: 8
- Fraud in REVIEW : 7
- Fraud in BLOCK  : 37

Capture:
- Flagged capture (REVIEW or BLOCK): 0.8462
- Auto-block capture (BLOCK only): 0.7115

## Binary evaluation (Flagged = REVIEW or BLOCK)
- Precision: 0.2280
- Recall   : 0.8462
- F1       : 0.3592
- TN/FP/FN/TP: 42521/149/8/44

## Binary evaluation (Auto-block only)
- Precision: 0.2483
- Recall   : 0.7115
- F1       : 0.3682
- TN/FP/FN/TP: 42558/112/15/37
