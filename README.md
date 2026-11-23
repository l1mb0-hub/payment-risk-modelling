# Payment Risk Modelling

This study uses 40,000 simulated payments to test whether logistic regression can reduce fraud losses under a review budget. The simulator increases fraud prevalence on day 80.

The model trains on days 0–69. Days 70–89 select the lowest-cost threshold among candidates that alert on at most 10% of validation transactions. Days 90–119 evaluate the fixed threshold against an always-approve policy. The test alert rate can exceed 10% because the limit applies during validation.

The cost model charges €4 per alert and €150 per missed fraud, assuming alerts block fraud perfectly. The holdout also reports ROC AUC, average precision, Brier score and shifts in fraud prevalence and merchant risk. All transactions and costs are illustrative.

## Holdout results

On 9,929 simulated test payments, the selected threshold (0.13) flagged 9.90% of transactions. The assumed cost was €11.82 per payment versus €16.77 for always approving. ROC AUC was 0.766, average precision was 0.307 and the Brier score was 0.093.

## Run locally

```bash
python -m pip install -r requirements.txt
python -m pytest -q
python -m payment_risk_lab.main
```

The run writes the synthetic transactions, metrics and precision–recall plot to `payment_risk_lab/outputs/`.
