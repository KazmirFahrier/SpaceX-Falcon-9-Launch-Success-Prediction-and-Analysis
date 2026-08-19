# Model card

## Intended use

This study asks how well Falcon 9 first stage landing success can be predicted from historical
information available before launch. It is an educational retrospective analysis, not an
operational SpaceX system.

## Evaluation design

All results use five expanding windows. The first model trains on flights 1 through 40 and tests
on 41 through 50. The training history then expands by ten flights until the final test window,
flights 81 through 90. Preprocessing is fitted independently inside every training window.

## Results

| Model | BalancedAccuracy | BalancedAccuracyCILow | BalancedAccuracyCIHigh | ROCAUC | Brier | F1 |
| --- | --- | --- | --- | --- | --- | --- |
| Random forest | 0.689 | 0.537 | 0.852 | 0.636 | 0.157 | 0.889 |
| Logistic regression | 0.669 | 0.531 | 0.821 | 0.576 | 0.15 | 0.905 |
| Gradient boosting | 0.521 | 0.375 | 0.691 | 0.566 | 0.234 | 0.779 |
| Historical rate baseline | 0.5 | 0.5 | 0.5 | 0.64 | 0.207 | 0.876 |
| Time only logistic | 0.5 | 0.5 | 0.5 | 0.394 | 0.186 | 0.876 |

The strongest aggregate result is **Random forest** with balanced accuracy
**0.689** versus **0.500** for the historical
rate baseline. Its 95% bootstrap interval is **[0.537,
0.852]**, so the small sample does not support a precise ranking.

## Calibration

Raw logistic Brier score is **0.150**. Sigmoid calibration produced
**0.178** and therefore did not improve squared probability error in this
backtest. The interactive application retains the raw logistic probability and displays this
limitation.

## Important limitations

1. The dataset contains only 90 historical launches.
2. Landing success increases sharply over time, creating distribution shift.
3. The final ten launches contain no failures, so fold specific discrimination metrics are not
   identifiable in that window.
4. Flight number represents historical era and should not be interpreted causally.
5. Launch site, booster configuration, and historical period are confounded.
6. Bootstrap intervals quantify sampling variation but not uncertainty from dataset construction.

## Reproduction

Run `python scripts/generate_report.py` after installing the project. Generated CSV files and
interactive figures are written beneath `reports/`.
