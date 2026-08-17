# Model card

## Model details

- Candidate: standardization plus class-balanced logistic regression
- Baseline: exact normalized invoice number, same vendor, and effectively identical amount
- Features: nine bounded pairwise similarities/equalities
- Split: chronological 70/15/15 by complete invoice group
- Threshold: 0.905, selected on validation with a minimum 0.80 precision constraint
- Artifact integrity: SHA-256 in `models/manifest.json`

## Test results

On 950 synthetic test candidates (90 positive, 860 negative), the candidate achieved average
precision 0.9830, ROC-AUC 0.9984, F1 0.9727, precision 0.9570, and recall 0.9889. The exact baseline
achieved F1 0.6466 and recall 0.4778. These are generator-specific measurements, not real-world
performance estimates.

## Error analysis

The candidate produced four false positives and one false negative. The missed case was a weak
resubmission; its slice recall was 22/23 (95.65%). Exact, formatting-shift, and amount-shift slices
had 100% recall in this small synthetic holdout. Similar legitimate recurring invoices remain the
main false-positive risk.

## Ethical and operational risks

False positives can delay valid supplier payments; false negatives can allow duplicate payment.
Risk scores can also create unjustified suspicion of suppliers or employees. The service therefore
uses neutral “candidate” language, never labels fraud, exposes evidence, and requires human review.
Monitor errors across supplier size, geography, currency, document channel, and correction type when
legally collected real data becomes available.

## Limitations and next steps

The model has no OCR, line-item semantics, graph signals, calibrated real-world probability, or
independently adjudicated labels. Next steps are shadow-mode evaluation on legally sourced data,
vendor/time holdouts, probability calibration, reviewer agreement measurement, drift monitoring,
and documented override/case-closing feedback loops.

