# Data card

## Dataset

- Name: FaturaIz deterministic synthetic invoice scenarios
- Generator: `src/faturaiz/synthetic.py`
- Seed: `20260817`
- Published run: 4,204 rows, 3,600 base groups, 604 labelled resubmissions
- Period: synthetic dates from 2024-01-01 through 2025-09-17
- License: MIT
- Personal data: none
- Real entities or transactions: none

## Generation

Synthetic suppliers receive random or recurring amounts, purchase-order tokens, bank fingerprints,
and service descriptions. Resubmissions are injected as exact copies, formatting shifts, amount
shifts, or deliberately weak reference changes. Similar legitimate recurring invoices are generated
as hard negatives. Labels come directly from the generation process, not from human adjudication.

## Quality evidence

The generated frame has 4,204 unique invoice IDs, zero null cells, zero non-positive amounts, and
604 groups containing two records. Candidate recall is 100% in each generated partition. See
`reports/data_quality.json` for machine-readable results.

## Intended and prohibited use

The data supports software tests and a portfolio demonstration. It must not be represented as a
real company ledger, used to estimate commercial savings, or used to validate production accuracy.
Real deployment data needs documented legal authority, retention limits, supplier confidentiality
controls, representative labels, and review for credit notes and legitimate re-billing.
