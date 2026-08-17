# Architecture

## Data flow

1. Strict Pydantic schemas reject extra fields, duplicate IDs, invalid dates, unsafe strings, and
   out-of-range amounts at the HTTP boundary.
2. Candidate generation blocks on vendor account or bank token plus currency. Date, amount, and
   invoice-similarity prefilters bound the pair set.
3. Nine features describe identifier, amount, date, purchase-order, bank, and description evidence.
4. A balanced logistic regression returns a review probability and signed feature contributions.
5. A threshold selected only on validation creates the model signal. The policy layer produces
   `PASS`, `REVIEW`, or `HOLD`; `HOLD` still requires human verification.
6. Output replaces vendor identifiers with peppered SHA-256 references and omits payload logging.

## Why logistic regression

The target is an operational control, not maximum benchmark complexity. Logistic regression is
small, deterministic, fast on CPU, and gives exact signed contributions after standardization. The
published model materially exceeds the exact-key baseline on synthetic recall and F1 without hiding
the four false positives and one false negative.

## Leakage controls

The generator assigns each original and resubmission to one group. Groups are sorted by their first
date and placed wholly in train, validation, or test. Candidate pairs are built only after the split,
so a record or underlying group cannot appear in multiple partitions. The decision threshold sees
validation labels only; the test partition is evaluated once.

## Operational boundaries

FaturaIz assumes structured invoice metadata already exists. OCR, invoice extraction, payment
execution, fraud adjudication, and regulatory decisions are outside scope. The service is stateless;
production audit storage and case management belong in an access-controlled downstream system.

