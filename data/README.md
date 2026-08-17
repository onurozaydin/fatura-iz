# Data contract

FaturaIz expects one row per received invoice with these fields:

| Field | Type | Rule |
|---|---|---|
| `invoice_id` | string | unique, 1–80 characters |
| `vendor_account_id` | string | stable ERP supplier key |
| `invoice_number` | string | 1–80 characters |
| `invoice_date` | ISO date | valid calendar date |
| `amount` | decimal | greater than zero, at most 1 billion |
| `currency` | string | three uppercase letters |
| `po_number` | string | optional, at most 80 characters |
| `bank_fingerprint` | string | stable token; never raw bank details |
| `description` | string | optional, at most 500 characters |

The committed CSV is a 120-row synthetic sample. `group_id`, used only to create and evaluate
synthetic labels, is deliberately removed. The API does not accept labels.

Do not send raw bank account numbers, tax identifiers, contact details, or secrets. Tokenize
commercially sensitive identifiers upstream and apply retention and access controls.

