# Threat model

## Assets

Invoice metadata, supplier relationships, payment amounts, model artifact, decision threshold,
review outcomes, and any downstream case notes.

## Threats and controls

| Threat | Control in this repository | Production requirement |
|---|---|---|
| Sensitive identifier exposure | API output pseudonymizes vendor IDs; payload is not logged | secret pepper, encryption, RBAC, retention policy |
| Oversized/hostile request | strict schema and 500-record limit | gateway rate limits and body-size cap |
| Model artifact replacement | SHA-256 verified before `joblib` load | signed artifact registry and deployment provenance |
| Unsafe deserialization | only packaged, digest-pinned local artifact is loaded | never accept uploaded artifacts; isolate build pipeline |
| Decision automation harm | human-review states and disclaimer | separation of duties and approval workflow |
| Container privilege | non-root user; read-only/cap-drop CI smoke | network policy, image scanning, immutable deployment |
| Membership/label leakage | group split before pair generation | independent temporal and supplier holdouts |

SHA-256 detects accidental or unauthorized changes only when the manifest itself is trusted. It is
not a substitute for signatures, provenance, or access control.
