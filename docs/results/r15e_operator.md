# R15e: how often an operator confirms the scout brain's false alarms

200 ringed spots the scout brain flagged at ≥ 0.5 on VisDrone val (native scale), answered in a median 0.75 s each; 4 "can't tell".

| Truth | Spots on a person | Operator says person | Spots without | Operator says person (false accept) |
|---|---|---|---|---|
| clear cases (visual check of the false alarms) | 127 | 117 = 92% [86, 96] | 43 | 10 = 23% [13, 38] |
| matched by R15b's scoring rule | 100 | 96 = 96% [90, 98] | 100 | 47 = 47% [38, 57] |
| a labelled person within the ring | 110 | 102 = 93% [86, 96] | 90 | 41 = 46% [36, 56] |

Brackets: 95% Wilson intervals. One operator, crops from the scout's own view (a confirmer would look from closer), answered quickly as on a live feed. VisDrone leaves many riders and seated or shadowed people unlabelled, so the two label-based rows count real people as false alarms; the clear-cases row keeps only spots that are plainly a person or plainly not one (`docs/results/r15e_adjudication.json`; the visual check was not blind to the answers).
