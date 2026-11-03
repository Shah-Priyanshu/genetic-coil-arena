# R15f: does a careful look cut the false alarms an operator confirms?

Waiting for the operator session (200 new spots, close-up and wider view, buttons locked for the first 2 s). Scored by `scripts/r15f_careful_score.py` once the answers are in.

Read-out, fixed before any answer was read: ENOUGH if the clear-case false accept is at most 5% and the clear-case hit at least 85%; NOT ENOUGH if the false accept is between 5% and 15% or the hit falls below 85%; NO GAIN if the false accept is 15% or more. Truth for the clear cases: `docs/results/r15f_adjudication.json` (26 person, 38 not a person, 36 ambiguous among the 100 R15b false alarms), committed before the session.
