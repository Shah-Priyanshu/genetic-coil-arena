# R7e probe: independently trained squads through a shared meaning space

Generated 2026-10-29 by job `r7e-interlingua`.

Exploratory probe (not a pre-registered goal). Squad A's centre drives squad B's specialists on the 6 held-out goals; the bridge reads A's symbols into eat/do-not-eat with A's G12 decoder and writes them in B's dialect with B's G12 writer. 34 pairs (each G12-sample run with the next), 50 episodes per goal. Mean ± 95% t-CI over pairs.

<!-- analysis:start -->
**What the probe shows: separately trained squads can work together through a shared meaning space, up to what that space can express.**

- **Without a bridge they cannot talk at all.** A's centre driving B's specialists scores −0.07, 0% of B's own centre: different dialects.
- **With the bridge they reach 71% ± 8 of native.** The bridge reads A's symbols into eat / do-not-eat with A's decoder and writes them in B's dialect with B's writer.
- **The bridge's reader is essentially perfect.** Writing the *true* status gives 73%. The bridged score tracks it pair by pair (correlation 0.99, mean gap −0.47), and a hand-coded centre written in B's dialect gives the same 73%.
- **So the ceiling is the meaning space, not the translation.** It names only 8 coarse food situations, so it loses the exact positions the native centre uses.
- **The spread follows each target squad's writer quality** (G12's write efficacy 0.15–1.14). 10 of 34 pairs reach ≥ 90% of native; the worst ~1%.

**For the ecosystem:** parts trained separately can be joined after the fact, without retraining either, by fitting small decoders onto a shared, human-defined vocabulary. The open lever is a richer vocabulary (finer situations, or a learned one), which would raise both the bridge and the G12 writer.
<!-- analysis:end -->

| Condition | Mission score | Share of B's native score |
|---|---|---|
| B's own centre (native) | 21.63 ± 0.56 | 100% ± 0 |
| A's centre, no bridge | -0.07 ± 2.03 | -0% ± 9 |
| A's centre through the bridge | 15.35 ± 1.79 | 71% ± 8 |
| hand-coded centre written in B's dialect | 15.81 ± 1.79 | 73% ± 8 |
| true status written in B's dialect | 15.81 ± 1.79 | 73% ± 8 |

Bridge gain over no bridge: +15.42 ± 2.67.

Reproduce: `python scripts/r7e_interlingua.py --from-json runs:r7b3-train-* runs:r7b4-train-* runs:r7b5-train-* runs:r7c-train-*`.
