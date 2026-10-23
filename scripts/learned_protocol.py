"""Does a learned protocol match a hand-coded one at the same bit budget? (R6)

Thin wrapper around ``genetic_coil_arena.experiments.learned_protocol_report`` (also
available as ``genetic-coil learned-protocol``); see that module for the flags.

    python scripts/learned_protocol.py --radii 3 --runs 4 --generations 1000 \
        --workers 4 --out docs/results/learned_protocol.md
    python scripts/learned_protocol.py --scale gpu      # the 20-core preset
"""

from genetic_coil_arena.experiments.learned_protocol_report import main

if __name__ == "__main__":
    main()
