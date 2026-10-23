"""Can a new specialist join a squad that already speaks a learned language? (R7)

Thin wrapper around ``genetic_coil_arena.experiments.add_specialist_report``; see that
module for the design and the flags.

    python scripts/add_specialist.py --scale smoke --out outputs/add_specialist_smoke.md
    python scripts/add_specialist.py --scale cpu --out docs/results/add_specialist_pilot.md
    python scripts/add_specialist.py --scale gpu      # <= 16 workers on the 20-core PC
"""

from genetic_coil_arena.experiments.add_specialist_report import main

if __name__ == "__main__":
    main()
