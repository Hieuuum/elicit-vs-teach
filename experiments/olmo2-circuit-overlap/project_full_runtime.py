"""CPU-only projection after seven-stage local timing backups are complete."""

import argparse
import json
from pathlib import Path
import sys


def main() -> None:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from geode.circuits.stage_projection import build_projection

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", default="geode-store/olmo2-sanity-20260910")
    parser.add_argument("--workload", default="/tmp/olmo-full-workload.json")
    parser.add_argument("--hourly-rate", type=float)
    parser.add_argument("--allow-partial", action="store_true")
    args = parser.parse_args()
    result = build_projection(
        args.run, args.workload, hourly_rate=args.hourly_rate, allow_partial=args.allow_partial
    )
    print(
        json.dumps(
            {
                key: result[key]
                for key in (
                    "status",
                    "point_hours",
                    "point_compute_usd",
                    "row_linear_probe_base_hours",
                    "generation",
                )
            }
        )
    )


if __name__ == "__main__":
    main()
