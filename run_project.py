"""Run the whole Copilot, or one Part, in dependency order.

    python run_project.py              # Parts A, B, C and D
    python run_project.py --part c     # one Part (its inputs must already exist)

Run ``python generate_data.py`` once first.
"""
from __future__ import annotations

import argparse
import logging

from src import analytics, customer_models, data_pipeline, policy_rag, refund_agent, triage
from src.config import OUTPUTS, RAW, setup_logging
from src.validate_checkpoints import validate

logger = logging.getLogger("copilot.run")


def part_a() -> None:
    data_pipeline.run()
    validate()
    print("ALL CHECKPOINTS PASSED")
    analytics.run()


def part_d() -> None:
    policy_rag.run()
    refund_agent.run()


PARTS = {"a": part_a, "b": customer_models.run, "c": triage.run_all, "d": part_d}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--part", choices=sorted(PARTS), help="run only this Part (default: all)")
    args = parser.parse_args()
    setup_logging()
    if not (RAW / "orders.csv").exists():
        parser.error("data/raw is empty: run 'python generate_data.py' first")
    OUTPUTS.mkdir(exist_ok=True)
    for name in ([args.part] if args.part else sorted(PARTS)):
        logger.info("=== Part %s ===", name.upper())
        PARTS[name]()
    logger.info("Finished. Computed figures are in outputs/metrics/.")


if __name__ == "__main__":
    main()
