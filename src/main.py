"""
This module provides functions for...
"""

import logging
import warnings

from cli import parse_arguments

warnings.filterwarnings("ignore")

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def main():
    args = parse_arguments()
    logging.info("Starting with args %s...", args)
    # Do work here
    logging.info("Finished.")


if __name__ == "__main__":
    main()
