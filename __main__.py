"""Entry point to program."""

import sys

from driver import Driver

__author__ = "Grant Holmes"
__email__ = "g.holmes429@gmail.com"


def main(arguments):
    """Main func."""
    driver = Driver()
    driver.run(arguments)


if __name__ == "__main__":
    main(sys.argv[1:])
