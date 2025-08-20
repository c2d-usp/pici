import argparse
from pathlib import Path
import sys
import unittest


def main():
    """
    Run every test in the tests/ directory.
    """
    parser = argparse.ArgumentParser(
        description="Run unittest suite from inside tests/"
    )
    parser.add_argument("-p", "--pattern", default="test*.py", help="Test file pattern")
    parser.add_argument(
        "-v", "--verbosity", type=int, default=1, help="Verbosity (1-3)"
    )
    parser.add_argument("--failfast", action="store_true", help="Stop on first failure")
    args = parser.parse_args()

    tests_dir = Path(__file__).resolve().parent
    project_root = tests_dir.parent

    sys.path.insert(0, str(project_root))

    suite = unittest.defaultTestLoader.discover(
        start_dir=str(tests_dir),
        pattern=args.pattern,
    )

    result = unittest.TextTestRunner(
        verbosity=args.verbosity, failfast=args.failfast
    ).run(suite)

    raise SystemExit(0 if result.wasSuccessful() else 1)


if __name__ == "__main__":
    main()
