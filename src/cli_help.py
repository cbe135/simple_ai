"""Shared --help/--default support for the simple_ai_* console scripts.

``--default`` prints the CLI parameter defaults (argparse default= values) and
exits. It intentionally does NOT print config.yaml defaults.
"""

import argparse
import sys


def add_default_flag(parser):
    parser.add_argument(
        "--default",
        action="store_true",
        help="Print the default values of all CLI parameters and exit "
             "(does NOT include config.yaml defaults).",
    )


def parse_with_default(parser, argv=None):
    """parse_args(), but relax required args and exit early when --default is set."""
    argv = list(sys.argv[1:] if argv is None else argv)
    if "--default" in argv:
        for action in parser._actions:
            action.required = False
    args = parser.parse_args(argv)
    if getattr(args, "default", False):
        print(f"{parser.prog} — default parameter values\n")
        print("CLI parameter defaults:")
        seen = set()
        for action in parser._actions:
            if not action.option_strings or action.dest in seen:
                continue
            seen.add(action.dest)
            if action.default is argparse.SUPPRESS:
                continue
            flag = "/".join(action.option_strings)
            print(f"  {flag} = {action.default!r}")
        sys.exit(0)
    return args
