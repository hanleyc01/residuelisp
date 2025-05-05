import argparse

from perf import perf


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="ResidueLisp",
        description="An interpreter for a VSA encoding of a subset of LISP",
        epilog="Thanks for using <3",
    )
    parser.add_argument("-i", "--interpret", type=str, help="Interpret a file")
    parser.add_argument(
        "-p",
        "--perf",
        action="store_true",
        help="Run performance tests comparing the two",
    )
    args = parser.parse_args()

    if args.perf:
        perf()

    if args.interpret is not None:
        pass


# from language import (EncodingEnvironment, IntegerEncodingScheme, encode,
#                       interpret)
# from syntax import lex, parse
# from vsa import FHRR


# def main() -> None:
#     src = "(car meow meow)"
#     dim = 100
#     vsa = FHRR
#     interpret(src, vsa, dim, IntegerEncodingScheme.ListIntegers)

if __name__ == "__main__":
    main()
