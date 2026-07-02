import argparse

from perf import perf
from language import (EncodingEnvironment, IntegerEncodingScheme, encode, interpret)
from syntax import lex, parse
from vsa import FHRR


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
        try:
            with open(args.interpret) as f:
                src = f.read()
        except:
            print(f"Failed to open {args.interpret}, interpreting it as code!")
            src = args.interpret

        dim = 1000
        vsa = FHRR
        print(interpret(src, vsa, dim, IntegerEncodingScheme.ListIntegers))


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
