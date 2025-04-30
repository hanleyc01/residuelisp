from language import (EncodingEnvironment, IntegerEncodingScheme, encode,
                      interpret)
from syntax import lex, parse
from vsa import FHRR


def main() -> None:
    src = "bar bazz"
    dim = 100
    vsa = FHRR
    interpret(src, vsa, dim, IntegerEncodingScheme.ListIntegers)


if __name__ == "__main__":
    main()
