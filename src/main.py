from vsa import RHC


def main() -> None:
    dim = 100
    left = RHC.encode(dim, 1)
    right = RHC.encode(dim, 3)

    print(left.sim(right))


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
