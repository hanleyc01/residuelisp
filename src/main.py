from language import EncodingEnvironment, IntegerEncodingScheme, encode
from syntax import lex, parse
from vsa import FHRR


def main() -> None:
    tokens = lex("(cons nil nil)")
    intr = parse(tokens)
    encoding_env = EncodingEnvironment(FHRR, 100, IntegerEncodingScheme.ListIntegers)
    vector_symbol = encode(intr, encoding_env)


if __name__ == "__main__":
    main()
