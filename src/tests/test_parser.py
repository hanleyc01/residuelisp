import sys

from syntax import *


def test_embedded_list() -> None:
    src = "(lambda (foo) bar)"
    lexed_src = list(lex(src))
    print(
        "###################################################################",
        file=sys.stderr,
    )
    print(
        f"""
    lexed_src = {lexed_src}
    """,
        file=sys.stderr,
    )
    print(
        "###################################################################",
        file=sys.stderr,
    )

    assert len(lexed_src) == 7
