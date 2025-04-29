from syntax.parser import Token, TokenKind, lex


def token_eq(s: str, kind: TokenKind) -> bool:
    return next(lex(s)) == Token(s, kind)


def test_lex_int() -> None:
    assert token_eq("123", TokenKind.Int)


def test_lex_ident() -> None:
    assert token_eq("abc", TokenKind.Ident)


def test_lex_lparen() -> None:
    assert token_eq("(", TokenKind.Lparen)


def test_lex_rparen() -> None:
    assert token_eq(")", TokenKind.Rparen)


def test_lex_dot() -> None:
    assert token_eq(".", TokenKind.Dot)


def test_lex_quote_mark() -> None:
    assert token_eq("'", TokenKind.QuoteMark)


def test_lex_true() -> None:
    assert token_eq("#t", TokenKind.Truth)


def test_lex_false() -> None:
    assert token_eq("#f", TokenKind.Falsity)


def test_plus() -> None:
    assert token_eq("+", TokenKind.Plus)


def test_minus() -> None:
    assert token_eq("-", TokenKind.Minus)


def test_star() -> None:
    assert token_eq("*", TokenKind.Star)


def test_slash() -> None:
    assert token_eq("/", TokenKind.Slash)


def test_error() -> None:
    assert next(lex("@@@@@@@")).kind == TokenKind.Error


def test_define() -> None:
    assert token_eq("define", TokenKind.Define)


def test_lambda() -> None:
    assert token_eq("lambda", TokenKind.Lambda)


def test_cons() -> None:
    assert token_eq("cons", TokenKind.Cons)


def test_nil() -> None:
    assert token_eq("nil", TokenKind.Nil)


def test_and() -> None:
    assert token_eq("and", TokenKind.And)


def test_or() -> None:
    assert token_eq("or", TokenKind.Or)


def test_not() -> None:
    assert token_eq("not", TokenKind.Not)
