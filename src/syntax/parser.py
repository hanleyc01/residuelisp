"""Parser for a subset of [R5RS](https://conservatory.scheme.org/schemers/Documents/Standards/R5RS/r5rs.pdf)
specification of the Scheme programming language.
"""

from __future__ import annotations

import itertools
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum, auto
from typing import Iterator

import more_itertools


class TokenKind(Enum):
    """Tags for tokens."""

    And = auto()
    Comment = auto()
    Define = auto()
    Dot = auto()
    Else = auto()
    Falsity = auto()
    Ident = auto()
    If = auto()
    Int = auto()
    Lambda = auto()
    Let = auto()
    Lparen = auto()
    Minus = auto()
    Or = auto()
    Plus = auto()
    QuoteMark = auto()
    Rparen = auto()
    Slash = auto()
    Star = auto()
    Truth = auto()
    Unquote = auto()
    Error = auto()
    Cons = auto()
    Nil = auto()
    Not = auto()

    def __str__(self) -> str:
        return self.name


KEYWORDS = {
    "define": TokenKind.Define,
    "lambda": TokenKind.Lambda,
    "and": TokenKind.And,
    "or": TokenKind.Or,
    "cons": TokenKind.Cons,
    "nil": TokenKind.Nil,
    "not": TokenKind.Not,
    "if": TokenKind.If,
    "let": TokenKind.Let,
}
"""Reserved keywords."""


OPERATORS = {
    "+": TokenKind.Plus,
    "-": TokenKind.Minus,
    "/": TokenKind.Slash,
    "*": TokenKind.Star,
}
"""Mathematical operators."""


VALUES = {"#f": TokenKind.Truth, "#t": TokenKind.Falsity}
"""Constant values, excluding "nil"."""


@dataclass
class Token:
    """Tagged union of tokens.

    Arguments:
        cont (str): A string reference to the contents in the raw source.
        kind (TokenKind): What kind of token it is.
    """

    cont: str
    kind: TokenKind


# helper function, returns None instead of throwing
def _next[T](iter: "more_itertools.peekable[T]") -> T | None:
    return next(iter, None)


# helper function, returns None instead of throwing
def _peek[T](iter: "more_itertools.peekable[T]") -> T | None:
    return iter.peek(None)


# consume whitespace until we reach a new character
def _eat_whitespace(conts: "more_itertools.peekable[str]") -> None:
    curr = _peek(conts)
    while curr is not None and curr.isspace():
        _next(conts)
        curr = _peek(conts)


# lex a digit
def _digit(d: str, conts: "more_itertools.peekable[str]") -> Token | None:
    ds = [d]
    curr = _next(conts)
    while curr is not None and curr.isdigit():
        ds.append(curr)
        curr = _next(conts)
    return Token("".join(ds), TokenKind.Int)


# lex an identifier or keyword
def _key_or_ident(c: str, conts: "more_itertools.peekable[str]") -> Token | None:
    word = [c]
    curr = _next(conts)
    while curr is not None and curr.isalpha():
        word.append(curr)
        curr = _next(conts)
    symbol = "".join(word)
    kind = KEYWORDS.get(symbol)
    if kind is None:
        kind = TokenKind.Ident
    return Token(symbol, kind)


# parse a token
def _token(conts: "more_itertools.peekable[str]") -> Token | None:
    _eat_whitespace(conts)
    curr = _next(conts)
    if curr is None:
        return None

    match curr:
        case "(":
            return Token(curr, TokenKind.Lparen)
        case ")":
            return Token(curr, TokenKind.Rparen)
        case "+":
            return Token(curr, TokenKind.Plus)
        case "-":
            return Token(curr, TokenKind.Minus)
        case "*":
            return Token(curr, TokenKind.Star)
        case "/":
            return Token(curr, TokenKind.Slash)
        case ".":
            return Token(curr, TokenKind.Dot)
        case "'":
            return Token(curr, TokenKind.QuoteMark)
        case "#":
            curr = _peek(conts)
            match curr:
                case "f":
                    _next(conts)
                    return Token("#f", TokenKind.Falsity)
                case "t":
                    _next(conts)
                    return Token("#t", TokenKind.Truth)
                case _:
                    _next(conts)
                    return Token(f"Required 'f' or 't' after '#'", TokenKind.Error)
        case d if d.isdigit():
            return _digit(d, conts)
        case c if c.isalpha():
            return _key_or_ident(c, conts)
        case _:
            return Token(f"Unrecognized token: {curr}", TokenKind.Error)


def lex(conts: str) -> Iterator[Token]:
    """Lex the contents of a raw source file to a string.

    Arguments:
        conts (str): The raw string source of scheme text.

    Returns:
        An iterator of tokens.
    """
    conts_iter = more_itertools.peekable(conts.lower())
    tokens = []
    current = _token(conts_iter)
    while current is not None:
        tokens.append(current)
        current = _token(conts_iter)
    return iter(tokens)


@dataclass
class Program:
    """⟨program⟩ −→ ⟨command or definition⟩*"""

    commands: list[Command]


type Command = Expression | Definition
"""
⟨command or definition⟩ −→ ⟨command⟩
| ⟨definition⟩
"""


class Expression(ABC):
    """
    ⟨expression⟩ −→ ⟨variable⟩
    | ⟨literal⟩
    | ⟨procedure call⟩
    | ⟨lambda expression⟩
    | ⟨conditional⟩
    | ⟨let⟩
    """

    pass


@dataclass
class ExprVar(Expression):
    """⟨variable⟩ −→ identifier"""

    var: str


@dataclass
class ExprBool(Expression):
    """⟨literal⟩ −→ #t | #f"""

    val: bool


@dataclass
class ExprInt(Expression):
    """⟨literal⟩ −→ int"""

    val: int


@dataclass
class ExprProc(Expression):
    """
    ⟨procedure call⟩ −→ (⟨operator⟩ ⟨operand⟩*)
    """

    rator: Expression
    rands: list[Expression]


@dataclass
class ExprLambda(Expression):
    """
    ⟨lambda expression⟩ −→ (lambda ⟨formals⟩ ⟨body⟩)
    ⟨formals⟩ −→ (⟨variable⟩*)
    """

    args: list[str]
    body: Expression


@dataclass
class ExprIf(Expression):
    """
    ⟨conditional⟩ −→ (if ⟨test⟩ ⟨consequent⟩ ⟨alternate⟩)
    ⟨test⟩ −→ ⟨expression⟩
    ⟨consequent⟩ −→ ⟨expression⟩
    ⟨alternate⟩ −→ ⟨expression⟩
    """

    test: Expression
    consequent: Expression
    alternate: Expression


@dataclass
class ExprLet(Expression):
    """
    ⟨let⟩ −→ (let (⟨variable⟩ ⟨value⟩) ⟨body⟩
    """

    x: str
    val: Expression
    body: Expression


@dataclass
class ExprNil(Expression):
    pass


@dataclass
class ExprCons(Expression):
    pass


class Definition(ABC):
    """
    ⟨definition⟩ −→ (define ⟨variable⟩ ⟨expression⟩)
    | (define (⟨variable⟩ ⟨def formals⟩) ⟨body⟩)
    """

    pass


class DefFunc(Definition):
    """(define ⟨variable⟩ ⟨expression⟩)"""

    name: str
    params: list[str]
    body: Expression


class DefExpr(Definition):
    """(define (⟨variable⟩ ⟨def formals⟩) ⟨body⟩)"""

    name: str
    body: Expression


@dataclass
class ParserError(Exception):
    """An exception indicating that a parsing error has occurred."""

    msg: str
    tok: Token


@dataclass
class IntrAtom:
    x: Token


@dataclass
class IntrList:
    xs: list[Intr]


type Intr = IntrAtom | IntrList


# convert list of tokens to intermediate representation
def _split_on(tokens: "more_itertools.peekable[Token]") -> Intr:
    curr = _next(tokens)
    if curr is None:
        return IntrList([])

    if curr.kind == TokenKind.Lparen:
        l = []
        while (nxt := _peek(tokens)) is not None and nxt.kind != TokenKind.Rparen:
            l.append(_split_on(tokens))
        foo = _next(tokens)
        return IntrList(l)

    elif curr.kind == TokenKind.Rparen:
        raise ParserError("Unexpected rparen", curr)

    else:
        return IntrAtom(curr)


def _atom(token: Token) -> Expression:
    match token.kind:
        case TokenKind.Int:
            return ExprInt(int(token.cont))

        case TokenKind.Ident:
            return ExprVar(token.cont)

        case TokenKind.Truth:
            return ExprBool(True)

        case TokenKind.Falsity:
            return ExprBool(False)

        case TokenKind.Nil:
            return ExprNil()

        case TokenKind.Cons:
            return ExprCons()

        case _:
            raise ParserError("Unexpected token", token)


def parse(toks: Iterator[Token]) -> Intr:
    """Parse an iterator of tokens.

    Arguments:
        toks (Iterator[Token]): An iterator of tokens, typically returned from
            `lex`.

    Returns:
        Returns intermediate representation.

    Raises:
        When parsing fails, raises `ParserError(str, Token)`.
    """
    splitted = _split_on(more_itertools.peekable(toks))
    return splitted
