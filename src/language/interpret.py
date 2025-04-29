"""Module containing functions for interpreting and manipulating vector
symbols.
"""

from typing import Any

from vsa import VSA, VSAdtype

from .encoding import EncodingEnvironment


def car(expr: VSA[VSAdtype], env: EncodingEnvironment) -> VSA[VSAdtype]:
    raise Exception("TODO")


def cdr(expr: VSA[VSAdtype], env: EncodingEnvironment) -> VSA[VSAdtype]:
    raise Exception("TODO")


def decode(
    expr: VSA[VSAdtype], env: EncodingEnvironment
) -> str | list[Any] | tuple[Any, ...]:
    return ""
