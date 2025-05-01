# VSA Lisp with Residue Numbers

This is an implementation of a LISP interpreted with VSAs, using 
Residue Hyperdimensional Computing (RHC) as the encoding for integers.

# Running and hacking

This project is built with [uv](https://docs.astral.sh/uv/). Follow
the instructions listed there for installation.

In order to run the tests, enter into your command line:
```bash
$ uv run ./src/main.py
```

In order to hack on this or contribute, make sure that you format the code
using `black` with the command `./.venv/bin/black ./src/*`. Similarly, make sure it passes
`mypy --strict`. To help with this, before submitting any changes,
run `./pre_commit.sh` which should format and type check the code.

# Documentation

Getting used to any larger project is complex and requires intimate knowledge
of the library beforehand. In order to avoid a chicken and egg paradox, here's
a short introduction to the project structure.

First, one can explore the documentation by running `./.venv/bin/pdoc ./src/*`.
This should be installed as a dev dependency.

Secondly, the project is organized into a couple of directories:
- `vsa`, which contains an abstract base class for VSAs, as well as an 
  implementation of different VSAs used as options for encoding.
- `syntax`, defines the lexer, parser, and intermediate representation of the 
  language.
- `language`, defines the encoding function as well as the actual interpretation
  and evaluation of the language.
- `perf`, runs some performance tests comparing the differing encoding schemes.
- Finally, `tests`, which are unit-tests that help make sure that the definitions
  run accurately.

One thing about this project that differs from most I've seen is the heavy use
of type-annotations. I'm sorry about if it gets a little unreadable. If you
want to familiarize yourself with how to read them, and their benefits,
see [PEP 484](https://peps.python.org/pep-0484/), and the 
Python [typing](https://docs.python.org/3/library/typing.html) module 
documentation.