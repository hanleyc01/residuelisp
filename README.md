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
using `black` with the command `black -l79`. Similarly, make sure it passes
`mypy --strict`. To help with this, before submitting any changes,
run `./build.sh` which should format and type check the code.