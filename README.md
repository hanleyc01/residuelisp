# VSA Lisp with Residue Numbers

Instructions for hacking, install [uv](https://docs.astral.sh/uv/#highlights),
```sh
# for macos and linux
$ curl -LsSf https://astral.sh/uv/install.sh | sh

# for windows
$ powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
```

Build the environment and dependencies,
```sh
$ uv build
```

Select the environment initialized at `.venv/bin/python` as the 
interpreter for the Jupyter Notebooks.

I've tried to keep styling consistent, so please when submitting commits,
make sure to verify that the code passes the following tests,
```sh
# install mypy and nbqa if you don't have them already
$ pip install -U nbqa
$ pip install mypy
 
# run it on the notebook to make sure it passes tests
$ nbqa mypy lisp.ipynb
```

Make sure also code is formatted using [black](https://black.readthedocs.io/en/stable/index.html).
```sh
$ pip install "black[jupyter]"
$ black -l 79 lisp.ipynb
```