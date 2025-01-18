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