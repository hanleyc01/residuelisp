./.venv/bin/black ./src/*
./.venv/bin/isort ./src/*
./.venv/bin/mypy --strict ./src 
./.venv/bin/pytest ./src/tests