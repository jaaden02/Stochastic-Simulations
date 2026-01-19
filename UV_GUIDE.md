# Using uv in this project

## Install uv (once)
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
# ensure ~/.local/bin is on PATH (zsh):
export PATH="$HOME/.local/bin:$PATH"
```

## Sync environment from lockfile
Run from project root:
```bash
uv sync --all-groups
```
This creates/refreshes `.venv` exactly as specified in `uv.lock`.

## Run commands without activating venv
```bash
uv run python src/use.py
uv run pytest tests/ -v
```
`uv run` uses the project’s `.venv` automatically.

## (Optional) Activate the venv manually
```bash
source .venv/bin/activate
python src/use.py
pytest tests/ -v
```

## Add dependencies
```bash
uv add <package>        # adds to pyproject + lock + installs
uv add --dev <package>  # dev-only dependency (tests/tools)
```

## Remove dependencies
```bash
uv remove <package>
```

## Update lockfile
```bash
uv lock           # refresh lock from pyproject
uv sync           # apply to .venv
```

## Clean rebuild (if env is broken)
```bash
chmod -R +w .venv 2>/dev/null || true
rm -rf .venv
uv sync --all-groups
```

## Common tips
- Use `uv run ...` when you don’t want to activate the venv.
- `uv sync` always aligns `.venv` to `uv.lock`.
- If you install with `uv pip install ...`, it is temporary; it will be removed on next `uv sync` unless added via `uv add`.
- To expose `uv` globally, you can symlink it: `sudo ln -sf "$HOME/.local/bin/uv" /usr/local/bin/uv`.
