# Contributing

## Daily workflow

1. `git pull` before you start.
2. Make a branch: `git checkout -b yourname/short-description`.
3. Commit small, with clear messages: `git add <files>; git commit -m "Part3: add dropout sweep"`.
4. `git push -u origin yourname/short-description`, then open a pull request on GitHub.
5. A teammate reviews and merges. Do not push straight to `main`.

## Notebooks

- Notebooks merge badly. Only one person edits a given notebook at a time; say so in the group chat.
- Clear or keep outputs consistently: outputs bloat the diff, so do not commit executed `*_run.ipynb` copies (they are git-ignored).
- Restart the kernel and run all cells before committing, so the notebook runs top to bottom.

## Do not commit

- `.venv/`, `__pycache__/`, API keys or passwords.
- Large new data files without telling the team first.
