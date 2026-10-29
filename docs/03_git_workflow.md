# Git Workflow for the Group (rubric: "frequent, meaningful commits showing steady progress")

The rubric rewards **version history showing collaborative work**. Don't upload everything in one commit.
Suggested split for a 5-member group (adjust names/roles):

```bash
git init && git branch -M main
git remote add origin https://github.com/<your-org>/<repo>.git
```

| # | Member | Commit (example message) | Files |
|---|---|---|---|
| 1 | A | `Initial commit: README, requirements, .gitignore` | README.md requirements.txt .gitignore |
| 2 | B | `Add dataset and data documentation` | data/ |
| 3 | B | `Add config and data loading/target construction` | src/__init__.py src/config.py src/data.py |
| 4 | C | `Add EDA and evaluation plotting functions` | src/plots.py |
| 5 | D | `Add lesson-style metrics and unit tests` | src/metrics.py tests/ |
| 5b | D | `Add model tuning, thresholding and baselines` | src/model.py |
| 6 | D | `Add main pipeline script` | main.py |
| 7 | C | `Generate figures and result files` | figures/ results/ models/ |
| 8 | E | `Add proposal and video script` | docs/ |
| 9 | A | `Add LaTeX mini-research paper (course template)` | paper/ |
| 10+ | all | `Fix typo in ...`, `Improve README ...`, `Rephrase section 5 ...` | — |

Each member commits from their own account (`git config user.name/user.email`) – ideally via branches + pull requests:

```bash
git checkout -b feature/eda-plots
git add src/plots.py && git commit -m "Add EDA plotting functions"
git push -u origin feature/eda-plots
```
