# Professional Git & GitHub Workflow Cheat Sheet

## 1. Daily Branching & Staging Workflow
```bash
# Create and switch to a feature branch
git checkout -b feature/model-training-pipeline

# Stage specific files
git add src/model.py configs/config.yaml

# Check status and staged differences
git status
git diff --staged

# Commit using Conventional Commit message convention
git commit -m "feat: implement LightGBM training pipeline with Optuna"
```

## 2. Conventional Commit Standards
- `feat:`: A new feature or model implementation.
- `fix:`: A bug fix (e.g. data leakage, shape mismatch).
- `docs:`: Documentation updates (README, docstrings).
- `refactor:`: Code refactoring without changing functionality.
- `test:`: Adding unit tests or dataset validation checks.
- `perf:`: Performance optimization (e.g. vectorization, CUDA acceleration).

## 3. Stashing & Context Switching
```bash
# Save uncommitted work temporarily without committing
git stash push -m "WIP: experimenting with learning rate"

# List stored stashes
git stash list

# Restore most recent stash and remove from stash list
git stash pop
```

## 4. History Rebase & Conflict Resolution
```bash
# Sync local branch with latest main branch cleanly via rebase
git checkout main
git pull origin main
git checkout feature/model-training-pipeline
git rebase main

# If conflicts occur:
# 1. Resolve conflict markers in code
# 2. Stage resolved files: git add <resolved-file>
# 3. Continue rebase: git rebase --continue
```

## 5. Clean History Squash (Interactive Rebase)
```bash
# Squash last 3 messy commits into one polished commit
git rebase -i HEAD~3
# Replace 'pick' with 'squash' or 's' for commits 2 and 3 in editor
```

## 6. Remote Repository Operations
```bash
# Push branch to GitHub and set upstream tracking
git push -u origin feature/model-training-pipeline

# Force push with lease (safe force push after rebase)
git push --force-with-lease
```