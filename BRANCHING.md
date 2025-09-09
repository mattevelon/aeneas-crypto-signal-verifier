# Git Branching Strategy

## GitFlow Model

This project follows the GitFlow branching model for organized development and releases.

## Branch Types

### Main Branches
- **main**: Production-ready code. Protected branch.
- **develop**: Integration branch for features. All feature branches merge here.

### Supporting Branches
- **feature/***: New features (branch from develop)
- **release/***: Release preparation (branch from develop)
- **hotfix/***: Emergency fixes (branch from main)
- **bugfix/***: Non-emergency bug fixes (branch from develop)

## Workflow

### Feature Development
```bash
# Create feature branch
git checkout develop
git pull origin develop
git checkout -b feature/telegram-handler

# Work on feature
git add .
git commit -m "feat: implement telegram message handler"

# Merge back to develop
git checkout develop
git merge --no-ff feature/telegram-handler
git push origin develop
git branch -d feature/telegram-handler
```

### Release Process
```bash
# Create release branch
git checkout -b release/1.0.0 develop

# Finish release
git checkout main
git merge --no-ff release/1.0.0
git tag -a v1.0.0 -m "Release version 1.0.0"
git checkout develop
git merge --no-ff release/1.0.0
```

### Hotfix Process
```bash
# Create hotfix from main
git checkout -b hotfix/critical-fix main

# Finish hotfix
git checkout main
git merge --no-ff hotfix/critical-fix
git tag -a v1.0.1 -m "Hotfix version 1.0.1"
git checkout develop
git merge --no-ff hotfix/critical-fix
```

## Branch Naming Conventions
- feature/short-description
- release/version-number
- hotfix/issue-description
- bugfix/bug-description

## Commit Message Format
See `.gitmessage` for commit message template.

## Pull Request Guidelines
1. All changes must go through PR review
2. PRs must pass all CI checks
3. Require at least 1 approval for merge
4. Delete branch after merge
