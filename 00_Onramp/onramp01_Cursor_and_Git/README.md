# AIE7 ROADMAP – ONRAMP SESSION #1

<details>
  <summary>📑 **Table of Contents**</summary>

- [Cursor](#cursor)
- [SSH Setup](#ssh-setup)
- [Git Workflow](#git-workflow)
  - [Create a Remote Architecture](#create-a-remote-architecture)
  - [Clone Your Local Repository](#clone-your-local-repository)
  - [Add the Upstream Remote (one-time)](#add-the-upstream-remote-one-time)
  - [Pull the Upstream Repo](#pull-the-upstream-repo)
  - [Use Sync Flow — Every Class](#use-sync-flow--every-class)
    - [Pull fresh starter code](#pull-fresh-starter-code)
    - [Develop locally](#develop-locally)
    - [Stage & commit changes](#stage--commit-changes)
    - [Push to your fork](#push-to-your-fork)

</details>

## Cursor
- Open & navigate multi-file projects with ease.  
- Ask the **AI Chat** panel for explanations, refactors, or bug fixes.  
- Scaffold quick apps, then iterate and debug until they run cleanly.  
- Chat about individual functions or whole scripts to deepen understanding.  

## SSH Setup
1. Generate an **SSH key pair** and save it to `~/.ssh/`.
2. Add the **public key** to GitHub.
3. Always clone/push with **SSH URLs** (`git@github.com:...`) for seamless, secure auth.

## Git Workflow
* Keep your fork current without overwriting the upstream repository.
* Guarantee you always start with the latest assignment skeleton.

### Create a Remote Architecture
**origin** = your personal fork (student repo)
**upstream** = class master repo (AI-Maker-Space)

#### Clone Your Local Repository
```bash
git clone <your_local_repo>
cd <your_local_repo>
```

#### Add the Upstream Remote (one-time)
```bash
git remote add upstream git@github.com:AI-Maker-Space/AIE7.git
```

#### Pull the Upstream Repo
```bash
git pull upstream main
```

### Use Sync Flow — Every Class
```bash
Go to your local repository
cd <your_local_repo>
```

#### Pull fresh starter code
```bash
git pull upstream main --no-rebase --no-edit -X ours
```

#### Develop locally
* edit, run, and debug in Cursor

#### Stage & commit changes
```bash
git add .
git commit -m "meaningful message"
```

#### Push to your fork
```bash
git push origin main
```
