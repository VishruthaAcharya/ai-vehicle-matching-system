# 🚀 GitHub Setup Guide

## Step 1: Initialize Git Repository

```bash
cd c:\Users\vishr\OneDrive\Desktop\vehicle-matching
git init
```

## Step 2: Add All Files

```bash
git add .
```

## Step 3: Create Initial Commit

```bash
git commit -m "Initial commit: Vehicle matching system with ETA prediction, demand forecasting, dynamic pricing"
```

## Step 4: Create GitHub Repository

1. Go to https://github.com/new
2. Repository name: `vehicle-matching`
3. Description: "AI-driven vehicle matching system with ETA prediction, demand forecasting, and dynamic pricing"
4. Choose: **Public** (or Private if preferred)
5. **DO NOT** initialize with README (you already have one)
6. Click **Create repository**

## Step 5: Connect Local Repo to GitHub

After creating the repo on GitHub, you'll see instructions. Run these in your terminal:

```bash
git branch -M main
git remote add origin https://github.com/YOUR_USERNAME/vehicle-matching.git
git push -u origin main
```

Replace `YOUR_USERNAME` with your actual GitHub username.

## Step 6: Verify

Go to `https://github.com/YOUR_USERNAME/vehicle-matching` - you should see all your files!

---

## 📁 Files That Will Be Pushed

### ✅ Included (tracked by git):
- All `.py` files (app.py, train_model.py, etc.)
- All `.md` files (README.md, documentation)
- `.gitignore` (git configuration)
- `requirements.txt` (dependencies)
- Model files (*.joblib)

### ❌ Excluded (in .gitignore):
- `.venv/` (virtual environment)
- `__pycache__/` (Python cache)
- `*.log` (log files)
- `*.db` (database files)
- `.env` (environment variables)

---

## 🔄 After First Push - Regular Updates

```bash
# Make changes to your files
# Then:

git add .
git commit -m "Description of changes"
git push
```

---

## 📋 Quick Command Reference

```bash
# See status of changes
git status

# View commit history
git log

# See what will be pushed
git diff --cached

# Undo last commit (before push)
git reset --soft HEAD~1

# Force push (use carefully!)
git push -f origin main
```

---

## ✅ Complete Checklist

- [ ] Run `git init` in project folder
- [ ] Run `git add .`
- [ ] Run `git commit -m "Initial commit..."`
- [ ] Create repo on GitHub
- [ ] Add remote: `git remote add origin https://...`
- [ ] Push: `git push -u origin main`
- [ ] Verify on GitHub website

---

Done! Your project is now on GitHub! 🎉
