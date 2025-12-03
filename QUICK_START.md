# 🚀 QUICK START GUIDE

## Activate Environment Every Time

### Option 1: Use Activation Script (Easiest)
```bash
source activate_env.sh
```

### Option 2: Manual Activation
```bash
cd /Users/muhammadsharjeel/Documents/ML_PROJECT
source venv/bin/activate
```

### Option 3: Add Alias to ~/.zshrc
```bash
echo 'alias mlproject="cd /Users/muhammadsharjeel/Documents/ML_PROJECT && source venv/bin/activate"' >> ~/.zshrc
source ~/.zshrc
# Then just type: mlproject
```

## Verify Environment is Active
```bash
which python  # Should show: .../ML_PROJECT/venv/bin/python
```

## For Jupyter Notebooks
1. Open notebook
2. Select kernel: **venv (Python 3.13.5)** or **mlproject-venv**
3. ❌ Don't use: base (anaconda3)

## All Packages Installed ✅
- numpy 2.3.5
- pandas 2.3.3
- seaborn 0.13.2
- matplotlib 3.10.7
- scikit-learn 1.7.2
- catboost 1.2.8
- xgboost 3.1.2

## Full Guide
See `ENVIRONMENT_GUIDE.md` for complete documentation.
