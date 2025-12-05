# 🐍 Python Environment Management Guide
## ML_PROJECT - Complete Environment Setup & Usage

---

## 📋 **QUICK START - Activate Environment Every Time**

### **Method 1: Manual Activation (Recommended)**
```bash
# Navigate to project directory
cd /Users/muhammadsharjeel/Documents/ML_PROJECT

# Activate virtual environment
source venv/bin/activate

# Verify activation (you should see (venv) in your prompt)
which python  # Should show: .../ML_PROJECT/venv/bin/python
```

### **Method 2: Auto-Activation Script**
Create a script to auto-activate when you enter the directory:

**For Zsh (your current shell):**
```bash
# Add to ~/.zshrc
echo 'alias mlproject="cd /Users/muhammadsharjeel/Documents/ML_PROJECT && source venv/bin/activate"' >> ~/.zshrc
source ~/.zshrc

# Then just type:
mlproject
```

**For VS Code/Cursor:**
- VS Code/Cursor will automatically detect and use the venv if you select it
- Make sure to select the correct Python interpreter: `venv/bin/python`

---

## 🔍 **DEEP ENVIRONMENT ANALYSIS**

### **Current Environment Status**

| Component | Details |
|-----------|---------|
| **Environment Type** | Virtual Environment (venv) |
| **Location** | `/Users/muhammadsharjeel/Documents/ML_PROJECT/venv` |
| **Python Version** | 3.13.5 (packaged by Anaconda, Inc.) |
| **Total Packages** | 59 packages installed |
| **Jupyter Kernel** | `mlproject-venv` (registered) |

### **Installed Packages**

✅ **Core ML Packages:**
- `numpy` 2.3.5
- `pandas` 2.3.3
- `seaborn` 0.13.2
- `matplotlib` 3.10.7
- `scikit-learn` 1.7.2
- `catboost` 1.2.8
- `xgboost` 3.1.2 ⚠️ (requires libomp - see troubleshooting)

✅ **Development Tools:**
- `ipykernel` 7.1.0 (for Jupyter notebooks)
- `ipython` 9.7.0
- `jupyter_client`, `jupyter_core`

### **Environment Structure**
```
venv/
├── bin/              # Executables (python, pip, etc.)
├── lib/              # Installed packages
├── include/          # Header files
├── conda-meta/       # Conda metadata (if created via conda)
└── pyvenv.cfg        # Configuration file
```

---

## 🚀 **HOW TO USE ENVIRONMENT IN DIFFERENT CASES**

### **Case 1: Working with Jupyter Notebooks**

**Step 1: Activate Environment**
```bash
cd /Users/muhammadsharjeel/Documents/ML_PROJECT
source venv/bin/activate
```

**Step 2: Select Kernel in Notebook**
- Open your notebook (e.g., `2. MODEL TRAINING.ipynb`)
- Click on kernel selector (top right)
- Select: **`venv (Python 3.13.5)`** or **`mlproject-venv`**
- ❌ **DO NOT** use: `base (Python 3.13.5)` from anaconda3

**Step 3: Verify Kernel**
```python
# Run this in a notebook cell:
import sys
print(sys.executable)  # Should show: .../ML_PROJECT/venv/bin/python
```

### **Case 2: Running Python Scripts**

**Method A: With Activated Environment**
```bash
# Activate first
source venv/bin/activate

# Run script
python src/pipeline/train_pipeline.py
```

**Method B: Direct Execution (No Activation)**
```bash
# Use venv's python directly
venv/bin/python src/pipeline/train_pipeline.py
```

### **Case 3: Installing New Packages**

**Always activate first:**
```bash
source venv/bin/activate
pip install package_name

# Update requirements.txt
pip freeze > requirements.txt
```

### **Case 4: VS Code / Cursor IDE**

**Step 1: Select Python Interpreter**
- Press `Cmd+Shift+P` (Mac) or `Ctrl+Shift+P` (Windows/Linux)
- Type: "Python: Select Interpreter"
- Choose: `./venv/bin/python` or `Python 3.13.5 ('venv': venv)`

**Step 2: Verify in Terminal**
- Open integrated terminal (`Ctrl+`` ` or `` ` ``)
- VS Code should auto-activate venv
- You should see `(venv)` in prompt

### **Case 5: Running Tests**

```bash
source venv/bin/activate
python -m pytest tests/  # If you have pytest
# OR
python -m unittest discover tests/
```

### **Case 6: Creating New Environment (If Needed)**

```bash
# Create new venv
python3 -m venv venv

# Activate it
source venv/bin/activate

# Install packages
pip install -r requirements.txt

# Register as Jupyter kernel
python -m ipykernel install --user --name=mlproject-venv --display-name="venv (Python 3.13.5)"
```

---

## 🔧 **TROUBLESHOOTING**

### **Issue 1: XGBoost Import Error**
**Error:** `XGBoostError: Library (libxgboost.dylib) could not be loaded`

**Solution:**
```bash
# Install OpenMP library (required for XGBoost on macOS)
brew install libomp

# If brew is not installed, install Homebrew first:
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
```

### **Issue 2: Package Not Found**
**Error:** `ModuleNotFoundError: No module named 'sklearn'`

**Solution:**
```bash
# Activate environment first
source venv/bin/activate

# Install missing package
pip install scikit-learn

# Verify installation
python -c "import sklearn; print(sklearn.__version__)"
```

### **Issue 3: Wrong Kernel Selected**
**Problem:** Notebook using wrong Python environment

**Solution:**
1. Check current kernel: Look at top-right of notebook
2. Click kernel selector
3. Choose: `venv (Python 3.13.5)` or `mlproject-venv`
4. Restart kernel if needed

### **Issue 4: Environment Not Activating**
**Problem:** `source venv/bin/activate` doesn't work

**Solution:**
```bash
# Check if venv exists
ls -la venv/bin/activate

# If missing, recreate:
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### **Issue 5: Multiple Python Versions**
**Problem:** Confusion between system Python, conda, and venv

**Solution:**
```bash
# Check which Python is active
which python
python --version

# Should show venv Python when activated:
# /Users/muhammadsharjeel/Documents/ML_PROJECT/venv/bin/python
# Python 3.13.5
```

---

## 📝 **BEST PRACTICES**

### ✅ **DO:**
1. **Always activate venv** before working on the project
2. **Check kernel** before running notebooks
3. **Update requirements.txt** after installing new packages
4. **Use venv/bin/python** for direct script execution
5. **Verify imports** work before running full scripts

### ❌ **DON'T:**
1. **Don't use system Python** directly (use venv)
2. **Don't use conda base** environment for this project
3. **Don't install packages globally** (always in venv)
4. **Don't commit venv/** to git (already in .gitignore)

---

## 🔄 **DAILY WORKFLOW CHECKLIST**

```bash
# 1. Navigate to project
cd /Users/muhammadsharjeel/Documents/ML_PROJECT

# 2. Activate environment
source venv/bin/activate

# 3. Verify activation (optional)
which python
pip list | grep -E "(numpy|pandas|sklearn)"

# 4. Work on your project
# - Run notebooks (select venv kernel)
# - Run scripts
# - Install packages if needed

# 5. Deactivate when done (optional)
deactivate
```

---

## 📦 **PACKAGE MANAGEMENT**

### **View Installed Packages**
```bash
source venv/bin/activate
pip list                    # All packages
pip list | grep sklearn     # Specific package
pip show package_name       # Package details
```

### **Install from requirements.txt**
```bash
source venv/bin/activate
pip install -r requirements.txt
```

### **Update requirements.txt**
```bash
source venv/bin/activate
pip freeze > requirements.txt
```

### **Install New Package**
```bash
source venv/bin/activate
pip install package_name
# Add to requirements.txt manually or:
pip freeze > requirements.txt
```

---

## 🎯 **QUICK REFERENCE COMMANDS**

| Task | Command |
|------|---------|
| **Activate venv** | `source venv/bin/activate` |
| **Deactivate venv** | `deactivate` |
| **Check Python** | `which python` |
| **List packages** | `pip list` |
| **Install package** | `pip install package_name` |
| **Run script** | `python script.py` |
| **Check kernel** | `jupyter kernelspec list` |
| **Register kernel** | `python -m ipykernel install --user --name=mlproject-venv` |

---

## 🔗 **ENVIRONMENT PATHS**

- **Venv Location:** `/Users/muhammadsharjeel/Documents/ML_PROJECT/venv`
- **Python Executable:** `/Users/muhammadsharjeel/Documents/ML_PROJECT/venv/bin/python`
- **Pip Executable:** `/Users/muhammadsharjeel/Documents/ML_PROJECT/venv/bin/pip`
- **Jupyter Kernel:** `/Users/muhammadsharjeel/Library/Jupyter/kernels/mlproject-venv`

---

## 📚 **ADDITIONAL RESOURCES**

- [Python venv Documentation](https://docs.python.org/3/library/venv.html)
- [Jupyter Kernels Guide](https://jupyter-client.readthedocs.io/en/stable/kernels.html)
- [VS Code Python Environments](https://code.visualstudio.com/docs/python/environments)

---

**Last Updated:** December 3, 2025  
**Environment:** Python 3.13.5 (venv)  
**Project:** ML_PROJECT

