#!/bin/bash
# Quick Reference - AI SmartGate ANPR Setup
# Copy and paste commands directly into terminal

# ============================================================
# STEP 1: Navigate to backend directory
# ============================================================
cd backend

# ============================================================
# STEP 2: Run the automated setup script
# ============================================================
bash setup_venv.sh

# This will automatically:
# - Create Python 3.10.14 virtual environment
# - Install all 128 dependencies
# - Create helper scripts
# - Verify installation

# ============================================================
# STEP 3: Activate the virtual environment
# ============================================================
source anpr_venv/bin/activate

# Or use the helper script:
bash activate_venv.sh

# ============================================================
# STEP 4: Run the ANPR system
# ============================================================
cd app
python camera_anpr.py

# Or use the helper script:
bash ../run_anpr.sh

# ============================================================
# STEP 5: Deactivate when done
# ============================================================
deactivate

# ============================================================
# TROUBLESHOOTING
# ============================================================

# Check Python version
python --version  # Should show 3.10.x

# List all installed packages
pip list | head -20

# Verify critical packages
python -c "import cv2, torch, depthai; print('✓ All critical packages OK')"

# Check virtual environment location
which python

# Reinstall specific package
pip install <package_name>

# ============================================================
# FILES CREATED/MODIFIED
# ============================================================

# 1. backend/requirements.txt (127 lines)
#    - Updated with all 128 dependencies
#    - Organized by category
#    - Pinned versions for stability

# 2. backend/setup_venv.sh (7.5 KB)
#    - Automated setup script
#    - Color-coded output
#    - Error handling

# 3. aismartgate-anpr.desktop (fixed)
#    - Updated paths
#    - Now executable
#    - Ready for application launcher

# 4. backend/SETUP_INSTRUCTIONS.md
#    - Detailed setup guide
#    - Troubleshooting section

# 5. SETUP_SUMMARY.md (root directory)
#    - Complete summary of changes

# ============================================================
# QUICK STATS
# ============================================================

# Total Dependencies: 128 packages
# Python Version: 3.10.14
# Installation Size: ~1.5-2 GB
# Setup Time: 10-30 minutes (first time)

# Main Categories:
# - Web Frameworks: 6 packages
# - Computer Vision: 4 packages
# - Deep Learning: 4 packages
# - ONNX/Optimization: 4 packages
# - OAK-D Camera: 2 packages
# - OCR: 1 package
# - Data Processing: 10+ packages
# - Support/Utilities: 90+ packages

# ============================================================
# DESKTOP ICON FIX
# ============================================================

# Desktop file location:
# /home/sairaspi/Desktop/smartgate-ver0/GIT+OAK/adapasaip-aismartgate_under_development/aismartgate-anpr.desktop

# Fixed paths:
# - Exec: Updated to correct start-app.sh path
# - Icon: Updated to correct car-detection-icon.svg path
# - Path: Updated to correct working directory

# ============================================================
# NOTES
# ============================================================

# 1. First time setup may take time (downloading large packages)
# 2. Requires internet connection for pip install
# 3. OAK-D camera should be connected via USB before running
# 4. Tesseract must be installed: sudo apt-get install tesseract-ocr
# 5. All packages are version-pinned for stability
# 6. No other files in the project were modified

# ============================================================
# CREATED: March 23, 2026
# STATUS: ✅ Complete & Ready to Use
# ============================================================
