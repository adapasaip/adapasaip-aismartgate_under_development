#!/bin/bash

# ============================================================
# AI SmartGate ANPR - Virtual Environment Setup Script
# Purpose: Create Python 3.10.14 virtual environment with all dependencies
# ============================================================

set -e  # Exit on error

# ============================================================
# Configuration
# ============================================================
PYTHON_VERSION="3.10.14"
VENV_NAME="anpr_venv"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_PATH="${SCRIPT_DIR}/${VENV_NAME}"
REQUIREMENTS_FILE="${SCRIPT_DIR}/requirements.txt"

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# ============================================================
# Helper Functions
# ============================================================
print_header() {
    echo -e "${BLUE}============================================================${NC}"
    echo -e "${BLUE}$1${NC}"
    echo -e "${BLUE}============================================================${NC}"
}

print_success() {
    echo -e "${GREEN}✓ $1${NC}"
}

print_error() {
    echo -e "${RED}✗ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠ $1${NC}"
}

print_info() {
    echo -e "${BLUE}ℹ $1${NC}"
}

# ============================================================
# Step 1: Verify Python Version
# ============================================================
print_header "Step 1: Verifying Python Installation"

if ! command -v python3.10 &> /dev/null; then
    print_error "Python 3.10 is not installed or not in PATH"
    echo "Please install Python 3.10.14 first:"
    echo "  Ubuntu/Debian: sudo apt-get install python3.10 python3.10-venv python3.10-dev"
    echo "  Or use pyenv: pyenv install 3.10.14"
    exit 1
fi

PYTHON_PATH=$(which python3.10)
PYTHON_ACTUAL_VERSION=$($PYTHON_PATH --version 2>&1 | awk '{print $2}')
print_success "Found Python: $PYTHON_PATH (version: $PYTHON_ACTUAL_VERSION)"

# ============================================================
# Step 2: Check if venv already exists
# ============================================================
print_header "Step 2: Checking for Existing Virtual Environment"

if [ -d "$VENV_PATH" ]; then
    print_warning "Virtual environment already exists at: $VENV_PATH"
    read -p "Do you want to remove it and create a fresh one? (y/n) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        print_info "Removing old virtual environment..."
        rm -rf "$VENV_PATH"
        print_success "Removed old virtual environment"
    else
        print_warning "Using existing virtual environment"
        SKIP_VENV_CREATION=1
    fi
else
    print_info "No existing virtual environment found"
fi

# ============================================================
# Step 3: Create Virtual Environment
# ============================================================
if [ "$SKIP_VENV_CREATION" != "1" ]; then
    print_header "Step 3: Creating Virtual Environment"
    
    print_info "Creating Python 3.10 virtual environment..."
    $PYTHON_PATH -m venv "$VENV_PATH"
    
    if [ -d "$VENV_PATH" ]; then
        print_success "Virtual environment created at: $VENV_PATH"
    else
        print_error "Failed to create virtual environment"
        exit 1
    fi
else
    print_header "Step 3: Using Existing Virtual Environment"
fi

# ============================================================
# Step 4: Activate Virtual Environment & Upgrade pip
# ============================================================
print_header "Step 4: Activating Environment & Upgrading pip"

source "$VENV_PATH/bin/activate"
print_success "Virtual environment activated"

print_info "Upgrading pip, setuptools, and wheel..."
python -m pip install --upgrade pip setuptools wheel --quiet
print_success "pip, setuptools, and wheel upgraded"

# ============================================================
# Step 5: Verify Requirements File
# ============================================================
print_header "Step 5: Verifying Requirements File"

if [ ! -f "$REQUIREMENTS_FILE" ]; then
    print_error "Requirements file not found: $REQUIREMENTS_FILE"
    print_warning "Using default minimal requirements"
    REQUIREMENTS_FILE=""
fi

if [ -n "$REQUIREMENTS_FILE" ]; then
    PACKAGE_COUNT=$(grep -c "^[a-zA-Z]" "$REQUIREMENTS_FILE" || echo 0)
    print_success "Found requirements file with $PACKAGE_COUNT packages"
    print_info "Location: $REQUIREMENTS_FILE"
fi

# ============================================================
# Step 6: Install Dependencies
# ============================================================
print_header "Step 6: Installing Dependencies from requirements.txt"

if [ -n "$REQUIREMENTS_FILE" ]; then
    print_info "Installing packages (this may take several minutes)..."
    
    if pip install -r "$REQUIREMENTS_FILE"; then
        print_success "All packages installed successfully"
    else
        print_error "Some packages failed to install"
        print_warning "Continuing anyway... You may need to install missing packages manually"
    fi
else
    print_warning "No requirements file found, skipping package installation"
fi

# ============================================================
# Step 7: Verify Installation
# ============================================================
print_header "Step 7: Verifying Installation"

# Check critical packages
CRITICAL_PACKAGES=("flask" "opencv-cv2" "numpy" "torch" "depthai" "onnxruntime")

for package in "${CRITICAL_PACKAGES[@]}"; do
    if python -c "import ${package}" 2>/dev/null; then
        print_success "$package is installed"
    else
        print_warning "$package is not installed (may not be critical)"
    fi
done

# Show Python info
print_info "Python executable: $(which python)"
print_info "Python version: $(python --version)"
print_info "Virtual environment path: $VENV_PATH"

# ============================================================
# Step 8: Display Installation Summary
# ============================================================
print_header "Step 8: Installation Summary"

print_success "Virtual environment setup completed!"
echo ""
echo "To activate the virtual environment, run:"
echo -e "  ${YELLOW}source $VENV_PATH/bin/activate${NC}"
echo ""
echo "To deactivate later, run:"
echo -e "  ${YELLOW}deactivate${NC}"
echo ""
echo "To run the ANPR system:"
echo -e "  ${YELLOW}cd $SCRIPT_DIR/app${NC}"
echo -e "  ${YELLOW}python camera_anpr.py${NC}"
echo ""

# ============================================================
# Step 9: Optional - Create Activation Helper Script
# ============================================================
print_header "Step 9: Creating Helper Scripts"

# Create activate script
ACTIVATE_SCRIPT="${SCRIPT_DIR}/activate_venv.sh"
cat > "$ACTIVATE_SCRIPT" << 'EOF'
#!/bin/bash
# Quick activation script for the virtual environment
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/anpr_venv/bin/activate"
echo "Virtual environment activated!"
EOF

chmod +x "$ACTIVATE_SCRIPT"
print_success "Created activation helper: $ACTIVATE_SCRIPT"

# Create run script
RUN_SCRIPT="${SCRIPT_DIR}/run_anpr.sh"
cat > "$RUN_SCRIPT" << 'EOF'
#!/bin/bash
# Quick startup script for ANPR system
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/anpr_venv/bin/activate"
cd "$SCRIPT_DIR/app"
python camera_anpr.py "$@"
EOF

chmod +x "$RUN_SCRIPT"
print_success "Created startup helper: $RUN_SCRIPT"

echo ""
echo -e "${GREEN}Setup complete! Ready to use.${NC}"
