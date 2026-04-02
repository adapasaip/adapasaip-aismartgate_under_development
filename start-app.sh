#!/bin/bash
# ANPR application startup script

set -e
trap 'on_error $? $LINENO' ERR

on_error() {
    local exit_code=$1
    local line_number=$2
    echo -e "${RED}✗ Error on line $line_number (exit code: $exit_code)${NC}"
    echo -e "${RED}Setup failed${NC}"
    exit $exit_code
}

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$SCRIPT_DIR"
BACKEND_DIR="$PROJECT_ROOT/backend"
VENV_DIR="$BACKEND_DIR/anpr_venv"

GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

echo -e "${BLUE}================================${NC}"
echo -e "${BLUE}  ANPR Desktop Application${NC}"
echo -e "${BLUE}  Setup & Launch${NC}"
echo -e "${BLUE}================================${NC}"
echo ""

echo -e "${YELLOW}[1/6]${NC} Checking prerequisites..."

CONFIG_FILE="$PROJECT_ROOT/data/cameras-config.json"
if [ ! -f "$CONFIG_FILE" ]; then
    echo -e "${RED}✗ Config file not found${NC}"
    exit 1
fi
echo -e "${GREEN}✓${NC} Config file found"

if ! command -v python3.10 &> /dev/null; then
    echo -e "${RED}✗ Python 3.10 not installed${NC}"
    exit 1
fi
echo -e "${GREEN}✓${NC} Python 3.10 found"

if ! command -v node &> /dev/null; then
    echo -e "${RED}✗ Node.js not installed${NC}"
    exit 1
fi
echo -e "${GREEN}✓${NC} Node.js found"

echo ""

echo -e "${YELLOW}[2/6]${NC} Setting up Python virtual environment..."

VENV_BROKEN=false
if [ -d "$VENV_DIR" ]; then
    if ! "${VENV_DIR}/bin/python3" --version &>/dev/null; then
        VENV_BROKEN=true
    else
        echo -e "${GREEN}✓${NC} Venv exists"
    fi
fi

if [ "$VENV_BROKEN" = true ]; then
    echo -e "  Removing broken venv..."
    rm -rf "$VENV_DIR"
    echo -e "  Creating new venv..."
    python3.10 -m venv "$VENV_DIR" 2>&1
    [ $? -eq 0 ] && echo -e "${GREEN}✓${NC} Venv created" || exit 1
elif [ ! -d "$VENV_DIR" ]; then
    echo -e "  Creating venv..."
    python3.10 -m venv "$VENV_DIR" 2>&1
    [ $? -eq 0 ] && echo -e "${GREEN}✓${NC} Venv created" || exit 1
fi

if ! "${VENV_DIR}/bin/python3" --version &>/dev/null; then
    echo -e "${RED}✗ Venv not functional${NC}"
    exit 1
fi

echo -e "${GREEN}✓${NC} Venv ready"

echo ""

echo -e "${YELLOW}[3/6]${NC} Installing Python dependencies..."

if [ ! -f "$BACKEND_DIR/requirements.txt" ]; then
    echo -e "${RED}✗ requirements.txt not found${NC}"
    exit 1
fi

echo -e "  Upgrading pip..."
"${VENV_DIR}/bin/pip" install --quiet --upgrade pip 2>&1

echo -e "  Installing packages..."
if "${VENV_DIR}/bin/pip" install --quiet -r "${BACKEND_DIR}/requirements.txt" 2>&1; then
    echo -e "${GREEN}✓${NC} Python dependencies installed"
else
    echo -e "${RED}✗ Failed to install dependencies${NC}"
    exit 1
fi

echo ""

echo -e "${YELLOW}[4/6]${NC} Installing Node dependencies..."

cd "$PROJECT_ROOT"

if [ ! -d "node_modules" ]; then
    echo -e "  Running npm install..."
    npm install 2>&1 | grep -v "npm warn" | grep -v "deprecated" >/dev/null
    echo -e "${GREEN}✓${NC} Node dependencies installed"
else
    echo -e "${GREEN}✓${NC} Node dependencies exist"
fi

echo ""

echo -e "${YELLOW}[5/6]${NC} Building application..."

echo -e "  Building frontend and server..."
if npm run build:all 2>&1; then
    echo -e "${GREEN}✓${NC} Build complete"
else
    echo -e "${RED}✗ Build failed${NC}"
    exit 1
fi

echo ""

echo -e "${YELLOW}[6/6]${NC} Starting services..."
echo ""

pkill -f "camera_anpr.py" 2>/dev/null || true
pkill -f "node dist/index.js" 2>/dev/null || true
sleep 1

echo -e "${YELLOW}Starting Flask backend...${NC}"
cd "$BACKEND_DIR/app"

nohup "${VENV_DIR}/bin/python" camera_anpr.py > "$PROJECT_ROOT/backend.log" 2>&1 &
BACKEND_PID=$!

if kill -0 $BACKEND_PID 2>/dev/null; then
    echo -e "${GREEN}✓${NC} Flask started (PID: $BACKEND_PID)"
else
    echo -e "${RED}✗ Failed to start Flask${NC}"
    exit 1
fi

sleep 5

if ! kill -0 $BACKEND_PID 2>/dev/null; then
    echo -e "${RED}✗ Flask crashed${NC}"
    tail -20 "$PROJECT_ROOT/backend.log"
    exit 1
fi

echo -e "${YELLOW}Starting Express server...${NC}"
cd "$PROJECT_ROOT"

nohup node dist/index.js > "$PROJECT_ROOT/frontend.log" 2>&1 &
FRONTEND_PID=$!

if kill -0 $FRONTEND_PID 2>/dev/null; then
    echo -e "${GREEN}✓${NC} Express started (PID: $FRONTEND_PID)"
else
    echo -e "${RED}✗ Failed to start Express${NC}"
    exit 1
fi

sleep 5

if ! kill -0 $FRONTEND_PID 2>/dev/null; then
    echo -e "${RED}✗ Express crashed${NC}"
    tail -20 "$PROJECT_ROOT/frontend.log"
    exit 1
fi

echo ""

echo -e "${GREEN}================================${NC}"
echo -e "${GREEN}  ✓ ANPR Running${NC}"
echo -e "${GREEN}================================${NC}"
echo ""
echo -e "Services:"
echo -e "  Flask (ANPR):   ${BLUE}http://localhost:8000${NC}"
echo -e "  Express (Web):  ${BLUE}http://localhost:5000${NC}"
echo ""
echo -e "Logs: backend.log, frontend.log"
echo ""
echo -e "To stop: pkill -f camera_anpr.py && pkill -f 'node dist/index.js'"
echo ""
echo -e "${YELLOW}Press Ctrl+C to stop${NC}"
wait
