#!/bin/bash
# ============================================================
# SDN DRL Routing — Run Script
#
# Starts all components in the correct order:
#   1. Ryu SDN Controller
#   2. Mininet Custom Topology
#   3. DRL Agent (training/inference)
#   4. Live Visualization Dashboard
#
# Usage:
#   chmod +x run.sh
#   sudo ./run.sh                  # Full system
#   sudo ./run.sh --train-only     # Simulation training only (no Mininet)
#   sudo ./run.sh --dashboard-only # Dashboard only
# ============================================================

set -e

# Project root directory (where this script lives)
PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"
DATA_DIR="$PROJECT_DIR/data"
LOG_DIR="$DATA_DIR/logs"

# Configuration
CONTROLLER_PORT=6633
DASHBOARD_PORT=9000
DRL_MODE="live"       # 'live' or 'simulation'
DRL_EPISODES=1000

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# PIDs for cleanup
PIDS=()

# ============================================================
# Cleanup handler
# ============================================================
cleanup() {
    echo -e "\n${YELLOW}[Shutdown] Stopping all components...${NC}"

    for pid in "${PIDS[@]}"; do
        if kill -0 "$pid" 2>/dev/null; then
            echo "  Killing PID $pid"
            kill "$pid" 2>/dev/null || true
        fi
    done

    # Clean up Mininet
    echo "  Cleaning up Mininet..."
    mn -c 2>/dev/null || true

    echo -e "${GREEN}[Shutdown] All components stopped.${NC}"
    exit 0
}

trap cleanup SIGINT SIGTERM EXIT

# ============================================================
# Helper functions
# ============================================================
log_section() {
    echo -e "\n${BLUE}============================================================${NC}"
    echo -e "${BLUE}  $1${NC}"
    echo -e "${BLUE}============================================================${NC}\n"
}

wait_for_port() {
    local port=$1
    local timeout=$2
    local elapsed=0

    while ! ss -tlnp | grep -q ":$port "; do
        sleep 1
        elapsed=$((elapsed + 1))
        if [ $elapsed -ge $timeout ]; then
            echo -e "${RED}  Timeout waiting for port $port${NC}"
            return 1
        fi
    done
    echo -e "${GREEN}  Port $port is ready (${elapsed}s)${NC}"
    return 0
}

# ============================================================
# Pre-flight checks
# ============================================================
log_section "Pre-flight Checks"

# Create directories
mkdir -p "$LOG_DIR"
mkdir -p "$DATA_DIR/checkpoints"
mkdir -p "$DATA_DIR/results"

# Check for root (needed for Mininet)
if [ "$1" != "--train-only" ] && [ "$1" != "--dashboard-only" ]; then
    if [ "$EUID" -ne 0 ]; then
        echo -e "${RED}Error: This script must run as root (for Mininet)${NC}"
        echo "Usage: sudo ./run.sh"
        exit 1
    fi
fi

# Check dependencies
echo "Checking dependencies..."
python3 -c "import torch; print(f'  PyTorch: {torch.__version__}')" 2>/dev/null || \
    echo -e "${YELLOW}  Warning: PyTorch not found${NC}"
python3 -c "import flask; print(f'  Flask: {flask.__version__}')" 2>/dev/null || \
    echo -e "${YELLOW}  Warning: Flask not found${NC}"
python3 -c "import networkx; print(f'  NetworkX: {networkx.__version__}')" 2>/dev/null || \
    echo -e "${YELLOW}  Warning: NetworkX not found${NC}"

echo -e "${GREEN}Pre-flight checks passed.${NC}"

# ============================================================
# Handle special modes
# ============================================================
if [ "$1" == "--train-only" ]; then
    log_section "Simulation Training Mode"
    echo "Starting DRL training in simulation mode..."
    cd "$PROJECT_DIR"
    python3 -m drl.train --mode simulation --episodes $DRL_EPISODES
    exit 0
fi

if [ "$1" == "--dashboard-only" ]; then
    log_section "Dashboard Only Mode"
    echo "Starting visualization dashboard..."
    cd "$PROJECT_DIR"
    python3 -m visualization.dashboard --port $DASHBOARD_PORT
    exit 0
fi

# ============================================================
# Step 1: Clean previous Mininet state
# ============================================================
log_section "Step 1: Cleaning Previous State"
echo "Running mn -c to clear previous Mininet state..."
mn -c 2>/dev/null || true
echo -e "${GREEN}Cleanup complete.${NC}"

# ============================================================
# Step 2: Start Ryu Controller
# ============================================================
log_section "Step 2: Starting Ryu Controller"
echo "Starting Ryu controller on port $CONTROLLER_PORT..."

cd "$PROJECT_DIR"
ryu-manager controller/ryu_controller.py \
    --ofp-tcp-listen-port $CONTROLLER_PORT \
    --observe-links \
    > "$LOG_DIR/ryu.log" 2>&1 &
PIDS+=($!)
echo "  Ryu PID: ${PIDS[-1]}"

echo "Waiting for controller to initialize..."
sleep 3
wait_for_port $CONTROLLER_PORT 10

# ============================================================
# Step 3: Start Mininet Topology
# ============================================================
log_section "Step 3: Starting Mininet Topology"
echo "Launching custom topology (4 switches, 6 hosts)..."

cd "$PROJECT_DIR"
python3 -c "
from topology.custom_topology import create_network
net = create_network('127.0.0.1', $CONTROLLER_PORT)
net.start()
print('Network started. Waiting...')
import time
while True:
    time.sleep(60)
" > "$LOG_DIR/mininet.log" 2>&1 &
PIDS+=($!)
echo "  Mininet PID: ${PIDS[-1]}"

echo "Waiting for topology to converge..."
sleep 5
echo -e "${GREEN}Topology started.${NC}"

# ============================================================
# Step 4: Start DRL Agent
# ============================================================
log_section "Step 4: Starting DRL Agent"
echo "Starting DQN training in live mode..."

cd "$PROJECT_DIR"
python3 -m drl.train \
    --mode $DRL_MODE \
    --episodes $DRL_EPISODES \
    > "$LOG_DIR/drl.log" 2>&1 &
PIDS+=($!)
echo "  DRL PID: ${PIDS[-1]}"
echo -e "${GREEN}DRL agent started.${NC}"

# ============================================================
# Step 5: Start Visualization Dashboard
# ============================================================
log_section "Step 5: Starting Visualization Dashboard"
echo "Starting Flask dashboard on port $DASHBOARD_PORT..."

cd "$PROJECT_DIR"
python3 -m visualization.dashboard \
    --port $DASHBOARD_PORT \
    > "$LOG_DIR/dashboard.log" 2>&1 &
PIDS+=($!)
echo "  Dashboard PID: ${PIDS[-1]}"

sleep 2
echo -e "${GREEN}Dashboard started.${NC}"

# ============================================================
# Running
# ============================================================
log_section "System Running"
echo -e "All components are running:"
echo -e "  ${GREEN}✓${NC} Ryu Controller    — port $CONTROLLER_PORT"
echo -e "  ${GREEN}✓${NC} Mininet Topology   — 4 switches, 6 hosts"
echo -e "  ${GREEN}✓${NC} DRL Agent          — $DRL_MODE mode"
echo -e "  ${GREEN}✓${NC} Dashboard          — http://localhost:$DASHBOARD_PORT"
echo ""
echo -e "Logs: $LOG_DIR/"
echo ""
echo -e "${YELLOW}Press Ctrl+C to stop all components.${NC}"

# Wait forever (cleanup handler will stop everything)
while true; do
    sleep 10
done
