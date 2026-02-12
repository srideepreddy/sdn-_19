#!/bin/bash
# start_system.sh

PROJECT_DIR="/mnt/c/Users/sride/OneDrive/Attachments/Desktop/Desktop/sdn_drl_shareable"
LOG_DIR="$PROJECT_DIR/logs"
mkdir -p "$LOG_DIR"

export EVENTLET_NO_GREENDNS=yes
echo "Starting Ryu Controller..."
nohup python3 $PROJECT_DIR/ryu_runner.py $PROJECT_DIR/controller/ryu_controller.py \
    --ofp-tcp-listen-port 6633 --observe-links > "$LOG_DIR/ryu.log" 2>&1 &

sleep 5

echo "Starting Mininet Topology..."
nohup sudo python3 -c "
from topology.custom_topology import create_network
net = create_network('127.0.0.1', 6633)
net.start()
print('Network started.')
import time
while True:
    time.sleep(60)
" > "$LOG_DIR/mininet.log" 2>&1 &

sleep 5

echo "Starting Dashboard on http://localhost:9000 ..."
export PYTHONPATH=$PROJECT_DIR
python3 -m visualization.dashboard --port 9004
