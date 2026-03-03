"""
Stats Collector for SDN DRL Routing.

Collects port and flow statistics from the Ryu controller and writes
them to data/net_stats.json every second for consumption by the DRL
agent and the visualization dashboard.

Features:
    - Thread-safe JSON writes with atomic file operations
    - Computes link utilization from port byte counters
    - Estimates per-link delay from RTT measurements
    - Tracks packet counts and queue sizes
    - Maintains a rolling history of performance metrics
"""

import json
import os
import time
import threading
import tempfile
from collections import defaultdict
from typing import Dict, List, Any, Optional


# Path to the shared stats file
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data')
STATS_FILE = os.path.join(DATA_DIR, 'net_stats.json')


class StatsCollector:
    """
    Collects and stores network statistics for the SDN DRL system.

    Reads raw counters from the controller, computes derived metrics
    (utilization, delay, loss), and writes everything to a shared
    JSON file that other components can read.
    """

    def __init__(self, topology_info: Dict = None, history_size: int = 100):
        """
        Initialize the stats collector.

        Args:
            topology_info: Dictionary with topology structure
            history_size: Number of historical data points to keep
        """
        self.topology_info = topology_info or self._default_topology()
        self.history_size = history_size
        self.lock = threading.Lock()

        # Current link statistics
        self.link_stats = {}

        # Previous byte counts for utilization calculation
        self._prev_port_stats = {}
        self._prev_timestamp = time.time()

        # Performance history (rolling window)
        self.delay_history = []
        self.packet_loss_history = []
        self.throughput_history = []

        # Current DRL selected path
        self.selected_path = []

        # Ensure data directory exists
        os.makedirs(DATA_DIR, exist_ok=True)

    def _default_topology(self) -> Dict:
        """Return default topology info matching custom_topology.py."""
        return {
            "switches": ["s1", "s2", "s3", "s4"],
            "hosts": ["h1", "h2", "h3", "h4", "h5", "h6"],
            "links": [
                {"src": "s1", "dst": "s2", "bw": 100},
                {"src": "s3", "dst": "s4", "bw": 100},
                {"src": "s1", "dst": "s4", "bw": 50},
                {"src": "s2", "dst": "s3", "bw": 50},
                {"src": "s1", "dst": "s3", "bw": 10},
                {"src": "s2", "dst": "s4", "bw": 10},
            ],
            "host_switch_map": {
                "h1": "s1", "h2": "s1",
                "h3": "s2",
                "h4": "s3",
                "h5": "s4", "h6": "s4",
            }
        }

    def update_port_stats(self, dpid: int, port_stats: List[Dict]):
        """
        Update statistics from port stats reply.

        Computes link utilization by comparing current byte counts
        with the previous measurement.

        Args:
            dpid: Switch datapath ID
            port_stats: List of port statistics dictionaries with keys:
                        port_no, rx_bytes, tx_bytes, rx_packets, tx_packets,
                        rx_errors, tx_errors, rx_dropped, tx_dropped
        """
        with self.lock:
            current_time = time.time()
            elapsed = current_time - self._prev_timestamp
            if elapsed <= 0:
                elapsed = 1.0

            switch_name = f"s{dpid}"

            for stat in port_stats:
                port_no = stat.get('port_no', 0)

                # Skip local port (OFPP_LOCAL = 0xfffffffe)
                if port_no >= 0xfffe:
                    continue

                key = (switch_name, port_no)
                tx_bytes = stat.get('tx_bytes', 0)
                rx_bytes = stat.get('rx_bytes', 0)
                tx_packets = stat.get('tx_packets', 0)
                rx_packets = stat.get('rx_packets', 0)
                rx_dropped = stat.get('rx_dropped', 0)
                tx_dropped = stat.get('tx_dropped', 0)
                rx_errors = stat.get('rx_errors', 0)
                tx_errors = stat.get('tx_errors', 0)

                # Calculate utilization
                prev = self._prev_port_stats.get(key, {})
                prev_tx = prev.get('tx_bytes', 0)
                prev_rx = prev.get('rx_bytes', 0)

                # Bytes per second
                tx_rate = max(0, (tx_bytes - prev_tx)) / elapsed
                rx_rate = max(0, (rx_bytes - prev_rx)) / elapsed

                # Bits per second
                tx_bps = tx_rate * 8
                rx_bps = rx_rate * 8

                # Store current for next delta
                self._prev_port_stats[key] = {
                    'tx_bytes': tx_bytes,
                    'rx_bytes': rx_bytes,
                }

                # Build link stat key
                link_key = f"{switch_name}:port{port_no}"
                self.link_stats[link_key] = {
                    'tx_bps': tx_bps,
                    'rx_bps': rx_bps,
                    'tx_packets': tx_packets,
                    'rx_packets': rx_packets,
                    'rx_dropped': rx_dropped,
                    'tx_dropped': tx_dropped,
                    'rx_errors': rx_errors,
                    'tx_errors': tx_errors,
                    'utilization': 0.0,  # Updated below
                }

            self._prev_timestamp = current_time

    def update_link_utilization(self):
        """
        Compute link utilization as a fraction of link bandwidth.

        Maps port-level stats to the corresponding topology links.
        """
        with self.lock:
            for link in self.topology_info.get('links', []):
                src = link['src']
                dst = link['dst']
                bw_mbps = link['bw']
                bw_bps = bw_mbps * 1e6  # Convert Mbps to bps

                link_key = f"{src}-{dst}"

                # Find max utilization from either direction
                max_util = 0.0
                for stat_key, stat_val in self.link_stats.items():
                    if stat_key.startswith(src) or stat_key.startswith(dst):
                        traffic_bps = max(stat_val.get('tx_bps', 0),
                                          stat_val.get('rx_bps', 0))
                        util = min(1.0, traffic_bps / bw_bps) if bw_bps > 0 else 0.0
                        max_util = max(max_util, util)

                self.link_stats[link_key] = self.link_stats.get(link_key, {})
                self.link_stats[link_key]['utilization'] = max_util
                self.link_stats[link_key]['bandwidth_mbps'] = bw_mbps

    def update_performance_metrics(self, delay: float, packet_loss: float,
                                   throughput: float):
        """
        Append a performance measurement to the rolling history.

        Args:
            delay: Current measured delay (ms)
            packet_loss: Current packet loss ratio (0-1)
            throughput: Current throughput (Mbps)
        """
        with self.lock:
            self.delay_history.append(round(delay, 3))
            self.packet_loss_history.append(round(packet_loss, 4))
            self.throughput_history.append(round(throughput, 2))

            # Trim to history_size
            if len(self.delay_history) > self.history_size:
                self.delay_history = self.delay_history[-self.history_size:]
            if len(self.packet_loss_history) > self.history_size:
                self.packet_loss_history = self.packet_loss_history[-self.history_size:]
            if len(self.throughput_history) > self.history_size:
                self.throughput_history = self.throughput_history[-self.history_size:]

    def set_selected_path(self, path: List[str]):
        """
        Set the currently selected DRL routing path.

        Args:
            path: List of switch names representing the path (e.g., ['s1', 's2', 's4'])
        """
        with self.lock:
            self.selected_path = list(path)

    def get_stats_dict(self) -> Dict:
        """
        Build the complete stats dictionary for JSON serialization.

        Returns:
            Dictionary with topology, link_stats, selected_path, and performance
        """
        with self.lock:
            # Build per-link utilization map
            link_utilization = {}
            for link in self.topology_info.get('links', []):
                link_key = f"{link['src']}-{link['dst']}"
                stats = self.link_stats.get(link_key, {})
                link_utilization[link_key] = {
                    'utilization': stats.get('utilization', 0.0),
                    'bandwidth_mbps': link['bw'],
                    'tx_bps': stats.get('tx_bps', 0),
                    'rx_bps': stats.get('rx_bps', 0),
                }

            return {
                'topology': self.topology_info,
                'link_stats': link_utilization,
                'selected_path': list(self.selected_path),
                'performance': {
                    'delay': list(self.delay_history),
                    'packet_loss': list(self.packet_loss_history),
                    'throughput': list(self.throughput_history),
                },
                'timestamp': time.time(),
            }

    def write_stats(self):
        """
        Write current statistics to the shared JSON file.

        Uses atomic write (write to temp file, then rename) to prevent
        partial reads by other processes.
        """
        stats = self.get_stats_dict()

        try:
            # Atomic write: write to temp file first, then rename
            fd, tmp_path = tempfile.mkstemp(dir=DATA_DIR, suffix='.tmp')
            with os.fdopen(fd, 'w') as f:
                json.dump(stats, f, indent=2)

            # Atomic rename (on Linux, this is atomic)
            os.replace(tmp_path, STATS_FILE)

        except Exception as e:
            print(f"[StatsCollector] Error writing stats: {e}")
            # Clean up temp file if it exists
            try:
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)
            except Exception:
                pass

    def start_periodic_write(self, interval: float = 1.0):
        """
        Start a background thread that writes stats every `interval` seconds.

        Args:
            interval: Seconds between writes (default 1.0)
        """
        self._running = True

        def _writer():
            while self._running:
                self.update_link_utilization()
                self.write_stats()
                time.sleep(interval)

        thread = threading.Thread(target=_writer, daemon=True, name='StatsWriter')
        thread.start()
        print(f"[StatsCollector] Periodic writer started (interval={interval}s)")

    def stop_periodic_write(self):
        """Stop the background writer thread."""
        self._running = False


def read_stats(stats_file: str = None) -> Dict:
    """
    Read the current network stats from the shared JSON file.

    This is a utility function for use by the DRL agent and dashboard.

    Args:
        stats_file: Path to the stats JSON file (defaults to STATS_FILE)

    Returns:
        Dictionary with current network statistics
    """
    if stats_file is None:
        stats_file = STATS_FILE

    try:
        with open(stats_file, 'r') as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return {
            'topology': {},
            'link_stats': {},
            'selected_path': [],
            'performance': {'delay': [], 'packet_loss': [], 'throughput': []},
            'timestamp': 0,
        }


if __name__ == '__main__':
    """Test the stats collector standalone."""
    print("Testing StatsCollector...")

    collector = StatsCollector()

    # Simulate some stats updates
    for i in range(5):
        collector.update_performance_metrics(
            delay=10.0 + i * 2,
            packet_loss=0.01 * i,
            throughput=50.0 - i * 5
        )
        collector.set_selected_path(['s1', 's2', 's4'])
        collector.write_stats()
        print(f"  Written stats iteration {i + 1}")
        time.sleep(0.5)

    # Read back
    stats = read_stats()
    print(f"\nRead back stats:")
    print(f"  Topology switches: {stats['topology']['switches']}")
    print(f"  Selected path: {stats['selected_path']}")
    print(f"  Delay history: {stats['performance']['delay']}")
    print(f"  Throughput history: {stats['performance']['throughput']}")
    print("\nStatsCollector test completed!")
