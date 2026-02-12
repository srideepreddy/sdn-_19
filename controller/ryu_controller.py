"""
Ryu SDN Controller with DRL Integration.

This controller:
    1. Manages OpenFlow 1.3 switches
    2. Discovers network topology
    3. Collects port and flow statistics every second
    4. Queries the DRL agent for routing decisions
    5. Installs flow rules dynamically using OpenFlow
    6. Exposes selected path and stats via shared JSON file

Usage:
    ryu-manager controller/ryu_controller.py --ofp-tcp-listen-port 6633
"""

import json
import os
import time
import threading
from collections import defaultdict

from ryu.base import app_manager
from ryu.controller import ofp_event
from ryu.controller.handler import CONFIG_DISPATCHER, MAIN_DISPATCHER
from ryu.controller.handler import set_ev_cls
from ryu.ofproto import ofproto_v1_3
from ryu.lib.packet import packet, ethernet, arp, ipv4, tcp, udp
from ryu.lib import hub
from ryu.topology import event as topo_event
from ryu.topology.api import get_switch, get_link

import networkx as nx

# Import stats collector
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from controller.stats_collector import StatsCollector, read_stats

# Path to stats and DRL decision files
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data')
STATS_FILE = os.path.join(DATA_DIR, 'net_stats.json')
DRL_DECISION_FILE = os.path.join(DATA_DIR, 'drl_decision.json')


class DRLController(app_manager.RyuApp):
    """
    Ryu-based SDN controller with DRL agent integration.

    Responsibilities:
        - Switch setup and table-miss flow entry
        - MAC learning for host discovery
        - Topology graph construction using NetworkX
        - Port/flow stats collection every second
        - Querying DRL agent for best routing path
        - Installing OpenFlow flow rules along the chosen path
        - Writing all stats to data/net_stats.json
    """

    OFP_VERSIONS = [ofproto_v1_3.OFP_VERSION]

    def __init__(self, *args, **kwargs):
        super(DRLController, self).__init__(*args, **kwargs)

        # MAC address table: dpid -> {mac: port}
        self.mac_to_port = defaultdict(dict)

        # Host location: mac -> (dpid, port)
        self.host_table = {}

        # Network topology graph
        self.net_graph = nx.Graph()

        # Switch datapaths: dpid -> datapath
        self.datapaths = {}

        # Port statistics: (dpid, port) -> stats dict
        self.port_stats = defaultdict(dict)

        # Flow statistics: dpid -> [flow_entries]
        self.flow_stats = defaultdict(list)

        # Link bandwidth map: (dpid1, dpid2) -> bw_mbps
        self.link_bw = {}

        # Stats collector
        self.stats_collector = StatsCollector()

        # Currently selected DRL path (list of switch names)
        self.current_path = []

        # Start stats collection monitor
        self.monitor_thread = hub.spawn(self._monitor_loop)

        self.logger.info("DRL Controller initialized")

    # =========================================================
    # Switch Setup
    # =========================================================

    @set_ev_cls(ofp_event.EventOFPSwitchFeatures, CONFIG_DISPATCHER)
    def switch_features_handler(self, ev):
        """
        Handle switch connection and install table-miss flow entry.

        The table-miss entry sends unmatched packets to the controller
        as packet-in messages.
        """
        datapath = ev.msg.datapath
        ofproto = datapath.ofproto
        parser = datapath.ofproto_parser
        dpid = datapath.id

        self.datapaths[dpid] = datapath

        # Install table-miss flow entry
        match = parser.OFPMatch()
        actions = [parser.OFPActionOutput(
            ofproto.OFPP_CONTROLLER,
            ofproto.OFPCML_NO_BUFFER
        )]
        self._add_flow(datapath, 0, match, actions)

        self.logger.info(f"Switch s{dpid} connected — table-miss entry installed")

    # =========================================================
    # Packet-In Handler
    # =========================================================

    @set_ev_cls(ofp_event.EventOFPPacketIn, MAIN_DISPATCHER)
    def packet_in_handler(self, ev):
        """
        Handle packet-in messages.

        Learns host MAC locations, computes routing path (via DRL or
        shortest path), and installs flow rules along the path.
        """
        msg = ev.msg
        datapath = msg.datapath
        ofproto = datapath.ofproto
        parser = datapath.ofproto_parser
        dpid = datapath.id
        in_port = msg.match['in_port']

        pkt = packet.Packet(msg.data)
        eth = pkt.get_protocol(ethernet.ethernet)

        if eth is None:
            return

        src_mac = eth.src
        dst_mac = eth.dst
        ethertype = eth.ethertype

        # Ignore LLDP
        if ethertype == 0x88cc:
            return

        # Learn source MAC
        self.mac_to_port[dpid][src_mac] = in_port
        self.host_table[src_mac] = (dpid, in_port)

        # If destination known, compute and install path
        if dst_mac in self.host_table:
            dst_dpid, dst_port = self.host_table[dst_mac]
            src_dpid = dpid

            if src_dpid == dst_dpid:
                # Same switch — direct forwarding
                out_port = dst_port
                actions = [parser.OFPActionOutput(out_port)]
                match = parser.OFPMatch(
                    in_port=in_port,
                    eth_dst=dst_mac,
                    eth_src=src_mac
                )
                self._add_flow(datapath, 1, match, actions,
                               idle_timeout=30, hard_timeout=120)
            else:
                # Cross-switch routing — query DRL agent
                path = self._get_routing_path(src_dpid, dst_dpid)

                if path and len(path) >= 2:
                    self.logger.info(
                        f"Installing path: {' -> '.join(f's{p}' for p in path)}"
                        f" for {src_mac} -> {dst_mac}"
                    )
                    self._install_path(path, src_mac, dst_mac,
                                       in_port, dst_port)

                    # Update selected path for visualization
                    self.current_path = [f"s{p}" for p in path]
                    self.stats_collector.set_selected_path(self.current_path)
                else:
                    # No path found — flood
                    out_port = ofproto.OFPP_FLOOD
        else:
            # Destination unknown — flood
            out_port = ofproto.OFPP_FLOOD

        # Send the buffered packet
        if dst_mac not in self.host_table or \
                (dst_mac in self.host_table and
                 self.host_table[dst_mac][0] == dpid):
            actions = [parser.OFPActionOutput(
                self.mac_to_port[dpid].get(dst_mac, ofproto.OFPP_FLOOD)
            )]
            data = msg.data if msg.buffer_id == ofproto.OFP_NO_BUFFER else None
            out = parser.OFPPacketOut(
                datapath=datapath,
                buffer_id=msg.buffer_id,
                in_port=in_port,
                actions=actions,
                data=data
            )
            datapath.send_msg(out)

    # =========================================================
    # Statistics Collection
    # =========================================================

    @set_ev_cls(ofp_event.EventOFPPortStatsReply, MAIN_DISPATCHER)
    def port_stats_reply_handler(self, ev):
        """Handle port statistics reply from switches."""
        body = ev.msg.body
        dpid = ev.msg.datapath.id

        port_stats_list = []
        for stat in sorted(body, key=lambda x: x.port_no):
            port_stats_list.append({
                'port_no': stat.port_no,
                'rx_bytes': stat.rx_bytes,
                'tx_bytes': stat.tx_bytes,
                'rx_packets': stat.rx_packets,
                'tx_packets': stat.tx_packets,
                'rx_errors': stat.rx_errors,
                'tx_errors': stat.tx_errors,
                'rx_dropped': stat.rx_dropped,
                'tx_dropped': stat.tx_dropped,
            })

        # Update stats collector
        self.stats_collector.update_port_stats(dpid, port_stats_list)

        # Calculate performance metrics from aggregated stats
        total_tx = sum(s['tx_packets'] for s in port_stats_list)
        total_dropped = sum(s['rx_dropped'] + s['tx_dropped']
                           for s in port_stats_list)
        total_bytes = sum(s['tx_bytes'] for s in port_stats_list)

        # Estimate packet loss
        packet_loss = (total_dropped / max(total_tx, 1))

        # Estimate throughput in Mbps
        throughput = (total_bytes * 8) / 1e6  # cumulative, not rate

        # Simple delay estimation (placeholder — real delay needs ping)
        delay = 2.0 + (packet_loss * 50)  # Base 2ms + congestion penalty

        self.stats_collector.update_performance_metrics(delay, packet_loss,
                                                        throughput)

    @set_ev_cls(ofp_event.EventOFPFlowStatsReply, MAIN_DISPATCHER)
    def flow_stats_reply_handler(self, ev):
        """Handle flow statistics reply from switches."""
        body = ev.msg.body
        dpid = ev.msg.datapath.id

        self.flow_stats[dpid] = []
        for stat in body:
            self.flow_stats[dpid].append({
                'priority': stat.priority,
                'packet_count': stat.packet_count,
                'byte_count': stat.byte_count,
                'duration_sec': stat.duration_sec,
            })

    # =========================================================
    # Topology Discovery
    # =========================================================

    @set_ev_cls(topo_event.EventSwitchEnter)
    def switch_enter_handler(self, ev):
        """Handle switch entry — update topology graph."""
        self._update_topology()

    @set_ev_cls(topo_event.EventSwitchLeave)
    def switch_leave_handler(self, ev):
        """Handle switch leave — update topology graph."""
        self._update_topology()

    @set_ev_cls(topo_event.EventLinkAdd)
    def link_add_handler(self, ev):
        """Handle link addition — update topology graph."""
        self._update_topology()

    @set_ev_cls(topo_event.EventLinkDelete)
    def link_delete_handler(self, ev):
        """Handle link deletion — update topology graph."""
        self._update_topology()

    def _update_topology(self):
        """Rebuild the network topology graph from current switch/link state."""
        self.net_graph.clear()

        # Add switches
        switches = get_switch(self, None)
        for sw in switches:
            dpid = sw.dp.id
            self.net_graph.add_node(dpid, type='switch', name=f's{dpid}')

        # Add links
        links = get_link(self, None)
        for link in links:
            src_dpid = link.src.dpid
            dst_dpid = link.dst.dpid
            src_port = link.src.port_no
            dst_port = link.dst.port_no

            self.net_graph.add_edge(
                src_dpid, dst_dpid,
                src_port=src_port,
                dst_port=dst_port,
                weight=1
            )

        self.logger.info(
            f"Topology updated: {self.net_graph.number_of_nodes()} switches, "
            f"{self.net_graph.number_of_edges()} links"
        )

    # =========================================================
    # Routing — DRL Integration
    # =========================================================

    def _get_routing_path(self, src_dpid, dst_dpid):
        """
        Get routing path between two switches.

        First tries to read a decision from the DRL agent (via shared file).
        Falls back to shortest path if DRL decision is not available.

        Args:
            src_dpid: Source switch datapath ID
            dst_dpid: Destination switch datapath ID

        Returns:
            List of switch DPIDs along the path
        """
        # Try DRL agent decision
        path = self._read_drl_decision(src_dpid, dst_dpid)
        if path:
            self.logger.info(f"Using DRL path: {path}")
            return path

        # Fallback to shortest path
        return self._shortest_path(src_dpid, dst_dpid)

    def _read_drl_decision(self, src_dpid, dst_dpid):
        """
        Read the DRL agent's routing decision from the shared file.

        The DRL agent writes its decision to data/drl_decision.json.

        Returns:
            List of DPIDs if valid decision exists, None otherwise
        """
        try:
            if os.path.exists(DRL_DECISION_FILE):
                with open(DRL_DECISION_FILE, 'r') as f:
                    decision = json.load(f)

                # Check if decision matches our request
                if (decision.get('src_dpid') == src_dpid and
                        decision.get('dst_dpid') == dst_dpid and
                        time.time() - decision.get('timestamp', 0) < 10):
                    return decision.get('path', None)
        except (json.JSONDecodeError, FileNotFoundError):
            pass

        return None

    def _shortest_path(self, src_dpid, dst_dpid):
        """
        Compute shortest path using Dijkstra.

        Args:
            src_dpid: Source switch DPID
            dst_dpid: Destination switch DPID

        Returns:
            List of switch DPIDs, or None if no path exists
        """
        try:
            if self.net_graph.has_node(src_dpid) and \
                    self.net_graph.has_node(dst_dpid):
                return nx.shortest_path(self.net_graph, src_dpid, dst_dpid)
        except nx.NetworkXNoPath:
            self.logger.warning(
                f"No path found: s{src_dpid} -> s{dst_dpid}"
            )
        return None

    # =========================================================
    # Flow Rule Installation
    # =========================================================

    def _add_flow(self, datapath, priority, match, actions,
                  idle_timeout=0, hard_timeout=0):
        """
        Add a flow entry to a switch.

        Args:
            datapath: Switch datapath object
            priority: Flow priority
            match: OFPMatch object
            actions: List of OFPAction objects
            idle_timeout: Idle timeout in seconds
            hard_timeout: Hard timeout in seconds
        """
        ofproto = datapath.ofproto
        parser = datapath.ofproto_parser

        inst = [parser.OFPInstructionActions(
            ofproto.OFPIT_APPLY_ACTIONS, actions
        )]

        mod = parser.OFPFlowMod(
            datapath=datapath,
            priority=priority,
            match=match,
            instructions=inst,
            idle_timeout=idle_timeout,
            hard_timeout=hard_timeout
        )
        datapath.send_msg(mod)

    def _install_path(self, path, src_mac, dst_mac, first_port, last_port):
        """
        Install flow rules along a path between two hosts.

        For each switch in the path, installs a flow rule matching
        on src/dst MAC to forward to the next hop's port.

        Args:
            path: List of switch DPIDs (e.g., [1, 2, 4])
            src_mac: Source host MAC address
            dst_mac: Destination host MAC address
            first_port: Ingress port on the first switch
            last_port: Egress port on the last switch (to host)
        """
        for i, dpid in enumerate(path):
            if dpid not in self.datapaths:
                self.logger.warning(f"Switch s{dpid} not found in datapaths")
                continue

            datapath = self.datapaths[dpid]
            parser = datapath.ofproto_parser

            # Determine output port
            if i == len(path) - 1:
                # Last switch — forward to destination host port
                out_port = last_port
            else:
                # Intermediate — forward to next switch's port
                next_dpid = path[i + 1]
                edge_data = self.net_graph.get_edge_data(dpid, next_dpid)
                if edge_data:
                    out_port = edge_data.get('src_port', 1)
                else:
                    self.logger.warning(
                        f"No edge data: s{dpid} -> s{next_dpid}"
                    )
                    continue

            actions = [parser.OFPActionOutput(out_port)]
            match = parser.OFPMatch(eth_src=src_mac, eth_dst=dst_mac)

            self._add_flow(datapath, 1, match, actions,
                           idle_timeout=30, hard_timeout=120)

    # =========================================================
    # Monitoring Loop
    # =========================================================

    def _monitor_loop(self):
        """
        Background monitoring loop.

        Requests port and flow statistics from all switches every second.
        Also triggers the stats collector to write to JSON.
        """
        # Start the stats collector periodic writer
        self.stats_collector.start_periodic_write(interval=1.0)

        while True:
            for dpid, datapath in self.datapaths.items():
                self._request_stats(datapath)
            hub.sleep(1)

    def _request_stats(self, datapath):
        """
        Request port and flow statistics from a switch.

        Args:
            datapath: Switch datapath to query
        """
        parser = datapath.ofproto_parser

        # Request port stats
        req = parser.OFPPortStatsRequest(datapath, 0, datapath.ofproto.OFPP_ANY)
        datapath.send_msg(req)

        # Request flow stats
        req = parser.OFPFlowStatsRequest(datapath)
        datapath.send_msg(req)

    # =========================================================
    # Network State API (for DRL agent)
    # =========================================================

    def get_network_state(self):
        """
        Get current network state for the DRL agent.

        Returns:
            Dictionary with link utilizations, delays, packet counts,
            queue sizes, and topology information.
        """
        return self.stats_collector.get_stats_dict()


# Standalone execution
if __name__ == '__main__':
    from ryu.cmd import manager

    sys.argv.append('--ofp-tcp-listen-port')
    sys.argv.append('6633')
    sys.argv.append('--observe-links')
    sys.argv.append(__file__)

    manager.main()
