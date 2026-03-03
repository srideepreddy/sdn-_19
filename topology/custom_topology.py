"""
Custom Mininet Topology for SDN DRL Routing.

Topology: 4 switches, 6 hosts, multiple redundant paths with variable bandwidths.

Architecture:
      h1  h2          h3
       \\  /            |
       s1 ----100M---- s2
       |  \\___50M___/  |
      10M            10M
       |  /---50M---\\  |
       s3 ----100M---- s4
       |               / \\
       h4            h5   h6

Switch Connections (redundant paths):
  s1 -- s2 : 100 Mbps (primary backbone)
  s1 -- s3 : 10 Mbps  (low-bandwidth alternate)
  s1 -- s4 : 50 Mbps  (mid-bandwidth cross-link)
  s2 -- s3 : 50 Mbps  (mid-bandwidth cross-link)
  s2 -- s4 : 10 Mbps  (low-bandwidth alternate)
  s3 -- s4 : 100 Mbps (primary backbone)

Host Assignments:
  h1 -> s1, h2 -> s1
  h3 -> s2
  h4 -> s3
  h5 -> s4, h6 -> s4

Usage:
  sudo python custom_topology.py
  or
  sudo mn --custom custom_topology.py --topo sdndrl
"""

from mininet.topo import Topo
from mininet.net import Mininet
from mininet.node import RemoteController, OVSSwitch
from mininet.link import TCLink
from mininet.cli import CLI
from mininet.log import setLogLevel, info
import sys
import os


class SDNDRLTopology(Topo):
    """
    Custom topology with 4 switches, 6 hosts, and redundant paths.

    Multiple paths exist between any pair of hosts, with different
    bandwidths to create meaningful routing decisions for the DRL agent.
    """

    def build(self):
        """Build the network topology."""

        info("*** Creating SDN DRL Custom Topology\n")

        # =============================================
        # Create 4 OpenFlow Switches
        # =============================================
        s1 = self.addSwitch('s1', dpid='0000000000000001', protocols='OpenFlow13')
        s2 = self.addSwitch('s2', dpid='0000000000000002', protocols='OpenFlow13')
        s3 = self.addSwitch('s3', dpid='0000000000000003', protocols='OpenFlow13')
        s4 = self.addSwitch('s4', dpid='0000000000000004', protocols='OpenFlow13')

        info("*** Added 4 switches: s1, s2, s3, s4\n")

        # =============================================
        # Create 6 Hosts
        # =============================================
        h1 = self.addHost('h1', ip='10.0.0.1/24', mac='00:00:00:00:00:01')
        h2 = self.addHost('h2', ip='10.0.0.2/24', mac='00:00:00:00:00:02')
        h3 = self.addHost('h3', ip='10.0.0.3/24', mac='00:00:00:00:00:03')
        h4 = self.addHost('h4', ip='10.0.0.4/24', mac='00:00:00:00:00:04')
        h5 = self.addHost('h5', ip='10.0.0.5/24', mac='00:00:00:00:00:05')
        h6 = self.addHost('h6', ip='10.0.0.6/24', mac='00:00:00:00:00:06')

        info("*** Added 6 hosts: h1-h6\n")

        # =============================================
        # Host-to-Switch Links (1 Gbps, low delay)
        # =============================================
        self.addLink(h1, s1, bw=1000, delay='1ms', loss=0, use_htb=True)
        self.addLink(h2, s1, bw=1000, delay='1ms', loss=0, use_htb=True)
        self.addLink(h3, s2, bw=1000, delay='1ms', loss=0, use_htb=True)
        self.addLink(h4, s3, bw=1000, delay='1ms', loss=0, use_htb=True)
        self.addLink(h5, s4, bw=1000, delay='1ms', loss=0, use_htb=True)
        self.addLink(h6, s4, bw=1000, delay='1ms', loss=0, use_htb=True)

        info("*** Added 6 host-switch links\n")

        # =============================================
        # Switch-to-Switch Links (Variable Bandwidth)
        # These create multiple redundant paths with
        # different capacities for DRL routing decisions
        # =============================================

        # Primary backbone links (high bandwidth)
        self.addLink(s1, s2, bw=100, delay='2ms', loss=0, use_htb=True)  # s1-s2: 100 Mbps
        self.addLink(s3, s4, bw=100, delay='2ms', loss=0, use_htb=True)  # s3-s4: 100 Mbps

        # Cross-links (medium bandwidth)
        self.addLink(s1, s4, bw=50, delay='5ms', loss=0, use_htb=True)   # s1-s4: 50 Mbps
        self.addLink(s2, s3, bw=50, delay='5ms', loss=0, use_htb=True)   # s2-s3: 50 Mbps

        # Alternate links (low bandwidth — creates congestion scenarios)
        self.addLink(s1, s3, bw=10, delay='10ms', loss=0, use_htb=True)  # s1-s3: 10 Mbps
        self.addLink(s2, s4, bw=10, delay='10ms', loss=0, use_htb=True)  # s2-s4: 10 Mbps

        info("*** Added 6 switch-switch links with variable bandwidths\n")
        info("*** Topology Summary:\n")
        info("***   Switches: 4 (s1-s4)\n")
        info("***   Hosts: 6 (h1-h6)\n")
        info("***   Links: 12 total (6 host-switch + 6 switch-switch)\n")
        info("***   Redundant paths available between all host pairs\n")


def create_network(controller_ip='127.0.0.1', controller_port=6633):
    """
    Create and return a Mininet network with the custom topology.

    Args:
        controller_ip: IP address of the Ryu controller
        controller_port: Port of the Ryu controller

    Returns:
        Configured Mininet network instance
    """
    topo = SDNDRLTopology()

    net = Mininet(
        topo=topo,
        controller=lambda name: RemoteController(
            name,
            ip=controller_ip,
            port=controller_port
        ),
        switch=OVSSwitch,
        link=TCLink,
        autoSetMacs=False,
        autoStaticArp=True
    )

    return net


def run_topology(controller_ip='127.0.0.1', controller_port=6633, cli=True):
    """
    Start the network topology.

    Args:
        controller_ip: Ryu controller IP
        controller_port: Ryu controller port
        cli: Whether to drop into Mininet CLI
    """
    setLogLevel('info')

    info("*** Starting SDN DRL Network\n")
    net = create_network(controller_ip, controller_port)
    net.start()

    info("*** Network started successfully\n")
    info("*** Testing basic connectivity...\n")
    net.pingAll()

    if cli:
        info("\n*** Entering Mininet CLI\n")
        info("*** Useful commands:\n")
        info("***   pingall            - Test connectivity\n")
        info("***   h1 ping h6         - Ping between hosts\n")
        info("***   iperf h1 h6        - Bandwidth test\n")
        info("***   h1 iperf -s &      - Start iperf server on h1\n")
        info("***   h6 iperf -c 10.0.0.1 -t 30  - Generate traffic\n")
        info("***   links              - Show all links\n")
        info("***   dump               - Show all nodes\n")
        CLI(net)

    net.stop()
    info("*** Network stopped\n")


# Register topology for --custom flag usage
topos = {'sdndrl': (lambda: SDNDRLTopology())}


if __name__ == '__main__':
    """Run standalone for testing."""
    controller_ip = sys.argv[1] if len(sys.argv) > 1 else '127.0.0.1'
    controller_port = int(sys.argv[2]) if len(sys.argv) > 2 else 6633
    run_topology(controller_ip, controller_port)
