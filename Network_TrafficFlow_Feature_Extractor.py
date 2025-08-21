#!/usr/bin/env python3
"""
CICIDS2017 Traffic Flow Feature Extractor
==========================================

This program extracts network traffic flow features from large CICIDS2017 PCAP files
for zero-day cyberattack detection. Optimized for 10+ GB files with memory management.
"""

import os
import sys
import time
import argparse
import logging
from datetime import datetime
from pathlib import Path
from collections import defaultdict, Counter
import gc

import numpy as np
import pandas as pd
from scapy.all import PcapReader, IP, TCP, UDP, ICMP
import psutil
import warnings

# Import progress bar library with error handling
try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False
    print("Warning: tqdm not available. Progress bars will be disabled.")
    # Create dummy tqdm class for compatibility
    class tqdm:
        def __init__(self, iterable=None, desc=None, unit=None):
            self.iterable = iterable
            self.desc = desc
            self.unit = unit
        
        def __iter__(self):
            if self.iterable:
                return iter(self.iterable)
            return iter([])
        
        def update(self, n=1):
            pass
        
        def close(self):
            pass

# Import visualization libraries with error handling
try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    plt.style.use('dark_background')  # Dark theme for cybersecurity aesthetic
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    print("Warning: matplotlib/seaborn not available. Static plots will be disabled.")

try:
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    print("Warning: plotly not available. Interactive dashboard will be disabled.")

try:
    import networkx as nx
    NETWORKX_AVAILABLE = True
except ImportError:
    NETWORKX_AVAILABLE = False
    print("Warning: networkx not available. Network topology graph will be disabled.")

from collections import Counter

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

class CICIDSPacketExtractor:
    """
    Extracts features from every individual packet in CICIDS2017 PCAP files.
    Designed to handle large files (10+ GB) efficiently with per-packet analysis.
    """
    
    def __init__(self, chunk_size=100000, enable_visualizations=True, packet_analysis_mode=True):
        """
        Initialize the packet extractor.
        
        Args:
            chunk_size (int): Number of packets to process in each chunk
            enable_visualizations (bool): Whether to generate visual outputs
            packet_analysis_mode (bool): If True, analyze every packet individually
        """
        # Configuration parameters - optimized for performance
        self.chunk_size = chunk_size
        self.enable_visualizations = enable_visualizations
        self.packet_analysis_mode = packet_analysis_mode
        
        # Flow storage - tracks network conversations (for flow context)
        self.flows = defaultdict(self._create_flow_dict)
        
        # Feature definitions - list of all features we extract
        if packet_analysis_mode:
            self.feature_names = self._get_packet_feature_names()
        else:
            self.feature_names = self._get_flow_feature_names()
        
        # Data collection for visualizations (with limits for performance)
        self.packet_timeline = []  # Limited to 5000 samples for performance
        self.protocol_counts = Counter()  # Count of each protocol type
        self.ip_connections = defaultdict(set)  # Limited to 1000 IPs for performance
        self.port_activity = Counter()  # Port usage statistics
        self.packet_sizes = []  # Limited to 10000 samples for performance
        self.flow_durations = []  # List of flow durations
        
        # Enhanced data collection for new features
        self.http_requests = []  # HTTP request tracking
        self.tls_handshakes = []  # TLS handshake tracking
        self.burst_patterns = []  # Traffic burst analysis
        self.bandwidth_timeline = []  # Bandwidth tracking
        self.connection_events = []  # Connection start/end events
        self.active_sessions = set()  # Currently active sessions
        
        # Performance tracking
        self.last_cleanup_time = 0
        self.visualization_sample_rate = 1000  # Sample every 1000th packet for visualization
        self.packet_sample_counter = 0
        
        # Packet-level analysis storage
        self.packet_features = []  # Store features for each packet
        self.packet_metadata = []  # Store metadata for each packet
        
        # Setup logging system
        self._setup_logging()
        self.logger = logging.getLogger(__name__)
        
    def _setup_logging(self):
        """Setup logging configuration for tracking processing."""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('cicids_extraction.log'),
                logging.StreamHandler(sys.stdout)
            ]
        )
        
    def _create_flow_dict(self):
        """Create a new flow dictionary with default values for tracking network flows."""
        return {
            # Additional flow identification
            'src_ip': '',  # Source IP address
            'dst_ip': '',  # Destination IP address
            'src_port': 0,  # Source port number
            'dst_port': 0,  # Destination port number
            'protocol': '',  # Protocol type (TCP/UDP/etc)
            
            # Timing information for flow analysis
            'start_time': 0,  # When flow started
            'end_time': 0,  # When flow ended
            'duration': 0,  # Total flow duration
            
            # Forward direction statistics (source to destination)
            'fwd_packets': 0,  # Number of forward packets
            'fwd_bytes': 0,  # Total bytes in forward direction
            'fwd_packet_sizes': [],  # List of forward packet sizes
            'fwd_iat': [],  # Inter-arrival times for forward packets
            
            # Backward direction statistics (destination to source)
            'bwd_packets': 0,  # Number of backward packets
            'bwd_bytes': 0,  # Total bytes in backward direction
            'bwd_packet_sizes': [],  # List of backward packet sizes
            'bwd_iat': [],  # Inter-arrival times for backward packets
            
            # TCP flag counts for protocol analysis
            'fin_flags': 0,  # FIN flag count
            'syn_flags': 0,  # SYN flag count
            'rst_flags': 0,  # RST flag count
            'psh_flags': 0,  # PSH flag count
            'ack_flags': 0,  # ACK flag count
            'urg_flags': 0,  # URG flag count
            'ece_flags': 0,  # ECE flag count
            'cwr_flags': 0,  # CWR flag count
            
            # Enhanced temporal analysis
            'active_periods': [],  # Periods of activity
            'idle_periods': [],  # Periods of inactivity
            'last_packet_time': 0,  # Timestamp of last packet
            'burst_count': 0,  # Number of traffic bursts detected
            'peak_bandwidth': 0,  # Maximum bandwidth observed
            'connection_frequency': 0,  # Rate of new connections
            
            # Application layer analysis
            'http_requests': 0,  # Number of HTTP requests
            'tls_handshakes': 0,  # Number of TLS handshakes
            'dns_queries': 0,  # Number of DNS queries
            
            # Session tracking
            'session_overlap_count': 0,  # Number of overlapping sessions
            'time_to_first_byte': 0,  # Response time metric
            'traffic_variance': 0,  # Traffic pattern variance
        }
    
    def _get_packet_feature_names(self):
        """Define the packet-level feature names that will be extracted from each packet."""
        return [
            # Basic packet information (8 features)
            'packet_id',  # Sequential packet number
            'timestamp',  # Packet timestamp
            'packet_size',  # Total packet size in bytes
            'header_size',  # Size of headers
            'payload_size',  # Size of payload data
            'protocol_type',  # Protocol (TCP=1, UDP=2, ICMP=3, OTHER=0)
            'has_payload',  # Whether packet has payload (1/0)
            'is_fragmented',  # Whether packet is fragmented (1/0)
            
            # IP layer features (10 features)
            'ip_version',  # IP version (4 or 6)
            'ip_header_length',  # IP header length
            'ip_tos',  # Type of service
            'ip_total_length',  # Total IP packet length
            'ip_identification',  # IP identification field
            'ip_flags',  # IP flags
            'ip_fragment_offset',  # Fragment offset
            'ip_ttl',  # Time to live
            'ip_protocol',  # IP protocol number
            'ip_checksum',  # IP header checksum
            
            # TCP/UDP specific features (12 features)
            'src_port',  # Source port
            'dst_port',  # Destination port
            'tcp_seq_num',  # TCP sequence number (0 for UDP)
            'tcp_ack_num',  # TCP acknowledgment number (0 for UDP)
            'tcp_window_size',  # TCP window size (0 for UDP)
            'tcp_flags',  # TCP flags combined (0 for UDP)
            'tcp_urgent_pointer',  # TCP urgent pointer (0 for UDP)
            'tcp_options_length',  # Length of TCP options (0 for UDP)
            'udp_length',  # UDP length (0 for TCP)
            'udp_checksum',  # UDP checksum (0 for TCP)
            'transport_checksum',  # Transport layer checksum
            'transport_header_length',  # Transport header length
            
            # Flow context features (8 features)
            'flow_packet_count',  # Number of packets in this flow so far
            'flow_byte_count',  # Number of bytes in this flow so far
            'flow_duration',  # Flow duration up to this packet
            'packet_direction',  # Direction in flow (1=forward, 0=backward)
            'time_since_last_packet',  # Time since last packet in flow
            'packet_rate',  # Current packet rate in flow
            'byte_rate',  # Current byte rate in flow
            'is_flow_start',  # Whether this is first packet in flow (1/0)
            
            # Enhanced application layer features (5 NEW features)
            'http_request_indicator',  # Whether this packet contains HTTP request (1/0)
            'tls_handshake_indicator',  # Whether this packet is TLS handshake (1/0)
            'dns_query_indicator',  # Whether this packet is DNS query (1/0)
            'application_data_ratio',  # Ratio of application data to total packet
            'content_type_indicator',  # Type of content detected (0-5)
            
            # Enhanced temporal pattern features (7 NEW features)
            'burst_indicator',  # Whether packet is part of burst (1/0)
            'idle_period_duration',  # Duration of idle period before this packet
            'peak_bandwidth_ratio',  # Current bandwidth vs peak bandwidth
            'traffic_variance_score',  # Local traffic variance score
            'connection_frequency_score',  # Connection establishment frequency
            'session_overlap_indicator',  # Whether overlapping sessions exist (1/0)
            'time_to_first_byte',  # Response time indicator for this flow
            
            # Original statistical features (8 features)
            'packet_size_ratio',  # This packet size / average packet size in flow
            'inter_arrival_time',  # Time since previous packet in flow
            'payload_entropy',  # Entropy of payload data (if available)
            'header_anomaly_score',  # Anomaly score for header values
            'port_reputation_score',  # Reputation score for destination port
            'packet_frequency_score',  # Frequency score in current time window
            'size_deviation_score',  # How much this packet deviates from flow average
            'timing_regularity_score',  # Regularity of packet timing in flow
        ]
    
    def _get_flow_feature_names(self):
        """Define the original flow-level feature names for backward compatibility."""
        return [
            # Basic flow statistics (8 features)
            'total_packets',  # Total number of packets in flow
            'total_bytes',  # Total bytes transferred
            'duration',  # Flow duration in seconds
            'packets_per_second',  # Rate of packets
            'bytes_per_second',  # Rate of bytes
            'avg_packet_size',  # Average packet size
            'flow_direction_ratio',  # Ratio of forward to total bytes
            'inter_arrival_time_avg',  # Average time between packets
            
            # Protocol distribution (5 features)
            'tcp_percentage',  # Whether flow uses TCP
            'udp_percentage',  # Whether flow uses UDP
            'icmp_percentage',  # Whether flow uses ICMP
            'fwd_packets_ratio',  # Ratio of forward packets
            'bwd_packets_ratio',  # Ratio of backward packets
            
            # Temporal patterns (7+ features)
            'packet_size_variance',  # Variation in packet sizes
            'iat_variance',  # Variation in inter-arrival times
            'active_periods_count',  # Number of active periods
            'idle_periods_count',  # Number of idle periods
            'max_packet_size',  # Largest packet in flow
            'min_packet_size',  # Smallest packet in flow
            'flow_bytes_per_sec',  # Bytes per second rate
            
            # Additional flow characteristics (5+ features)
            'syn_flag_count',  # Number of SYN flags
            'fin_flag_count',  # Number of FIN flags
            'rst_flag_count',  # Number of RST flags
            'ack_flag_count',  # Number of ACK flags
            'avg_fwd_packet_size',  # Average forward packet size
            'avg_bwd_packet_size',  # Average backward packet size
            'fwd_iat_mean',  # Mean forward inter-arrival time
            'bwd_iat_mean'  # Mean backward inter-arrival time
        ]
    
    def get_flow_key(self, packet):
        """
        Generate a unique flow key from packet information.
        This creates a unique identifier for each network conversation.
        
        Args:
            packet: Scapy packet object
            
        Returns:
            tuple: Flow identifier (src_ip, dst_ip, src_port, dst_port, protocol) or None
        """
        # Check if packet has IP layer
        if not packet.haslayer(IP):
            return None
            
        try:
            # Extract IP layer information
            ip_layer = packet[IP]
            src_ip = str(ip_layer.src)  # Convert to string to avoid unhashable type error
            dst_ip = str(ip_layer.dst)  # Convert to string to avoid unhashable type error
            protocol = int(ip_layer.proto)
            
            # Extract port information based on protocol
            src_port = dst_port = 0
            if packet.haslayer(TCP):
                src_port = int(packet[TCP].sport)
                dst_port = int(packet[TCP].dport)
            elif packet.haslayer(UDP):
                src_port = int(packet[UDP].sport)
                dst_port = int(packet[UDP].dport)
                
            # Create bidirectional flow key (normalize direction for consistency)
            flow_tuple = tuple(sorted([(src_ip, src_port), (dst_ip, dst_port)]))
            return (*flow_tuple[0], *flow_tuple[1], protocol)
            
        except Exception as e:
            # Log error and return None if packet parsing fails
            return None
    
    def process_packet(self, packet, packet_time, packet_id=0):
        """
        Process a single packet and extract features.
        This function handles both packet-level and flow-level analysis.
        
        Args:
            packet: Scapy packet object
            packet_time: Packet timestamp
            packet_id: Sequential packet number
        """
        # Get unique flow identifier
        flow_key = self.get_flow_key(packet)
        if not flow_key:
            return
            
        # Get or create flow record
        flow = self.flows[flow_key]
        
        # In packet analysis mode, extract features from this individual packet
        if self.packet_analysis_mode:
            try:
                # Extract packet features
                packet_features = self.extract_packet_features(packet, packet_time, packet_id, flow_key, flow)
                self.packet_features.append(packet_features)
                
                # Store packet metadata
                packet_metadata = {
                    'packet_id': packet_id,
                    'timestamp': packet_time,
                    'src_ip': str(packet[IP].src) if packet.haslayer(IP) else '',
                    'dst_ip': str(packet[IP].dst) if packet.haslayer(IP) else '',
                    'protocol': self._get_protocol_name(packet),
                    'packet_size': len(packet)
                }
                self.packet_metadata.append(packet_metadata)
                
            except Exception as e:
                self.logger.warning(f"Error extracting packet features for packet {packet_id}: {str(e)}")
        
        # Update flow statistics (for flow context)
        self._update_flow_statistics(packet, packet_time, flow)
        
        # Collect data for visualizations if enabled
        if self.enable_visualizations:
            self.collect_visualization_data(packet, packet_time)
    
    def _get_protocol_name(self, packet):
        """Get protocol name as string."""
        if packet.haslayer(TCP):
            return 'TCP'
        elif packet.haslayer(UDP):
            return 'UDP'
        elif packet.haslayer(ICMP):
            return 'ICMP'
        else:
            return 'OTHER'
    
    def _update_flow_statistics(self, packet, packet_time, flow):
        """Update flow statistics for context (used in packet analysis mode)."""
        try:
            # Initialize flow on first packet
            if flow['start_time'] == 0:
                flow['start_time'] = packet_time
                if packet.haslayer(IP):
                    flow['src_ip'] = str(packet[IP].src)  # Convert to string
                    flow['dst_ip'] = str(packet[IP].dst)  # Convert to string
                else:
                    flow['src_ip'] = ''
                    flow['dst_ip'] = ''
                
                # Set port and protocol information
                if packet.haslayer(TCP):
                    flow['src_port'] = int(packet[TCP].sport)
                    flow['dst_port'] = int(packet[TCP].dport)
                    flow['protocol'] = 'TCP'
                elif packet.haslayer(UDP):
                    flow['src_port'] = int(packet[UDP].sport)
                    flow['dst_port'] = int(packet[UDP].dport)
                    flow['protocol'] = 'UDP'
                else:
                    flow['src_port'] = 0
                    flow['dst_port'] = 0
                    flow['protocol'] = 'OTHER'
            
            # Update flow timing information
            flow['end_time'] = packet_time
            flow['duration'] = packet_time - flow['start_time']
            
            # Determine packet direction (forward or backward)
            is_forward = (packet.haslayer(IP) and 
                         str(packet[IP].src) == flow['src_ip'])
            packet_size = len(packet)
            
            # Update statistics based on packet direction
            if is_forward:
                flow['fwd_packets'] += 1
                flow['fwd_bytes'] += packet_size
                flow['fwd_packet_sizes'].append(packet_size)
                
                # Calculate inter-arrival time
                if flow['last_packet_time'] > 0:
                    iat = packet_time - flow['last_packet_time']
                    flow['fwd_iat'].append(iat)
            else:
                flow['bwd_packets'] += 1
                flow['bwd_bytes'] += packet_size
                flow['bwd_packet_sizes'].append(packet_size)
                
                # Calculate inter-arrival time
                if flow['last_packet_time'] > 0:
                    iat = packet_time - flow['last_packet_time']
                    flow['bwd_iat'].append(iat)
            
            # Update last packet time
            flow['last_packet_time'] = packet_time
            
            # Extract and count TCP flags if present
            if packet.haslayer(TCP):
                tcp_flags = packet[TCP].flags
                if tcp_flags & 0x01: flow['fin_flags'] += 1  # FIN flag
                if tcp_flags & 0x02: flow['syn_flags'] += 1  # SYN flag
                if tcp_flags & 0x04: flow['rst_flags'] += 1  # RST flag
                if tcp_flags & 0x08: flow['psh_flags'] += 1  # PSH flag
                if tcp_flags & 0x10: flow['ack_flags'] += 1  # ACK flag
                if tcp_flags & 0x20: flow['urg_flags'] += 1  # URG flag
                
        except Exception as e:
            # Continue processing even if flow update fails
            pass
    
    def extract_packet_features(self, packet, packet_time, packet_id, flow_key, flow):
        """
        Extract comprehensive features from a single packet.
        
        Args:
            packet: Scapy packet object
            packet_time: Packet timestamp
            packet_id: Sequential packet number
            flow_key: Flow identifier
            flow: Flow state dictionary
            
        Returns:
            list: Feature values for this packet
        """
        try:
            features = []
            
            # Basic packet information (8 features)
            packet_size = len(packet)
            header_size = len(packet) - len(packet.payload) if hasattr(packet, 'payload') else len(packet)
            payload_size = len(packet.payload) if hasattr(packet, 'payload') else 0
            
            features.extend([
                packet_id,  # packet_id
                packet_time,  # timestamp
                packet_size,  # packet_size
                header_size,  # header_size
                payload_size,  # payload_size
                self._get_protocol_type(packet),  # protocol_type
                1 if payload_size > 0 else 0,  # has_payload
                1 if packet.haslayer(IP) and (packet[IP].flags & 0x1) else 0,  # is_fragmented
            ])
            
            # IP layer features (10 features)
            if packet.haslayer(IP):
                ip = packet[IP]
                features.extend([
                    int(ip.version) if hasattr(ip, 'version') else 4,  # ip_version
                    int(ip.ihl * 4) if hasattr(ip, 'ihl') else 20,  # ip_header_length
                    int(ip.tos) if hasattr(ip, 'tos') else 0,  # ip_tos
                    int(ip.len) if hasattr(ip, 'len') else packet_size,  # ip_total_length
                    int(ip.id) if hasattr(ip, 'id') else 0,  # ip_identification
                    int(ip.flags) if hasattr(ip, 'flags') else 0,  # ip_flags
                    int(ip.frag) if hasattr(ip, 'frag') else 0,  # ip_fragment_offset
                    int(ip.ttl) if hasattr(ip, 'ttl') else 64,  # ip_ttl
                    int(ip.proto) if hasattr(ip, 'proto') else 0,  # ip_protocol
                    int(ip.chksum) if hasattr(ip, 'chksum') and ip.chksum else 0,  # ip_checksum
                ])
            else:
                # Default values for non-IP packets
                features.extend([4, 20, 0, packet_size, 0, 0, 0, 64, 0, 0])
            
            # TCP/UDP specific features (12 features)
            if packet.haslayer(TCP):
                tcp = packet[TCP]
                features.extend([
                    int(tcp.sport) if hasattr(tcp, 'sport') else 0,  # src_port
                    int(tcp.dport) if hasattr(tcp, 'dport') else 0,  # dst_port
                    int(tcp.seq) if hasattr(tcp, 'seq') else 0,  # tcp_seq_num
                    int(tcp.ack) if hasattr(tcp, 'ack') else 0,  # tcp_ack_num
                    int(tcp.window) if hasattr(tcp, 'window') else 0,  # tcp_window_size
                    int(tcp.flags) if hasattr(tcp, 'flags') else 0,  # tcp_flags
                    int(tcp.urgptr) if hasattr(tcp, 'urgptr') else 0,  # tcp_urgent_pointer
                    len(tcp.options) * 4 if hasattr(tcp, 'options') and tcp.options else 0,  # tcp_options_length
                    0,  # udp_length (not applicable)
                    0,  # udp_checksum (not applicable)
                    int(tcp.chksum) if hasattr(tcp, 'chksum') and tcp.chksum else 0,  # transport_checksum
                    int(tcp.dataofs * 4) if hasattr(tcp, 'dataofs') else 20,  # transport_header_length
                ])
            elif packet.haslayer(UDP):
                udp = packet[UDP]
                features.extend([
                    int(udp.sport) if hasattr(udp, 'sport') else 0,  # src_port
                    int(udp.dport) if hasattr(udp, 'dport') else 0,  # dst_port
                    0,  # tcp_seq_num (not applicable)
                    0,  # tcp_ack_num (not applicable)
                    0,  # tcp_window_size (not applicable)
                    0,  # tcp_flags (not applicable)
                    0,  # tcp_urgent_pointer (not applicable)
                    0,  # tcp_options_length (not applicable)
                    int(udp.len) if hasattr(udp, 'len') else packet_size - 20,  # udp_length
                    int(udp.chksum) if hasattr(udp, 'chksum') and udp.chksum else 0,  # udp_checksum
                    int(udp.chksum) if hasattr(udp, 'chksum') and udp.chksum else 0,  # transport_checksum
                    8,  # transport_header_length (UDP header is 8 bytes)
                ])
            else:
                # Default values for other protocols
                features.extend([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
            
            # Flow context features (8 features)
            total_packets = flow['fwd_packets'] + flow['bwd_packets']
            total_bytes = flow['fwd_bytes'] + flow['bwd_bytes']
            flow_duration = packet_time - flow['start_time'] if flow['start_time'] > 0 else 0
            
            # Determine packet direction
            is_forward = 1 if (packet.haslayer(IP) and 
                              str(packet[IP].src) == flow['src_ip']) else 0
            
            # Calculate timing metrics
            time_since_last = packet_time - flow['last_packet_time'] if flow['last_packet_time'] > 0 else 0
            packet_rate = total_packets / max(flow_duration, 0.001)
            byte_rate = total_bytes / max(flow_duration, 0.001)
            is_flow_start = 1 if total_packets == 0 else 0
            
            features.extend([
                total_packets + 1,  # flow_packet_count (including this packet)
                total_bytes + packet_size,  # flow_byte_count (including this packet)
                flow_duration,  # flow_duration
                is_forward,  # packet_direction
                time_since_last,  # time_since_last_packet
                packet_rate,  # packet_rate
                byte_rate,  # byte_rate
                is_flow_start,  # is_flow_start
            ])
            
            # Enhanced application layer features (5 NEW features)
            http_request_indicator = self._detect_http_request(packet)
            tls_handshake_indicator = self._detect_tls_handshake(packet)
            dns_query_indicator = self._detect_dns_query(packet)
            application_data_ratio = payload_size / max(packet_size, 1)
            content_type_indicator = self._detect_content_type(packet)
            
            features.extend([
                http_request_indicator,  # http_request_indicator
                tls_handshake_indicator,  # tls_handshake_indicator
                dns_query_indicator,  # dns_query_indicator
                application_data_ratio,  # application_data_ratio
                content_type_indicator,  # content_type_indicator
            ])
            
            # Enhanced temporal pattern features (7 NEW features)
            burst_indicator = self._detect_burst_pattern(packet_time, flow)
            idle_period_duration = time_since_last if time_since_last > 1.0 else 0
            peak_bandwidth_ratio = byte_rate / max(flow.get('peak_bandwidth', 1), 1)
            traffic_variance_score = self._calculate_traffic_variance(flow)
            connection_frequency_score = self._calculate_connection_frequency(packet_time)
            session_overlap_indicator = len(self.active_sessions) > 1
            time_to_first_byte_metric = self._calculate_ttfb(packet, flow)
            
            features.extend([
                burst_indicator,  # burst_indicator
                idle_period_duration,  # idle_period_duration
                peak_bandwidth_ratio,  # peak_bandwidth_ratio
                traffic_variance_score,  # traffic_variance_score
                connection_frequency_score,  # connection_frequency_score
                1 if session_overlap_indicator else 0,  # session_overlap_indicator
                time_to_first_byte_metric,  # time_to_first_byte
            ])
            
            # Original statistical features (8 features)
            avg_packet_size = total_bytes / max(total_packets, 1)
            packet_size_ratio = packet_size / max(avg_packet_size, 1)
            
            # Calculate payload entropy if payload exists
            payload_entropy = self._calculate_entropy(packet.payload) if hasattr(packet, 'payload') and len(packet.payload) > 0 else 0
            
            # Simple anomaly scores
            header_anomaly_score = self._calculate_header_anomaly_score(packet)
            port_reputation_score = self._calculate_port_reputation_score(packet)
            packet_frequency_score = 1.0  # Placeholder
            size_deviation_score = abs(packet_size - avg_packet_size) / max(avg_packet_size, 1)
            timing_regularity_score = 1.0  # Placeholder
            
            features.extend([
                packet_size_ratio,  # packet_size_ratio
                time_since_last,  # inter_arrival_time
                payload_entropy,  # payload_entropy
                header_anomaly_score,  # header_anomaly_score
                port_reputation_score,  # port_reputation_score
                packet_frequency_score,  # packet_frequency_score
                size_deviation_score,  # size_deviation_score
                timing_regularity_score,  # timing_regularity_score
            ])
            
            return features
            
        except Exception as e:
            # Return default features if extraction fails
            return [0] * len(self.feature_names)
    
    def _get_protocol_type(self, packet):
        """Get numeric protocol type."""
        if packet.haslayer(TCP):
            return 1
        elif packet.haslayer(UDP):
            return 2
        elif packet.haslayer(ICMP):
            return 3
        else:
            return 0
    
    def _calculate_entropy(self, data):
        """Calculate Shannon entropy of data."""
        if not data or len(data) == 0:
            return 0
        
        # Convert to bytes if not already
        if hasattr(data, 'load'):
            data = data.load
        elif isinstance(data, str):
            data = data.encode()
        
        # Count byte frequencies
        byte_counts = Counter(data)
        data_len = len(data)
        
        # Calculate entropy
        entropy = 0
        for count in byte_counts.values():
            probability = count / data_len
            if probability > 0:
                entropy -= probability * np.log2(probability)
        
        return entropy
    
    def _calculate_header_anomaly_score(self, packet):
        """Calculate simple header anomaly score."""
        anomaly_score = 0
        
        if packet.haslayer(IP):
            ip = packet[IP]
            # Check for unusual TTL values
            if ip.ttl < 10 or ip.ttl > 255:
                anomaly_score += 0.3
            # Check for unusual TOS values
            if ip.tos > 0:
                anomaly_score += 0.2
        
        if packet.haslayer(TCP):
            tcp = packet[TCP]
            # Check for unusual flag combinations
            if tcp.flags & 0x3F == 0:  # No flags set
                anomaly_score += 0.2
            if tcp.flags & 0x06 == 0x06:  # SYN+RST
                anomaly_score += 0.4
        
        return min(anomaly_score, 1.0)
    
    def _calculate_port_reputation_score(self, packet):
        """Calculate port reputation score (higher = more suspicious)."""
        well_known_ports = {20, 21, 22, 23, 25, 53, 80, 110, 143, 443, 993, 995}
        
        if packet.haslayer(TCP):
            port = packet[TCP].dport
        elif packet.haslayer(UDP):
            port = packet[UDP].dport
        else:
            return 0.5
        
        if port in well_known_ports:
            return 0.1  # Low suspicion for well-known ports
        elif port < 1024:
            return 0.3  # Medium suspicion for other system ports
        elif port > 49152:
            return 0.7  # Higher suspicion for dynamic ports
        else:
            return 0.5  # Medium suspicion for registered ports
    
    def collect_visualization_data(self, packet, packet_time):
        """
        Collect data for visualizations during packet processing.
        Optimized with sampling to reduce memory usage and improve performance.
        
        Args:
            packet: Scapy packet object
            packet_time: Packet timestamp
        """
        # Skip if packet doesn't have IP layer
        if not packet.haslayer(IP):
            return
            
        # Increment sample counter
        self.packet_sample_counter += 1
        
        # Sample only every Nth packet for performance
        if self.packet_sample_counter % self.visualization_sample_rate != 0:
            return
            
        # Collect timeline data (limited samples for performance)
        if len(self.packet_timeline) < 5000:
            self.packet_timeline.append({
                'time': packet_time,
                'size': len(packet),
                'protocol': packet[IP].proto
            })
        
        # Count protocol types
        if packet.haslayer(TCP):
            self.protocol_counts['TCP'] += 1
        elif packet.haslayer(UDP):
            self.protocol_counts['UDP'] += 1
        elif packet.haslayer(ICMP):
            self.protocol_counts['ICMP'] += 1
        else:
            self.protocol_counts['OTHER'] += 1
        
        # Track network connections (limited to prevent memory issues)
        src_ip = packet[IP].src
        dst_ip = packet[IP].dst
        if len(self.ip_connections) < 1000:  # Reduced limit for performance
            self.ip_connections[src_ip].add(dst_ip)
        
        # Count port activity
        if packet.haslayer(TCP):
            self.port_activity[packet[TCP].dport] += 1
        elif packet.haslayer(UDP):
            self.port_activity[packet[UDP].dport] += 1
        
        # Sample packet sizes (limited for performance)
        if len(self.packet_sizes) < 10000:  # Reduced limit for performance
            self.packet_sizes.append(len(packet))
    
    def cleanup_old_flows(self, current_time):
        """
        Remove flows that have timed out to manage memory.
        Optimized to reduce cleanup frequency for better performance.
        
        Args:
            current_time: Current packet timestamp
        """
        # Only cleanup every 5 minutes to reduce overhead
        if current_time - self.last_cleanup_time < 300:  # 5 minutes
            return
            
        self.last_cleanup_time = current_time
        
        # Calculate timeout threshold
        timeout_threshold = current_time - self.timeout_seconds
        
        # Find expired flows
        expired_flows = [
            flow_key for flow_key, flow in self.flows.items()
            if flow['end_time'] < timeout_threshold
        ]
        
        # Remove expired flows
        for flow_key in expired_flows:
            del self.flows[flow_key]
        
        # Log cleanup if flows were removed
        if expired_flows:
            self.logger.info(f"Cleaned up {len(expired_flows)} expired flows")
            
        # Force garbage collection after cleanup
        if len(expired_flows) > 1000:
            gc.collect()
    
    def get_memory_usage(self):
        """Get current memory usage in MB for monitoring."""
        process = psutil.Process(os.getpid())
        return process.memory_info().rss / 1024 / 1024
    
    def process_pcap_file(self, pcap_file_path):
        """
        Process a large PCAP file and extract traffic flow features.
        This is the main processing function that handles the entire PCAP file.
        
        Args:
            pcap_file_path (str): Path to the PCAP file
            
        Returns:
            pandas.DataFrame: Extracted features
        """
        self.logger.info(f"Starting to process PCAP file: {pcap_file_path}")
        
        # Validate file exists and get size information
        if not os.path.exists(pcap_file_path):
            raise FileNotFoundError(f"PCAP file not found: {pcap_file_path}")
        
        file_size_gb = os.path.getsize(pcap_file_path) / (1024**3)
        self.logger.info(f"File size: {file_size_gb:.2f} GB")
        
        # Initialize processing counters
        packet_count = 0
        chunk_count = 0
        start_time = time.time()
        
        try:
            # Open PCAP file with PcapReader for memory efficiency
            with PcapReader(pcap_file_path) as pcap_reader:
                
                # Simple progress tracking without tqdm to avoid errors
                print("Processing packets...")
                last_report_time = time.time()
                
                # Process each packet in the file
                packet_counter = 0
                packet_id = 0
                for packet in pcap_reader:
                    try:
                        # Process individual packet
                        packet_time = float(packet.time)
                        self.process_packet(packet, packet_time, packet_id)
                        
                        # Update counters and progress
                        packet_count += 1
                        packet_counter += 1
                        packet_id += 1
                        
                        # Simple progress reporting every 10,000 packets
                        if packet_counter % 10000 == 0:
                            current_time = time.time()
                            elapsed = current_time - last_report_time
                            rate = 10000 / elapsed if elapsed > 0 else 0
                            print(f"Processed {packet_count:,} packets (Rate: {rate:.0f} packets/sec)")
                            last_report_time = current_time
                        
                        # Periodic maintenance and status updates
                        if packet_count % self.chunk_size == 0:
                            chunk_count += 1
                            
                            # Monitor memory usage
                            memory_mb = self.get_memory_usage()
                            active_flows = len(self.flows)
                            packets_analyzed = len(self.packet_features) if self.packet_analysis_mode else "N/A"
                            
                            # Log progress information
                            if self.packet_analysis_mode:
                                self.logger.info(
                                    f"Chunk {chunk_count}: Processed {packet_count:,} packets, "
                                    f"Packet features: {packets_analyzed:,}, Active flows: {active_flows:,}, Memory: {memory_mb:.1f} MB"
                                )
                            else:
                                self.logger.info(
                                    f"Chunk {chunk_count}: Processed {packet_count:,} packets, "
                                    f"Active flows: {active_flows:,}, Memory: {memory_mb:.1f} MB"
                                )
                            
                            # Force garbage collection if memory usage is high
                            if memory_mb > 4000:  # 4GB threshold for packet analysis
                                gc.collect()
                                self.logger.info("Performed garbage collection")
                        
                    except Exception as e:
                        self.logger.warning(f"Error processing packet {packet_count}: {str(e)}")
                        continue
                
                print(f"Completed processing {packet_count:,} packets")
        
        except Exception as e:
            self.logger.error(f"Error reading PCAP file: {str(e)}")
            raise
        
        # Extract features from collected data
        if self.packet_analysis_mode:
            self.logger.info("Creating packet-level dataset...")
            
            # Create DataFrames from packet data
            if self.packet_features and self.packet_metadata:
                df_features = pd.DataFrame(self.packet_features, columns=self.feature_names)
                df_metadata = pd.DataFrame(self.packet_metadata)
                
                # Combine metadata and features
                result_df = pd.concat([df_metadata, df_features], axis=1)
                
                self.logger.info(f"Packet analysis completed!")
                self.logger.info(f"Total packets processed: {packet_count:,}")
                self.logger.info(f"Total packet features extracted: {len(result_df):,}")
            else:
                self.logger.warning("No packet features extracted!")
                result_df = pd.DataFrame()
                
        else:
            # Original flow-based analysis
            self.logger.info("Extracting features from flows...")
            feature_matrix = []
            flow_metadata = []
            
            # Process each flow to extract features - simple iteration without tqdm
            total_flows = len(self.flows)
            print(f"Extracting features from {total_flows:,} flows...")
            
            flow_counter = 0
            for flow_key, flow in self.flows.items():
                try:
                    # Extract numerical features using original flow method
                    features = self.extract_features_from_flow(flow)
                    feature_matrix.append(features)
                    
                    # Store flow metadata for analysis
                    flow_metadata.append({
                        'src_ip': flow['src_ip'],
                        'dst_ip': flow['dst_ip'],
                        'src_port': flow['src_port'],
                        'dst_port': flow['dst_port'],
                        'protocol': flow['protocol'],
                        'duration': flow['duration'],
                        'total_packets': flow['fwd_packets'] + flow['bwd_packets'],
                        'total_bytes': flow['fwd_bytes'] + flow['bwd_bytes']
                    })
                    
                    # Collect flow durations for visualization
                    if self.enable_visualizations:
                        self.flow_durations.append(flow['duration'])
                    
                    # Simple progress reporting
                    flow_counter += 1
                    if flow_counter % 1000 == 0:
                        print(f"Processed {flow_counter:,}/{total_flows:,} flows")
                    
                except Exception as e:
                    self.logger.warning(f"Error extracting features from flow: {str(e)}")
                    continue
            
            print(f"Completed processing {flow_counter:,} flows")
            
            # Create DataFrames from extracted data
            if feature_matrix:
                df_features = pd.DataFrame(feature_matrix, columns=self.feature_names)
                df_metadata = pd.DataFrame(flow_metadata)
                
                # Ensure no duplicate column names before concatenation
                if not df_metadata.empty and not df_features.empty:
                    # Check for overlapping columns and rename if necessary
                    overlapping_columns = set(df_metadata.columns) & set(df_features.columns)
                    if overlapping_columns:
                        self.logger.warning(f"Found overlapping columns: {overlapping_columns}")
                        for col in overlapping_columns:
                            if col in df_metadata.columns:
                                df_metadata = df_metadata.rename(columns={col: f"meta_{col}"})
                
                # Combine metadata and features into single DataFrame
                if not df_metadata.empty and not df_features.empty:
                    result_df = pd.concat([df_metadata, df_features], axis=1)
                elif not df_features.empty:
                    result_df = df_features
                else:
                    result_df = pd.DataFrame()  # Empty dataframe if no features extracted
            else:
                result_df = pd.DataFrame()
                
            # Log final flow processing statistics
            flows_extracted = len(result_df)
            self.logger.info(f"Processing completed!")
            self.logger.info(f"Total packets processed: {packet_count:,}")
            self.logger.info(f"Total flows extracted: {flows_extracted:,}")
        
        # Log final processing statistics
        processing_time = time.time() - start_time
        self.logger.info(f"Processing time: {processing_time:.2f} seconds")
        self.logger.info(f"Packets per second: {packet_count/processing_time:.1f}")
        
        return result_df
    
    def extract_features_from_flow(self, flow):
        """
        Extract numerical features from a single flow (for backward compatibility).
        This converts flow statistics into ML-ready numerical features.
        
        Args:
            flow (dict): Flow statistics dictionary
            
        Returns:
            list: Feature values for this flow
        """
        features = []
        
        # Calculate basic flow statistics (8 features)
        total_packets = flow['fwd_packets'] + flow['bwd_packets']
        total_bytes = flow['fwd_bytes'] + flow['bwd_bytes']
        duration = max(flow['duration'], 0.000001)  # Avoid division by zero
        
        # Add basic statistics to feature list
        features.extend([
            total_packets,                                    # total_packets
            total_bytes,                                      # total_bytes
            duration,                                         # duration
            total_packets / duration,                         # packets_per_second
            total_bytes / duration,                           # bytes_per_second
            total_bytes / max(total_packets, 1),              # avg_packet_size
            flow['fwd_bytes'] / max(total_bytes, 1),          # flow_direction_ratio
            np.mean(flow['fwd_iat'] + flow['bwd_iat']) if (flow['fwd_iat'] + flow['bwd_iat']) else 0  # inter_arrival_time_avg
        ])
        
        # Calculate protocol distribution features (5 features)
        features.extend([
            1.0 if flow['protocol'] == 'TCP' else 0.0,       # tcp_percentage
            1.0 if flow['protocol'] == 'UDP' else 0.0,       # udp_percentage
            1.0 if flow['protocol'] == 'ICMP' else 0.0,      # icmp_percentage
            flow['fwd_packets'] / max(total_packets, 1),      # fwd_packets_ratio
            flow['bwd_packets'] / max(total_packets, 1),      # bwd_packets_ratio
        ])
        
        # Calculate temporal pattern features (7+ features)
        all_packet_sizes = flow['fwd_packet_sizes'] + flow['bwd_packet_sizes']
        all_iats = flow['fwd_iat'] + flow['bwd_iat']
        
        features.extend([
            np.var(all_packet_sizes) if all_packet_sizes else 0,      # packet_size_variance
            np.var(all_iats) if all_iats else 0,                      # iat_variance
            len(flow['active_periods']),                              # active_periods_count
            len(flow['idle_periods']),                                # idle_periods_count
            max(all_packet_sizes) if all_packet_sizes else 0,        # max_packet_size
            min(all_packet_sizes) if all_packet_sizes else 0,        # min_packet_size
            total_bytes / duration,                                   # flow_bytes_per_sec
        ])
        
        # Calculate additional flow characteristics (5+ features)
        features.extend([
            flow['syn_flags'],                                        # syn_flag_count
            flow['fin_flags'],                                        # fin_flag_count
            flow['rst_flags'],                                        # rst_flag_count
            flow['ack_flags'],                                        # ack_flag_count
            np.mean(flow['fwd_packet_sizes']) if flow['fwd_packet_sizes'] else 0,  # avg_fwd_packet_size
            np.mean(flow['bwd_packet_sizes']) if flow['bwd_packet_sizes'] else 0,  # avg_bwd_packet_size
            np.mean(flow['fwd_iat']) if flow['fwd_iat'] else 0,       # fwd_iat_mean
            np.mean(flow['bwd_iat']) if flow['bwd_iat'] else 0,       # bwd_iat_mean
        ])
        
        return features
    
    def generate_visualizations(self, output_dir, features_df):
        """
        Generate comprehensive visualizations of the PCAP analysis.
        This creates charts and graphs to demonstrate PCAP processing capabilities.
        
        Args:
            output_dir (str): Directory to save visualization files
            features_df (pandas.DataFrame): Extracted features
        """
        if not self.enable_visualizations:
            return
            
        self.logger.info("Generating visualizations...")
        
        # Create visualizations subdirectory
        viz_dir = os.path.join(output_dir, "visualizations")
        os.makedirs(viz_dir, exist_ok=True)
        
        # Track number of successful visualizations
        viz_count = 0
        
        # Generate matplotlib-based visualizations
        if MATPLOTLIB_AVAILABLE:
            # Create protocol distribution chart
            if self._create_protocol_distribution_chart(viz_dir):
                viz_count += 1
            
            # Create traffic timeline chart
            if self._create_traffic_timeline(viz_dir):
                viz_count += 1
            
            # Create packet size distribution
            if self._create_packet_size_distribution(viz_dir):
                viz_count += 1
            
            # Create port activity heatmap
            if self._create_port_activity_heatmap(viz_dir):
                viz_count += 1
            
            # Create feature correlation matrix
            if self._create_feature_correlation_matrix(viz_dir, features_df):
                viz_count += 1
            
            # Create CNN-ready traffic heatmap
            if self._create_cnn_traffic_heatmap(viz_dir, features_df):
                viz_count += 1
        
        # Generate network topology if networkx is available
        if NETWORKX_AVAILABLE and MATPLOTLIB_AVAILABLE:
            if self._create_network_topology(viz_dir):
                viz_count += 1
        
        # Generate interactive dashboard if plotly is available
        if PLOTLY_AVAILABLE:
            if self._create_interactive_dashboard(viz_dir, features_df):
                viz_count += 1
        
        # Generate text-based summary if no visualization libraries available
        if viz_count == 0:
            self._create_text_summary(viz_dir, features_df)
            viz_count = 1
        
        self.logger.info(f"{viz_count} visualizations saved to: {viz_dir}")
    
    def _create_text_summary(self, viz_dir, features_df):
        """Create a text-based summary when visualization libraries aren't available."""
        try:
            summary_file = os.path.join(viz_dir, "analysis_summary.txt")
            
            with open(summary_file, 'w') as f:
                f.write("CICIDS2017 PCAP Analysis Summary\n")
                f.write("=" * 50 + "\n\n")
                
                # Write protocol distribution
                if self.protocol_counts:
                    f.write("Protocol Distribution:\n")
                    total_packets = sum(self.protocol_counts.values())
                    for protocol, count in self.protocol_counts.most_common():
                        percentage = (count / total_packets) * 100
                        f.write(f"   {protocol}: {count:,} packets ({percentage:.1f}%)\n")
                    f.write("\n")
                
                # Write packet size statistics
                if self.packet_sizes:
                    f.write("Packet Size Statistics:\n")
                    f.write(f"   Total packets analyzed: {len(self.packet_sizes):,}\n")
                    f.write(f"   Average size: {np.mean(self.packet_sizes):.1f} bytes\n")
                    f.write(f"   Median size: {np.median(self.packet_sizes):.1f} bytes\n")
                    f.write(f"   Min size: {min(self.packet_sizes)} bytes\n")
                    f.write(f"   Max size: {max(self.packet_sizes)} bytes\n")
                    f.write(f"   Standard deviation: {np.std(self.packet_sizes):.1f} bytes\n")
                    f.write("\n")
                
                # Write port activity information
                if self.port_activity:
                    f.write("Top 10 Most Active Ports:\n")
                    for i, (port, count) in enumerate(self.port_activity.most_common(10), 1):
                        f.write(f"   {i}. Port {port}: {count:,} packets\n")
                    f.write("\n")
                
                # Write network activity summary
                if self.ip_connections:
                    f.write("Network Activity:\n")
                    f.write(f"   Unique source IPs: {len(self.ip_connections)}\n")
                    total_connections = sum(len(dsts) for dsts in self.ip_connections.values())
                    f.write(f"   Total connections: {total_connections:,}\n")
                    f.write("\n")
                
                # Write feature summary
                if not features_df.empty:
                    f.write("Flow Feature Summary:\n")
                    f.write(f"   Total flows extracted: {len(features_df):,}\n")
                    
                    try:
                        # Safe conversion with error handling
                        if 'total_packets' in features_df.columns and not features_df['total_packets'].empty:
                            avg_packets = float(features_df['total_packets'].mean())
                            f.write(f"   Avg packets per flow: {avg_packets:.1f}\n")
                        
                        if 'total_bytes' in features_df.columns and not features_df['total_bytes'].empty:
                            avg_bytes = float(features_df['total_bytes'].mean())
                            f.write(f"   Avg bytes per flow: {avg_bytes:.1f}\n")
                        
                        if 'duration' in features_df.columns and not features_df['duration'].empty:
                            avg_duration = float(features_df['duration'].mean())
                            f.write(f"   Avg flow duration: {avg_duration:.2f} seconds\n")
                        
                        feature_count = len([col for col in features_df.columns if col not in ['src_ip', 'dst_ip', 'src_port', 'dst_port', 'protocol']])
                        f.write(f"\n   Features extracted: {feature_count}\n")
                        
                    except Exception as e:
                        f.write(f"   Error calculating statistics: {str(e)}\n")
                        f.write(f"   Dataset shape: {features_df.shape}\n")
                
                # Write installation instructions
                f.write("\n" + "=" * 50 + "\n")
                f.write("Install visualization libraries for charts:\n")
                f.write("   pip install matplotlib seaborn plotly networkx\n")
            
            return True
            
        except Exception as e:
            self.logger.warning(f"Error creating text summary: {str(e)}")
            return False
    
    def _create_protocol_distribution_chart(self, viz_dir):
        """Create protocol distribution pie chart and bar chart."""
        if not MATPLOTLIB_AVAILABLE:
            return False
            
        try:
            # Create figure with two subplots
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
            fig.suptitle('Network Protocol Analysis', fontsize=16, color='white')
            
            # Extract protocol data
            protocols = list(self.protocol_counts.keys())
            counts = list(self.protocol_counts.values())
            colors = ['#ff6b6b', '#4ecdc4', '#45b7d1', '#96ceb4']
            
            # Create pie chart
            wedges, texts, autotexts = ax1.pie(counts, labels=protocols, autopct='%1.1f%%', 
                                              colors=colors, startangle=90)
            ax1.set_title('Protocol Distribution', color='white')
            
            # Create bar chart
            ax2.bar(protocols, counts, color=colors[:len(protocols)])
            ax2.set_title('Protocol Packet Counts', color='white')
            ax2.set_ylabel('Packet Count', color='white')
            ax2.tick_params(colors='white')
            
            # Save chart
            plt.tight_layout()
            plt.savefig(os.path.join(viz_dir, 'protocol_distribution.png'), 
                       facecolor='black', dpi=300, bbox_inches='tight')
            plt.close()
            return True
            
        except Exception as e:
            self.logger.warning(f"Error creating protocol chart: {str(e)}")
            return False
    
    def _create_traffic_timeline(self, viz_dir):
        """Create traffic timeline visualization showing packets and bytes over time."""
        if not MATPLOTLIB_AVAILABLE:
            return False
            
        try:
            if not self.packet_timeline:
                return False
                
            # Convert timeline data to DataFrame
            df_timeline = pd.DataFrame(self.packet_timeline)
            df_timeline['time'] = pd.to_datetime(df_timeline['time'], unit='s')
            
            # Resample data to 1-second intervals
            df_resampled = df_timeline.set_index('time').resample('1S').agg({
                'size': ['count', 'sum'],
                'protocol': 'count'
            }).fillna(0)
            
            # Create figure with subplots
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10))
            fig.suptitle('Network Traffic Timeline', fontsize=16, color='white')
            
            # Plot packets per second
            ax1.plot(df_resampled.index, df_resampled[('size', 'count')], 
                    color='#ff6b6b', linewidth=2, alpha=0.8)
            ax1.fill_between(df_resampled.index, df_resampled[('size', 'count')], 
                           alpha=0.3, color='#ff6b6b')
            ax1.set_title('Packets per Second', color='white')
            ax1.set_ylabel('Packets/sec', color='white')
            ax1.tick_params(colors='white')
            ax1.grid(True, alpha=0.3)
            
            # Plot bytes per second
            ax2.plot(df_resampled.index, df_resampled[('size', 'sum')], 
                    color='#4ecdc4', linewidth=2, alpha=0.8)
            ax2.fill_between(df_resampled.index, df_resampled[('size', 'sum')], 
                           alpha=0.3, color='#4ecdc4')
            ax2.set_title('Bytes per Second', color='white')
            ax2.set_ylabel('Bytes/sec', color='white')
            ax2.set_xlabel('Time', color='white')
            ax2.tick_params(colors='white')
            ax2.grid(True, alpha=0.3)
            
            # Save chart
            plt.tight_layout()
            plt.savefig(os.path.join(viz_dir, 'traffic_timeline.png'), 
                       facecolor='black', dpi=300, bbox_inches='tight')
            plt.close()
            return True
            
        except Exception as e:
            self.logger.warning(f"Error creating timeline: {str(e)}")
            return False
    
    def _create_network_topology(self, viz_dir):
        """Create network topology graph showing IP connections."""
        if not (NETWORKX_AVAILABLE and MATPLOTLIB_AVAILABLE):
            return False
            
        try:
            if len(self.ip_connections) == 0:
                return False
                
            # Create network graph
            G = nx.Graph()
            
            # Add edges (connections between IPs) with limits for visualization
            edge_count = 0
            for src_ip, dst_ips in self.ip_connections.items():
                if edge_count > 500:  # Limit for performance
                    break
                for dst_ip in list(dst_ips)[:10]:  # Limit connections per IP
                    G.add_edge(src_ip, dst_ip)
                    edge_count += 1
            
            if len(G.nodes()) == 0:
                return False
                
            # Create visualization
            plt.figure(figsize=(15, 15))
            plt.title('Network Topology Graph', fontsize=16, color='white', pad=20)
            
            # Use spring layout for better visualization
            pos = nx.spring_layout(G, k=1, iterations=50)
            
            # Draw network nodes
            nx.draw_networkx_nodes(G, pos, node_color='#ff6b6b', 
                                 node_size=50, alpha=0.8)
            
            # Draw network edges
            nx.draw_networkx_edges(G, pos, edge_color='#4ecdc4', 
                                 alpha=0.5, width=0.5)
            
            # Add labels for major nodes only
            major_nodes = sorted(G.nodes(), key=lambda x: G.degree(x), reverse=True)[:10]
            labels = {node: node for node in major_nodes}
            nx.draw_networkx_labels(G, pos, labels, font_size=8, 
                                  font_color='white', alpha=0.8)
            
            # Remove axes and save
            plt.axis('off')
            plt.tight_layout()
            plt.savefig(os.path.join(viz_dir, 'network_topology.png'), 
                       facecolor='black', dpi=300, bbox_inches='tight')
            plt.close()
            return True
            
        except Exception as e:
            self.logger.warning(f"Error creating topology: {str(e)}")
            return False
    
    def _create_packet_size_distribution(self, viz_dir):
        """Create packet size distribution histogram and statistics."""
        if not MATPLOTLIB_AVAILABLE:
            return False
            
        try:
            if not self.packet_sizes:
                return False
                
            # Create figure with subplots
            plt.figure(figsize=(12, 8))
            plt.suptitle('Packet Size Distribution Analysis', fontsize=16, color='white')
            
            # Create subplot grid
            gs = plt.GridSpec(2, 2, hspace=0.3, wspace=0.3)
            
            # Create histogram
            ax1 = plt.subplot(gs[0, :])
            n, bins, patches = ax1.hist(self.packet_sizes, bins=50, color='#ff6b6b', 
                                       alpha=0.7, edgecolor='white', linewidth=0.5)
            ax1.set_title('Packet Size Histogram', color='white')
            ax1.set_xlabel('Packet Size (bytes)', color='white')
            ax1.set_ylabel('Frequency', color='white')
            ax1.tick_params(colors='white')
            ax1.grid(True, alpha=0.3)
            
            # Create box plot
            ax2 = plt.subplot(gs[1, 0])
            bp = ax2.boxplot(self.packet_sizes, patch_artist=True)
            bp['boxes'][0].set_facecolor('#4ecdc4')
            bp['boxes'][0].set_alpha(0.7)
            ax2.set_title('Size Distribution', color='white')
            ax2.set_ylabel('Packet Size (bytes)', color='white')
            ax2.tick_params(colors='white')
            ax2.grid(True, alpha=0.3)
            
            # Add statistics text
            ax3 = plt.subplot(gs[1, 1])
            stats_text = f"""
            Total Packets: {len(self.packet_sizes):,}
            Mean Size: {np.mean(self.packet_sizes):.1f} bytes
            Median Size: {np.median(self.packet_sizes):.1f} bytes
            Std Deviation: {np.std(self.packet_sizes):.1f} bytes
            Min Size: {min(self.packet_sizes)} bytes
            Max Size: {max(self.packet_sizes)} bytes
            """
            ax3.text(0.1, 0.5, stats_text, transform=ax3.transAxes, 
                    color='white', fontsize=10, verticalalignment='center',
                    bbox=dict(boxstyle="round,pad=0.3", facecolor='#333', alpha=0.8))
            ax3.set_title('Statistics', color='white')
            ax3.axis('off')
            
            # Save chart
            plt.tight_layout()
            plt.savefig(os.path.join(viz_dir, 'packet_size_distribution.png'), 
                       facecolor='black', dpi=300, bbox_inches='tight')
            plt.close()
            return True
            
        except Exception as e:
            self.logger.warning(f"Error creating size distribution: {str(e)}")
            return False
    
    def _create_port_activity_heatmap(self, viz_dir):
        """Create port activity analysis with multiple charts."""
        if not MATPLOTLIB_AVAILABLE:
            return False
            
        try:
            if not self.port_activity:
                return False
                
            # Get top 20 most active ports
            top_ports = dict(self.port_activity.most_common(20))
            
            # Categorize ports by type
            well_known = {p: c for p, c in top_ports.items() if p <= 1023}
            registered = {p: c for p, c in top_ports.items() if 1024 <= p <= 49151}
            dynamic = {p: c for p, c in top_ports.items() if p > 49151}
            
            # Create figure with subplots
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
            fig.suptitle('Port Activity Analysis', fontsize=16, color='white')
            
            # Create top ports bar chart
            ax1 = axes[0, 0]
            ports = list(top_ports.keys())
            counts = list(top_ports.values())
            bars = ax1.bar(range(len(ports)), counts, color='#ff6b6b', alpha=0.8)
            ax1.set_title('Top 20 Active Ports', color='white')
            ax1.set_xlabel('Port Rank', color='white')
            ax1.set_ylabel('Packet Count', color='white')
            ax1.tick_params(colors='white')
            ax1.grid(True, alpha=0.3)
            
            # Create port categories pie chart
            ax2 = axes[0, 1]
            categories = ['Well-Known (≤1023)', 'Registered (1024-49151)', 'Dynamic (>49151)']
            cat_counts = [sum(well_known.values()), sum(registered.values()), sum(dynamic.values())]
            colors = ['#ff6b6b', '#4ecdc4', '#45b7d1']
            
            if sum(cat_counts) > 0:
                ax2.pie(cat_counts, labels=categories, autopct='%1.1f%%', 
                       colors=colors, startangle=90)
                ax2.set_title('Port Categories', color='white')
            
            # Create well-known ports detail chart
            ax3 = axes[1, 0]
            if well_known:
                wk_ports = list(well_known.keys())
                wk_counts = list(well_known.values())
                ax3.bar(range(len(wk_ports)), wk_counts, color='#96ceb4', alpha=0.8)
                ax3.set_title('Well-Known Ports Detail', color='white')
                ax3.set_xlabel('Port', color='white')
                ax3.set_ylabel('Count', color='white')
                ax3.set_xticks(range(len(wk_ports)))
                ax3.set_xticklabels(wk_ports, rotation=45)
                ax3.tick_params(colors='white')
                ax3.grid(True, alpha=0.3)
            
            # Add port statistics text
            ax4 = axes[1, 1]
            stats_text = f"""
            Total Unique Ports: {len(self.port_activity)}
            Most Active Port: {max(self.port_activity, key=self.port_activity.get)}
            ({max(self.port_activity.values())} packets)
            
            Well-Known Ports: {len(well_known)}
            Registered Ports: {len(registered)}
            Dynamic Ports: {len(dynamic)}
            
            Avg Packets/Port: {np.mean(list(self.port_activity.values())):.1f}
            """
            ax4.text(0.1, 0.5, stats_text, transform=ax4.transAxes, 
                    color='white', fontsize=10, verticalalignment='center',
                    bbox=dict(boxstyle="round,pad=0.3", facecolor='#333', alpha=0.8))
            ax4.set_title('Port Statistics', color='white')
            ax4.axis('off')
            
            # Save chart
            plt.tight_layout()
            plt.savefig(os.path.join(viz_dir, 'port_activity_heatmap.png'), 
                       facecolor='black', dpi=300, bbox_inches='tight')
            plt.close()
            return True
            
        except Exception as e:
            self.logger.warning(f"Error creating port heatmap: {str(e)}")
            return False
    
    def _create_feature_correlation_matrix(self, viz_dir, features_df):
        """Create feature correlation matrix heatmap."""
        if not MATPLOTLIB_AVAILABLE:
            return False
            
        try:
            if features_df.empty:
                return False
                
            # Select only numerical features
            numerical_features = features_df.select_dtypes(include=[np.number])
            if numerical_features.empty:
                return False
                
            # Calculate correlation matrix
            corr_matrix = numerical_features.corr()
            
            # Create figure
            plt.figure(figsize=(15, 12))
            plt.title('Traffic Flow Features Correlation Matrix', 
                     fontsize=16, color='white', pad=20)
            
            # Create heatmap with fallback for missing seaborn
            if 'seaborn' in sys.modules:
                import seaborn as sns
                mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
                sns.heatmap(corr_matrix, mask=mask, annot=False, cmap='RdYlBu_r', 
                           center=0, square=True, linewidths=0.5, 
                           cbar_kws={"shrink": .8})
            else:
                # Fallback to matplotlib only
                plt.imshow(corr_matrix, cmap='RdYlBu_r', aspect='auto', interpolation='nearest')
                plt.colorbar(label='Correlation Coefficient', shrink=0.8)
                
                # Add feature labels
                tick_positions = range(len(corr_matrix.columns))
                plt.xticks(tick_positions, corr_matrix.columns, rotation=45, ha='right')
                plt.yticks(tick_positions, corr_matrix.columns)
            
            # Style and save
            plt.xticks(rotation=45, ha='right', color='white')
            plt.yticks(rotation=0, color='white')
            plt.tight_layout()
            plt.savefig(os.path.join(viz_dir, 'feature_correlation_matrix.png'), 
                       facecolor='black', dpi=300, bbox_inches='tight')
            plt.close()
            return True
            
        except Exception as e:
            self.logger.warning(f"Error creating correlation matrix: {str(e)}")
            return False
    
    def _create_cnn_traffic_heatmap(self, viz_dir, features_df):
        """Create CNN-ready traffic pattern heatmap (224x224 pixels)."""
        if not MATPLOTLIB_AVAILABLE:
            return False
            
        try:
            if features_df.empty or len(features_df) < 10:
                return False
                
            # Select key features for CNN visualization
            key_features = [
                'total_packets', 'total_bytes', 'duration', 'packets_per_second',
                'avg_packet_size', 'packet_size_variance', 'iat_variance',
                'syn_flag_count', 'fin_flag_count', 'ack_flag_count'
            ]
            
            # Filter to only available features
            available_features = [f for f in key_features if f in features_df.columns]
            if not available_features:
                return False
                
            # Get feature data and normalize to 0-1 range
            feature_data = features_df[available_features].values
            feature_data = (feature_data - feature_data.min(axis=0)) / (feature_data.max(axis=0) - feature_data.min(axis=0) + 1e-8)
            
            # Create 224x224 heatmap for CNN input
            target_size = 224
            
            # Create 2D representation from feature data
            if len(feature_data) >= target_size:
                # Use first portion of data to create grid
                grid_data = feature_data[:target_size**2//len(available_features)]
                
                # Pad with zeros if needed
                needed_points = target_size * target_size // len(available_features)
                if len(grid_data) < needed_points:
                    padding = np.zeros((needed_points - len(grid_data), len(available_features)))
                    grid_data = np.vstack([grid_data, padding])
                
                # Reshape to 2D grid
                heatmap_data = grid_data.flatten()[:target_size*target_size]
                heatmap_data = heatmap_data.reshape(target_size, target_size)
            else:
                # For smaller datasets, tile the data
                small_grid = np.tile(feature_data.flatten(), 
                                   (target_size*target_size // len(feature_data.flatten()) + 1))
                heatmap_data = small_grid[:target_size*target_size].reshape(target_size, target_size)
            
            # Create visualization
            plt.figure(figsize=(10, 10))
            plt.title('CNN-Ready Traffic Pattern Heatmap (224x224)', 
                     fontsize=14, color='white', pad=20)
            
            # Display heatmap
            im = plt.imshow(heatmap_data, cmap='plasma', aspect='auto', interpolation='bilinear')
            plt.colorbar(im, label='Normalized Feature Intensity', shrink=0.8)
            
            # Remove axis ticks for cleaner look
            plt.xticks([])
            plt.yticks([])
            
            # Add grid lines to show structure
            for i in range(0, target_size, target_size//8):
                plt.axhline(y=i, color='white', alpha=0.2, linewidth=0.5)
                plt.axvline(x=i, color='white', alpha=0.2, linewidth=0.5)
            
            # Save chart and raw data
            plt.tight_layout()
            plt.savefig(os.path.join(viz_dir, 'cnn_traffic_heatmap.png'), 
                       facecolor='black', dpi=150, bbox_inches='tight')
            plt.close()
            
            # Save raw array for direct CNN use
            np.save(os.path.join(viz_dir, 'cnn_traffic_array.npy'), heatmap_data)
            return True
            
        except Exception as e:
            self.logger.warning(f"Error creating CNN heatmap: {str(e)}")
            return False
    
    def _create_interactive_dashboard(self, viz_dir, features_df):
        """Create interactive dashboard using Plotly."""
        if not PLOTLY_AVAILABLE:
            return False
            
        try:
            if features_df.empty:
                return False
                
            # Create subplot layout
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=('Traffic Volume Over Time', 'Protocol Distribution', 
                               'Packet Size vs Duration', 'Top Flow Statistics'),
                specs=[[{"secondary_y": True}, {"type": "pie"}],
                      [{"type": "scatter"}, {"type": "bar"}]]
            )
            
            # Add traffic timeline if data available
            if self.packet_timeline:
                timeline_df = pd.DataFrame(self.packet_timeline)
                timeline_df['datetime'] = pd.to_datetime(timeline_df['time'], unit='s')
                
                fig.add_trace(
                    go.Scatter(x=timeline_df['datetime'], y=timeline_df['size'],
                             mode='lines', name='Packet Size', line=dict(color='#ff6b6b')),
                    row=1, col=1
                )
            
            # Add protocol pie chart
            if self.protocol_counts:
                fig.add_trace(
                    go.Pie(labels=list(self.protocol_counts.keys()), 
                          values=list(self.protocol_counts.values()),
                          name="Protocols"),
                    row=1, col=2
                )
            
            # Add scatter plot of packet size vs duration
            if 'avg_packet_size' in features_df.columns and 'duration' in features_df.columns:
                fig.add_trace(
                    go.Scatter(x=features_df['duration'], y=features_df['avg_packet_size'],
                             mode='markers', name='Flows',
                             marker=dict(color='#4ecdc4', opacity=0.6)),
                    row=2, col=1
                )
            
            # Add top flows bar chart
            if 'total_bytes' in features_df.columns:
                top_flows = features_df.nlargest(10, 'total_bytes')
                fig.add_trace(
                    go.Bar(x=list(range(len(top_flows))), y=top_flows['total_bytes'],
                          name='Top Flows by Bytes', marker_color='#45b7d1'),
                    row=2, col=2
                )
            
            # Update layout with dark theme
            fig.update_layout(
                title_text="Interactive Network Traffic Dashboard",
                title_font_size=20,
                template="plotly_dark",
                height=800,
                showlegend=True
            )
            
            # Save interactive HTML file
            fig.write_html(os.path.join(viz_dir, 'interactive_dashboard.html'))
            return True
            
        except Exception as e:
            self.logger.warning(f"Error creating interactive dashboard: {str(e)}")
            return False


def main():
    """Main function to run the traffic flow extractor."""
    print("=" * 60)
    print("CICIDS2017 Traffic Flow Feature Extractor")
    print("Zero-Day Detection Framework")
    print("=" * 60)
    print()
    
    # Get PCAP file path from user
    while True:
        pcap_path = input("Enter the path to your CICIDS2017 PCAP file: ").strip()
        
        # Validate user input
        if not pcap_path:
            print("Please enter a valid file path.")
            continue
            
        if not os.path.exists(pcap_path):
            print(f"File not found: {pcap_path}")
            print("   Please check the path and try again.")
            continue
            
        if not pcap_path.lower().endswith(('.pcap', '.pcapng')):
            print("Warning: File doesn't have .pcap or .pcapng extension.")
            confirm = input("   Continue anyway? (y/n): ").lower()
            if confirm != 'y':
                continue
        
        break
    
    # Get output directory from user
    output_dir = input("Enter output directory (press Enter for current directory): ").strip()
    if not output_dir:
        output_dir = "."
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Get analysis mode preference
    print("\nAnalysis Mode:")
    analysis_mode = input("   Analyze every packet individually? (Y/n): ").strip().lower()
    packet_analysis_mode = analysis_mode != 'n'
    
    if packet_analysis_mode:
        print("   Mode: PACKET-LEVEL ANALYSIS")
        print("   Will extract 46 features from every individual packet")
        print("   Output: One row per packet with comprehensive packet features")
    else:
        print("   Mode: FLOW-LEVEL ANALYSIS")
        print("   Will extract 28 features from network flows")
        print("   Output: One row per flow with aggregated flow statistics")
    
    # Get configuration options from user
    print("\nConfiguration Options:")
    chunk_size = input("   Chunk size for processing (default: 100000 for better performance): ").strip()
    chunk_size = int(chunk_size) if chunk_size.isdigit() else 100000
    
    # Get visualization preferences
    print("\nVisualization Options:")
    enable_viz = input("   Generate visualizations? (Y/n): ").strip().lower()
    enable_visualizations = enable_viz != 'n'
    
    if enable_visualizations:
        print("   Will generate: Protocol charts, traffic timeline, network topology")
        print("   Will generate: CNN-ready heatmaps, correlation matrices, interactive dashboard")
    
    # Display final configuration
    print(f"\nConfiguration:")
    print(f"   Input file: {pcap_path}")
    print(f"   Output directory: {output_dir}")
    print(f"   Analysis mode: {'PACKET-LEVEL' if packet_analysis_mode else 'FLOW-LEVEL'}")
    print(f"   Features to extract: {46 if packet_analysis_mode else 28}")
    print(f"   Chunk size: {chunk_size:,} packets")
    print(f"   Visualizations: {'Enabled' if enable_visualizations else 'Disabled'}")
    
    print(f"\nStarting extraction...")
    if packet_analysis_mode:
        print(f"   Note: Packet-level analysis will create VERY large output files")
        print(f"   Expected output size: ~1GB per million packets")
    print(f"   Progress and memory usage will be displayed")
    if enable_visualizations:
        print(f"   Visualizations will be saved in 'visualizations/' subdirectory")
    print()
    
    try:
        # Initialize the extractor
        extractor = CICIDSPacketExtractor(
            chunk_size=chunk_size,
            enable_visualizations=enable_visualizations,
            packet_analysis_mode=packet_analysis_mode
        )
        
        # Process the PCAP file
        features_df = extractor.process_pcap_file(pcap_path)
        
        # Save results to CSV file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        if packet_analysis_mode:
            output_file = os.path.join(output_dir, f"cicids_packet_features_{timestamp}.csv")
        else:
            output_file = os.path.join(output_dir, f"cicids_flow_features_{timestamp}.csv")
        
        print(f"\nSaving results to: {output_file}")
        features_df.to_csv(output_file, index=False)
        
        # Generate visualizations if enabled
        if enable_visualizations and not features_df.empty:
            print(f"\nGenerating visualizations...")
            extractor.generate_visualizations(output_dir, features_df)
        
        # Display summary statistics
        print(f"\nExtraction Summary:")
        if packet_analysis_mode:
            print(f"   Total packets analyzed: {len(features_df):,}")
            print(f"   Feature columns: {len(extractor.feature_names)}")
        else:
            print(f"   Total flows extracted: {len(features_df):,}")
            print(f"   Feature columns: {len(extractor.feature_names)}")
        print(f"   Output file size: {os.path.getsize(output_file) / (1024*1024):.2f} MB")
        
        # Show visualization file count
        if enable_visualizations:
            viz_dir = os.path.join(output_dir, "visualizations")
            if os.path.exists(viz_dir):
                viz_files = len([f for f in os.listdir(viz_dir) if f.endswith(('.png', '.html'))])
                print(f"   Visualization files generated: {viz_files}")
        
        # Display analysis statistics
        if packet_analysis_mode:
            print(f"\nPacket Analysis Statistics:")
            if len(features_df) > 0:
                try:
                    # Safe conversion with error handling
                    if 'packet_size' in features_df.columns and not features_df['packet_size'].empty:
                        avg_packet_size = float(features_df['packet_size'].mean())
                        max_packet_size = int(features_df['packet_size'].max())
                        min_packet_size = int(features_df['packet_size'].min())
                        
                        print(f"   Avg packet size: {avg_packet_size:.1f} bytes")
                        print(f"   Max packet size: {max_packet_size:,} bytes")
                        print(f"   Min packet size: {min_packet_size} bytes")
                    
                    if 'protocol_type' in features_df.columns and not features_df['protocol_type'].empty:
                        protocol_dist = features_df['protocol_type'].value_counts()
                        total_packets = len(features_df)
                        print(f"\nProtocol Distribution:")
                        protocol_names = {0: 'OTHER', 1: 'TCP', 2: 'UDP', 3: 'ICMP'}
                        for proto_num, count in protocol_dist.items():
                            proto_name = protocol_names.get(proto_num, f'UNKNOWN({proto_num})')
                            percentage = (count / total_packets) * 100
                            print(f"   {proto_name}: {count:,} packets ({percentage:.1f}%)")
                            
                except Exception as e:
                    print(f"   Error calculating packet statistics: {str(e)}")
                    print(f"   Dataset shape: {features_df.shape}")
        else:
            print(f"\nFlow Analysis Statistics:")
            if len(features_df) > 0:
                try:
                    # Safe conversion with error handling for flow statistics
                    if 'total_packets' in features_df.columns and not features_df['total_packets'].empty:
                        avg_packets = float(features_df['total_packets'].mean())
                        max_packets = int(features_df['total_packets'].max())
                        print(f"   Avg packets per flow: {avg_packets:.1f}")
                        print(f"   Max packets in flow: {max_packets:,}")
                    
                    if 'total_bytes' in features_df.columns and not features_df['total_bytes'].empty:
                        avg_bytes = float(features_df['total_bytes'].mean())
                        print(f"   Avg bytes per flow: {avg_bytes:.1f}")
                    
                    if 'duration' in features_df.columns and not features_df['duration'].empty:
                        avg_duration = float(features_df['duration'].mean())
                        print(f"   Avg duration: {avg_duration:.2f} seconds")
                        
                except Exception as e:
                    print(f"   Error calculating flow statistics: {str(e)}")
                    print(f"   Dataset shape: {features_df.shape}")
            
            # Show protocol distribution
            if extractor.protocol_counts:
                print(f"\nProtocol Distribution:")
                total_protocol_packets = sum(extractor.protocol_counts.values())
                for protocol, count in extractor.protocol_counts.most_common():
                    percentage = (count / total_protocol_packets) * 100
                    print(f"   {protocol}: {count:,} packets ({percentage:.1f}%)")
        
        print(f"\nFeature extraction completed successfully!")
        print(f"Log file: cicids_extraction.log")
        
        # Show visualization file details
        if enable_visualizations:
            print(f"\nVisualization Files Generated:")
            viz_dir = os.path.join(output_dir, "visualizations")
            if os.path.exists(viz_dir):
                print(f"   Static Images: {viz_dir}/")
                print(f"      • protocol_distribution.png")
                print(f"      • traffic_timeline.png")
                print(f"      • network_topology.png")
                print(f"      • packet_size_distribution.png")
                print(f"      • port_activity_heatmap.png")
                print(f"      • feature_correlation_matrix.png")
                print(f"      • cnn_traffic_heatmap.png (224x224 for CNN)")
                print(f"   Interactive: {viz_dir}/interactive_dashboard.html")
                print(f"   CNN Data: {viz_dir}/cnn_traffic_array.npy")
                print(f"\n   Open interactive_dashboard.html in your browser for interactive analysis!")
        
    except KeyboardInterrupt:
        print(f"\nProcessing interrupted by user.")
        print(f"   Partial results may be available in log file.")
        
    except Exception as e:
        print(f"\nError during processing: {str(e)}")
        print(f"   Check the log file for detailed error information.")
        raise


if __name__ == "__main__":
    main()