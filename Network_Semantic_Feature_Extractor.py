#!/usr/bin/env python3

"""
CICIDS2017 Semantic Network Analysis Feature Extractor
======================================================

This interactive version combines sophisticated semantic analysis with user-friendly prompts.
Includes advanced threat scoring, interactive dashboards, and comprehensive visualizations.
"""

import os
import sys
import time
import logging
from datetime import datetime
from pathlib import Path
from collections import defaultdict, Counter
import gc
import re
import math
import statistics
import json
import hashlib
import ipaddress
import string
import urllib.parse
from html.parser import HTMLParser
import base64

import numpy as np
import pandas as pd
from scapy.all import PcapReader, IP, TCP, UDP, ICMP, Raw, DNS, rdpcap
import psutil
import warnings
import tldextract

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
    print("Warning: networkx not available. Network graphs will be disabled.")

try:
    import nltk
    from nltk.tokenize import word_tokenize
    from nltk.corpus import stopwords
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
    NLTK_AVAILABLE = True
except ImportError:
    NLTK_AVAILABLE = False
    print("Warning: NLTK not available. Some text analysis features will be limited.")

try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.cluster import KMeans
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    print("Warning: scikit-learn not available. Advanced text clustering will be disabled.")

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")


def print_banner():
    """Print the application banner."""
    print("=" * 75)
    print("CICIDS2017 Semantic Network Analysis Feature Extractor")
    print("Advanced Interactive Zero-Day Detection Framework")
    print("=" * 75)
    print()


def get_user_input():
    """Get user configuration through interactive prompts."""
    config = {}
    
    # Get PCAP file path
    while True:
        pcap_path = input("Enter the path to your PCAP file: ").strip()
        if pcap_path.startswith('"') and pcap_path.endswith('"'):
            pcap_path = pcap_path[1:-1]  # Remove quotes
        if pcap_path.startswith("'") and pcap_path.endswith("'"):
            pcap_path = pcap_path[1:-1]  # Remove quotes
            
        if os.path.exists(pcap_path):
            config['pcap_file'] = pcap_path
            break
        else:
            print(f"Error: File '{pcap_path}' not found. Please try again.")
    
    # Get output directory
    default_output = "Generated_Data"
    output_dir = input(f"Enter output directory (press Enter for '{default_output}'): ").strip()
    if not output_dir:
        output_dir = default_output
    config['output_dir'] = output_dir
    
    print()
    print("Analysis Mode:")
    
    # Get deep packet inspection preference
    while True:
        deep_inspection = input("  Enable deep packet inspection? (Y/n): ").strip().lower()
        if deep_inspection in ['', 'y', 'yes']:
            config['deep_inspection'] = True
            print("    Mode: DEEP PACKET INSPECTION")
            print("    Will perform comprehensive semantic analysis of packet payloads")
            break
        elif deep_inspection in ['n', 'no']:
            config['deep_inspection'] = False
            print("    Mode: STANDARD INSPECTION")
            print("    Will perform basic flow-level analysis")
            break
        else:
            print("    Please enter Y or n")
    
    print("    Output: Detailed semantic features for threat detection")
    print()
    
    print("Configuration Options:")
    
    # Get chunk size
    while True:
        try:
            chunk_input = input("  Chunk size for processing (default: 10000 for better performance): ").strip()
            if chunk_input == '':
                config['chunk_size'] = 10000
                break
            chunk_size = int(chunk_input)
            if chunk_size > 0:
                config['chunk_size'] = chunk_size
                break
            else:
                print("    Chunk size must be positive")
        except ValueError:
            print("    Please enter a valid number")
    
    print()
    print("Visualization Options:")
    
    # Get visualization preference
    while True:
        viz_choice = input("  Generate visualizations? (Y/n): ").strip().lower()
        if viz_choice in ['', 'y', 'yes']:
            config['enable_visualizations'] = True
            print("    Will generate: URL analysis, DNS analysis, HTTP content analysis")
            print("    Will generate: Command payload analysis, threat score visualization, interactive dashboard")
            break
        elif viz_choice in ['n', 'no']:
            config['enable_visualizations'] = False
            print("    Visualizations disabled")
            break
        else:
            print("    Please enter Y or n")
    
    print()
    print("Configuration:")
    print(f"  Input file: {config['pcap_file']}")
    print(f"  Output directory: {config['output_dir']}")
    print(f"  Analysis mode: {'DEEP INSPECTION' if config['deep_inspection'] else 'STANDARD'}")
    print(f"  Chunk size: {config['chunk_size']:,} packets")
    print(f"  Visualizations: {'Enabled' if config['enable_visualizations'] else 'Disabled'}")
    print()
    
    print("Starting semantic analysis...")
    print("  Note: Deep inspection may take longer to process")
    print("  Will analyze URL patterns, DNS queries, HTTP content, command payloads")
    print("  Progress and memory usage will be displayed")
    print(f"  Visualizations will be saved in '{config['output_dir']}/visualizations/' subdirectory")
    print()
    
    return config


class SemanticNetworkAnalyzer:
    """
    Advanced semantic network analyzer with comprehensive threat detection capabilities.
    Extracts 33 semantic features and generates interactive visualizations.
    """
    
    def __init__(self, chunk_size=10000, enable_visualizations=True, deep_inspection=True):
        """
        Initialize the advanced semantic analyzer.
        
        Args:
            chunk_size (int): Number of packets to process in each chunk
            enable_visualizations (bool): Whether to generate visual outputs
            deep_inspection (bool): Whether to perform deep payload inspection
        """
        # Configuration parameters
        self.chunk_size = chunk_size
        self.enable_visualizations = enable_visualizations
        self.deep_inspection = deep_inspection
        self.timeout_seconds = 1800  # 30 minutes flow timeout
        self.min_packets_for_analysis = 2

        # Storage for flows and packets
        self.flows = defaultdict(self._create_flow_dict)
        self.packet_features = []
        self.packet_metadata = []
        
        # Semantic feature names
        self.feature_names = self._get_semantic_feature_names()
        
        # Data collections for visualizations and analysis
        self.urls = []
        self.dns_queries = []
        self.http_requests = []
        self.http_responses = []
        self.payload_samples = []
        self.user_agents = []
        self.domains_seen = Counter()
        self.suspicious_patterns = Counter()
        
        # Enhanced tracking for semantic analysis
        self.command_keywords = set([
            'select', 'insert', 'update', 'delete', 'exec', 'shell', 'cmd', 'powershell',
            'bash', 'wget', 'curl', 'chmod', 'system', 'eval', 'execute', 'ping', 'nslookup',
            'tracert', 'dir', 'ls', 'whoami', 'net', 'reg', 'rundll', 'wmic'
        ])
        
        # Initialize dictionaries for reputation and categorization
        self.tld_reputation = self._initialize_tld_reputation()
        self.suspicious_url_keywords = self._initialize_suspicious_keywords()
        self.http_status_categories = self._initialize_http_status_categories()
        
        # Setup logging system
        self._setup_logging()
        self.logger = logging.getLogger(__name__)
        
        # Performance tracking
        self.last_cleanup_time = 0
        self.sample_counter = 0
        self.sample_rate = 100
        
        # Statistics tracking
        self.packets_processed = 0
        self.flows_created = 0
        self.flows_analyzed = 0
        self.features_extracted = 0
        self.errors_encountered = 0
        
    def _setup_logging(self):
        """Setup logging configuration for tracking processing."""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('semantic_analysis.log'),
                logging.StreamHandler(sys.stdout)
            ]
        )
    
    def _create_flow_dict(self):
        """Create a new flow dictionary with default values for tracking network conversations."""
        return {
            # Flow identification
            'src_ip': '',
            'dst_ip': '',
            'src_port': 0,
            'dst_port': 0,
            'protocol': '',
            
            # Timing information
            'start_time': 0,
            'end_time': 0,
            'duration': 0,
            'last_packet_time': 0,
            
            # Basic flow statistics
            'fwd_packets': 0,
            'bwd_packets': 0,
            'fwd_bytes': 0,
            'bwd_bytes': 0,
            
            # HTTP/HTTPS specific tracking
            'http_requests': [],
            'http_responses': [],
            'urls_requested': [],
            'user_agents': [],
            'content_types': [],
            'http_methods': [],
            'status_codes': [],
            
            # DNS specific tracking
            'dns_queries': [],
            'dns_responses': [],
            'unique_domains': set(),
            'failed_queries': 0,
            
            # Payload analysis
            'command_keywords_detected': set(),
            'obfuscation_indicators': 0,
            'encryption_indicators': 0,
            'base64_strings': [],
            'script_languages': set(),
            
            # Semantic metrics
            'domain_entropy_values': [],
            'url_length_values': [],
            'query_frequency': 0,
            'suspicious_pattern_count': 0,
            'suspicious_tld_count': 0,
            'suspicious_keywords_count': 0,
            'subdomain_count': 0,
            
            # Aggregated feature values
            'semantic_features': {},
            
            # Raw data storage for deep inspection
            'payloads': [] if self.deep_inspection else None,
            
            # Flag to indicate if this flow has valuable semantic data
            'has_semantic_data': False
        }
    
    def _get_semantic_feature_names(self):
        """Define the semantic feature names for network traffic analysis."""
        return [
            # URL Analysis Features (8 features)
            'domain_entropy',  # How random/suspicious domain looks
            'subdomain_count',  # Number of subdomains
            'suspicious_keywords',  # Count of suspicious words in URL
            'domain_age_category',  # Category representing domain age
            'url_length',  # Average URL length
            'special_char_ratio',  # % of special characters in URLs
            'numeric_ratio',  # % of numbers in URLs
            'tld_reputation',  # Reputation score of TLDs used

            # DNS Query Features (6 features)
            'query_entropy',  # Randomness of domain names
            'dga_probability',  # Likelihood of Domain Generation Algorithm
            'query_frequency',  # Queries per minute
            'unique_domains',  # Number of different domains
            'failed_queries',  # DNS failures (domains don't exist)
            'subdomain_entropy',  # Randomness in subdomains

            # HTTP Content Features (6 features)
            'user_agent_entropy',  # How unique/fake user agent is
            'header_anomalies',  # Unusual HTTP headers
            'content_type_variety',  # Different file types requested
            'status_code_distribution',  # Mix of success/error codes
            'referrer_consistency',  # How consistent referrer headers are
            'cookie_complexity',  # Complexity of session cookies

            # Command & Payload Features (5 features)
            'command_keywords',  # SQL, PowerShell, bash commands
            'obfuscation_indicators',  # Base64, hex encoding detected
            'script_language_mix',  # Multiple scripting languages
            'dangerous_functions',  # exec(), eval(), system() calls
            'encryption_indicators',  # Encrypted/encoded content blocks

            # Additional Semantic Features (8 features)
            'redirect_frequency',  # Frequency of HTTP redirects
            'https_ratio',  # Ratio of HTTPS to HTTP
            'path_depth',  # Depth of URL paths
            'param_complexity',  # Complexity of URL parameters
            'host_diversity',  # Diversity of hosts contacted
            'mime_type_entropy',  # Entropy of MIME types
            'certificate_validity',  # SSL certificate validity indicators
            'semantic_consistency'  # Consistency of semantic patterns
        ]
    
    def _initialize_tld_reputation(self):
        """Initialize TLD reputation scores (lower score = more suspicious)."""
        reputation = defaultdict(lambda: 0.5)
        
        # Common legitimate TLDs
        for tld in ['com', 'org', 'net', 'edu', 'gov', 'mil']:
            reputation[tld] = 0.9
            
        # Country TLDs with mixed reputation
        for tld in ['io', 'co', 'me', 'tv', 'ai', 'app']:
            reputation[tld] = 0.7
            
        # Suspicious TLDs often used in malicious campaigns
        for tld in ['tk', 'ml', 'ga', 'cf', 'gq', 'xyz', 'top', 'info', 'biz', 'pw']:
            reputation[tld] = 0.2
            
        return reputation
    
    def _initialize_suspicious_keywords(self):
        """Initialize dictionary of suspicious keywords for URL analysis."""
        keywords = {
            'high': [
                'admin', 'login', 'banking', 'verify', 'secure', 'account', 'password',
                'signin', 'update', 'alert', 'confirm', 'authorize', 'wallet', 'access',
                'recovery', 'authorize'
            ],
            'medium': [
                'payment', 'credit', 'bank', 'paypal', 'ebay', 'amazon', 'apple',
                'microsoft', 'google', 'facebook', 'verification', 'verify',
                'authenticate', 'support', 'security'
            ],
            'low': [
                'download', 'free', 'prize', 'winner', 'offer', 'gift', 'bonus',
                'limited', 'expire', 'click', 'install', 'upgrade'
            ]
        }
        return keywords
    
    def _initialize_http_status_categories(self):
        """Initialize HTTP status code categories for analysis."""
        return {
            'success': list(range(200, 300)),  # 2xx Success
            'redirect': list(range(300, 400)),  # 3xx Redirection
            'client_error': list(range(400, 500)),  # 4xx Client Error
            'server_error': list(range(500, 600))  # 5xx Server Error
        }
    
    def get_flow_key(self, packet):
        """Generate a unique flow key from packet information."""
        if not packet.haslayer(IP):
            return None
            
        try:
            ip_layer = packet[IP]
            src_ip = str(ip_layer.src)
            dst_ip = str(ip_layer.dst)
            protocol = int(ip_layer.proto)
            
            src_port = dst_port = 0
            if packet.haslayer(TCP):
                src_port = int(packet[TCP].sport)
                dst_port = int(packet[TCP].dport)
            elif packet.haslayer(UDP):
                src_port = int(packet[UDP].sport)
                dst_port = int(packet[UDP].dport)
                
            flow_tuple = tuple(sorted([(src_ip, src_port), (dst_ip, dst_port)]))
            return (*flow_tuple[0], *flow_tuple[1], protocol)
            
        except Exception as e:
            self.logger.debug(f"Error generating flow key: {str(e)}")
            return None

    def process_packet(self, packet, packet_time, packet_id=0):
        """Process a single packet and extract semantic features."""
        flow_key = self.get_flow_key(packet)
        if not flow_key:
            return
            
        flow = self.flows[flow_key]
        
        # Initialize flow on first packet
        if flow['start_time'] == 0:
            flow['start_time'] = packet_time
            self.flows_created += 1
            
            if packet.haslayer(IP):
                flow['src_ip'] = str(packet[IP].src)
                flow['dst_ip'] = str(packet[IP].dst)
                
            if packet.haslayer(TCP):
                flow['src_port'] = int(packet[TCP].sport)
                flow['dst_port'] = int(packet[TCP].dport)
                flow['protocol'] = 'TCP'
            elif packet.haslayer(UDP):
                flow['src_port'] = int(packet[UDP].sport)
                flow['dst_port'] = int(packet[UDP].dport)
                flow['protocol'] = 'UDP'
            else:
                flow['protocol'] = 'OTHER'
                
        # Update flow timing and statistics
        flow['end_time'] = packet_time
        flow['duration'] = packet_time - flow['start_time']
        
        is_forward = (packet.haslayer(IP) and str(packet[IP].src) == flow['src_ip'])
        packet_size = len(packet)
        
        if is_forward:
            flow['fwd_packets'] += 1
            flow['fwd_bytes'] += packet_size
        else:
            flow['bwd_packets'] += 1
            flow['bwd_bytes'] += packet_size
            
        flow['last_packet_time'] = packet_time
        
        # Extract semantic features from packet payload
        self._extract_semantic_features(packet, flow, is_forward)
        
        # Deep packet inspection
        if self.deep_inspection:
            packet_features = self._extract_packet_semantic_features(packet, flow, is_forward)
            if packet_features:
                self.packet_features.append(packet_features)
                
                packet_metadata = {
                    'packet_id': packet_id,
                    'timestamp': packet_time,
                    'src_ip': str(packet[IP].src) if packet.haslayer(IP) else '',
                    'dst_ip': str(packet[IP].dst) if packet.haslayer(IP) else '',
                    'protocol': self._get_protocol_name(packet),
                    'size': len(packet)
                }
                self.packet_metadata.append(packet_metadata)
                
        # Collect visualization data
        if self.enable_visualizations:
            self._collect_visualization_data(packet, flow)
            
        self.packets_processed += 1
        
        # Mark flow as having semantic data
        if (flow['urls_requested'] or flow['dns_queries'] or 
            flow['http_requests'] or flow['command_keywords_detected']):
            flow['has_semantic_data'] = True
            
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
            
    def _extract_semantic_features(self, packet, flow, is_forward):
        """Extract semantic features from packet payload."""
        try:
            if not packet.haslayer(Raw):
                return
                
            payload_bytes = bytes(packet[Raw].load)
            payload_text = None
            
            try:
                payload_text = payload_bytes.decode('utf-8', errors='ignore')
            except Exception:
                pass
                
            # Store payload for deep inspection
            if self.deep_inspection and payload_text and len(payload_text) > 10:
                if len(flow.get('payloads', [])) < 100:
                    flow['payloads'].append(payload_text)
                    
            # Process HTTP traffic
            if packet.haslayer(TCP):
                dst_port = packet[TCP].dport if is_forward else packet[TCP].sport
                src_port = packet[TCP].sport if is_forward else packet[TCP].dport
                
                if dst_port in (80, 8080, 443, 8443) or src_port in (80, 8080, 443, 8443):
                    self._analyze_http_traffic(payload_text, payload_bytes, flow, is_forward, dst_port)
                    
            # Process DNS traffic
            if packet.haslayer(UDP) and (packet[UDP].dport == 53 or packet[UDP].sport == 53):
                self._analyze_dns_traffic(packet, flow)
                
            # Find URLs in payload
            if payload_text:
                urls = self._find_urls_in_payload(payload_text)
                for url in urls:
                    if url not in flow['urls_requested']:
                        flow['urls_requested'].append(url)
                        self._analyze_url_semantics(url, flow)
                
                # Analyze commands and obfuscation
                self._analyze_command_payload(payload_text, flow)
                self._detect_obfuscation(payload_text, payload_bytes, flow)
                
                # Sample payload for visualization
                self.sample_counter += 1
                if self.sample_counter % self.sample_rate == 0:
                    if len(self.payload_samples) < 1000:
                        self.payload_samples.append({
                            'text': payload_text[:200],
                            'size': len(payload_text),
                            'is_http': packet.haslayer(TCP) and (
                                packet[TCP].dport in (80, 8080, 443, 8443) or
                                packet[TCP].sport in (80, 8080, 443, 8443)
                            ),
                            'is_dns': packet.haslayer(UDP) and (
                                packet[UDP].dport == 53 or packet[UDP].sport == 53
                            ),
                            'entropy': self._calculate_entropy(payload_text)
                        })
                        
        except Exception as e:
            self.logger.debug(f"Error extracting semantic features: {str(e)}")
            self.errors_encountered += 1
            
    def _find_urls_in_payload(self, payload_text):
        """Extract URLs from any payload text."""
        url_pattern = r'https?://[^\s<>"\']+|www\.[^\s<>"\']+' 
        return re.findall(url_pattern, payload_text)
        
    def _analyze_http_traffic(self, payload_text, payload_bytes, flow, is_forward, dst_port):
        """Analyze HTTP/HTTPS traffic for semantic features."""
        if not payload_text:
            return
            
        # Check for HTTP request
        is_request = False
        http_method = None
        for method in ('GET', 'POST', 'PUT', 'DELETE', 'HEAD', 'OPTIONS', 'CONNECT', 'TRACE', 'PATCH'):
            if payload_text.startswith(method + ' '):
                is_request = True
                http_method = method
                break
                
        # Process HTTP request
        if is_request and http_method:
            try:
                url_match = re.search(f"{http_method} (.*?) HTTP", payload_text)
                if url_match:
                    url = url_match.group(1).strip()
                    
                    full_url = url
                    if not url.startswith('http'):
                        host_header = re.search(r"Host: (.*?)[\r\n]", payload_text)
                        host = host_header.group(1).strip() if host_header else flow['dst_ip']
                        
                        if dst_port in (80, 8080):
                            full_url = f"http://{host}{url}"
                        elif dst_port in (443, 8443):
                            full_url = f"https://{host}{url}"
                    
                    if full_url not in flow['urls_requested']:
                        flow['urls_requested'].append(full_url)
                    
                    self._analyze_url_semantics(full_url, flow)
                    
                    if len(self.urls) < 1000:
                        self.urls.append(full_url)
            except Exception as e:
                self.logger.debug(f"Error extracting URL: {str(e)}")
                
            # Extract User-Agent
            try:
                ua_match = re.search(r"User-Agent: (.*?)[\r\n]", payload_text)
                if ua_match:
                    user_agent = ua_match.group(1).strip()
                    flow['user_agents'].append(user_agent)
                    
                    if len(self.user_agents) < 500:
                        self.user_agents.append(user_agent)
            except Exception as e:
                self.logger.debug(f"Error extracting User-Agent: {str(e)}")
                
            # Store HTTP request
            flow['http_requests'].append({
                'method': http_method,
                'url': url if 'url' in locals() else '',
                'user_agent': user_agent if 'user_agent' in locals() else '',
                'headers': self._extract_http_headers(payload_text)
            })
            
            flow['http_methods'].append(http_method)
            
            if len(self.http_requests) < 1000:
                self.http_requests.append({
                    'method': http_method,
                    'url': url if 'url' in locals() else '',
                    'user_agent': user_agent if 'user_agent' in locals() else ''
                })
                
        # Process HTTP response
        elif payload_text.startswith('HTTP/'):
            try:
                status_match = re.search(r"HTTP/[\d\.]+ (\d+)", payload_text)
                if status_match:
                    status_code = int(status_match.group(1))
                    flow['status_codes'].append(status_code)
            except Exception as e:
                self.logger.debug(f"Error extracting status code: {str(e)}")
                
            try:
                ct_match = re.search(r"Content-Type: (.*?)[\r\n]", payload_text)
                if ct_match:
                    content_type = ct_match.group(1).strip()
                    flow['content_types'].append(content_type)
            except Exception as e:
                self.logger.debug(f"Error extracting Content-Type: {str(e)}")
                
            flow['http_responses'].append({
                'status_code': status_code if 'status_code' in locals() else 0,
                'content_type': content_type if 'content_type' in locals() else '',
                'headers': self._extract_http_headers(payload_text)
            })
            
            if len(self.http_responses) < 1000:
                self.http_responses.append({
                    'status_code': status_code if 'status_code' in locals() else 0,
                    'content_type': content_type if 'content_type' in locals() else ''
                })
                
    def _extract_http_headers(self, payload_text):
        """Extract HTTP headers from payload text."""
        headers = {}
        header_section_match = re.search(r"(.*?)\r\n\r\n", payload_text, re.DOTALL)
        
        if header_section_match:
            header_lines = header_section_match.group(1).split('\r\n')
            for line in header_lines[1:]:
                if ': ' in line:
                    key, value = line.split(': ', 1)
                    headers[key.strip()] = value.strip()
                    
        return headers
        
    def _analyze_url_semantics(self, url, flow):
        """Analyze URL for semantic patterns and features."""
        try:
            parsed = urllib.parse.urlparse(url)
            extracted = tldextract.extract(parsed.netloc or parsed.path)
            subdomain = extracted.subdomain
            domain = extracted.domain
            tld = extracted.suffix
            
            # Check for suspicious TLDs
            if tld in self.tld_reputation and self.tld_reputation[tld] < 0.5:
                flow['suspicious_tld_count'] += 1
                
            # Count subdomains
            subdomain_count = len(subdomain.split('.')) if subdomain else 0
            flow['subdomain_count'] = subdomain_count
            
            # Calculate domain entropy
            if domain:
                domain_entropy = self._calculate_entropy(domain)
                flow['domain_entropy_values'].append(domain_entropy)
                self.domains_seen[domain] += 1
                
            # Check URL length
            url_length = len(url)
            flow['url_length_values'].append(url_length)
            
            # Calculate ratios
            special_chars = sum(1 for c in url if not c.isalnum() and c not in '/:.-_')
            special_char_ratio = special_chars / max(len(url), 1)
            
            numeric_chars = sum(1 for c in url if c.isdigit())
            numeric_ratio = numeric_chars / max(len(url), 1)
            
            # Check for suspicious keywords
            suspicious_keywords_count = 0
            
            for keyword in self.suspicious_url_keywords['high']:
                if keyword in url.lower():
                    suspicious_keywords_count += 3
                    self.suspicious_patterns[f"high:{keyword}"] += 1
                    
            for keyword in self.suspicious_url_keywords['medium']:
                if keyword in url.lower():
                    suspicious_keywords_count += 2
                    self.suspicious_patterns[f"medium:{keyword}"] += 1
                    
            for keyword in self.suspicious_url_keywords['low']:
                if keyword in url.lower():
                    suspicious_keywords_count += 1
                    self.suspicious_patterns[f"low:{keyword}"] += 1
                    
            flow['suspicious_keywords_count'] += suspicious_keywords_count
            
            path_depth = len([p for p in parsed.path.split('/') if p])
            param_count = len(urllib.parse.parse_qs(parsed.query))
            
            return {
                'domain': domain,
                'tld': tld,
                'subdomain_count': subdomain_count,
                'domain_entropy': domain_entropy if 'domain_entropy' in locals() else 0,
                'url_length': url_length,
                'special_char_ratio': special_char_ratio,
                'numeric_ratio': numeric_ratio,
                'suspicious_keywords_count': suspicious_keywords_count,
                'path_depth': path_depth,
                'param_count': param_count
            }
            
        except Exception as e:
            self.logger.debug(f"Error analyzing URL semantics: {str(e)}")
            self.errors_encountered += 1
            return {}
            
    def _analyze_dns_traffic(self, packet, flow):
        """Analyze DNS traffic for semantic features."""
        try:
            if packet.haslayer(DNS):
                dns = packet[DNS]
                
                if dns.qr == 0:  # DNS query
                    qname = ""
                    if dns.qd:
                        try:
                            qname = dns.qd.qname.decode('utf-8', errors='ignore')
                        except:
                            if hasattr(dns.qd, 'qname'):
                                qname = str(dns.qd.qname)
                    
                    if not qname and hasattr(dns, 'qd') and dns.qd:
                        try:
                            qname = str(dns.qd[0].qname)
                        except:
                            pass
                            
                    if qname:
                        if qname.endswith('.'):
                            qname = qname[:-1]
                            
                        flow['dns_queries'].append(qname)
                        flow['unique_domains'].add(qname)
                        
                        domain_entropy = self._calculate_entropy(qname)
                        
                        try:
                            extracted = tldextract.extract(qname)
                            domain = extracted.domain
                            tld = extracted.suffix
                            
                            dga_score = 0
                            if domain_entropy > 3.5:
                                dga_score += 0.5
                            if len(domain) > 15 or len(domain) < 4:
                                dga_score += 0.2
                            consonants = sum(1 for c in domain if c.lower() in 'bcdfghjklmnpqrstvwxyz')
                            vowels = sum(1 for c in domain if c.lower() in 'aeiou')
                            if vowels > 0 and consonants / vowels > 3:
                                dga_score += 0.3
                                
                            if len(self.dns_queries) < 1000:
                                self.dns_queries.append({
                                    'query': qname,
                                    'entropy': domain_entropy,
                                    'dga_score': dga_score
                                })
                                
                        except Exception as e:
                            self.logger.debug(f"Error processing DNS domain: {str(e)}")
                            
                elif dns.qr == 1:  # DNS response
                    if dns.rcode != 0:
                        flow['failed_queries'] += 1
                        
                    answers = []
                    if dns.ancount > 0:
                        for i in range(dns.ancount):
                            try:
                                dnsrr = dns.an[i]
                                answer = {
                                    'name': '',
                                    'type': dnsrr.type if hasattr(dnsrr, 'type') else 0,
                                    'ttl': dnsrr.ttl if hasattr(dnsrr, 'ttl') else 0
                                }
                                
                                if hasattr(dnsrr, 'rrname'):
                                    try:
                                        answer['name'] = dnsrr.rrname.decode('utf-8', errors='ignore')
                                    except:
                                        answer['name'] = str(dnsrr.rrname)
                                        
                                answers.append(answer)
                            except Exception as e:
                                self.logger.debug(f"Error processing DNS answer: {str(e)}")
                    
                    flow['dns_responses'].append({
                        'answers': answers,
                        'rcode': dns.rcode
                    })
                    
        except Exception as e:
            self.logger.debug(f"Error analyzing DNS traffic: {str(e)}")
            self.errors_encountered += 1
            
    def _analyze_command_payload(self, payload_text, flow):
        """Analyze payload for command patterns."""
        try:
            detected_commands = set()
            for cmd in self.command_keywords:
                pattern = r'\b' + re.escape(cmd) + r'\b'
                if re.search(pattern, payload_text, re.IGNORECASE):
                    detected_commands.add(cmd)
                    flow['command_keywords_detected'].add(cmd)
                    
            # Detect script languages
            script_languages = set()
            
            if any(x in payload_text for x in ['function(', 'var ', 'let ', 'const ', 'document.', 'window.']):
                script_languages.add('javascript')
                
            if any(x in payload_text for x in ['$PSVersionTable', '-ExecutionPolicy', 'Get-Process', 'Set-Item']):
                script_languages.add('powershell')
                
            if any(x in payload_text for x in ['import ', 'def ', '__main__', 'class ', 'print(']):
                script_languages.add('python')
                
            if any(x in payload_text for x in ['#!/bin/bash', 'echo ', 'grep ', 'awk ', '$(']):
                script_languages.add('bash')
                
            if any(x in payload_text.upper() for x in ['SELECT ', 'INSERT INTO', 'UPDATE ', 'DELETE FROM']):
                script_languages.add('sql')
                
            flow['script_languages'].update(script_languages)
            
            dangerous_funcs = ['eval(', 'exec(', 'system(', 'shell_exec(', 'subprocess.', 'os.system']
            dangerous_count = sum(payload_text.count(func) for func in dangerous_funcs)
            
            return {
                'command_keywords': detected_commands,
                'script_languages': script_languages,
                'dangerous_functions': dangerous_count
            }
            
        except Exception as e:
            self.logger.debug(f"Error analyzing command payload: {str(e)}")
            self.errors_encountered += 1
            return {}
            
    def _detect_obfuscation(self, payload_text, payload_bytes, flow):
        """Detect obfuscation techniques in payload."""
        obfuscation_score = 0
        encryption_score = 0
        
        try:
            # Check for base64 encoded content
            base64_pattern = r'[A-Za-z0-9+/]{20,}={0,2}'
            base64_matches = re.findall(base64_pattern, payload_text)
            
            if base64_matches:
                valid_base64 = []
                for match in base64_matches:
                    try:
                        decoded = base64.b64decode(match)
                        if len(match) >= 30:
                            valid_base64.append(match)
                            flow['base64_strings'].append(match[:50])
                            obfuscation_score += 1
                    except:
                        pass
                        
                if valid_base64:
                    flow['obfuscation_indicators'] += len(valid_base64)
                    
            # Check for hex encoded content
            hex_pattern = r'\\x[0-9a-fA-F]{2}|0x[0-9a-fA-F]{2}'
            hex_matches = re.findall(hex_pattern, payload_text)
            if len(hex_matches) > 5:
                obfuscation_score += 1
                flow['obfuscation_indicators'] += 1
                
            # Check for unicode escapes
            unicode_pattern = r'\\u[0-9a-fA-F]{4}'
            unicode_matches = re.findall(unicode_pattern, payload_text)
            if len(unicode_matches) > 5:
                obfuscation_score += 1
                flow['obfuscation_indicators'] += 1
                
            # Check for high entropy
            entropy = self._calculate_entropy(payload_text)
            if entropy > 7.0:
                encryption_score += 2
                flow['encryption_indicators'] += 1
            elif entropy > 6.0:
                encryption_score += 1
                
            # Check for even distribution of bytes
            if len(payload_bytes) > 100:
                byte_counts = Counter(payload_bytes)
                byte_values = list(byte_counts.values())
                
                if len(byte_values) > 10:
                    std_dev = statistics.stdev(byte_values) if len(byte_values) > 1 else 0
                    mean = statistics.mean(byte_values)
                    
                    if mean > 0 and std_dev / mean < 0.5:
                        encryption_score += 1
                        flow['encryption_indicators'] += 1
                        
            return {
                'obfuscation_score': obfuscation_score,
                'encryption_score': encryption_score,
                'entropy': entropy if 'entropy' in locals() else 0
            }
            
        except Exception as e:
            self.logger.debug(f"Error detecting obfuscation: {str(e)}")
            self.errors_encountered += 1
            return {
                'obfuscation_score': 0,
                'encryption_score': 0,
                'entropy': 0
            }
            
    def _calculate_entropy(self, data):
        """Calculate Shannon entropy of data."""
        if not data:
            return 0
            
        if isinstance(data, str):
            data = data.encode('utf-8', errors='ignore')
            
        byte_counts = Counter(data)
        data_len = len(data)
        
        entropy = 0
        for count in byte_counts.values():
            probability = count / data_len
            entropy -= probability * math.log2(probability)
            
        return entropy
        
    def _extract_packet_semantic_features(self, packet, flow, is_forward):
        """Extract detailed semantic features from a single packet."""
        if not self.deep_inspection:
            return None
            
        try:
            features = {}
            
            # Initialize with default values
            for feature_name in self.feature_names:
                features[feature_name] = 0
                
            # Extract payload if available
            payload_text = None
            if packet.haslayer(Raw):
                payload_bytes = bytes(packet[Raw].load)
                try:
                    payload_text = payload_bytes.decode('utf-8', errors='ignore')
                except:
                    pass
                    
            # Extract URL semantics
            if packet.haslayer(TCP) and payload_text:
                dst_port = packet[TCP].dport if is_forward else packet[TCP].sport
                
                if dst_port in (80, 8080, 443, 8443):
                    http_methods = ('GET', 'POST', 'PUT', 'DELETE', 'HEAD', 'OPTIONS', 'CONNECT', 'TRACE', 'PATCH')
                    for method in http_methods:
                        if payload_text.startswith(method + ' '):
                            url_match = re.search(f"{method} (.*?) HTTP", payload_text)
                            if url_match:
                                url = url_match.group(1).strip()
                                
                                full_url = url
                                if not url.startswith('http'):
                                    host_header = re.search(r"Host: (.*?)[\r\n]", payload_text)
                                    host = host_header.group(1).strip() if host_header else flow['dst_ip']
                                    
                                    if dst_port in (80, 8080):
                                        full_url = f"http://{host}{url}"
                                    elif dst_port in (443, 8443):
                                        full_url = f"https://{host}{url}"
                                        
                                url_semantics = self._analyze_url_semantics(full_url, flow)
                                
                                if url_semantics:
                                    features['domain_entropy'] = url_semantics.get('domain_entropy', 0)
                                    features['subdomain_count'] = url_semantics.get('subdomain_count', 0)
                                    features['suspicious_keywords'] = url_semantics.get('suspicious_keywords_count', 0)
                                    features['url_length'] = url_semantics.get('url_length', 0)
                                    features['special_char_ratio'] = url_semantics.get('special_char_ratio', 0)
                                    features['numeric_ratio'] = url_semantics.get('numeric_ratio', 0)
                                    
                                    tld = url_semantics.get('tld', '')
                                    features['tld_reputation'] = self.tld_reputation[tld] if tld in self.tld_reputation else 0.5
                                    
                                    features['path_depth'] = url_semantics.get('path_depth', 0)
                                    features['param_complexity'] = url_semantics.get('param_count', 0)
                                    
                            headers = self._extract_http_headers(payload_text)
                            
                            if 'User-Agent' in headers:
                                ua = headers['User-Agent']
                                features['user_agent_entropy'] = self._calculate_entropy(ua)
                                
                            uncommon_headers = [h for h in headers.keys() if h not in {
                                'Host', 'User-Agent', 'Accept', 'Accept-Language', 'Accept-Encoding',
                                'Connection', 'Referer', 'Cookie', 'Content-Type', 'Content-Length'
                            }]
                            features['header_anomalies'] = len(uncommon_headers)
                            
                            if 'Referer' in headers:
                                features['referrer_consistency'] = 1.0
                                
                            if 'Cookie' in headers:
                                cookie = headers['Cookie']
                                cookie_parts = cookie.split(';')
                                features['cookie_complexity'] = len(cookie_parts)
                                
                            break
                            
                    if payload_text.startswith('HTTP/'):
                        status_match = re.search(r"HTTP/[\d\.]+ (\d+)", payload_text)
                        if status_match:
                            status_code = int(status_match.group(1))
                            
                            if 200 <= status_code < 300:
                                features['status_code_distribution'] = 0.25
                            elif 300 <= status_code < 400:
                                features['status_code_distribution'] = 0.5
                                features['redirect_frequency'] = 1.0
                            elif 400 <= status_code < 500:
                                features['status_code_distribution'] = 0.75
                            else:
                                features['status_code_distribution'] = 1.0
                                
                        ct_match = re.search(r"Content-Type: (.*?)[\r\n]", payload_text)
                        if ct_match:
                            content_type = ct_match.group(1).strip()
                            features['content_type_variety'] = 1.0
                            features['mime_type_entropy'] = self._calculate_entropy(content_type)
                            
            # Extract DNS semantics
            if packet.haslayer(UDP) and (packet[UDP].dport == 53 or packet[UDP].sport == 53):
                if packet.haslayer(DNS):
                    dns = packet[DNS]
                    
                    if dns.qr == 0 and dns.qd:
                        qname = ""
                        try:
                            qname = dns.qd.qname.decode('utf-8', errors='ignore')
                        except:
                            if hasattr(dns.qd, 'qname'):
                                qname = str(dns.qd.qname)
                                
                        if qname:
                            features['query_entropy'] = self._calculate_entropy(qname)
                            
                            parts = qname.split('.')
                            if len(parts) > 2:
                                subdomains = '.'.join(parts[:-2])
                                features['subdomain_entropy'] = self._calculate_entropy(subdomains)
                                
                            try:
                                extracted = tldextract.extract(qname)
                                domain = extracted.domain
                                
                                dga_score = 0
                                domain_entropy = self._calculate_entropy(domain)
                                if domain_entropy > 3.5:
                                    dga_score += 0.5
                                if len(domain) > 15 or len(domain) < 4:
                                    dga_score += 0.2
                                consonants = sum(1 for c in domain if c.lower() in 'bcdfghjklmnpqrstvwxyz')
                                vowels = sum(1 for c in domain if c.lower() in 'aeiou')
                                if vowels > 0 and consonants / vowels > 3:
                                    dga_score += 0.3
                                    
                                features['dga_probability'] = min(dga_score, 1.0)
                            except:
                                features['dga_probability'] = 0
                                
                    elif dns.qr == 1:
                        if dns.rcode != 0:
                            features['failed_queries'] = 1.0
                            
            # Command and payload analysis
            if payload_text:
                payload_analysis = self._analyze_command_payload(payload_text, flow)
                if payload_analysis:
                    features['command_keywords'] = len(payload_analysis.get('command_keywords', []))
                    features['script_language_mix'] = len(payload_analysis.get('script_languages', []))
                    features['dangerous_functions'] = payload_analysis.get('dangerous_functions', 0)
                    
                obfuscation_data = self._detect_obfuscation(payload_text, 
                                                        payload_bytes if 'payload_bytes' in locals() else b'', 
                                                        flow)
                if obfuscation_data:
                    features['obfuscation_indicators'] = obfuscation_data.get('obfuscation_score', 0)
                    features['encryption_indicators'] = obfuscation_data.get('encryption_score', 0)
                    
            # Flow-level statistical features
            features['query_frequency'] = len(flow['dns_queries']) / max(flow['duration'], 0.1) if flow['duration'] > 0 else 0
            features['unique_domains'] = len(flow['unique_domains'])
            features['host_diversity'] = len({u.split('/')[2] for u in flow['urls_requested'] if '/' in u and u.count('/') >= 2})
            
            https_count = sum(1 for u in flow['urls_requested'] if u.startswith('https://'))
            http_count = sum(1 for u in flow['urls_requested'] if u.startswith('http://'))
            features['https_ratio'] = https_count / max(https_count + http_count, 1)
            
            content_types = set(flow['content_types'])
            methods = set(flow['http_methods'])
            features['semantic_consistency'] = 1.0 - (len(content_types) + len(methods)) / max(
                len(flow['http_requests']) + len(flow['http_responses']), 1)
                
            return features
            
        except Exception as e:
            self.logger.debug(f"Error extracting packet semantic features: {str(e)}")
            self.errors_encountered += 1
            return None
            
    def _collect_visualization_data(self, packet, flow):
        """Collect data for visualizations during packet processing."""
        self.sample_counter += 1
        
        if self.sample_counter % self.sample_rate != 0:
            return
            
        if packet.haslayer(TCP) and packet.haslayer(Raw):
            payload = packet[Raw].load
            try:
                payload_text = payload.decode('utf-8', errors='ignore')
                for method in ('GET', 'POST', 'PUT', 'DELETE', 'HEAD', 'OPTIONS'):
                    if payload_text.startswith(method + ' '):
                        flow['http_methods'].append(method)
                        break
            except:
                pass
                
    def cleanup_old_flows(self, current_time):
        """Remove flows that have timed out to manage memory."""
        if current_time - self.last_cleanup_time < 300:
            return
            
        self.last_cleanup_time = current_time
        timeout_threshold = current_time - self.timeout_seconds
        
        expired_flows = [
            flow_key for flow_key, flow in self.flows.items()
            if flow['end_time'] < timeout_threshold
        ]
        
        for flow_key in expired_flows:
            del self.flows[flow_key]
            
        if expired_flows:
            self.logger.info(f"Cleaned up {len(expired_flows)} expired flows")
            
        if len(expired_flows) > 1000:
            gc.collect()
            
    def get_memory_usage(self):
        """Get current memory usage in MB for monitoring."""
        process = psutil.Process(os.getpid())
        return process.memory_info().rss / 1024 / 1024
        
    def process_pcap_file(self, pcap_file_path):
        """Process a PCAP file and extract semantic features."""
        self.logger.info(f"Starting to process PCAP file: {pcap_file_path}")
        
        if not os.path.exists(pcap_file_path):
            raise FileNotFoundError(f"PCAP file not found: {pcap_file_path}")
            
        file_size_gb = os.path.getsize(pcap_file_path) / (1024 ** 3)
        self.logger.info(f"File size: {file_size_gb:.2f} GB")
        
        packet_count = 0
        chunk_count = 0
        start_time = time.time()
        
        try:
            with PcapReader(pcap_file_path) as pcap_reader:
                print("Processing packets...")
                last_report_time = time.time()
                
                packet_counter = 0
                packet_id = 0
                for packet in pcap_reader:
                    try:
                        packet_time = float(packet.time)
                        self.process_packet(packet, packet_time, packet_id)
                        
                        self.cleanup_old_flows(packet_time)
                        
                        packet_count += 1
                        packet_counter += 1
                        packet_id += 1
                        
                        if packet_counter % 10000 == 0:
                            current_time = time.time()
                            elapsed = current_time - last_report_time
                            rate = 10000 / elapsed if elapsed > 0 else 0
                            print(f"Processed {packet_count:,} packets (Rate: {rate:.0f} packets/sec)")
                            last_report_time = current_time
                            
                        if packet_count % self.chunk_size == 0:
                            chunk_count += 1
                            
                            memory_mb = self.get_memory_usage()
                            active_flows = len(self.flows)
                            
                            self.logger.info(
                                f"Chunk {chunk_count}: Processed {packet_count:,} packets, "
                                f"Active flows: {active_flows:,}, Memory: {memory_mb:.1f} MB"
                            )
                            
                            if memory_mb > 4000:
                                gc.collect()
                                self.logger.info("Performed garbage collection")
                                
                    except Exception as e:
                        self.logger.warning(f"Error processing packet {packet_count}: {str(e)}")
                        self.errors_encountered += 1
                        continue
                        
        except Exception as e:
            self.logger.error(f"Error reading PCAP file: {str(e)}")
            raise
            
        # Extract features from flows
        self.logger.info("Extracting semantic features from flows...")
        
        feature_matrix = []
        flow_metadata = []
        
        total_flows = len(self.flows)
        print(f"Extracting features from {total_flows:,} flows...")
        
        flow_counter = 0
        successful_flows = 0
        
        flows_with_data = sum(1 for flow in self.flows.values() if flow['has_semantic_data'] or 
                         (flow['fwd_packets'] + flow['bwd_packets'] >= self.min_packets_for_analysis))
                         
        print(f"Found {flows_with_data:,} flows with useful semantic data")
        
        for flow_key, flow in self.flows.items():
            try:
                if not flow['has_semantic_data'] and (flow['fwd_packets'] + flow['bwd_packets'] < self.min_packets_for_analysis):
                    continue
                    
                semantic_features = self.extract_semantic_features_from_flow(flow)
                feature_matrix.append(semantic_features)
                
                flow_metadata.append({
                    'src_ip': flow['src_ip'],
                    'dst_ip': flow['dst_ip'],
                    'src_port': flow['src_port'],
                    'dst_port': flow['dst_port'],
                    'protocol': flow['protocol'],
                    'duration': flow['duration'],
                    'total_packets': flow['fwd_packets'] + flow['bwd_packets'],
                    'total_bytes': flow['fwd_bytes'] + flow['bwd_bytes'],
                    'http_requests': len(flow['http_requests']),
                    'dns_queries': len(flow['dns_queries'])
                })
                
                self.flows_analyzed += 1
                successful_flows += 1
                
                flow_counter += 1
                if flow_counter % 1000 == 0:
                    print(f"Processed {flow_counter:,}/{total_flows:,} flows")
                    
            except Exception as e:
                self.logger.warning(f"Error extracting features from flow: {str(e)}")
                self.errors_encountered += 1
                continue
                
        print(f"Completed processing {successful_flows:,}/{total_flows:,} flows")
        
        # Create DataFrames from extracted data
        if feature_matrix:
            df_features = pd.DataFrame(feature_matrix, columns=self.feature_names)
            df_metadata = pd.DataFrame(flow_metadata)
            
            result_df = pd.concat([df_metadata, df_features], axis=1)
            self.features_extracted = len(feature_matrix)
        else:
            columns = ['src_ip', 'dst_ip', 'src_port', 'dst_port', 'protocol', 
                       'duration', 'total_packets', 'total_bytes', 'http_requests', 
                       'dns_queries'] + self.feature_names
            result_df = pd.DataFrame(columns=columns)
            
        # Log final processing statistics
        processing_time = time.time() - start_time
        self.logger.info(f"Processing time: {processing_time:.2f} seconds")
        self.logger.info(f"Packets per second: {packet_count / processing_time:.1f}")
        self.logger.info(f"Total flows extracted: {len(result_df):,}")
        self.logger.info(f"Errors encountered: {self.errors_encountered}")
        
        self.logger.info(f"Summary: Processed {packet_count:,} packets, analyzed {successful_flows:,} flows")
        
        return result_df
        
    def extract_semantic_features_from_flow(self, flow):
        """Extract semantic features from a flow."""
        try:
            features = []
            
            # URL Analysis Features (8 features)
            avg_domain_entropy = statistics.mean(flow['domain_entropy_values']) if flow['domain_entropy_values'] else 0
            features.append(avg_domain_entropy)
            
            features.append(flow.get('subdomain_count', 0))
            features.append(flow.get('suspicious_keywords_count', 0))
            
            domain_age_category = 0.5
            features.append(domain_age_category)
            
            avg_url_length = statistics.mean(flow['url_length_values']) if flow['url_length_values'] else 0
            features.append(avg_url_length)
            
            special_char_ratio = 0.1
            if flow['urls_requested']:
                special_chars = sum(1 for url in flow['urls_requested'] for c in url if not c.isalnum() and c not in '/:.-_')
                total_chars = sum(len(url) for url in flow['urls_requested'])
                if total_chars > 0:
                    special_char_ratio = special_chars / total_chars
            features.append(special_char_ratio)
            
            numeric_ratio = 0.05
            if flow['urls_requested']:
                numeric_chars = sum(1 for url in flow['urls_requested'] for c in url if c.isdigit())
                total_chars = sum(len(url) for url in flow['urls_requested'])
                if total_chars > 0:
                    numeric_ratio = numeric_chars / total_chars
            features.append(numeric_ratio)
            
            tld_reputation = 0.7
            if flow['urls_requested']:
                tld_scores = []
                for url in flow['urls_requested']:
                    try:
                        extracted = tldextract.extract(url)
                        tld = extracted.suffix
                        if tld in self.tld_reputation:
                            tld_scores.append(self.tld_reputation[tld])
                    except:
                        pass
                if tld_scores:
                    tld_reputation = statistics.mean(tld_scores)
            features.append(tld_reputation)
            
            # DNS Query Features (6 features)
            query_entropy = 0
            if flow['dns_queries']:
                entropy_values = [self._calculate_entropy(query) for query in flow['dns_queries']]
                query_entropy = statistics.mean(entropy_values) if entropy_values else 0
            features.append(query_entropy)
            
            dga_probability = 0
            if flow['dns_queries']:
                dga_scores = []
                for query in flow['dns_queries']:
                    try:
                        extracted = tldextract.extract(query)
                        domain = extracted.domain
                        
                        dga_score = 0
                        domain_entropy = self._calculate_entropy(domain)
                        if domain_entropy > 3.5:
                            dga_score += 0.5
                        if len(domain) > 15 or len(domain) < 4:
                            dga_score += 0.2
                        consonants = sum(1 for c in domain if c.lower() in 'bcdfghjklmnpqrstvwxyz')
                        vowels = sum(1 for c in domain if c.lower() in 'aeiou')
                        if vowels > 0 and consonants / vowels > 3:
                            dga_score += 0.3
                            
                        dga_scores.append(min(dga_score, 1.0))
                    except:
                        pass
                        
                if dga_scores:
                    dga_probability = statistics.mean(dga_scores)
            features.append(dga_probability)
            
            duration_minutes = flow['duration'] / 60 if flow['duration'] > 0 else 0.1
            query_freq = len(flow['dns_queries']) / duration_minutes if duration_minutes > 0 else 0
            features.append(query_freq)
            
            unique_domains = len(flow['unique_domains'])
            features.append(unique_domains)
            
            features.append(flow.get('failed_queries', 0))
            
            subdomain_entropy = 0
            if flow['dns_queries']:
                subdomain_entropies = []
                for query in flow['dns_queries']:
                    parts = query.split('.')
                    if len(parts) > 2:
                        subdomains = '.'.join(parts[:-2])
                        subdomain_entropies.append(self._calculate_entropy(subdomains))
                if subdomain_entropies:
                    subdomain_entropy = statistics.mean(subdomain_entropies)
            features.append(subdomain_entropy)
            
            # HTTP Content Features (6 features)
            user_agent_entropy = 0
            if flow['user_agents']:
                ua_entropies = [self._calculate_entropy(ua) for ua in flow['user_agents']]
                user_agent_entropy = statistics.mean(ua_entropies) if ua_entropies else 0
            features.append(user_agent_entropy)
            
            header_anomalies = 0
            for req in flow['http_requests']:
                headers = req.get('headers', {})
                uncommon_headers = [h for h in headers.keys() if h not in {
                    'Host', 'User-Agent', 'Accept', 'Accept-Language', 'Accept-Encoding',
                    'Connection', 'Referer', 'Cookie', 'Content-Type', 'Content-Length'
                }]
                header_anomalies += len(uncommon_headers)
            features.append(header_anomalies)
            
            content_type_variety = len(set(flow['content_types']))
            features.append(content_type_variety)
            
            status_code_distribution = 0.5
            if flow['status_codes']:
                status_counts = Counter(flow['status_codes'])
                total_responses = len(flow['status_codes'])
                
                distribution = (
                    (sum(status_counts.get(code, 0) for code in range(200, 300)) * 0.2) +
                    (sum(status_counts.get(code, 0) for code in range(300, 400)) * 0.4) +
                    (sum(status_counts.get(code, 0) for code in range(400, 500)) * 0.7) +
                    (sum(status_counts.get(code, 0) for code in range(500, 600)) * 1.0)
                ) / max(total_responses, 1)
                
                status_code_distribution = distribution
            features.append(status_code_distribution)
            
            referrer_consistency = 1.0
            referrers = []
            for req in flow['http_requests']:
                headers = req.get('headers', {})
                if 'Referer' in headers:
                    referrers.append(headers['Referer'])
                    
            if referrers:
                referrer_consistency = len(set(referrers)) / len(referrers)
            features.append(referrer_consistency)
            
            cookie_complexity = 0
            cookie_parts = []
            for req in flow['http_requests']:
                headers = req.get('headers', {})
                if 'Cookie' in headers:
                    cookie = headers['Cookie']
                    parts = cookie.split(';')
                    cookie_parts.extend(parts)
                    
            if cookie_parts:
                cookie_complexity = len(cookie_parts)
            features.append(cookie_complexity)
            
            # Command & Payload Features (5 features)
            command_keywords_count = len(flow['command_keywords_detected'])
            features.append(command_keywords_count)
            
            features.append(flow.get('obfuscation_indicators', 0))
            
            script_language_count = len(flow['script_languages'])
            features.append(script_language_count)
            
            dangerous_functions = 0
            for payload in flow.get('payloads', []):
                for func in ['eval(', 'exec(', 'system(', 'shell_exec(', 'subprocess.', 'os.system']:
                    dangerous_functions += payload.count(func)
            features.append(dangerous_functions)
            
            features.append(flow.get('encryption_indicators', 0))
            
            # Additional Semantic Features (8 features)
            redirect_frequency = 0
            if flow['status_codes']:
                redirect_count = sum(1 for code in flow['status_codes'] if 300 <= code < 400)
                redirect_frequency = redirect_count / len(flow['status_codes'])
            features.append(redirect_frequency)
            
            https_ratio = 0.5
            if flow['urls_requested']:
                https_count = sum(1 for url in flow['urls_requested'] if url.startswith('https://'))
                http_count = sum(1 for url in flow['urls_requested'] if url.startswith('http://'))
                if https_count + http_count > 0:
                    https_ratio = https_count / (https_count + http_count)
            features.append(https_ratio)
            
            path_depth = 0
            if flow['urls_requested']:
                path_depths = []
                for url in flow['urls_requested']:
                    try:
                        parsed = urllib.parse.urlparse(url)
                        depth = len([p for p in parsed.path.split('/') if p])
                        path_depths.append(depth)
                    except:
                        pass
                if path_depths:
                    path_depth = statistics.mean(path_depths)
            features.append(path_depth)
            
            param_complexity = 0
            if flow['urls_requested']:
                param_counts = []
                for url in flow['urls_requested']:
                    try:
                        parsed = urllib.parse.urlparse(url)
                        params = urllib.parse.parse_qs(parsed.query)
                        param_counts.append(len(params))
                    except:
                        pass
                if param_counts:
                    param_complexity = statistics.mean(param_counts)
            features.append(param_complexity)
            
            host_diversity = 0
            if flow['urls_requested']:
                hosts = set()
                for url in flow['urls_requested']:
                    try:
                        parsed = urllib.parse.urlparse(url)
                        if parsed.netloc:
                            hosts.add(parsed.netloc)
                    except:
                        pass
                host_diversity = len(hosts)
            features.append(host_diversity)
            
            mime_type_entropy = 0
            if flow['content_types']:
                mime_type_entropy = self._calculate_entropy(''.join(flow['content_types']))
            features.append(mime_type_entropy)
            
            certificate_validity = 0.5
            features.append(certificate_validity)
            
            semantic_consistency = 0.5
            if flow['http_requests']:
                methods = [req.get('method', '') for req in flow['http_requests']]
                method_consistency = len(set(methods)) / len(methods) if methods else 1.0
                semantic_consistency = 1.0 - method_consistency
            features.append(semantic_consistency)
            
            return features
            
        except Exception as e:
            self.logger.warning(f"Error in feature extraction: {str(e)}")
            self.errors_encountered += 1
            return [0] * len(self.feature_names)
            
    def generate_visualizations(self, output_dir, features_df):
        """Generate comprehensive visualizations of the semantic analysis."""
        if not self.enable_visualizations:
            return
            
        self.logger.info("Generating visualizations...")
        
        viz_dir = os.path.join(output_dir, "visualizations")
        os.makedirs(viz_dir, exist_ok=True)
        
        viz_count = 0
        
        # Generate matplotlib-based visualizations
        if MATPLOTLIB_AVAILABLE:
            if self._create_url_feature_visualization(viz_dir):
                viz_count += 1
                
            if self._create_dns_analysis_visualization(viz_dir):
                viz_count += 1
                
            if self._create_http_content_visualization(viz_dir):
                viz_count += 1
                
            if self._create_command_payload_visualization(viz_dir):
                viz_count += 1
                
            if self._create_feature_correlation_matrix(viz_dir, features_df):
                viz_count += 1
                
        # Generate network graph
        if NETWORKX_AVAILABLE and MATPLOTLIB_AVAILABLE:
            if self._create_semantic_network_graph(viz_dir):
                viz_count += 1
                
        # Generate interactive dashboard
        if PLOTLY_AVAILABLE:
            if self._create_interactive_dashboard(viz_dir, features_df):
                viz_count += 1
                
            if self._create_threat_score_visualization(viz_dir, features_df):
                viz_count += 1
                
        # Generate text-based summary
        if self._create_text_summary(viz_dir, features_df):
            viz_count += 1
                
        self.logger.info(f"{viz_count} visualizations saved to: {viz_dir}")
        
    def _create_url_feature_visualization(self, viz_dir):
        """Create visualization for URL analysis features."""
        try:
            if not self.urls:
                return False
                
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
            fig.suptitle('URL Semantic Analysis', fontsize=16, color='white')
            
            # URL length distribution
            ax1 = axes[0, 0]
            url_lengths = []
            for flow in self.flows.values():
                url_lengths.extend(flow['url_length_values'])
                
            if url_lengths:
                ax1.hist(url_lengths, bins=30, color='#4ecdc4', alpha=0.7, edgecolor='white')
                ax1.set_title('URL Length Distribution', color='white')
                ax1.set_xlabel('Length (characters)', color='white')
                ax1.set_ylabel('Frequency', color='white')
                ax1.tick_params(colors='white')
                ax1.grid(True, alpha=0.3)
                
                ax1.axvline(x=100, color='#ff6b6b', linestyle='--', alpha=0.8)
                ax1.text(105, ax1.get_ylim()[1] * 0.9, 'Suspicious Length Threshold',
                         color='#ff6b6b', fontsize=10, va='top')
                         
            # Domain entropy distribution
            ax2 = axes[0, 1]
            domain_entropies = []
            for flow in self.flows.values():
                domain_entropies.extend(flow['domain_entropy_values'])
                
            if domain_entropies:
                ax2.hist(domain_entropies, bins=30, color='#ff6b6b', alpha=0.7, edgecolor='white')
                ax2.set_title('Domain Entropy Distribution', color='white')
                ax2.set_xlabel('Entropy', color='white')
                ax2.set_ylabel('Frequency', color='white')
                ax2.tick_params(colors='white')
                ax2.grid(True, alpha=0.3)
                
                ax2.axvline(x=3.5, color='#4ecdc4', linestyle='--', alpha=0.8)
                ax2.text(3.6, ax2.get_ylim()[1] * 0.9, 'High Entropy Threshold',
                         color='#4ecdc4', fontsize=10, va='top')
                         
            # Top suspicious keywords
            ax3 = axes[1, 0]
            if self.suspicious_patterns:
                pattern_counts = self.suspicious_patterns.most_common(10)
                pattern_names = [p[0].split(':')[1] for p in pattern_counts]
                pattern_values = [p[1] for p in pattern_counts]
                pattern_severity = [p[0].split(':')[0] for p in pattern_counts]
                
                colors = []
                for severity in pattern_severity:
                    if severity == 'high':
                        colors.append('#ff6b6b')
                    elif severity == 'medium':
                        colors.append('#ffa94d')
                    else:
                        colors.append('#ffd166')
                        
                bars = ax3.barh(pattern_names, pattern_values, color=colors, alpha=0.7)
                ax3.set_title('Top 10 Suspicious Keywords', color='white')
                ax3.set_xlabel('Frequency', color='white')
                ax3.set_ylabel('Keyword', color='white')
                ax3.tick_params(colors='white')
                ax3.grid(True, alpha=0.3)
                
            # TLD distribution
            ax4 = axes[1, 1]
            tlds = []
            for url in self.urls:
                try:
                    extracted = tldextract.extract(url)
                    if extracted.suffix:
                        tlds.append(extracted.suffix)
                except:
                    pass
                    
            if tlds:
                tld_counts = Counter(tlds).most_common(10)
                tld_names = [t[0] for t in tld_counts]
                tld_values = [t[1] for t in tld_counts]
                
                ax4.pie(tld_values, labels=tld_names, autopct='%1.1f%%',
                        colors=plt.cm.tab10.colors, startangle=90)
                ax4.set_title('Top 10 TLDs', color='white')
                
            plt.tight_layout()
            plt.savefig(os.path.join(viz_dir, 'url_analysis.png'),
                        facecolor='black', dpi=300, bbox_inches='tight')
            plt.close()
            return True
            
        except Exception as e:
            self.logger.warning(f"Error creating URL visualization: {str(e)}")
            return False

    def _create_dns_analysis_visualization(self, viz_dir):
        """Create visualization for DNS analysis features."""
        try:
            if not self.dns_queries:
                return False
                
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
            fig.suptitle('DNS Query Semantic Analysis', fontsize=16, color='white')
            
            # Query entropy distribution
            ax1 = axes[0, 0]
            query_entropies = [q['entropy'] for q in self.dns_queries if 'entropy' in q]
            if query_entropies:
                ax1.hist(query_entropies, bins=30, color='#4ecdc4', alpha=0.7, edgecolor='white')
                ax1.set_title('DNS Query Entropy Distribution', color='white')
                ax1.set_xlabel('Entropy', color='white')
                ax1.set_ylabel('Frequency', color='white')
                ax1.tick_params(colors='white')
                ax1.grid(True, alpha=0.3)
                
                ax1.axvline(x=4.0, color='#ff6b6b', linestyle='--', alpha=0.8)
                ax1.text(4.1, ax1.get_ylim()[1] * 0.9, 'DGA Threshold',
                         color='#ff6b6b', fontsize=10, va='top')
                         
            # DGA probability distribution
            ax2 = axes[0, 1]
            dga_scores = [q['dga_score'] for q in self.dns_queries if 'dga_score' in q]
            if dga_scores:
                ax2.hist(dga_scores, bins=30, color='#ff6b6b', alpha=0.7, edgecolor='white')
                ax2.set_title('DGA Probability Distribution', color='white')
                ax2.set_xlabel('DGA Probability Score', color='white')
                ax2.set_ylabel('Frequency', color='white')
                ax2.tick_params(colors='white')
                ax2.grid(True, alpha=0.3)
                
                ax2.axvline(x=0.7, color='#4ecdc4', linestyle='--', alpha=0.8)
                ax2.text(0.71, ax2.get_ylim()[1] * 0.9, 'High Suspicion Threshold',
                         color='#4ecdc4', fontsize=10, va='top')
                         
            # Top queried domains
            ax3 = axes[1, 0]
            if self.domains_seen:
                domain_counts = self.domains_seen.most_common(10)
                domain_names = [d[0] for d in domain_counts]
                domain_values = [d[1] for d in domain_counts]
                
                bars = ax3.bar(range(len(domain_names)), domain_values, color='#45b7d1', alpha=0.7)
                ax3.set_title('Top 10 Queried Domains', color='white')
                ax3.set_xlabel('Domain', color='white')
                ax3.set_ylabel('Query Count', color='white')
                ax3.set_xticks(range(len(domain_names)))
                ax3.set_xticklabels(domain_names, rotation=45, ha='right')
                ax3.tick_params(colors='white')
                ax3.grid(True, alpha=0.3)
                
            # Domain length distribution
            ax4 = axes[1, 1]
            domain_lengths = []
            for q in self.dns_queries:
                try:
                    query = q['query']
                    extracted = tldextract.extract(query)
                    if extracted.domain:
                        domain_lengths.append(len(extracted.domain))
                except:
                    pass
                    
            if domain_lengths:
                ax4.hist(domain_lengths, bins=30, color='#4ecdc4', alpha=0.7, edgecolor='white')
                ax4.set_title('Domain Length Distribution', color='white')
                ax4.set_xlabel('Length (characters)', color='white')
                ax4.set_ylabel('Frequency', color='white')
                ax4.tick_params(colors='white')
                ax4.grid(True, alpha=0.3)
                    
            plt.tight_layout()
            plt.savefig(os.path.join(viz_dir, 'dns_analysis.png'),
                        facecolor='black', dpi=300, bbox_inches='tight')
            plt.close()
            return True
            
        except Exception as e:
            self.logger.warning(f"Error creating DNS visualization: {str(e)}")
            return False

    def _create_http_content_visualization(self, viz_dir):
        """Create visualization for HTTP content analysis."""
        try:
            if not self.http_requests and not self.http_responses:
                return False
                
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
            fig.suptitle('HTTP Content Analysis', fontsize=16, color='white')
            
            # HTTP method distribution
            ax1 = axes[0, 0]
            methods = [req['method'] for req in self.http_requests if 'method' in req]
            if methods:
                method_counts = Counter(methods)
                method_names = list(method_counts.keys())
                method_values = list(method_counts.values())
                
                ax1.pie(method_values, labels=method_names, autopct='%1.1f%%',
                        colors=plt.cm.Set3.colors, startangle=90)
                ax1.set_title('HTTP Method Distribution', color='white')
                
            # Status code distribution
            ax2 = axes[0, 1]
            status_codes = [resp['status_code'] for resp in self.http_responses if 'status_code' in resp and resp['status_code'] > 0]
            if status_codes:
                # Group by status code category
                success_count = sum(1 for code in status_codes if 200 <= code < 300)
                redirect_count = sum(1 for code in status_codes if 300 <= code < 400)
                client_error_count = sum(1 for code in status_codes if 400 <= code < 500)
                server_error_count = sum(1 for code in status_codes if 500 <= code < 600)
                
                categories = ['2xx Success', '3xx Redirect', '4xx Client Error', '5xx Server Error']
                counts = [success_count, redirect_count, client_error_count, server_error_count]
                colors = ['#4ecdc4', '#ffd166', '#ff6b6b', '#ff4757']
                
                bars = ax2.bar(categories, counts, color=colors, alpha=0.7)
                ax2.set_title('HTTP Status Code Categories', color='white')
                ax2.set_ylabel('Count', color='white')
                ax2.tick_params(colors='white')
                ax2.grid(True, alpha=0.3)
                
            # User agent entropy
            ax3 = axes[1, 0]
            user_agents = [req['user_agent'] for req in self.http_requests if 'user_agent' in req and req['user_agent']]
            if user_agents:
                ua_entropies = [self._calculate_entropy(ua) for ua in user_agents]
                ax3.hist(ua_entropies, bins=20, color='#45b7d1', alpha=0.7, edgecolor='white')
                ax3.set_title('User Agent Entropy Distribution', color='white')
                ax3.set_xlabel('Entropy', color='white')
                ax3.set_ylabel('Frequency', color='white')
                ax3.tick_params(colors='white')
                ax3.grid(True, alpha=0.3)
                
            # Content type distribution
            ax4 = axes[1, 1]
            content_types = [resp['content_type'] for resp in self.http_responses if 'content_type' in resp and resp['content_type']]
            if content_types:
                # Extract main content type (before semicolon)
                main_types = [ct.split(';')[0].strip() for ct in content_types]
                type_counts = Counter(main_types).most_common(8)
                type_names = [t[0] for t in type_counts]
                type_values = [t[1] for t in type_counts]
                
                bars = ax4.barh(type_names, type_values, color='#ff6b6b', alpha=0.7)
                ax4.set_title('Top Content Types', color='white')
                ax4.set_xlabel('Frequency', color='white')
                ax4.tick_params(colors='white')
                ax4.grid(True, alpha=0.3)
                
            plt.tight_layout()
            plt.savefig(os.path.join(viz_dir, 'http_content_analysis.png'),
                        facecolor='black', dpi=300, bbox_inches='tight')
            plt.close()
            return True
            
        except Exception as e:
            self.logger.warning(f"Error creating HTTP visualization: {str(e)}")
            return False

    def _create_command_payload_visualization(self, viz_dir):
        """Create visualization for command and payload analysis."""
        try:
            if not self.payload_samples:
                return False
                
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
            fig.suptitle('Command & Payload Semantic Analysis', fontsize=16, color='white')
            
            # Payload entropy distribution
            ax1 = axes[0, 0]
            payload_entropies = [sample['entropy'] for sample in self.payload_samples if 'entropy' in sample]
            if payload_entropies:
                ax1.hist(payload_entropies, bins=30, color='#ff6b6b', alpha=0.7, edgecolor='white')
                ax1.set_title('Payload Entropy Distribution', color='white')
                ax1.set_xlabel('Entropy', color='white')
                ax1.set_ylabel('Frequency', color='white')
                ax1.tick_params(colors='white')
                ax1.grid(True, alpha=0.3)
                
                ax1.axvline(x=5.0, color='#ffd166', linestyle='--', alpha=0.8)
                ax1.axvline(x=7.0, color='#ff6b6b', linestyle='--', alpha=0.8)
                ax1.text(5.1, ax1.get_ylim()[1] * 0.9, 'Medium Suspicion',
                         color='#ffd166', fontsize=10, va='top')
                ax1.text(7.1, ax1.get_ylim()[1] * 0.8, 'High Suspicion (Encryption)',
                         color='#ff6b6b', fontsize=10, va='top')
                         
            # Payload size distribution
            ax2 = axes[0, 1]
            payload_sizes = [sample['size'] for sample in self.payload_samples if 'size' in sample]
            if payload_sizes:
                ax2.hist(payload_sizes, bins=30, color='#4ecdc4', alpha=0.7, edgecolor='white')
                ax2.set_title('Payload Size Distribution', color='white')
                ax2.set_xlabel('Size (bytes)', color='white')
                ax2.set_ylabel('Frequency', color='white')
                ax2.tick_params(colors='white')
                ax2.grid(True, alpha=0.3)
                
            # Command keyword distribution
            ax3 = axes[1, 0]
            command_keywords = []
            for flow in self.flows.values():
                command_keywords.extend(flow['command_keywords_detected'])
                
            if command_keywords:
                keyword_counts = Counter(command_keywords).most_common(10)
                keyword_names = [k[0] for k in keyword_counts]
                keyword_values = [k[1] for k in keyword_counts]
                
                bars = ax3.barh(keyword_names, keyword_values, color='#ff6b6b', alpha=0.7)
                ax3.set_title('Top 10 Command Keywords', color='white')
                ax3.set_xlabel('Frequency', color='white')
                ax3.set_ylabel('Keyword', color='white')
                ax3.tick_params(colors='white')
                ax3.grid(True, alpha=0.3)
                
                high_severity = ['shell', 'exec', 'eval', 'system', 'powershell', 'chmod']
                medium_severity = ['select', 'insert', 'update', 'delete', 'wget', 'curl']
                
                for i, keyword in enumerate(keyword_names):
                    if keyword in high_severity:
                        bars[i].set_color('#ff6b6b')
                    elif keyword in medium_severity:
                        bars[i].set_color('#ffa94d')
                    else:
                        bars[i].set_color('#ffd166')
                        
            # Script language distribution
            ax4 = axes[1, 1]
            script_languages = []
            for flow in self.flows.values():
                script_languages.extend(flow['script_languages'])
                
            if script_languages:
                language_counts = Counter(script_languages)
                language_names = list(language_counts.keys())
                language_values = [language_counts[name] for name in language_names]
                
                ax4.pie(language_values, labels=language_names, autopct='%1.1f%%',
                        colors=plt.cm.tab10.colors, startangle=90)
                ax4.set_title('Script Language Distribution', color='white')
                
            plt.tight_layout()
            plt.savefig(os.path.join(viz_dir, 'command_payload_analysis.png'),
                        facecolor='black', dpi=300, bbox_inches='tight')
            plt.close()
            return True
            
        except Exception as e:
            self.logger.warning(f"Error creating payload visualization: {str(e)}")
            return False

    def _create_feature_correlation_matrix(self, viz_dir, features_df):
        """Create correlation matrix heatmap for semantic features."""
        try:
            if features_df.empty:
                return False
                
            feature_cols = [col for col in features_df.columns if col in self.feature_names]
            feature_df = features_df[feature_cols]
            
            corr_matrix = feature_df.corr()
            
            plt.figure(figsize=(15, 13))
            plt.title('Semantic Feature Correlation Matrix', fontsize=16, color='white', pad=20)
            
            if 'seaborn' in sys.modules:
                import seaborn as sns
                mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
                sns.heatmap(corr_matrix, mask=mask, annot=False, cmap='RdYlBu_r',
                            center=0, square=True, linewidths=0.5,
                            cbar_kws={"shrink": .8})
            else:
                plt.imshow(corr_matrix, cmap='RdYlBu_r', aspect='auto', interpolation='nearest')
                plt.colorbar(label='Correlation Coefficient', shrink=0.8)
                
                tick_positions = range(len(corr_matrix.columns))
                plt.xticks(tick_positions, corr_matrix.columns, rotation=90, ha='right')
                plt.yticks(tick_positions, corr_matrix.columns)
                
            plt.xticks(rotation=90, ha='right', color='white')
            plt.yticks(rotation=0, color='white')
            plt.tight_layout()
            plt.savefig(os.path.join(viz_dir, 'feature_correlation_matrix.png'),
                        facecolor='black', dpi=300, bbox_inches='tight')
            plt.close()
            return True
            
        except Exception as e:
            self.logger.warning(f"Error creating correlation matrix: {str(e)}")
            return False

    def _create_semantic_network_graph(self, viz_dir):
        """Create network graph based on semantic relationships."""
        if not (NETWORKX_AVAILABLE and MATPLOTLIB_AVAILABLE):
            return False
            
        try:
            G = nx.Graph()
            
            # Add nodes for domains
            domains = []
            for flow in self.flows.values():
                for url in flow['urls_requested']:
                    try:
                        extracted = tldextract.extract(url)
                        full_domain = f"{extracted.domain}.{extracted.suffix}"
                        if full_domain not in domains:
                            domains.append(full_domain)
                    except:
                        pass
                        
                for query in flow['dns_queries']:
                    try:
                        extracted = tldextract.extract(query)
                        full_domain = f"{extracted.domain}.{extracted.suffix}"
                        if full_domain not in domains:
                            domains.append(full_domain)
                    except:
                        pass
                        
            # Add domain nodes
            for domain in domains[:100]:
                G.add_node(domain, type='domain')
                
            # Add edges based on flow relationships
            edge_count = 0
            for flow_key, flow in self.flows.items():
                src_ip = flow['src_ip']
                dst_ip = flow['dst_ip']
                
                G.add_node(src_ip, type='ip')
                G.add_node(dst_ip, type='ip')
                
                G.add_edge(src_ip, dst_ip, weight=flow['fwd_packets'] + flow['bwd_packets'])
                edge_count += 1
                
                for url in flow['urls_requested'][:5]:
                    try:
                        extracted = tldextract.extract(url)
                        full_domain = f"{extracted.domain}.{extracted.suffix}"
                        if full_domain in domains:
                            G.add_edge(dst_ip, full_domain, weight=1)
                            edge_count += 1
                    except:
                        pass
                        
                if edge_count > 300:
                    break
                    
            if len(G.nodes()) == 0 or len(G.edges()) == 0:
                return False
                
            plt.figure(figsize=(15, 15))
            plt.title('Semantic Network Relationships', fontsize=16, color='white', pad=20)
            
            pos = nx.spring_layout(G, k=0.3, iterations=50)
            
            ip_nodes = [n for n, attr in G.nodes(data=True) if attr.get('type') == 'ip']
            domain_nodes = [n for n, attr in G.nodes(data=True) if attr.get('type') == 'domain']
            
            nx.draw_networkx_nodes(G, pos, nodelist=ip_nodes,
                                   node_color='#ff6b6b', node_size=50, alpha=0.8)
            nx.draw_networkx_nodes(G, pos, nodelist=domain_nodes,
                                   node_color='#4ecdc4', node_size=80, alpha=0.8)
                                   
            nx.draw_networkx_edges(G, pos, width=0.5, alpha=0.5, edge_color='#aaa')
            
            high_degree_nodes = [n for n, d in G.degree() if d > 3]
            labels = {node: node for node in high_degree_nodes}
            nx.draw_networkx_labels(G, pos, labels, font_size=8,
                                    font_color='white', alpha=0.8)
                                    
            plt.plot([], [], 'o', color='#ff6b6b', label='IP Address')
            plt.plot([], [], 'o', color='#4ecdc4', label='Domain')
            plt.legend(frameon=False, fontsize=10, labelcolor='white')
            
            plt.axis('off')
            plt.tight_layout()
            plt.savefig(os.path.join(viz_dir, 'semantic_network_graph.png'),
                        facecolor='black', dpi=300, bbox_inches='tight')
            plt.close()
            return True
            
        except Exception as e:
            self.logger.warning(f"Error creating network graph: {str(e)}")
            return False

    def _create_interactive_dashboard(self, viz_dir, features_df):
        """Create interactive dashboard for semantic analysis."""
        if not PLOTLY_AVAILABLE:
            return False

        try:
            if features_df.empty:
                return False

            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=(
                    'URL & Domain Analysis',
                    'Threat Score Distribution',
                    'Feature Importance',
                    'Protocol Distribution'
                ),
                specs=[
                    [{"type": "scatter"}, {"type": "histogram"}],
                    [{"type": "bar"}, {"type": "pie"}]
                ]
            )

            # URL and Domain Analysis scatter plot
            if 'domain_entropy' in features_df.columns and 'url_length' in features_df.columns:
                fig.add_trace(
                    go.Scatter(
                        x=features_df['domain_entropy'],
                        y=features_df['url_length'],
                        mode='markers',
                        marker=dict(
                            color=features_df['suspicious_keywords'],
                            colorscale='Viridis',
                            colorbar=dict(title='Suspicious Keywords'),
                            size=8,
                            opacity=0.7
                        ),
                        text=features_df['dst_ip'] if 'dst_ip' in features_df.columns else None,
                        name='Domain Analysis'
                    ),
                    row=1, col=1
                )

                fig.add_shape(
                    type="rect",
                    x0=3.5, y0=0,
                    x1=8, y1=50,
                    line=dict(color="rgba(255,0,0,0.1)", width=0),
                    fillcolor="rgba(255,0,0,0.1)",
                    row=1, col=1
                )

                fig.add_shape(
                    type="rect",
                    x0=0, y0=100,
                    x1=8, y1=500,
                    line=dict(color="rgba(255,0,0,0.1)", width=0),
                    fillcolor="rgba(255,0,0,0.1)",
                    row=1, col=1
                )

                fig.add_annotation(
                    x=4, y=25,
                    text="High Entropy Domain",
                    showarrow=False,
                    font=dict(color="white"),
                    row=1, col=1
                )

                fig.add_annotation(
                    x=4, y=200,
                    text="Long URL (Suspicious)",
                    showarrow=False,
                    font=dict(color="white"),
                    row=1, col=1
                )

            # Threat Score Distribution
            if not features_df.empty:
                threat_features = [
                    'domain_entropy', 'suspicious_keywords', 'dga_probability',
                    'query_frequency', 'failed_queries', 'obfuscation_indicators',
                    'encryption_indicators', 'command_keywords'
                ]

                available_features = [f for f in threat_features if f in features_df.columns]

                if available_features:
                    threat_scores = []
                    for _, row in features_df.iterrows():
                        score = sum(row[f] / features_df[f].max() if features_df[f].max() > 0 else 0
                                    for f in available_features)
                        threat_scores.append(min(10, score * 10 / len(available_features)))

                    fig.add_trace(
                        go.Histogram(
                            x=threat_scores,
                            nbinsx=20,
                            marker_color='rgba(255, 107, 107, 0.7)',
                            name='Threat Score'
                        ),
                        row=1, col=2
                    )

                    fig.add_shape(
                        type="line",
                        x0=3, y0=0,
                        x1=3, y1=features_df.shape[0] // 3,
                        line=dict(color="yellow", width=2, dash="dash"),
                        row=1, col=2
                    )

                    fig.add_shape(
                        type="line",
                        x0=7, y0=0,
                        x1=7, y1=features_df.shape[0] // 3,
                        line=dict(color="red", width=2, dash="dash"),
                        row=1, col=2
                    )

                    fig.add_annotation(
                        x=3, y=features_df.shape[0] // 4,
                        text="Warning",
                        showarrow=False,
                        font=dict(color="yellow"),
                        row=1, col=2
                    )

                    fig.add_annotation(
                        x=7, y=features_df.shape[0] // 4,
                        text="Critical",
                        showarrow=False,
                        font=dict(color="red"),
                        row=1, col=2
                    )

            # Feature Importance
            if not features_df.empty:
                feature_cols = [col for col in features_df.columns if col in self.feature_names]
                if feature_cols:
                    variances = features_df[feature_cols].var()
                    top_features = variances.nlargest(10)

                    fig.add_trace(
                        go.Bar(
                            x=top_features.index,
                            y=top_features.values,
                            marker_color='rgba(78, 205, 196, 0.7)',
                            name='Feature Importance'
                        ),
                        row=2, col=1
                    )

            # Protocol Distribution Pie Chart
            protocol_counts = {}
            for flow in self.flows.values():
                protocol = flow['protocol']
                protocol_counts[protocol] = protocol_counts.get(protocol, 0) + 1

            if protocol_counts:
                labels = list(protocol_counts.keys())
                values = list(protocol_counts.values())

                fig.add_trace(
                    go.Pie(
                        labels=labels,
                        values=values,
                        marker=dict(colors=['#ff6b6b', '#4ecdc4', '#45b7d1']),
                        name='Protocol Distribution'
                    ),
                    row=2, col=2
                )

            fig.update_layout(
                title_text="Interactive Semantic Analysis Dashboard",
                title_font_size=20,
                template="plotly_dark",
                height=800,
                showlegend=True,
                legend=dict(orientation="h", y=-0.1)
            )

            fig.update_xaxes(title_text="Domain Entropy", row=1, col=1)
            fig.update_yaxes(title_text="URL Length", row=1, col=1)
            fig.update_xaxes(title_text="Threat Score (0-10)", row=1, col=2)
            fig.update_yaxes(title_text="Count", row=1, col=2)
            fig.update_xaxes(title_text="Feature", row=2, col=1)
            fig.update_yaxes(title_text="Variance (Importance)", row=2, col=1)

            fig.write_html(os.path.join(viz_dir, 'interactive_dashboard.html'))
            return True

        except Exception as e:
            self.logger.warning(f"Error creating interactive dashboard: {str(e)}")
            return False

    def _create_threat_score_visualization(self, viz_dir, features_df):
        """Create threat score visualization with detailed breakdown."""
        if not PLOTLY_AVAILABLE:
            return False

        try:
            if features_df.empty:
                return False

            threat_categories = {
                'URL Analysis': [
                    'domain_entropy', 'subdomain_count', 'suspicious_keywords',
                    'url_length', 'special_char_ratio', 'numeric_ratio',
                    'tld_reputation'
                ],
                'DNS Analysis': [
                    'query_entropy', 'dga_probability', 'query_frequency',
                    'unique_domains', 'failed_queries', 'subdomain_entropy'
                ],
                'HTTP Content': [
                    'user_agent_entropy', 'header_anomalies', 'content_type_variety',
                    'status_code_distribution', 'referrer_consistency', 'cookie_complexity'
                ],
                'Payload Analysis': [
                    'command_keywords', 'obfuscation_indicators', 'script_language_mix',
                    'dangerous_functions', 'encryption_indicators'
                ]
            }

            scores_data = []
            for idx, row in features_df.iterrows():
                flow_scores = {'id': idx}

                for category, features in threat_categories.items():
                    available_features = [f for f in features if f in features_df.columns]

                    if available_features:
                        category_score = sum(row[f] / features_df[f].max() if features_df[f].max() > 0 else 0
                                             for f in available_features) / len(available_features)
                        flow_scores[category] = min(100, category_score * 100)
                    else:
                        flow_scores[category] = 0

                weights = {
                    'URL Analysis': 0.25,
                    'DNS Analysis': 0.25,
                    'HTTP Content': 0.2,
                    'Payload Analysis': 0.3
                }

                overall_score = sum(flow_scores[cat] * weights[cat] for cat in weights.keys())
                flow_scores['Overall Threat'] = overall_score

                if 'src_ip' in features_df.columns:
                    flow_scores['src_ip'] = row['src_ip']
                if 'dst_ip' in features_df.columns:
                    flow_scores['dst_ip'] = row['dst_ip']

                scores_data.append(flow_scores)

            scores_df = pd.DataFrame(scores_data)
            scores_df = scores_df.sort_values('Overall Threat', ascending=False)
            top_threats = scores_df.head(100)

            # Create radar chart for top 10 threats
            fig = go.Figure()

            categories = list(threat_categories.keys()) + ['Overall Threat']

            for i in range(min(10, len(top_threats))):
                row = top_threats.iloc[i]

                values = [row[cat] for cat in categories]
                values.append(values[0])

                fig.add_trace(go.Scatterpolar(
                    r=values,
                    theta=categories + [categories[0]],
                    fill='toself',
                    name=f"Flow {row['id']} ({row['src_ip']} → {row['dst_ip']})" if 'src_ip' in row and 'dst_ip' in row
                    else f"Flow {row['id']}"
                ))

            fig.update_layout(
                title="Top 10 Threats - Category Breakdown",
                polar=dict(
                    radialaxis=dict(
                        visible=True,
                        range=[0, 100]
                    )
                ),
                showlegend=True,
                template="plotly_dark",
                height=800,
                width=1000
            )

            fig.write_html(os.path.join(viz_dir, 'threat_radar_chart.html'))

            # Create threat score distribution visualization
            fig2 = go.Figure()

            fig2.add_trace(go.Histogram(
                x=scores_df['Overall Threat'],
                nbinsx=20,
                marker_color='rgba(255, 107, 107, 0.7)',
                name='Overall Threat Score'
            ))

            for category in threat_categories.keys():
                fig2.add_trace(go.Histogram(
                    x=scores_df[category],
                    nbinsx=20,
                    opacity=0.6,
                    name=category
                ))

            fig2.update_layout(
                title="Threat Score Distributions by Category",
                xaxis_title="Threat Score (0-100)",
                yaxis_title="Count",
                barmode='overlay',
                template="plotly_dark",
                height=600,
                width=1000
            )

            fig2.write_html(os.path.join(viz_dir, 'threat_score_distribution.html'))

            return True

        except Exception as e:
            self.logger.warning(f"Error creating threat score visualization: {str(e)}")
            return False

    def _create_text_summary(self, viz_dir, features_df):
        """Create comprehensive text-based summary."""
        try:
            summary_file = os.path.join(viz_dir, "semantic_analysis_summary.txt")

            with open(summary_file, 'w') as f:
                f.write("CICIDS2017 Semantic Network Analysis Summary\n")
                f.write("=" * 50 + "\n\n")

                # Write general statistics
                f.write("General Statistics:\n")
                f.write(f"Total flows analyzed: {len(self.flows):,}\n")
                f.write(f"Total HTTP requests: {len(self.http_requests):,}\n")
                f.write(f"Total DNS queries: {len(self.dns_queries):,}\n")
                f.write(f"Unique domains observed: {len(self.domains_seen):,}\n")
                f.write("\n")

                # Write URL analysis statistics
                if self.urls:
                    f.write("URL Analysis:\n")

                    domain_entropies = []
                    for url in self.urls:
                        try:
                            extracted = tldextract.extract(url)
                            domain = extracted.domain
                            if domain:
                                domain_entropies.append(self._calculate_entropy(domain))
                        except:
                            pass

                    if domain_entropies:
                        f.write(f"Domain entropy (avg): {sum(domain_entropies) / len(domain_entropies):.2f}\n")
                        f.write(f"High entropy domains (>3.5): {sum(1 for e in domain_entropies if e > 3.5)}\n")

                    url_lengths = [len(url) for url in self.urls]
                    if url_lengths:
                        f.write(f"URL length (avg): {sum(url_lengths) / len(url_lengths):.1f} characters\n")
                        f.write(f"Long URLs (>100 chars): {sum(1 for l in url_lengths if l > 100)}\n")

                    tlds = []
                    for url in self.urls:
                        try:
                            extracted = tldextract.extract(url)
                            if extracted.suffix:
                                tlds.append(extracted.suffix)
                        except:
                            pass

                    if tlds:
                        f.write("Top 5 TLDs:\n")
                        for tld, count in Counter(tlds).most_common(5):
                            f.write(f"   {tld}: {count:,}\n")

                    if self.suspicious_patterns:
                        f.write("Top 5 suspicious keywords:\n")
                        for pattern, count in self.suspicious_patterns.most_common(5):
                            severity, keyword = pattern.split(':')
                            f.write(f"   {keyword} ({severity} severity): {count:,}\n")

                    f.write("\n")

                # Write DNS analysis statistics
                if self.dns_queries:
                    f.write("DNS Analysis:\n")

                    query_entropies = [q['entropy'] for q in self.dns_queries if 'entropy' in q]
                    if query_entropies:
                        f.write(f"Query entropy (avg): {sum(query_entropies) / len(query_entropies):.2f}\n")
                        f.write(f"High entropy queries (>4.0): {sum(1 for e in query_entropies if e > 4.0)}\n")

                    dga_scores = [q['dga_score'] for q in self.dns_queries if 'dga_score' in q]
                    if dga_scores:
                        f.write(f"DGA probability (avg): {sum(dga_scores) / len(dga_scores):.2f}\n")
                        f.write(f"High DGA probability queries (>0.7): {sum(1 for s in dga_scores if s > 0.7)}\n")

                    if self.domains_seen:
                        f.write("Top 5 queried domains:\n")
                        for domain, count in self.domains_seen.most_common(5):
                            f.write(f"   {domain}: {count:,}\n")

                    f.write("\n")

                # Write HTTP analysis statistics
                if self.http_requests or self.http_responses:
                    f.write("HTTP Analysis:\n")

                    methods = [req['method'] for req in self.http_requests if 'method' in req]
                    if methods:
                        method_counts = Counter(methods)
                        f.write("HTTP method distribution:\n")
                        for method, count in method_counts.most_common():
                            f.write(f"   {method}: {count:,}\n")

                    status_codes = [resp['status_code'] for resp in self.http_responses if 'status_code' in resp]
                    if status_codes:
                        success_count = sum(1 for code in status_codes if 200 <= code < 300)
                        redirect_count = sum(1 for code in status_codes if 300 <= code < 400)
                        client_error_count = sum(1 for code in status_codes if 400 <= code < 500)
                        server_error_count = sum(1 for code in status_codes if 500 <= code < 600)

                        f.write("HTTP status code distribution:\n")
                        f.write(f"   2xx (Success): {success_count:,}\n")
                        f.write(f"   3xx (Redirect): {redirect_count:,}\n")
                        f.write(f"   4xx (Client Error): {client_error_count:,}\n")
                        f.write(f"   5xx (Server Error): {server_error_count:,}\n")

                    f.write("\n")

                # Write Command & Payload analysis statistics
                f.write("Command & Payload Analysis:\n")

                all_commands = []
                for flow in self.flows.values():
                    all_commands.extend(flow['command_keywords_detected'])

                if all_commands:
                    f.write("Top 5 command keywords:\n")
                    for cmd, count in Counter(all_commands).most_common(5):
                        f.write(f"   {cmd}: {count:,}\n")

                all_languages = []
                for flow in self.flows.values():
                    all_languages.extend(flow['script_languages'])

                if all_languages:
                    f.write("Script language distribution:\n")
                    for lang, count in Counter(all_languages).most_common():
                        f.write(f"   {lang}: {count:,}\n")

                total_obfuscation = sum(flow['obfuscation_indicators'] for flow in self.flows.values())
                f.write(f"Total obfuscation indicators: {total_obfuscation:,}\n")

                total_encryption = sum(flow['encryption_indicators'] for flow in self.flows.values())
                f.write(f"Total encryption indicators: {total_encryption:,}\n")

                f.write("\n")

                # Feature summary if available
                if not features_df.empty:
                    f.write("Semantic Feature Summary:\n")

                    feature_cols = [col for col in features_df.columns if col in self.feature_names]
                    if feature_cols:
                        for feature in feature_cols:
                            values = features_df[feature]
                            f.write(
                                f"   {feature}: min={values.min():.2f}, avg={values.mean():.2f}, max={values.max():.2f}\n")

                f.write("\n")

                # Threat summary
                f.write("Threat Summary:\n")

                threat_features = [
                    'domain_entropy', 'suspicious_keywords', 'dga_probability',
                    'query_frequency', 'failed_queries', 'obfuscation_indicators',
                    'encryption_indicators', 'command_keywords'
                ]

                if not features_df.empty:
                    available_features = [f for f in threat_features if f in features_df.columns]

                    if available_features:
                        threat_scores = []
                        for _, row in features_df.iterrows():
                            score = sum(row[f] / features_df[f].max() if features_df[f].max() > 0 else 0
                                        for f in available_features)
                            threat_scores.append(min(10, score * 10 / len(available_features)))

                        f.write(f"Average threat score (0-10): {sum(threat_scores) / len(threat_scores):.2f}\n")
                        f.write(f"High threat flows (score > 7): {sum(1 for s in threat_scores if s > 7)}\n")
                        f.write(f"Medium threat flows (score 3-7): {sum(1 for s in threat_scores if 3 <= s <= 7)}\n")
                        f.write(f"Low threat flows (score < 3): {sum(1 for s in threat_scores if s < 3)}\n")

                f.write("\n" + "=" * 50 + "\n")
                f.write("Install visualization libraries for graphical reports:\n")
                f.write("   pip install matplotlib seaborn plotly networkx nltk scikit-learn\n")

            return True

        except Exception as e:
            self.logger.warning(f"Error creating text summary: {str(e)}")
            return False


def main():
    """Main function to run the advanced interactive semantic network analyzer."""
    try:
        # Print banner and get user configuration
        print_banner()
        config = get_user_input()
        
        # Create output directory
        os.makedirs(config['output_dir'], exist_ok=True)
        
        # Initialize analyzer with advanced capabilities
        analyzer = SemanticNetworkAnalyzer(
            chunk_size=config['chunk_size'],
            enable_visualizations=config['enable_visualizations'],
            deep_inspection=config['deep_inspection']
        )
        
        # Get file size for estimation
        file_size_gb = os.path.getsize(config['pcap_file']) / (1024 ** 3)
        print(f"Starting to process PCAP file: {config['pcap_file']}")
        print(f"File size: {file_size_gb:.2f} GB")
        print()
        
        # Process PCAP file
        features_df = analyzer.process_pcap_file(config['pcap_file'])
        
        # Save features to CSV
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        features_file = os.path.join(config['output_dir'], f'semantic_features_{timestamp}.csv')
        features_df.to_csv(features_file, index=False)
        print(f"Features saved to: {features_file}")
        
        # Generate visualizations if enabled
        if config['enable_visualizations']:
            analyzer.generate_visualizations(config['output_dir'], features_df)
        
        # Print comprehensive summary
        print()
        print("Analysis Complete!")
        print(f"- Packets processed: {analyzer.packets_processed:,}")
        print(f"- Flows analyzed: {len(features_df):,}")
        print(f"- Features extracted: {len(features_df):,}")
        print(f"- Output directory: {config['output_dir']}")
        
        # Show semantic analysis statistics
        if len(features_df) > 0:
            print(f"\nSemantic Analysis Statistics:")
            try:
                if 'domain_entropy' in features_df.columns and not features_df['domain_entropy'].empty:
                    avg_entropy = float(features_df['domain_entropy'].mean())
                    high_entropy_domains = sum(1 for e in features_df['domain_entropy'] if e > 3.5)
                    print(f"   Avg domain entropy: {avg_entropy:.2f}")
                    print(f"   High entropy domains (>3.5): {high_entropy_domains:,}")
                
                if 'suspicious_keywords' in features_df.columns and not features_df['suspicious_keywords'].empty:
                    total_suspicious = int(features_df['suspicious_keywords'].sum())
                    print(f"   Suspicious keywords detected: {total_suspicious:,}")
                
                if 'dga_probability' in features_df.columns and not features_df['dga_probability'].empty:
                    high_dga_prob = sum(1 for p in features_df['dga_probability'] if p > 0.7)
                    print(f"   Potential DGA domains (prob >0.7): {high_dga_prob:,}")
                
                if 'command_keywords' in features_df.columns and not features_df['command_keywords'].empty:
                    command_count = int(features_df['command_keywords'].sum())
                    print(f"   Command keywords detected: {command_count:,}")
                
                if 'obfuscation_indicators' in features_df.columns and not features_df['obfuscation_indicators'].empty:
                    obfuscation_count = int(features_df['obfuscation_indicators'].sum())
                    print(f"   Obfuscation indicators: {obfuscation_count:,}")
                
            except Exception as e:
                print(f"   Error calculating statistics: {str(e)}")
                print(f"   Dataset shape: {features_df.shape}")
        
        # Show additional analysis info
        if hasattr(analyzer, 'urls') and analyzer.urls:
            print(f"\nURL Analysis:")
            print(f"   Total URLs analyzed: {len(analyzer.urls):,}")
        
        if hasattr(analyzer, 'dns_queries') and analyzer.dns_queries:
            print(f"\nDNS Analysis:")
            print(f"   Total DNS queries analyzed: {len(analyzer.dns_queries):,}")
            print(f"   Unique domains: {len(analyzer.domains_seen):,}")
        
        if hasattr(analyzer, 'suspicious_patterns') and analyzer.suspicious_patterns:
            print(f"\nTop Suspicious Patterns:")
            for pattern, count in analyzer.suspicious_patterns.most_common(5):
                severity, keyword = pattern.split(':')
                print(f"   {keyword} ({severity} severity): {count:,}")
        
        print(f"\nAdvanced semantic analysis completed successfully!")
        print(f"Log file: semantic_analysis.log")
        
        # Show visualization file details
        if config['enable_visualizations']:
            print(f"\nVisualization Files Generated:")
            viz_dir = os.path.join(config['output_dir'], "visualizations")
            if os.path.exists(viz_dir):
                print(f"   Interactive Dashboard: {viz_dir}/interactive_dashboard.html")
                print(f"   Threat Analysis: {viz_dir}/threat_radar_chart.html")
                print(f"   Threat Scores: {viz_dir}/threat_score_distribution.html")
                if MATPLOTLIB_AVAILABLE:
                    print(f"   Static Images: {viz_dir}/")
                    print(f"      • url_analysis.png")
                    print(f"      • dns_analysis.png")
                    print(f"      • http_content_analysis.png")
                    print(f"      • command_payload_analysis.png")
                    print(f"      • feature_correlation_matrix.png")
                    print(f"      • semantic_network_graph.png")
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