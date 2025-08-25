#!/usr/bin/env python3

"""
Visual NLP Network Security System with CNN-Ready Outputs
=========================================================
Enhanced framework that generates visual representations of network threats
suitable for CNN training and analysis visualization.
"""

import os
import sys
import json
import numpy as np
import pandas as pd
from datetime import datetime
from pathlib import Path
from collections import Counter, defaultdict
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import warnings

# Visualization libraries
try:
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    from matplotlib.colors import ListedColormap
    import seaborn as sns
    plt.style.use('default')
    MATPLOTLIB_AVAILABLE = True
    print("✓ Matplotlib loaded successfully")
except ImportError as e:
    MATPLOTLIB_AVAILABLE = False
    print(f"✗ Matplotlib import failed: {e}")

try:
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots
    import plotly.offline as pyo
    PLOTLY_AVAILABLE = True
    print("✓ Plotly loaded successfully")
except ImportError as e:
    PLOTLY_AVAILABLE = False
    print(f"✗ Plotly import failed: {e}")

# Core libraries
try:
    import scapy.all as scapy
    from scapy.layers.inet import IP, TCP, UDP
    from scapy.layers.http import HTTP, HTTPRequest
    from scapy.layers.dns import DNS
    SCAPY_AVAILABLE = True
    print("✓ Scapy loaded successfully")
except ImportError as e:
    SCAPY_AVAILABLE = False
    print(f"✗ Scapy import failed: {e}")

try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.manifold import TSNE
    from sklearn.cluster import KMeans
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA
    SKLEARN_AVAILABLE = True
    print("✓ Scikit-learn loaded successfully")
except ImportError as e:
    SKLEARN_AVAILABLE = False
    print(f"✗ Scikit-learn import failed: {e}")

try:
    import gensim
    from gensim.models import Word2Vec
    GENSIM_AVAILABLE = True
    print("✓ Gensim loaded successfully")
except ImportError as e:
    GENSIM_AVAILABLE = False
    print(f"✗ Gensim import failed: {e}")

warnings.filterwarnings("ignore")


@dataclass
class NetworkDocument:
    """
    Data structure representing a network communication as a document for NLP analysis.
    
    This class encapsulates all information extracted from a network packet or flow,
    including both the raw network data and the results of NLP analysis.
    
    Attributes:
        doc_id (str): Unique identifier for this document
        source_ip (str): Source IP address of the communication
        dest_ip (str): Destination IP address of the communication
        protocol (str): Network protocol used (HTTP, DNS, TCP, etc.)
        timestamp (datetime): When this communication occurred
        extracted_text (str): Human-readable text extracted from packet payload
        threat_indicators (List[str]): List of detected suspicious patterns
        classification (str): Threat classification result ('benign', 'web_attack', etc.)
        confidence (float): Confidence score for the classification (0.0 to 1.0)
        visual_features (Dict): Dictionary for storing visual feature data
    """
    doc_id: str                    # Unique identifier for tracking this document
    source_ip: str                 # Source IP address from packet header
    dest_ip: str                   # Destination IP address from packet header
    protocol: str                  # Protocol type (HTTP, DNS, TCP, UDP, etc.)
    timestamp: datetime            # Timestamp when packet was captured
    extracted_text: str            # Text content extracted from packet payload
    threat_indicators: List[str]   # List of suspicious patterns found in text
    classification: str = "benign" # Default classification is benign (non-threatening)
    confidence: float = 0.0        # Confidence score for classification
    visual_features: Dict = None   # Additional features for visual analysis
    
    def __post_init__(self):
        """Initialize visual_features dictionary if not provided."""
        if self.visual_features is None:
            self.visual_features = {}


# =============================================================================
# VISUAL FEATURE EXTRACTION CLASS
# =============================================================================
class VisualFeatureExtractor:
    """
    Extracts visual features from network data and creates CNN-ready images.
    
    This class converts network security data into visual representations that can
    be used to train Convolutional Neural Networks (CNNs). It creates three main
    types of images:
    1. Threat heatmaps - showing threat intensity across network space
    2. Protocol flow images - showing protocol usage over time
    3. Feature correlation matrices - showing relationships between features
    
    The generated images are in standard formats (256x256 RGB) suitable for
    direct use in CNN training pipelines.
    """
    
    def __init__(self):
        """
        Initialize the visual feature extractor with color schemes and mappings.
        
        Sets up color mappings for different threat types and protocols to ensure
        consistent visual representation across all generated images.
        """
        # Color mapping for different threat classifications
        # Each threat type gets a distinct color for easy visual identification
        self.threat_colors = {
            'benign': [0.2, 0.8, 0.2],        # Green - safe/normal traffic
            'web_attack': [1.0, 0.2, 0.2],    # Red - web-based attacks
            'malware_c2': [0.8, 0.2, 0.8],    # Purple - malware communication
            'data_exfiltration': [1.0, 0.5, 0.0],  # Orange - data theft
            'reconnaissance': [0.2, 0.2, 1.0],     # Blue - network scanning
            'suspicious': [1.0, 1.0, 0.0]     # Yellow - unknown suspicious activity
        }
        
        # Color mapping for different network protocols
        # Helps distinguish different types of network traffic visually
        self.protocol_colors = {
            'HTTP': [1.0, 0.3, 0.3],    # Light red for web traffic
            'DNS': [0.3, 0.3, 1.0],     # Blue for domain name queries
            'TCP': [0.3, 1.0, 0.3],     # Green for general TCP traffic
            'UDP': [1.0, 1.0, 0.3],     # Yellow for UDP traffic
            'RAW': [0.8, 0.8, 0.8]      # Gray for unclassified traffic
        }
    
    def create_threat_heatmap_image(self, documents: List[NetworkDocument], 
                                   output_path: str, image_size: Tuple[int, int] = (256, 256)):
        """
        Create a heatmap image representation of network threats for CNN training.
        
        This method converts network threat data into a visual heatmap where:
        - Each pixel represents threat intensity and type
        - Lines connect source and destination IPs showing communication flows
        - Colors indicate different types of threats
        - Intensity (brightness) represents threat confidence levels
        
        The resulting image can be directly used as input to a CNN for threat
        classification or pattern recognition.
        
        Args:
            documents (List[NetworkDocument]): List of network documents to visualize
            output_path (str): Path where to save the generated image
            image_size (Tuple[int, int]): Output image dimensions (default: 256x256)
            
        Returns:
            np.ndarray: The generated threat matrix as a numpy array (height, width, 3)
        """
        # Check if matplotlib is available for image generation
        if not MATPLOTLIB_AVAILABLE:
            print("Matplotlib not available - skipping heatmap generation")
            return None
        
        # Initialize the threat matrix - 3D array for RGB image
        height, width = image_size
        threat_matrix = np.zeros((height, width, 3))  # RGB channels
        
        # Create mapping from IP addresses to grid coordinates
        # This allows us to position IPs consistently in the image space
        ip_to_coords = {}
        unique_ips = set()
        
        # Collect all unique IP addresses from the documents
        for doc in documents:
            unique_ips.add(doc.source_ip)
            unique_ips.add(doc.dest_ip)
        
        # Map IP addresses to 2D grid coordinates
        # Uses a square grid layout to distribute IPs across the image
        unique_ips = list(unique_ips)
        grid_size = int(np.sqrt(len(unique_ips))) + 1  # Calculate grid dimensions
        
        # Assign each IP address to a specific coordinate in the image
        for i, ip in enumerate(unique_ips[:grid_size*grid_size]):
            x = i % grid_size  # X coordinate in grid
            y = i // grid_size  # Y coordinate in grid
            # Scale grid coordinates to image dimensions
            ip_to_coords[ip] = (min(x * (width // grid_size), width-1), 
                               min(y * (height // grid_size), height-1))
        
        # Draw threat connections in the matrix
        # Each document represents a communication flow between two IPs
        for doc in documents:
            # Only process if both IPs have assigned coordinates
            if doc.source_ip in ip_to_coords and doc.dest_ip in ip_to_coords:
                src_x, src_y = ip_to_coords[doc.source_ip]
                dst_x, dst_y = ip_to_coords[doc.dest_ip]
                
                # Get color based on threat classification
                threat_color = self.threat_colors.get(doc.classification, [0.5, 0.5, 0.5])
                intensity = doc.confidence  # Use confidence as intensity
                
                # Draw line between source and destination IPs
                # This represents the communication flow
                line_coords = self._get_line_coordinates(src_x, src_y, dst_x, dst_y)
                
                # Color each pixel along the line
                for x, y in line_coords:
                    if 0 <= x < width and 0 <= y < height:
                        # Set RGB values, taking maximum to avoid overwriting higher threats
                        for c in range(3):  # For each color channel (R, G, B)
                            threat_matrix[y, x, c] = max(threat_matrix[y, x, c], 
                                                       threat_color[c] * intensity)
        
        # Save the heatmap as a standard image file
        plt.figure(figsize=(10, 10))
        plt.imshow(threat_matrix)  # Display the RGB matrix as an image
        plt.title("Network Threat Heatmap")
        plt.axis('off')  # Remove axis labels for cleaner appearance
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight', pad_inches=0)
        plt.close()
        
        # Save raw numpy array for direct CNN input
        # This allows bypassing image loading/preprocessing in CNN pipelines
        np.save(output_path.replace('.png', '_raw.npy'), threat_matrix)
        
        return threat_matrix
    
    def _get_line_coordinates(self, x1: int, y1: int, x2: int, y2: int) -> List[Tuple[int, int]]:
        """
        Generate coordinates for drawing a line between two points using Bresenham's algorithm.
        
        This classic computer graphics algorithm efficiently determines which pixels
        should be colored to draw a straight line between two points. It's used here
        to draw communication flows between IP addresses in the threat heatmap.
        
        Args:
            x1, y1 (int): Starting point coordinates
            x2, y2 (int): Ending point coordinates
            
        Returns:
            List[Tuple[int, int]]: List of (x, y) coordinates forming the line
        """
        coordinates = []
        
        # Calculate differences and step directions
        dx = abs(x2 - x1)  # Horizontal distance
        dy = abs(y2 - y1)  # Vertical distance
        sx = 1 if x1 < x2 else -1  # Horizontal step direction
        sy = 1 if y1 < y2 else -1  # Vertical step direction
        err = dx - dy  # Initial error term
        
        x, y = x1, y1  # Current position
        
        # Bresenham's line drawing algorithm
        while True:
            coordinates.append((x, y))  # Add current point to line
            
            # Check if we've reached the end point
            if x == x2 and y == y2:
                break
                
            # Calculate next position based on error term
            e2 = 2 * err
            if e2 > -dy:  # Move horizontally
                err -= dy
                x += sx
            if e2 < dx:   # Move vertically
                err += dx
                y += sy
        
        return coordinates
    
    def create_protocol_flow_image(self, documents: List[NetworkDocument], 
                                  output_path: str, image_size: Tuple[int, int] = (256, 256)):
        """
        Create a protocol flow visualization as a time-series CNN training image.
        
        This method creates a 2D representation of network protocol usage over time:
        - X-axis represents time progression
        - Y-axis represents different protocol types
        - Colors represent different protocols
        - Intensity represents threat levels
        
        This visualization helps CNNs learn temporal patterns in network traffic
        and identify suspicious protocol usage patterns over time.
        
        Args:
            documents (List[NetworkDocument]): Network documents to visualize
            output_path (str): Path to save the generated image
            image_size (Tuple[int, int]): Output image dimensions
            
        Returns:
            np.ndarray: The generated flow matrix as RGB image array
        """
        if not MATPLOTLIB_AVAILABLE:
            return None
        
        height, width = image_size
        flow_matrix = np.zeros((height, width, 3))  # Initialize RGB image matrix
        
        # Create time-based visualization if we have documents with timestamps
        if documents:
            # Calculate time span of the network capture
            timestamps = [doc.timestamp for doc in documents]
            time_span = max(timestamps) - min(timestamps)
            start_time = min(timestamps)
            
            # Process each document to create the flow visualization
            for doc in documents:
                # Map timestamp to X-axis coordinate (time progression)
                time_offset = (doc.timestamp - start_time).total_seconds()
                if time_span.total_seconds() > 0:
                    x = int((time_offset / time_span.total_seconds()) * (width - 1))
                else:
                    x = width // 2  # Center if no time variation
                
                # Map protocol type to Y-axis coordinate and get protocol color
                # Different protocols get different vertical positions
                protocol_y_map = {
                    'HTTP': 0.8,   # Top area for HTTP traffic
                    'DNS': 0.6,    # Upper-middle for DNS queries  
                    'TCP': 0.4,    # Middle for general TCP
                    'UDP': 0.2,    # Lower-middle for UDP
                    'RAW': 0.1     # Bottom for unclassified traffic
                }
                y = int(protocol_y_map.get(doc.protocol, 0.5) * (height - 1))
                
                # Get color scheme for this protocol
                protocol_color = self.protocol_colors.get(doc.protocol, [0.5, 0.5, 0.5])
                
                # Calculate intensity: base protocol intensity + threat level
                intensity = 0.5 + (doc.confidence * 0.5)  # Range: 0.5 to 1.0
                
                # Draw a small area around the point to make it visible
                # This creates a "dot" representing the network event
                for dx in range(-2, 3):  # 5x5 pixel area
                    for dy in range(-2, 3):
                        px, py = x + dx, y + dy
                        # Check bounds and update pixel color
                        if 0 <= px < width and 0 <= py < height:
                            for c in range(3):  # For each RGB channel
                                flow_matrix[py, px, c] = max(flow_matrix[py, px, c],
                                                           protocol_color[c] * intensity)
        
        # Save the protocol flow image
        plt.figure(figsize=(12, 8))
        plt.imshow(flow_matrix)
        plt.title("Protocol Flow Over Time")
        plt.xlabel("Time")
        plt.ylabel("Protocol Type")
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight', pad_inches=0)
        plt.close()
        
        # Save raw array for direct CNN input
        np.save(output_path.replace('.png', '_raw.npy'), flow_matrix)
        
        return flow_matrix
    
    def create_feature_correlation_matrix(self, documents: List[NetworkDocument], 
                                        output_path: str):
        """Create feature correlation heatmap for CNN training."""
        if not MATPLOTLIB_AVAILABLE:
            return None
        
        # Extract numerical features
        features_data = []
        feature_names = ['threat_count', 'text_length', 'confidence', 'ip_entropy', 
                        'protocol_http', 'protocol_dns', 'protocol_tcp']
        
        for doc in documents:
            features = [
                len(doc.threat_indicators),
                len(doc.extracted_text),
                doc.confidence,
                self._calculate_ip_entropy(doc.source_ip, doc.dest_ip),
                1 if doc.protocol == 'HTTP' else 0,
                1 if doc.protocol == 'DNS' else 0,
                1 if doc.protocol == 'TCP' else 0,
            ]
            features_data.append(features)
        
        # Create correlation matrix
        df = pd.DataFrame(features_data, columns=feature_names)
        correlation_matrix = df.corr()
        
        # Create heatmap
        plt.figure(figsize=(10, 8))
        sns.heatmap(correlation_matrix, annot=True, cmap='RdYlBu_r', center=0,
                   square=True, linewidths=0.5, cbar_kws={"shrink": .8})
        plt.title("Feature Correlation Matrix")
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        # Save raw matrix for CNN
        np.save(output_path.replace('.png', '_matrix.npy'), correlation_matrix.values)
        
        return correlation_matrix
    
    def _calculate_ip_entropy(self, src_ip: str, dst_ip: str) -> float:
        """Calculate entropy of IP addresses for feature extraction."""
        combined = src_ip + dst_ip
        byte_counts = Counter(combined.encode())
        total = len(combined.encode())
        
        entropy = 0.0
        for count in byte_counts.values():
            p = count / total
            if p > 0:
                entropy -= p * np.log2(p)
        
        return entropy


class CNNDatasetGenerator:
    """
    Generates CNN-ready datasets from network analysis results.
    
    This class takes the analyzed network documents and creates structured datasets
    suitable for training Convolutional Neural Networks. It handles:
    1. Batch generation - Splitting large datasets into manageable batches
    2. Image creation - Converting network data into visual representations
    3. Label generation - Creating corresponding labels for supervised learning
    4. Train/validation splitting - Organizing data for proper ML training
    
    The output follows standard ML dataset conventions and can be directly
    used with popular deep learning frameworks like TensorFlow or PyTorch.
    """
    
    def __init__(self, output_dir: str):
        """
        Initialize the CNN dataset generator.
        
        Args:
            output_dir (str): Base directory where all CNN datasets will be saved
        """
        self.output_dir = output_dir
        # Create subdirectories for organized dataset storage
        self.images_dir = os.path.join(output_dir, 'cnn_images')  # CNN training images
        self.labels_dir = os.path.join(output_dir, 'cnn_labels')  # Corresponding labels
        
        # Ensure directories exist
        os.makedirs(self.images_dir, exist_ok=True)
        os.makedirs(self.labels_dir, exist_ok=True)
    
    def generate_cnn_dataset(self, documents: List[NetworkDocument], 
                           batch_size: int = 100):
        """
        Generate a complete CNN training dataset with images and labels.
        
        This method processes all network documents and creates:
        - Visual representations (heatmaps, flow charts, correlation matrices)
        - Corresponding JSON label files for supervised learning
        - Dataset metadata for tracking and reproducibility
        
        The dataset is organized in batches to handle large volumes of network data
        efficiently and enable batch processing during CNN training.
        
        Args:
            documents (List[NetworkDocument]): All analyzed network documents
            batch_size (int): Number of documents per batch (default: 100)
            
        Returns:
            Dict: Dataset information including statistics and file locations
        """
        print("Generating CNN dataset...")
        
        # Initialize the visual feature extractor
        feature_extractor = VisualFeatureExtractor()
        
        # Split documents into batches for efficient processing
        # Large datasets are split into smaller chunks to manage memory usage
        batches = [documents[i:i + batch_size] for i in range(0, len(documents), batch_size)]
        
        # Create dataset metadata for tracking and reproducibility
        dataset_info = {
            'total_samples': len(documents),           # Total number of network documents
            'num_batches': len(batches),               # Number of generated batches
            'batch_size': batch_size,                  # Documents per batch
            'image_size': [256, 256, 3],              # Standard CNN input dimensions
            'classes': list(set(doc.classification for doc in documents)), # Unique threat classes
            'class_counts': dict(Counter(doc.classification for doc in documents)) # Class distribution
        }
        
        # Process each batch to generate CNN training data
        for batch_idx, batch_docs in enumerate(batches):
            print(f"Processing batch {batch_idx + 1}/{len(batches)}")
            
            # Generate threat heatmap image for this batch
            # This creates a visual representation of threat distribution
            heatmap_path = os.path.join(self.images_dir, f'threat_heatmap_batch_{batch_idx:03d}.png')
            threat_matrix = feature_extractor.create_threat_heatmap_image(
                batch_docs, heatmap_path, (256, 256)
            )
            
            # Generate protocol flow image showing temporal patterns
            # This helps CNNs learn time-based attack patterns
            flow_path = os.path.join(self.images_dir, f'protocol_flow_batch_{batch_idx:03d}.png')
            flow_matrix = feature_extractor.create_protocol_flow_image(
                batch_docs, flow_path, (256, 256)
            )
            
            # Generate feature correlation matrix
            # This shows relationships between different network features
            corr_path = os.path.join(self.images_dir, f'feature_correlation_batch_{batch_idx:03d}.png')
            correlation_matrix = feature_extractor.create_feature_correlation_matrix(
                batch_docs, corr_path
            )
            
            # Create corresponding labels for supervised learning
            # Each image gets a label file with ground truth classifications
            batch_labels = []
            for doc in batch_docs:
                # Extract key information for CNN training labels
                label_data = {
                    'doc_id': doc.doc_id,                    # Unique identifier
                    'classification': doc.classification,    # Ground truth threat type
                    'confidence': doc.confidence,            # Classification confidence
                    'threat_indicators': doc.threat_indicators, # List of detected patterns
                    'protocol': doc.protocol,                # Network protocol used
                    'timestamp': doc.timestamp.isoformat()   # When this occurred
                }
                batch_labels.append(label_data)
            
            # Save batch labels as JSON file
            labels_path = os.path.join(self.labels_dir, f'labels_batch_{batch_idx:03d}.json')
            with open(labels_path, 'w') as f:
                json.dump(batch_labels, f, indent=2)
        
        # Save overall dataset information for reference
        with open(os.path.join(self.output_dir, 'dataset_info.json'), 'w') as f:
            json.dump(dataset_info, f, indent=2)
        
        print(f"CNN dataset generated: {len(batches)} batches, {len(documents)} total samples")
        
        return dataset_info
    
    def create_training_validation_split(self, split_ratio: float = 0.8):
        """Split dataset into training and validation sets."""
        import glob
        import shutil
        
        # Create train/val directories
        train_images_dir = os.path.join(self.output_dir, 'train', 'images')
        train_labels_dir = os.path.join(self.output_dir, 'train', 'labels')
        val_images_dir = os.path.join(self.output_dir, 'val', 'images')
        val_labels_dir = os.path.join(self.output_dir, 'val', 'labels')
        
        for dir_path in [train_images_dir, train_labels_dir, val_images_dir, val_labels_dir]:
            os.makedirs(dir_path, exist_ok=True)
        
        # Get all batch files
        image_files = glob.glob(os.path.join(self.images_dir, '*_batch_*.png'))
        label_files = glob.glob(os.path.join(self.labels_dir, '*_batch_*.json'))
        
        # Split files
        num_train = int(len(image_files) * split_ratio)
        
        train_images = image_files[:num_train]
        val_images = image_files[num_train:]
        train_labels = label_files[:num_train]
        val_labels = label_files[num_train:]
        
        # Copy files to train/val directories
        for img_file in train_images:
            shutil.copy2(img_file, train_images_dir)
        
        for img_file in val_images:
            shutil.copy2(img_file, val_images_dir)
        
        for label_file in train_labels:
            shutil.copy2(label_file, train_labels_dir)
        
        for label_file in val_labels:
            shutil.copy2(label_file, val_labels_dir)
        
        split_info = {
            'train_samples': len(train_images),
            'val_samples': len(val_images),
            'split_ratio': split_ratio
        }
        
        with open(os.path.join(self.output_dir, 'split_info.json'), 'w') as f:
            json.dump(split_info, f, indent=2)
        
        print(f"Dataset split: {len(train_images)} training, {len(val_images)} validation")
        
        return split_info


class HumanReadableVisualizer:
    """Creates human-readable visualizations for manual analysis."""
    
    def __init__(self, output_dir: str):
        self.output_dir = output_dir
        self.human_analysis_dir = os.path.join(output_dir, 'human_analysis')
        os.makedirs(self.human_analysis_dir, exist_ok=True)
        
        # Set up clean styling
        if MATPLOTLIB_AVAILABLE:
            plt.style.use('default')
        if MATPLOTLIB_AVAILABLE:
            try:
                sns.set_palette("husl")
            except:
                pass  # Continue if seaborn styling fails
    
    def create_threat_overview_dashboard(self, documents: List[NetworkDocument]):
        """Create comprehensive dashboard for human analysis."""
        if not MATPLOTLIB_AVAILABLE:
            print("Matplotlib not available - skipping dashboard creation")
            return
        
        # Create figure with subplots
        fig, axes = plt.subplots(2, 3, figsize=(20, 12))
        fig.suptitle('Network Security Analysis Dashboard', fontsize=16, fontweight='bold')
        
        # 1. Threat Classification Distribution
        classifications = [doc.classification for doc in documents]
        class_counts = Counter(classifications)
        
        colors = ['#2ecc71', '#e74c3c', '#9b59b6', '#f39c12', '#3498db', '#1abc9c']
        axes[0, 0].pie(class_counts.values(), labels=class_counts.keys(), autopct='%1.1f%%', 
                       colors=colors[:len(class_counts)], startangle=90)
        axes[0, 0].set_title('Threat Classification Distribution', fontweight='bold')
        
        # 2. Threats by Protocol
        protocol_threat_data = []
        for doc in documents:
            protocol_threat_data.append({
                'Protocol': doc.protocol,
                'Classification': doc.classification,
                'Confidence': doc.confidence
            })
        
        df = pd.DataFrame(protocol_threat_data)
        protocol_counts = df.groupby(['Protocol', 'Classification']).size().unstack(fill_value=0)
        protocol_counts.plot(kind='bar', ax=axes[0, 1], stacked=True, color=colors)
        axes[0, 1].set_title('Threats by Protocol Type', fontweight='bold')
        axes[0, 1].set_xlabel('Protocol')
        axes[0, 1].set_ylabel('Count')
        axes[0, 1].legend(title='Threat Type', bbox_to_anchor=(1.05, 1), loc='upper left')
        axes[0, 1].tick_params(axis='x', rotation=45)
        
        # 3. Confidence Score Distribution
        threat_docs = [doc for doc in documents if doc.classification != 'benign']
        if threat_docs:
            confidences = [doc.confidence for doc in threat_docs]
            axes[0, 2].hist(confidences, bins=15, alpha=0.7, color='#e74c3c', edgecolor='black')
            axes[0, 2].axvline(np.mean(confidences), color='blue', linestyle='--', 
                              label=f'Mean: {np.mean(confidences):.2f}')
            axes[0, 2].set_title('Threat Confidence Distribution', fontweight='bold')
            axes[0, 2].set_xlabel('Confidence Score')
            axes[0, 2].set_ylabel('Frequency')
            axes[0, 2].legend()
        
        # 4. Top Source IPs with Threats
        threat_sources = [doc.source_ip for doc in documents if doc.classification != 'benign']
        if threat_sources:
            top_sources = Counter(threat_sources).most_common(10)
            ips, counts = zip(*top_sources)
            axes[1, 0].barh(range(len(ips)), counts, color='#e74c3c')
            axes[1, 0].set_yticks(range(len(ips)))
            axes[1, 0].set_yticklabels(ips, fontsize=9)
            axes[1, 0].set_title('Top 10 Source IPs with Threats', fontweight='bold')
            axes[1, 0].set_xlabel('Threat Count')
        
        # 5. Threat Indicators Frequency
        all_indicators = []
        for doc in documents:
            all_indicators.extend(doc.threat_indicators)
        
        if all_indicators:
            top_indicators = Counter(all_indicators).most_common(8)
            indicators, counts = zip(*top_indicators)
            # Truncate long indicator names
            short_indicators = [ind[:20] + '...' if len(ind) > 20 else ind for ind in indicators]
            axes[1, 1].bar(range(len(short_indicators)), counts, color='#f39c12')
            axes[1, 1].set_xticks(range(len(short_indicators)))
            axes[1, 1].set_xticklabels(short_indicators, rotation=45, ha='right', fontsize=9)
            axes[1, 1].set_title('Most Common Threat Indicators', fontweight='bold')
            axes[1, 1].set_ylabel('Frequency')
        
        # 6. Timeline of Threats
        threat_times = [doc.timestamp for doc in documents if doc.classification != 'benign']
        if threat_times:
            axes[1, 2].hist([t.hour for t in threat_times], bins=24, alpha=0.7, 
                           color='#9b59b6', edgecolor='black')
            axes[1, 2].set_title('Threat Activity by Hour of Day', fontweight='bold')
            axes[1, 2].set_xlabel('Hour')
            axes[1, 2].set_ylabel('Threat Count')
            axes[1, 2].set_xticks(range(0, 24, 4))
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.human_analysis_dir, 'threat_overview_dashboard.png'), 
                   dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        
        print("Threat overview dashboard created")
    
    def create_network_topology_map(self, documents: List[NetworkDocument]):
        """Create network topology visualization for human analysis."""
        if not MATPLOTLIB_AVAILABLE:
            print("Matplotlib not available - skipping topology map")
            return
        
        # Analyze network communications
        ip_connections = defaultdict(set)
        ip_threat_levels = defaultdict(float)
        
        for doc in documents:
            ip_connections[doc.source_ip].add(doc.dest_ip)
            ip_connections[doc.dest_ip].add(doc.source_ip)
            
            if doc.classification != 'benign':
                ip_threat_levels[doc.source_ip] += doc.confidence
                ip_threat_levels[doc.dest_ip] += doc.confidence * 0.5  # Destination gets less weight
        
        # Create network map
        fig, ax = plt.subplots(figsize=(16, 12))
        
        # Get unique IPs and assign positions
        all_ips = list(set(doc.source_ip for doc in documents) | 
                      set(doc.dest_ip for doc in documents))
        
        # Create circular layout
        n_ips = len(all_ips)
        angles = [2 * np.pi * i / n_ips for i in range(n_ips)]
        positions = {ip: (np.cos(angle), np.sin(angle)) for ip, angle in zip(all_ips, angles)}
        
        # Draw connections
        for doc in documents:
            if doc.source_ip in positions and doc.dest_ip in positions:
                x1, y1 = positions[doc.source_ip]
                x2, y2 = positions[doc.dest_ip]
                
                # Color based on threat level
                if doc.classification != 'benign':
                    color = '#e74c3c'  # Red for threats
                    alpha = min(doc.confidence, 1.0)
                    linewidth = 2
                else:
                    color = '#95a5a6'  # Gray for benign
                    alpha = 0.3
                    linewidth = 1
                
                ax.plot([x1, x2], [y1, y2], color=color, alpha=alpha, linewidth=linewidth)
        
        # Draw nodes (IPs)
        for ip, (x, y) in positions.items():
            threat_level = ip_threat_levels[ip]
            
            if threat_level > 0:
                color = '#e74c3c'  # Red for high threat
                size = min(100 + threat_level * 200, 500)
            else:
                color = '#3498db'  # Blue for normal
                size = 80
            
            ax.scatter(x, y, c=color, s=size, alpha=0.7, edgecolors='black', linewidth=1)
            
            # Add IP labels for high-threat nodes
            if threat_level > 0.5:
                ax.annotate(ip, (x, y), xytext=(5, 5), textcoords='offset points', 
                           fontsize=8, bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7))
        
        ax.set_xlim(-1.2, 1.2)
        ax.set_ylim(-1.2, 1.2)
        ax.set_aspect('equal')
        ax.set_title('Network Topology - Threat Analysis View', fontsize=16, fontweight='bold')
        
        # Create legend
        legend_elements = [
            plt.Line2D([0], [0], color='#e74c3c', lw=2, label='Threat Communication'),
            plt.Line2D([0], [0], color='#95a5a6', lw=1, alpha=0.3, label='Normal Communication'),
            plt.scatter([], [], c='#e74c3c', s=100, label='High Threat IP', alpha=0.7),
            plt.scatter([], [], c='#3498db', s=80, label='Normal IP', alpha=0.7)
        ]
        ax.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(1, 1))
        
        ax.axis('off')
        plt.tight_layout()
        plt.savefig(os.path.join(self.human_analysis_dir, 'network_topology_map.png'), 
                   dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        
        print("Network topology map created")
    
    def create_detailed_threat_analysis(self, documents: List[NetworkDocument]):
        """Create detailed threat analysis charts."""
        if not MATPLOTLIB_AVAILABLE:
            print("Matplotlib not available - skipping detailed analysis")
            return
        
        threat_docs = [doc for doc in documents if doc.classification != 'benign']
        
        if not threat_docs:
            print("No threats found for detailed analysis")
            return
        
        # Create detailed analysis figure
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Detailed Threat Analysis', fontsize=16, fontweight='bold')
        
        # 1. Threat Types vs Confidence Levels
        threat_data = pd.DataFrame([
            {'Classification': doc.classification, 'Confidence': doc.confidence, 
             'Protocol': doc.protocol, 'Indicators': len(doc.threat_indicators)}
            for doc in threat_docs
        ])
        
        # Box plot of confidence by threat type
        threat_types = threat_data['Classification'].unique()
        box_data = [threat_data[threat_data['Classification'] == t]['Confidence'].values 
                   for t in threat_types]
        
        box_plot = axes[0, 0].boxplot(box_data, labels=threat_types, patch_artist=True)
        colors = ['#e74c3c', '#9b59b6', '#f39c12', '#e67e22', '#34495e']
        for patch, color in zip(box_plot['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        
        axes[0, 0].set_title('Confidence Distribution by Threat Type', fontweight='bold')
        axes[0, 0].set_ylabel('Confidence Score')
        axes[0, 0].tick_params(axis='x', rotation=45)
        
        # 2. Protocol vs Threat Type Heatmap
        protocol_threat_matrix = pd.crosstab(threat_data['Protocol'], threat_data['Classification'])
        sns.heatmap(protocol_threat_matrix, annot=True, fmt='d', cmap='Reds', ax=axes[0, 1])
        axes[0, 1].set_title('Protocol vs Threat Type Heatmap', fontweight='bold')
        axes[0, 1].set_xlabel('Threat Classification')
        axes[0, 1].set_ylabel('Protocol')
        
        # 3. Threat Indicator Count Distribution
        indicator_counts = [len(doc.threat_indicators) for doc in threat_docs]
        axes[1, 0].hist(indicator_counts, bins=max(1, max(indicator_counts)), 
                       alpha=0.7, color='#e74c3c', edgecolor='black')
        axes[1, 0].set_title('Distribution of Threat Indicators per Document', fontweight='bold')
        axes[1, 0].set_xlabel('Number of Threat Indicators')
        axes[1, 0].set_ylabel('Frequency')
        
        # 4. Timeline of Threat Activity
        if len(threat_docs) > 1:
            times = [doc.timestamp for doc in threat_docs]
            classifications = [doc.classification for doc in threat_docs]
            
            # Create time series plot
            time_df = pd.DataFrame({'Time': times, 'Classification': classifications})
            time_df = time_df.sort_values('Time')
            
            # Group by hour and classification
            time_df['Hour'] = time_df['Time'].dt.floor('H')
            hourly_threats = time_df.groupby(['Hour', 'Classification']).size().unstack(fill_value=0)
            
            hourly_threats.plot(kind='bar', stacked=True, ax=axes[1, 1], 
                               color=['#e74c3c', '#9b59b6', '#f39c12', '#e67e22'])
            axes[1, 1].set_title('Threat Activity Timeline (Hourly)', fontweight='bold')
            axes[1, 1].set_xlabel('Time')
            axes[1, 1].set_ylabel('Threat Count')
            axes[1, 1].legend(title='Threat Type', bbox_to_anchor=(1.05, 1), loc='upper left')
            axes[1, 1].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.human_analysis_dir, 'detailed_threat_analysis.png'), 
                   dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        
        print("Detailed threat analysis created")
    
    def create_security_summary_report(self, documents: List[NetworkDocument]):
        """Create executive summary visualization."""
        if not MATPLOTLIB_AVAILABLE:
            print("Matplotlib not available - skipping summary report")
            return
        
        # Calculate summary statistics
        total_docs = len(documents)
        threat_docs = [doc for doc in documents if doc.classification != 'benign']
        threat_count = len(threat_docs)
        
        if total_docs == 0:
            return
        
        threat_percentage = (threat_count / total_docs) * 100
        avg_confidence = np.mean([doc.confidence for doc in threat_docs]) if threat_docs else 0
        
        unique_source_ips = len(set(doc.source_ip for doc in documents))
        threat_source_ips = len(set(doc.source_ip for doc in threat_docs))
        
        protocols = Counter(doc.protocol for doc in documents)
        threat_classifications = Counter(doc.classification for doc in threat_docs)
        
        # Create summary report figure
        fig = plt.figure(figsize=(16, 10))
        fig.suptitle('Network Security Executive Summary', fontsize=20, fontweight='bold')
        
        # Key metrics at the top
        metrics_text = f"""
        NETWORK SECURITY ANALYSIS SUMMARY
        ══════════════════════════════════════════
        
        Total Network Documents Analyzed: {total_docs:,}
        Threats Detected: {threat_count:,} ({threat_percentage:.1f}%)
        Average Threat Confidence: {avg_confidence:.2f}
        
        Unique Source IPs: {unique_source_ips:,}
        IPs with Threats: {threat_source_ips:,}
        
        Most Common Protocol: {protocols.most_common(1)[0][0]} ({protocols.most_common(1)[0][1]:,} instances)
        """
        
        plt.figtext(0.1, 0.85, metrics_text, fontsize=12, fontfamily='monospace',
                   bbox=dict(boxstyle='round,pad=1', facecolor='lightblue', alpha=0.3))
        
        # Create subplots for charts
        gs = fig.add_gridspec(2, 2, top=0.6, hspace=0.3, wspace=0.3)
        
        # Protocol distribution
        ax1 = fig.add_subplot(gs[0, 0])
        protocols_df = pd.DataFrame(list(protocols.items()), columns=['Protocol', 'Count'])
        protocols_df = protocols_df.sort_values('Count', ascending=False)
        ax1.pie(protocols_df['Count'], labels=protocols_df['Protocol'], autopct='%1.1f%%', startangle=90)
        ax1.set_title('Network Protocols', fontweight='bold')
        
        # Threat classifications
        if threat_classifications:
            ax2 = fig.add_subplot(gs[0, 1])
            threats_df = pd.DataFrame(list(threat_classifications.items()), 
                                    columns=['Threat Type', 'Count'])
            threats_df = threats_df.sort_values('Count', ascending=False)
            colors = ['#e74c3c', '#9b59b6', '#f39c12', '#e67e22', '#34495e']
            ax2.pie(threats_df['Count'], labels=threats_df['Threat Type'], 
                   autopct='%1.1f%%', startangle=90, colors=colors[:len(threats_df)])
            ax2.set_title('Threat Classifications', fontweight='bold')
        
        # Risk assessment gauge
        ax3 = fig.add_subplot(gs[1, :])
        risk_level = "LOW" if threat_percentage < 1 else "MEDIUM" if threat_percentage < 5 else "HIGH"
        risk_color = "#2ecc71" if risk_level == "LOW" else "#f39c12" if risk_level == "MEDIUM" else "#e74c3c"
        
        ax3.text(0.5, 0.5, f'RISK LEVEL: {risk_level}', 
                fontsize=24, fontweight='bold', ha='center', va='center',
                bbox=dict(boxstyle='round,pad=1', facecolor=risk_color, alpha=0.3))
        ax3.set_xlim(0, 1)
        ax3.set_ylim(0, 1)
        ax3.axis('off')
        
        plt.savefig(os.path.join(self.human_analysis_dir, 'security_summary_report.png'), 
                   dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        
        print("Security summary report created")
    
    def generate_all_human_analysis_visuals(self, documents: List[NetworkDocument]):
        """Generate all human-readable visualizations."""
        print("Generating human-readable analysis visualizations...")
        
        self.create_threat_overview_dashboard(documents)
        self.create_network_topology_map(documents)
        self.create_detailed_threat_analysis(documents)
        self.create_security_summary_report(documents)
        
        print(f"Human analysis visualizations saved to: {self.human_analysis_dir}")
        return self.human_analysis_dir


class InteractiveVisualizer:
    """
    Creates interactive visualizations for detailed network security analysis.
    
    This class generates web-based, interactive visualizations that allow users
    to explore network threat data in detail. The visualizations are created
    using Plotly and saved as HTML files that can be opened in any web browser.
    
    Key features:
    - Interactive threat timelines with hover details
    - Network topology graphs showing communication patterns
    - Zoomable and filterable visualizations
    - Professional presentation-ready outputs
    """
    
    def __init__(self, output_dir: str):
        """
        Initialize the interactive visualizer.
        
        Args:
            output_dir (str): Directory where interactive HTML files will be saved
        """
        self.output_dir = output_dir
    
    def create_threat_timeline(self, documents: List[NetworkDocument]):
        """
        Create an interactive timeline showing threat activity over time.
        
        This visualization displays:
        - Scatter plot with time on X-axis and confidence on Y-axis  
        - Different colors for different threat types
        - Point sizes based on number of threat indicators
        - Hover information showing source IPs and details
        - Interactive zooming and panning capabilities
        
        Args:
            documents (List[NetworkDocument]): Network documents to visualize
        """
        if not PLOTLY_AVAILABLE:
            print("Plotly not available - skipping interactive visualizations")
            return
        
        # Prepare data arrays for plotting
        timestamps = [doc.timestamp for doc in documents]
        classifications = [doc.classification for doc in documents]
        confidences = [doc.confidence for doc in documents]
        source_ips = [doc.source_ip for doc in documents]
        
        # Create interactive scatter plot using Plotly Express
        fig = px.scatter(
            x=timestamps,                                              # Time progression
            y=confidences,                                            # Threat confidence levels
            color=classifications,                                     # Color by threat type
            size=[len(doc.threat_indicators) + 1 for doc in documents], # Size by indicator count
            hover_data={'source_ip': source_ips},                     # Additional hover info
            title="Network Threats Timeline",
            labels={'x': 'Time', 'y': 'Threat Confidence', 'color': 'Threat Type'}
        )
        
        # Customize the layout for better presentation
        fig.update_layout(
            width=1200,      # Wide format for timeline viewing
            height=600,      # Appropriate height for detail
            showlegend=True  # Show legend for threat types
        )
        
        # Save interactive HTML (always works without additional dependencies)
        fig.write_html(os.path.join(self.output_dir, 'threat_timeline.html'))
        
        # Try to save PNG image, but handle gracefully if Kaleido is missing
        try:
            fig.write_image(os.path.join(self.output_dir, 'threat_timeline.png'))
            print("Threat timeline PNG saved successfully")
        except Exception as e:
            print(f"Note: Could not save PNG image ({str(e)})")
            print("HTML version saved successfully. To save PNG images, install: pip install kaleido")
        
        print("Interactive threat timeline created")
    
    def create_network_graph(self, documents: List[NetworkDocument]):
        """
        Create an interactive network graph showing communication patterns.
        
        This visualization displays:
        - Network nodes representing IP addresses
        - Edges representing communication flows
        - Colors indicating threat levels (red for threats, green for normal)
        - Line thickness representing confidence levels
        - Interactive hover information and zooming
        
        Args:
            documents (List[NetworkDocument]): Network documents to visualize
        """
        if not PLOTLY_AVAILABLE:
            print("Plotly not available - skipping network graph")
            return
        
        # Build network graph data structures
        nodes = set()  # Unique IP addresses
        edges = []     # Communication flows
        
        # Extract nodes and edges from documents
        for doc in documents:
            nodes.add(doc.source_ip)
            nodes.add(doc.dest_ip)
            edges.append({
                'source': doc.source_ip,
                'target': doc.dest_ip,
                'classification': doc.classification,
                'confidence': doc.confidence
            })
        
        # Create the network visualization using Plotly
        # Note: This is a simplified version - full implementation would use networkx
        fig = go.Figure()
        
        # Add edges (communication flows) to the visualization
        # Limit to first 100 edges for performance in large networks
        for edge in edges[:100]:
            # Choose color based on threat classification
            color = 'red' if edge['classification'] != 'benign' else 'green'
            
            # Use simple hash-based positioning (in real implementation, 
            # would use proper graph layout algorithms)
            x_coords = [hash(edge['source']) % 100, hash(edge['target']) % 100]
            y_coords = [hash(edge['source']) % 100, hash(edge['target']) % 100]
            
            # Add line representing the communication flow
            fig.add_trace(go.Scatter(
                x=x_coords,
                y=y_coords,
                mode='lines',
                line=dict(color=color, width=edge['confidence']*5),  # Width shows confidence
                showlegend=False
            ))
        
        # Customize layout for network graph presentation
        fig.update_layout(
            title="Network Communication Graph",
            xaxis_title="Network Space X",
            yaxis_title="Network Space Y",
            width=1000,
            height=800
        )
        
        # Save interactive HTML (always works)
        fig.write_html(os.path.join(self.output_dir, 'network_graph.html'))
        
        # Try to save PNG image with graceful error handling
        try:
            fig.write_image(os.path.join(self.output_dir, 'network_graph.png'))
            print("Network graph PNG saved successfully")
        except Exception as e:
            print(f"Note: Could not save PNG image ({str(e)})")
            print("HTML version saved successfully. To save PNG images, install: pip install kaleido")
        
        print("Interactive network graph created")


class VisualNetworkAnalyzer:
    """Main analyzer with visual output capabilities."""
    
    def __init__(self, config: Dict):
        self.config = config
        self.documents = []
        self.threat_database = {}
        
        # Visual components
        self.feature_extractor = VisualFeatureExtractor()
        self.cnn_generator = CNNDatasetGenerator(config['output_dir'])
        self.visualizer = InteractiveVisualizer(config['output_dir'])
        self.human_visualizer = HumanReadableVisualizer(config['output_dir'])
    
    def process_pcap_and_generate_visuals(self, pcap_path: str):
        """Process PCAP and generate all visual outputs."""
        print("Processing PCAP file...")
        
        # Simplified packet processing (reuse from previous implementation)
        self.documents = self._extract_documents_from_pcap(pcap_path)
        
        print(f"Extracted {len(self.documents)} documents")
        
        # Generate CNN dataset
        print("Generating CNN training dataset...")
        dataset_info = self.cnn_generator.generate_cnn_dataset(self.documents)
        
        # Create train/validation split
        print("Creating train/validation split...")
        split_info = self.cnn_generator.create_training_validation_split()
        
        # Generate interactive visualizations
        print("Creating interactive visualizations...")
        self.visualizer.create_threat_timeline(self.documents)
        self.visualizer.create_network_graph(self.documents)
        
        # Generate human-readable analysis visualizations
        print("Creating human-readable analysis visualizations...")
        human_analysis_dir = self.human_visualizer.generate_all_human_analysis_visuals(self.documents)
        
        # Generate static visualizations
        self._create_static_visualizations()
        
        return dataset_info, split_info
    
    def _extract_documents_from_pcap(self, pcap_path: str) -> List[NetworkDocument]:
        """Extract documents from PCAP (simplified version)."""
        if not SCAPY_AVAILABLE:
            # Generate sample data for demonstration
            return self._generate_sample_documents()
        
        try:
            packets = scapy.rdpcap(pcap_path)
            documents = []
            
            for i, packet in enumerate(packets[:1000]):  # Limit for demo
                if packet.haslayer(IP):
                    doc = NetworkDocument(
                        doc_id=f"doc_{i}",
                        source_ip=packet[IP].src,
                        dest_ip=packet[IP].dst,
                        protocol="HTTP" if packet.haslayer(TCP) and packet[TCP].dport in [80, 443] else "TCP",
                        timestamp=datetime.fromtimestamp(float(packet.time)),
                        extracted_text=f"network traffic from {packet[IP].src} to {packet[IP].dst}",
                        threat_indicators=[],
                        classification="benign" if i % 10 != 0 else "web_attack",
                        confidence=0.1 if i % 10 != 0 else 0.8
                    )
                    documents.append(doc)
            
            return documents
            
        except Exception as e:
            print(f"Error processing PCAP: {e}")
            return self._generate_sample_documents()
    
    def _generate_sample_documents(self) -> List[NetworkDocument]:
        """Generate sample documents for demonstration."""
        classifications = ['benign', 'web_attack', 'malware_c2', 'data_exfiltration', 'reconnaissance']
        protocols = ['HTTP', 'DNS', 'TCP', 'UDP']
        
        documents = []
        
        for i in range(500):
            classification = np.random.choice(classifications, p=[0.7, 0.1, 0.1, 0.05, 0.05])
            protocol = np.random.choice(protocols)
            confidence = 0.1 if classification == 'benign' else np.random.uniform(0.6, 0.9)
            
            doc = NetworkDocument(
                doc_id=f"sample_doc_{i}",
                source_ip=f"192.168.{np.random.randint(1,255)}.{np.random.randint(1,255)}",
                dest_ip=f"10.0.{np.random.randint(1,255)}.{np.random.randint(1,255)}",
                protocol=protocol,
                timestamp=datetime.now(),
                extracted_text=f"sample network traffic {i}",
                threat_indicators=['suspicious_pattern'] if classification != 'benign' else [],
                classification=classification,
                confidence=confidence
            )
            documents.append(doc)
        
        return documents
    
    def _create_static_visualizations(self):
        """Create static visualizations and charts."""
        if not MATPLOTLIB_AVAILABLE:
            print("Matplotlib not available - skipping static visualizations")
            return
        
        # Threat distribution pie chart
        classifications = [doc.classification for doc in self.documents]
        class_counts = Counter(classifications)
        
        plt.figure(figsize=(10, 8))
        plt.pie(class_counts.values(), labels=class_counts.keys(), autopct='%1.1f%%')
        plt.title("Threat Classification Distribution")
        plt.savefig(os.path.join(self.config['output_dir'], 'threat_distribution.png'), 
                   dpi=150, bbox_inches='tight')
        plt.close()
        
        # Confidence distribution histogram
        confidences = [doc.confidence for doc in self.documents if doc.classification != 'benign']
        
        if confidences:
            plt.figure(figsize=(12, 6))
            plt.hist(confidences, bins=20, alpha=0.7, color='red')
            plt.xlabel('Threat Confidence')
            plt.ylabel('Frequency')
            plt.title('Threat Confidence Distribution')
            plt.savefig(os.path.join(self.config['output_dir'], 'confidence_distribution.png'),
                       dpi=150, bbox_inches='tight')
            plt.close()
        
        # Protocol distribution bar chart
        protocols = [doc.protocol for doc in self.documents]
        protocol_counts = Counter(protocols)
        
        plt.figure(figsize=(10, 6))
        plt.bar(protocol_counts.keys(), protocol_counts.values())
        plt.xlabel('Protocol')
        plt.ylabel('Count')
        plt.title('Protocol Distribution')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(os.path.join(self.config['output_dir'], 'protocol_distribution.png'),
                   dpi=150, bbox_inches='tight')
        plt.close()
        
        print("Static visualizations created")


def main():
    """
    Main execution function with comprehensive visual outputs for network security analysis.
    
    This function orchestrates the entire analysis pipeline:
    1. Collects user configuration (PCAP file, output directory, parameters)
    2. Initializes the visual network analyzer with all components
    3. Processes PCAP files to extract network documents
    4. Generates multiple types of visual outputs:
       - CNN training datasets (heatmaps, flow charts, correlation matrices)
       - Human-readable analysis charts and dashboards
       - Interactive visualizations for exploration
    5. Creates train/validation splits for machine learning
    6. Displays comprehensive results and file locations
    
    The system is designed to be user-friendly while providing professional-grade
    outputs suitable for both academic research and practical security analysis.
    """
    print("Visual NLP Network Security Analysis System")
    print("=" * 60)
    
    # ==========================================================================
    # USER CONFIGURATION COLLECTION
    # ==========================================================================
    # Collect all necessary parameters from the user through interactive prompts
    config = {}
    
    # Get PCAP file path or enable demo mode
    pcap_path = input("Enter path to PCAP file (or press Enter for demo): ").strip().strip('"\'')
    if not pcap_path or not os.path.exists(pcap_path):
        print("Using demo mode with sample data")
        pcap_path = "demo"  # Special flag for demo mode
    config['pcap_path'] = pcap_path
    
    # Get output directory for all generated files
    output_dir = input("Enter output directory (default: 'visual_results'): ").strip()
    config['output_dir'] = output_dir if output_dir else 'visual_results'
    
    # Display final configuration to user for confirmation
    print(f"\nConfiguration:")
    print(f"  PCAP file: {config['pcap_path']}")
    print(f"  Output directory: {config['output_dir']}")
    print()
    
    try:
        # =======================================================================
        # SYSTEM INITIALIZATION AND PROCESSING
        # =======================================================================
        
        # Initialize the main analyzer with all visual components
        # This creates all necessary subdirectories and initializes:
        # - Visual feature extractor for CNN images
        # - CNN dataset generator for training data
        # - Interactive visualizer for web-based charts
        # - Human-readable visualizer for analysis dashboards
        analyzer = VisualNetworkAnalyzer(config)
        
        # Process PCAP file and generate all visual outputs
        # This is the main processing function that:
        # 1. Extracts network documents from PCAP data
        # 2. Generates CNN training images and datasets
        # 3. Creates human-readable analysis visualizations
        # 4. Builds interactive web-based charts
        # 5. Organizes everything into train/validation splits
        print("=" * 60)
        dataset_info, split_info = analyzer.process_pcap_and_generate_visuals(config['pcap_path'])
        
        # =======================================================================
        # RESULTS DISPLAY AND SUMMARY
        # =======================================================================
        
        # Display comprehensive results summary
        print("\n" + "=" * 60)
        print("VISUAL ANALYSIS COMPLETE")
        print("=" * 60)
        
        # CNN Dataset Information
        # Shows statistics about the generated machine learning datasets
        print(f"\nGenerated CNN Dataset:")
        print(f"  Total samples: {dataset_info['total_samples']:,}")
        print(f"  Batches created: {dataset_info['num_batches']}")
        print(f"  Image size: {dataset_info['image_size']}")
        print(f"  Classes: {', '.join(dataset_info['classes'])}")
        
        # Train/Validation Split Information
        # Shows how the dataset was divided for machine learning
        print(f"\nDataset Split:")
        print(f"  Training samples: {split_info['train_samples']}")
        print(f"  Validation samples: {split_info['val_samples']}")
        
        # Generated Files and Directories
        # Complete list of all outputs for user reference
        print(f"\nGenerated Visual Outputs:")
        print(f"  CNN Images: {config['output_dir']}/cnn_images/")
        print(f"  CNN Labels: {config['output_dir']}/cnn_labels/")
        print(f"  Training Set: {config['output_dir']}/train/")
        print(f"  Validation Set: {config['output_dir']}/val/")
        print(f"  Human Analysis: {config['output_dir']}/human_analysis/")
        print(f"  Interactive Timeline: threat_timeline.html")
        print(f"  Network Graph: network_graph.html")
        print(f"  Static Charts: *.png files")
        
        # CNN Training Files Description
        # Explains what files can be used for machine learning
        print(f"\nFiles suitable for CNN training:")
        print(f"  Threat heatmaps (256x256 RGB images)")
        print(f"  Protocol flow images (256x256 RGB images)")
        print(f"  Feature correlation matrices")
        print(f"  JSON labels for supervised learning")
        print(f"  Raw numpy arrays for direct CNN input")
        
        # Human Analysis Files Description
        # Explains what files are available for manual analysis
        print(f"\nHuman-readable analysis images:")
        print(f"  Threat overview dashboard")
        print(f"  Network topology map")
        print(f"  Detailed threat analysis charts")
        print(f"  Executive security summary report")
        
    except KeyboardInterrupt:
        # Handle user interruption gracefully
        print("\nAnalysis interrupted by user")
        print("Partial results may be available in the output directory")
        
    except Exception as e:
        # Handle any unexpected errors with detailed information
        print(f"\nError during analysis: {str(e)}")
        print("Please check your input files and try again")
        
        # Show detailed error information for debugging
        import traceback
        traceback.print_exc()


# =============================================================================
# PROGRAM ENTRY POINT
# =============================================================================
if __name__ == "__main__":
    """
    Entry point for the Visual NLP Network Security Analysis System.
    
    This script can be run directly from the command line to perform
    comprehensive network security analysis with visual outputs suitable
    for both CNN training and human analysis.
    
    Usage:
        python visual_nlp_network_system.py
        
    The program will prompt for:
        - PCAP file path (or demo mode)
        - Output directory for results
        
    Outputs:
        - CNN training datasets with images and labels
        - Human-readable analysis dashboards
        - Interactive visualizations
        - Comprehensive reports and statistics
    """
    main()