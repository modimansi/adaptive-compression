import os
import time
import json
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict


class CompressionAnalyzer:
    """
    Analyzer for compression performance and statistics
    
    This class manages compression results, provides statistical analysis
    and generates visualizations to help understand compression performance
    across different file types and sizes.
    """
    
    def __init__(self):
        """Initialize the analyzer"""
        self.results = []
        self.filename_map = {}  # Maps filenames to result indices
        
        # Method names mapping for consistent labeling
        self.method_names = {
            '1': 'RLE', '2': 'Dictionary', '3': 'Huffman', '4': 'Delta',
            '5': 'DEFLATE', '6': 'BZIP2', '7': 'LZMA', '8': 'ZStd',
            '9': 'LZ4', '10': 'Brotli', '11': 'LZHAM', '255': 'No Compression'
        }
    
    def add_result(self, filename, stats):
        """
        Add a compression result for analysis. If a file with the same name exists,
        it will be replaced with the new result.
        
        Args:
            filename (str): Name of the compressed file
            stats (dict): Compression statistics
        """
        base_filename = os.path.basename(filename)
        stats['filename'] = base_filename
        stats['extension'] = os.path.splitext(base_filename)[1].lower() or 'unknown'
        stats['filename_no_ext'] = os.path.splitext(base_filename)[0]
        stats['timestamp'] = time.time()
        
        # Format size for display
        stats['size_label'] = self._format_file_size(stats.get('original_size', 0))
        
        # If we already have a result with this filename, replace it
        if base_filename in self.filename_map:
            index = self.filename_map[base_filename]
            old_timestamp = self.results[index].get('timestamp', 0)
            new_timestamp = stats.get('timestamp', 0)
            
            # Only replace if the new entry is more recent
            if new_timestamp > old_timestamp:
                print(f"Replacing previous result for '{base_filename}'")
                self.results[index] = stats
        else:
            # Add new result
            self.results.append(stats)
            self.filename_map[base_filename] = len(self.results) - 1
            
            
    def save_results(self, filename):
        """
        Save results to a JSON file
        
        Args:
            filename (str): Path to save the results
        """
        with open(filename, 'w') as f:
            json.dump(self.results, f, indent=2)
    
    def load_results(self, filename):
        """
        Load results from a JSON file, filtering out duplicates by filename
        and keeping only the most recent entry for each filename.
        
        Args:
            filename (str): Path to load the results from
            
        Returns:
            int: Number of results loaded
        """
        try:
            with open(filename, 'r') as f:
                all_results = json.load(f)
            
            # Track duplicates for reporting
            original_count = len(all_results)
            
            # Group by filename and keep only the most recent entry
            filename_to_latest = {}
            for result in all_results:
                base_filename = result.get('filename', "unknown")
                timestamp = result.get('timestamp', 0)
                
                # If we haven't seen this filename before, or if this entry is more recent
                if (base_filename not in filename_to_latest or 
                    timestamp > filename_to_latest[base_filename].get('timestamp', 0)):
                    filename_to_latest[base_filename] = result
            
            # Rebuild results list with unique entries
            self.results = list(filename_to_latest.values())
            
            # Rebuild the filename map
            self.filename_map = {}
            for i, result in enumerate(self.results):
                base_filename = result.get('filename', f"file_{i}")
                self.filename_map[base_filename] = i
                
                # Ensure extension is set for older results
                if 'extension' not in result:
                    result['extension'] = os.path.splitext(base_filename)[1].lower() or 'unknown'
                    
                # Ensure filename_no_ext is set for older results
                if 'filename_no_ext' not in result:
                    result['filename_no_ext'] = os.path.splitext(base_filename)[0]
                    
                # Ensure size_label is set for older results
                if 'size_label' not in result:
                    result['size_label'] = self._format_file_size(result.get('original_size', 0))
            
            # Calculate number of duplicates removed
            duplicates_removed = original_count - len(self.results)
            
            if duplicates_removed > 0:
                print(f"Loaded {len(self.results)} unique results (removed {duplicates_removed} duplicates)")
            else:
                print(f"Loaded {len(self.results)} results (no duplicates found)")
                
            return len(self.results)
            
        except Exception as e:
            print(f"Error loading results: {e}")
            self.results = []
            self.filename_map = {}
            return 0
    def clear_results(self):
        """
        Clear all stored results
        """
        self.results = []
        self.filename_map = {}
    
    def get_summary_stats(self):
        """
        Get summary statistics for all results
        
        Returns:
            dict: Summary statistics
        """
        if not self.results:
            return {
                'total_files': 0,
                'total_original_size': 0,
                'total_compressed_size': 0,
                'average_ratio': 0,
                'average_percent_reduction': 0,
                'average_throughput': 0,
                'file_types': {}
            }
        
        # Count file types and compression by type
        file_types = defaultdict(int)
        type_compression = defaultdict(list)
        type_original_size = defaultdict(int)
        type_compressed_size = defaultdict(int)
        
        for result in self.results:
            ext = result.get('extension', 'unknown').lower()
            file_types[ext] += 1
            type_compression[ext].append(result.get('percent_reduction', 0))
            type_original_size[ext] += result.get('original_size', 0)
            type_compressed_size[ext] += result.get('compressed_size', 0)
        
        # Calculate average compression by file type
        type_avg_compression = {ext: sum(reductions)/len(reductions) if reductions else 0 
                             for ext, reductions in type_compression.items()}
        
        # Calculate compression ratio by file type
        type_ratio = {ext: type_compressed_size[ext]/type_original_size[ext] if type_original_size[ext] > 0 else 1.0
                   for ext in file_types.keys()}
        
        # Overall statistics
        total_original_size = sum(r.get('original_size', 0) for r in self.results)
        total_compressed_size = sum(r.get('compressed_size', 0) for r in self.results)
        
        summary = {
            'total_files': len(self.results),
            'total_original_size': total_original_size,
            'total_compressed_size': total_compressed_size,
            'average_ratio': sum(r.get('ratio', 0) for r in self.results) / len(self.results),
            'average_percent_reduction': sum(r.get('percent_reduction', 0) for r in self.results) / len(self.results),
            'average_throughput': sum(r.get('throughput_mb_per_sec', 0) for r in self.results) / len(self.results),
            'file_types': dict(file_types),
            'type_avg_compression': type_avg_compression,
            'type_ratio': type_ratio,
            'type_original_size': dict(type_original_size),
            'type_compressed_size': dict(type_compressed_size)
        }
        
        # Calculate overall ratio
        if total_original_size > 0:
            summary['overall_ratio'] = total_compressed_size / total_original_size
            summary['overall_percent_reduction'] = (1 - summary['overall_ratio']) * 100
        else:
            summary['overall_ratio'] = 1.0
            summary['overall_percent_reduction'] = 0.0
        
        # Format sizes for display
        summary['total_original_size_formatted'] = self._format_file_size(total_original_size)
        summary['total_compressed_size_formatted'] = self._format_file_size(total_compressed_size)
        
        return summary
    
    def remove_duplicates(self):
        """
        Remove duplicate entries from current results, keeping only the most recent 
        version of each file.
        
        Returns:
            int: Number of duplicates removed
        """
        if not self.results:
            return 0
        
        # Track initial count
        original_count = len(self.results)
        
        # Group by filename and keep only the most recent entry
        filename_to_latest = {}
        for result in self.results:
            base_filename = result.get('filename', "unknown")
            timestamp = result.get('timestamp', 0)
            
            # If we haven't seen this filename before, or if this entry is more recent
            if (base_filename not in filename_to_latest or 
                timestamp > filename_to_latest[base_filename].get('timestamp', 0)):
                filename_to_latest[base_filename] = result
        
        # Rebuild results list with unique entries
        self.results = list(filename_to_latest.values())
        
        # Rebuild the filename map
        self.filename_map = {}
        for i, result in enumerate(self.results):
            base_filename = result.get('filename', f"file_{i}")
            self.filename_map[base_filename] = i
        
        # Calculate number of duplicates removed
        duplicates_removed = original_count - len(self.results)
        
        if duplicates_removed > 0:
            print(f"Removed {duplicates_removed} duplicate entries")
        
        return duplicates_removed
    
    def get_method_usage_stats(self):
        """
        Get statistics on compression method usage
        
        Returns:
            dict: Method usage statistics
        """
        if not self.results:
            return {}
        
        method_counts = defaultdict(int)
        method_bytes_saved = defaultdict(int)
        file_type_method_usage = defaultdict(lambda: defaultdict(int))
        
        for result in self.results:
            chunk_stats = result.get('chunk_stats', {})
            method_usage = chunk_stats.get('method_usage', {})
            file_ext = result.get('extension', 'unknown')
            
            for method_id, count in method_usage.items():
                method_counts[method_id] += count
                # Track method usage per file type
                file_type_method_usage[file_ext][method_id] += count
        
        total_chunks = sum(method_counts.values())
        
        method_stats = {
            'method_counts': dict(method_counts),
            'method_percentages': {method_id: (count / total_chunks * 100) if total_chunks > 0 else 0 
                                for method_id, count in method_counts.items()},
            'total_chunks': total_chunks,
            'file_type_method_usage': {ext: dict(methods) for ext, methods in file_type_method_usage.items()}
        }
        
        return method_stats
    
    def plot_compression_ratio(self, figsize=(12, 7)):
        """
        Plot compression ratio grouped by file type and size with improved readability
        
        Args:
            figsize (tuple): Figure size (width, height)
            
        Returns:
            matplotlib.figure.Figure: The figure object
        """
        if not self.results:
            return None
        
        fig, ax = plt.subplots(figsize=figsize)
        
        # Group by file type
        file_types = defaultdict(list)
        for r in self.results:
            ext = r.get('extension', 'unknown')
            file_types[ext].append(r)
        
        # Set up colors for different file types
        colors = plt.cm.tab10.colors
        color_map = {ext: colors[i % len(colors)] for i, ext in enumerate(file_types.keys())}
        
        # Plot each file type as a group
        x_pos = 0
        x_ticks = []
        x_labels = []
        bars = []
        file_indices = []  # To store indices for each file
        
        for ext, results in file_types.items():
            # Sort by file size (smaller files first)
            results.sort(key=lambda r: r.get('original_size', 0))
            
            # Add a group label
            file_count = len(results)
            ax.text(x_pos + (file_count/2) - 0.5, 1.05, f"{ext} ({file_count})",
                   ha='center', va='bottom', fontsize=10, fontweight='bold',
                   bbox=dict(boxstyle='round,pad=0.3', fc=color_map[ext], alpha=0.3))
            
            # Track file indices for simple x-axis labels
            file_index = 1
            
            for r in results:
                ratio = r.get('ratio', 1.0)
                size_label = r.get('size_label', 'unknown')
                filename = r.get('filename_no_ext', 'unknown')
                
                # Truncate filename if too long
                if len(filename) > 15:
                    filename = filename[:12] + "..."
                
                # Create a simple index-based label
                label = f"{file_index}"
                file_indices.append(f"{file_index}: {filename} ({size_label})")
                
                bar = ax.bar(x_pos, ratio, width=0.8, color=color_map[ext], 
                        alpha=0.7, label=ext if ext not in [b.get_label() for b in bars] else "")
                
                # Add compression percentage on top of bar
                percent_reduction = r.get('percent_reduction', 0)
                
                # Only show percentage on top if significant compression occurred
                if percent_reduction > 5:
                    ax.text(x_pos, ratio + 0.02, f"{percent_reduction:.1f}%", 
                        ha='center', va='bottom', fontsize=8, rotation=0)
                
                x_ticks.append(x_pos)
                x_labels.append(label)
                if ext not in [b.get_label() for b in bars]:
                    bars.append(bar)
                
                x_pos += 1
                file_index += 1
            
            # Add space between groups
            x_pos += 1
        
        # Add a horizontal line at ratio = 1.0 (no compression)
        ax.axhline(y=1.0, color='red', linestyle='--', alpha=0.7, label='No Compression')
        
        # Add labels and title
        ax.set_ylabel('Compression Ratio (smaller is better)')
        ax.set_title('Compression Ratio by File Type and Size')
        
        # Set x-axis ticks and labels
        ax.set_xticks(x_ticks)
        ax.set_xticklabels(x_labels, fontsize=8)
        
        # Add grid and legend
        ax.grid(axis='y', linestyle='--', alpha=0.7)
        ax.legend(handles=bars + [plt.Line2D([0], [0], color='red', linestyle='--', alpha=0.7, label='No Compression')],
                title="File Types", loc='upper right')
        
        # Add a text box with summary statistics
        summary = self.get_summary_stats()
        summary_text = (f"Total files: {summary['total_files']}\n"
                     f"Average compression: {summary['average_percent_reduction']:.1f}%\n"
                     f"Overall ratio: {summary['overall_ratio']:.3f}")
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        ax.text(0.05, 0.95, summary_text, transform=ax.transAxes, fontsize=9,
               verticalalignment='top', bbox=props)
        
        # Add a text box with file index legend at the bottom
        if len(file_indices) > 0:
            # Create legend text - display at most 10 files
            legend_text = "\n".join(file_indices[:10])
            if len(file_indices) > 10:
                legend_text += f"\n... and {len(file_indices) - 10} more files"
                
            # Add legend box
            fig.text(0.5, 0.01, legend_text, ha='center', va='bottom', fontsize=8, 
                 bbox=dict(boxstyle='round', facecolor='lightgrey', alpha=0.5))
        
        plt.tight_layout()
        plt.subplots_adjust(bottom=0.15)  # Make room for file index legend
        return fig
    
    def plot_method_usage(self, figsize=(12, 7)):
        """
        Plot compression method usage broken down by file type
        
        Args:
            figsize (tuple): Figure size (width, height)
            
        Returns:
            matplotlib.figure.Figure: The figure object
        """
        method_stats = self.get_method_usage_stats()
        
        if not method_stats or 'file_type_method_usage' not in method_stats:
            return None
        
        # Create figure with two subplots - one for overall, one for breakdown
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
        
        # Overall method usage - pie chart (left subplot)
        method_counts = method_stats['method_counts']
        
        # Create data for pie chart
        labels = [self.method_names.get(str(method_id), f"Method {method_id}") 
                 for method_id in method_counts.keys()]
        sizes = list(method_counts.values())
        
        # Set colors
        colors = plt.cm.tab10.colors[:len(labels)]
        
        # Create pie chart
        wedges, texts, autotexts = ax1.pie(
            sizes, 
            labels=labels, 
            colors=colors,
            autopct='%1.1f%%',
            startangle=90,
            shadow=False
        )
        
        # Equal aspect ratio ensures that pie is drawn as a circle
        ax1.axis('equal')
        ax1.set_title('Overall Compression Method Usage')
        
        # Add legend to the left subplot
        ax1.legend(
            wedges, 
            [f"{label}: {count} chunks" for label, count in zip(labels, sizes)],
            title="Methods",
            loc="center left",
            bbox_to_anchor=(1, 0, 0.5, 1),
            fontsize='small'
        )
        
        # File type breakdown - stacked bar chart (right subplot)
        file_type_method_usage = method_stats['file_type_method_usage']
        
        # Get all method IDs used
        all_methods = sorted(method_counts.keys())
        
        # Prepare data for stacked bar chart
        file_types = list(file_type_method_usage.keys())
        x = np.arange(len(file_types))
        bar_width = 0.8
        bottom = np.zeros(len(file_types))
        
        # Plot each method as a segment of the stacked bars
        for method_id in all_methods:
            method_values = [file_type_method_usage[ext].get(method_id, 0) for ext in file_types]
            method_name = self.method_names.get(str(method_id), f"Method {method_id}")
            
            # Match color with the pie chart
            method_index = list(method_counts.keys()).index(method_id)
            color = colors[method_index % len(colors)]
            
            ax2.bar(x, method_values, bar_width, label=method_name, bottom=bottom, color=color)
            bottom += method_values
        
        # Configure right subplot
        ax2.set_xlabel('File Type')
        ax2.set_ylabel('Chunk Count')
        ax2.set_title('Compression Methods by File Type')
        ax2.set_xticks(x)
        ax2.set_xticklabels([ext if ext else 'unknown' for ext in file_types], rotation=45)
        ax2.grid(axis='y', linestyle='--', alpha=0.3)
        
        # Add total chunk counts on top of bars
        for i, ext in enumerate(file_types):
            total = sum(file_type_method_usage[ext].values())
            if total > 0:
                ax2.text(i, total + 0.5, str(total), ha='center', va='bottom')
        
        # Adjust layout
        plt.tight_layout()
        
        return fig
    
    def plot_size_comparison(self, figsize=(12, 7)):
        """
        Plot size comparison between original and compressed files
        grouped by file type with improved readability
        
        Args:
            figsize (tuple): Figure size (width, height)
            
        Returns:
            matplotlib.figure.Figure: The figure object
        """
        if not self.results:
            return None
        
        fig, ax = plt.subplots(figsize=figsize)
        
        # Group by file type
        file_types = defaultdict(list)
        for r in self.results:
            ext = r.get('extension', 'unknown')
            file_types[ext].append(r)
        
        # Set up colors for different file types
        colors = plt.cm.tab10.colors
        
        # Plot each file type as a group
        x_pos = 0
        x_ticks = []
        x_labels = []
        file_indices = []  # To store indices for each file
        
        # Track category positions for group labels
        category_positions = []
        category_labels = []
        
        for i, (ext, results) in enumerate(file_types.items()):
            # Sort by file size (smaller files first)
            results.sort(key=lambda r: r.get('original_size', 0))
            
            # Track category start position
            category_start = x_pos
            color = colors[i % len(colors)]
            
            # Track file indices for simple x-axis labels
            file_index = 1
            
            for r in results:
                original_size = r.get('original_size', 0) / 1024  # KB
                compressed_size = r.get('compressed_size', 0) / 1024  # KB
                filename = r.get('filename_no_ext', 'unknown')
                
                # Create a simple index-based label
                label = f"{file_index}"
                file_indices.append(f"{file_index}: {filename} ({r.get('size_label', '')})")
                
                # Create grouped bars
                width = 0.35
                rects1 = ax.bar(x_pos - width/2, original_size, width, 
                           label='Original' if x_pos == 0 else "", color='skyblue')
                rects2 = ax.bar(x_pos + width/2, compressed_size, width, 
                           label='Compressed' if x_pos == 0 else "", color='lightgreen')
                
                # Add size labels on top of significant bars only
                if original_size > 50:  # Only for bars that are visible
                    if compressed_size > 0:
                        reduction = (1 - compressed_size/original_size) * 100
                        ax.text(x_pos, max(original_size, compressed_size) * 1.05, 
                             f"{reduction:.1f}%", ha='center', va='bottom', fontsize=8)
                
                x_ticks.append(x_pos)
                x_labels.append(label)
                
                x_pos += 1
                file_index += 1
            
            # Record category midpoint for labels
            if results:
                category_positions.append((category_start + x_pos - 1) / 2)
                category_labels.append(f"{ext} ({len(results)})")
            
            # Add space between categories
            x_pos += 1
        
        # Add category labels
        for pos, label in zip(category_positions, category_labels):
            ax.annotate(
                label, 
                xy=(pos, 0), 
                xytext=(0, -30), 
                textcoords="offset points",
                ha='center', va='top',
                fontsize=10, fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.3', fc='yellow', alpha=0.3)
            )
        
        # Add labels and title
        ax.set_ylabel('Size (KB)')
        ax.set_title('Size Comparison: Original vs. Compressed by File Type')
        
        # Use logarithmic scale for y-axis if range is very large
        max_size = max([r.get('original_size', 0) for r in self.results], default=1) / 1024
        min_size = min([r.get('compressed_size', 0) for r in self.results if r.get('compressed_size', 0) > 0], default=1) / 1024
        
        if max_size / min_size > 1000:  # If range is more than 3 orders of magnitude
            ax.set_yscale('log')
            ax.text(0.02, 0.02, "Note: Using logarithmic scale", transform=ax.transAxes, 
                fontsize=8, style='italic')
        
        # Set x-axis ticks and labels
        ax.set_xticks(x_ticks)
        ax.set_xticklabels(x_labels, fontsize=8)
        
        # Add grid and legend
        ax.grid(axis='y', linestyle='--', alpha=0.7)
        ax.legend(loc='upper right')
        
        # Add a text box with summary statistics
        summary = self.get_summary_stats()
        original_total = summary['total_original_size'] / 1024  # KB
        compressed_total = summary['total_compressed_size'] / 1024  # KB
        summary_text = (f"Total original: {self._format_file_size(summary['total_original_size'])}\n"
                     f"Total compressed: {self._format_file_size(summary['total_compressed_size'])}\n"
                     f"Overall reduction: {summary['overall_percent_reduction']:.1f}%")
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        ax.text(0.05, 0.95, summary_text, transform=ax.transAxes, fontsize=9,
               verticalalignment='top', bbox=props)
        
        # Add a text box with file index legend at the bottom
        if len(file_indices) > 0:
            # Create legend text - display at most 10 files
            legend_text = "\n".join(file_indices[:10])
            if len(file_indices) > 10:
                legend_text += f"\n... and {len(file_indices) - 10} more files"
                
            # Add legend box
            fig.text(0.5, 0.01, legend_text, ha='center', va='bottom', fontsize=8, 
                 bbox=dict(boxstyle='round', facecolor='lightgrey', alpha=0.5))
        
        plt.tight_layout()
        plt.subplots_adjust(bottom=0.15)  # Make room for file index legend
        return fig
    
    def plot_throughput(self, figsize=(12, 7)):
        """
        Plot compression throughput grouped by file type and size with improved readability
        
        Args:
            figsize (tuple): Figure size (width, height)
            
        Returns:
            matplotlib.figure.Figure: The figure object
        """
        if not self.results:
            return None
        
        fig, ax = plt.subplots(figsize=figsize)
        
        # Group by file type
        file_types = defaultdict(list)
        for r in self.results:
            ext = r.get('extension', 'unknown')
            file_types[ext].append(r)
        
        # Set up colors for different file types
        colors = plt.cm.tab10.colors
        color_map = {ext: colors[i % len(colors)] for i, ext in enumerate(file_types.keys())}
        
        # Plot each file type as a group
        x_pos = 0
        x_ticks = []
        x_labels = []
        bars = []
        file_indices = []  # To store indices for each file
        
        for ext, results in file_types.items():
            # Sort by file size
            results.sort(key=lambda r: r.get('original_size', 0))
            
            # Add a group label
            file_count = len(results)
            max_throughput = max([r.get('throughput_mb_per_sec', 0) for r in results], default=0)
            ax.text(x_pos + (file_count/2) - 0.5, max_throughput + 2, f"{ext} ({file_count})",
                   ha='center', va='bottom', fontsize=10, fontweight='bold',
                   bbox=dict(boxstyle='round,pad=0.3', fc=color_map[ext], alpha=0.3))
            
            # Track file indices for simple x-axis labels
            file_index = 1
            
            for r in results:
                throughput = r.get('throughput_mb_per_sec', 0)
                size_label = r.get('size_label', 'unknown')
                filename = r.get('filename_no_ext', 'unknown')
                
                # Truncate filename if too long
                if len(filename) > 15:
                    filename = filename[:12] + "..."
                
                # Create a simple index-based label
                label = f"{file_index}"
                file_indices.append(f"{file_index}: {filename} ({size_label})")
                
                bar = ax.bar(x_pos, throughput, width=0.8, color=color_map[ext], 
                        alpha=0.7, label=ext if ext not in [b.get_label() for b in bars] else "")
                
                # Add throughput value on top of significant bars only
                if throughput > 5:
                    ax.text(x_pos, throughput + 0.5, f"{throughput:.1f}", 
                        ha='center', va='bottom', fontsize=8, rotation=0)
                
                x_ticks.append(x_pos)
                x_labels.append(label)
                if ext not in [b.get_label() for b in bars]:
                    bars.append(bar)
                
                x_pos += 1
                file_index += 1
            
            # Add space between groups
            x_pos += 1
        
        # Add labels and title
        ax.set_ylabel('Throughput (MB/s)')
        ax.set_title('Compression Throughput by File Type and Size')
        
        # Use logarithmic scale for y-axis to better show wide range of values
        if any(r.get('throughput_mb_per_sec', 0) < 0.1 for r in self.results):
            ax.set_yscale('symlog')  # Use symlog to handle zero values
            ax.set_ylim(bottom=0.01)  # Set a reasonable minimum
            ax.text(0.02, 0.02, "Note: Using logarithmic scale", transform=ax.transAxes, 
                fontsize=8, style='italic')
        
        # Set x-axis ticks and labels
        ax.set_xticks(x_ticks)
        ax.set_xticklabels(x_labels, fontsize=8)
        
        # Add grid and legend
        ax.grid(axis='y', linestyle='--', alpha=0.7)
        ax.legend(handles=bars, title="File Types", loc='upper right')
        
        # Add a text box with summary statistics
        summary = self.get_summary_stats()
        summary_text = (f"Average throughput: {summary['average_throughput']:.2f} MB/s\n"
                     f"Total files: {summary['total_files']}")
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        ax.text(0.05, 0.95, summary_text, transform=ax.transAxes, fontsize=9,
               verticalalignment='top', bbox=props)
        
        # Add a text box with file index legend at the bottom
        if len(file_indices) > 0:
            # Create legend text - display at most 10 files
            legend_text = "\n".join(file_indices[:10])
            if len(file_indices) > 10:
                legend_text += f"\n... and {len(file_indices) - 10} more files"
                
            # Add legend box
            fig.text(0.5, 0.01, legend_text, ha='center', va='bottom', fontsize=8, 
                 bbox=dict(boxstyle='round', facecolor='lightgrey', alpha=0.5))
        
        plt.tight_layout()
        plt.subplots_adjust(bottom=0.15)  # Make room for file index legend
        return fig
        
    def plot_file_type_summary(self, figsize=(12, 7)):
        """
        Plot a summary of compression performance by file type
        
        Args:
            figsize (tuple): Figure size (width, height)
            
        Returns:
            matplotlib.figure.Figure: The figure object
        """
        summary = self.get_summary_stats()
        
        if not summary['file_types']:
            return None
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
        
        # Extract data
        file_types = list(summary['file_types'].keys())
        file_counts = list(summary['file_types'].values())
        
        # Sort by file count (descending)
        sorted_indices = np.argsort(file_counts)[::-1]
        file_types = [file_types[i] for i in sorted_indices]
        file_counts = [file_counts[i] for i in sorted_indices]
        
        # File type distribution - pie chart (left subplot)
        colors = plt.cm.tab10.colors[:len(file_types)]
        
        # Create pie chart
        wedges, texts, autotexts = ax1.pie(
            file_counts, 
            labels=file_types, 
            colors=colors,
            autopct='%1.1f%%',
            startangle=90,
            shadow=False
        )
        
        # Equal aspect ratio ensures that pie is drawn as a circle
        ax1.axis('equal')
        ax1.set_title('File Type Distribution')
        
        # Compression performance by file type - bar chart (right subplot)
        x = np.arange(len(file_types))
        
        # Get compression ratios by file type
        ratios = [summary['type_ratio'].get(ext, 1.0) for ext in file_types]
        
        # Create bar chart
        bars = ax2.bar(x, ratios, color=colors)
        
        # Add a horizontal line at ratio = 1.0 (no compression)
        ax2.axhline(y=1.0, color='red', linestyle='--', alpha=0.7)
        
        # Add percentage labels
        for i, (ratio, ext) in enumerate(zip(ratios, file_types)):
            if ratio < 1.0:
                reduction = (1 - ratio) * 100
                ax2.text(i, ratio + 0.02, f"{reduction:.1f}%", 
                    ha='center', va='bottom', fontsize=8)
        
        # Configure right subplot
        ax2.set_xlabel('File Type')
        ax2.set_ylabel('Compression Ratio (smaller is better)')
        ax2.set_title('Compression Performance by File Type')
        ax2.set_xticks(x)
        ax2.set_xticklabels(file_types, rotation=45)
        ax2.grid(axis='y', linestyle='--', alpha=0.7)
        
        # Overall statistics text box
        stats_text = (f"Total files: {summary['total_files']}\n"
                     f"Total size: {summary['total_original_size_formatted']}\n"
                     f"Overall compression: {summary['overall_percent_reduction']:.1f}%")
        
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        fig.text(0.5, 0.01, stats_text, ha='center', va='bottom', fontsize=10,
               bbox=props)
        
        plt.tight_layout()
        plt.subplots_adjust(bottom=0.15)  # Make room for stats text
        
        return fig
    
    def _format_file_size(self, size_bytes):
        """
        Format file size in bytes to a human-readable format
        
        Args:
            size_bytes (int): Size in bytes
            
        Returns:
            str: Formatted size (e.g. "4.2 MB")
        """
        if size_bytes == 0:
            return "0 B"
           
        size_names = ["B", "KB", "MB", "GB", "TB"]
        i = 0
        while size_bytes >= 1024 and i < len(size_names) - 1:
            size_bytes /= 1024.0
            i += 1
            
        return f"{size_bytes:.1f} {size_names[i]}"


if __name__ == "__main__":
    # Simple test of the CompressionAnalyzer
    def test_compression_analyzer():
        analyzer = CompressionAnalyzer()
        
        # Add some test results
        analyzer.add_result("file1.txt", {
            'original_size': 1000,
            'compressed_size': 600,
            'ratio': 0.6,
            'percent_reduction': 40.0,
            'elapsed_time': 0.1,
            'throughput_mb_per_sec': 10.0,
            'chunk_stats': {
                'total_chunks': 10,
                'method_usage': {
                    '1': 3,
                    '2': 4,
                    '3': 2,
                    '4': 1,
                    '255': 0
                }
            }
        })
        
        analyzer.add_result("file2.dat", {
            'original_size': 2000,
            'compressed_size': 1500,
            'ratio': 0.75,
            'percent_reduction': 25.0,
            'elapsed_time': 0.2,
            'throughput_mb_per_sec': 9.5,
            'chunk_stats': {
                'total_chunks': 20,
                'method_usage': {
                    '1': 5,
                    '2': 5,
                    '3': 5,
                    '4': 3,
                    '255': 2
                }
            }
        })
        
        # Add a duplicate file to test replacement
        analyzer.add_result("file1.txt", {
            'original_size': 1200,
            'compressed_size': 700,
            'ratio': 0.58,
            'percent_reduction': 42.0,
            'elapsed_time': 0.15,
            'throughput_mb_per_sec': 8.0,
            'chunk_stats': {
                'total_chunks': 15,
                'method_usage': {
                    '1': 4,
                    '2': 5,
                    '3': 3,
                    '4': 2,
                    '255': 1
                }
            }
        })
        
        # Get summary stats
        summary = analyzer.get_summary_stats()
        print("Summary Statistics:")
        for key, value in summary.items():
            print(f"  {key}: {value}")
        
        # Get method usage stats
        method_stats = analyzer.get_method_usage_stats()
        print("\nMethod Usage Statistics:")
        for key, value in method_stats.items():
            print(f"  {key}: {value}")
        
        # Plot all visualizations
        plots = [
            ("compression_ratio", analyzer.plot_compression_ratio),
            ("method_usage", analyzer.plot_method_usage),
            ("size_comparison", analyzer.plot_size_comparison),
            ("throughput", analyzer.plot_throughput),
            ("file_type_summary", analyzer.plot_file_type_summary)
        ]
        
        for name, plot_func in plots:
            fig = plot_func()
            if fig:
                plt.figure(fig.number)
                plt.savefig(f"test_{name}.png")
                plt.close()
                print(f"Saved {name} plot to test_{name}.png")
        
        # Test saving and loading
        analyzer.save_results("test_compression_results.json")
        print("\nSaved results to test_compression_results.json")
        
        # Create a new analyzer and load the results
        new_analyzer = CompressionAnalyzer()
        num_loaded = new_analyzer.load_results("test_compression_results.json")
        
        # Verify loaded results
        new_summary = new_analyzer.get_summary_stats()
        print(f"Loaded {num_loaded} results:")
        for key in ['total_files', 'total_original_size', 'average_ratio']:
            print(f"  {key}: {new_summary[key]}")
        
        # Clean up
        os.remove("test_compression_results.json")
        for name, _ in plots:
            filename = f"test_{name}.png"
            if os.path.exists(filename):
                os.remove(filename)
    
    # Run the test
    test_compression_analyzer()
