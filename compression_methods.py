import heapq
from collections import Counter, defaultdict
import numpy as np
from abc import ABC, abstractmethod


class CompressionMethod(ABC):
    """
    Abstract base class for all compression methods
    """
    @property
    @abstractmethod
    def type_id(self):
        """Return the unique type identifier for this compression method"""
        pass
    
    @abstractmethod
    def compress(self, data):
        """
        Compress the given data using this method
        
        Args:
            data (bytes): Data to compress
            
        Returns:
            bytes: Compressed data
        """
        pass
    
    @abstractmethod
    def decompress(self, data, original_length):
        """
        Decompress the data
        
        Args:
            data (bytes): Compressed data
            original_length (int): Original length of the uncompressed data
            
        Returns:
            bytes: Decompressed data
        """
        pass
    
    def should_use(self, data, threshold=0.9):
        """
        Determine if this compression method should be used based on a quick analysis
        
        Args:
            data (bytes): Data to analyze
            threshold (float): Threshold for making decision (compression ratio)
            
        Returns:
            bool: True if this method should be used, False otherwise
        """
        # Default implementation just returns True
        # Specific methods can override this for more intelligent decisions
        return True
    
    def calculate_overhead(self):
        """
        Calculate the overhead (in bytes) this method adds
        
        Returns:
            int: Number of bytes of overhead
        """
        # Default overhead is 0, subclasses can override
        return 0


class RLECompression(CompressionMethod):
    """
    Run-Length Encoding compression method
    """
    @property
    def type_id(self):
        return 1
    
    def compress(self, data):
        """
        Compress using Run-Length Encoding
        
        Args:
            data (bytes): Data to compress
            
        Returns:
            bytes: Compressed data
        """
        if not data:
            return b''
        
        compressed = bytearray()
        current_byte = data[0]
        count = 1
        
        for i in range(1, len(data)):
            if data[i] == current_byte and count < 255:
                count += 1
            else:
                # Store the current run
                compressed.append(current_byte)
                compressed.append(count)
                
                # Start a new run
                current_byte = data[i]
                count = 1
        
        # Don't forget the last run
        compressed.append(current_byte)
        compressed.append(count)
        
        # Debug info
        print(f"RLE compression: {len(data)} bytes -> {len(compressed)} bytes")
        
        return bytes(compressed)
    
    def decompress(self, data, original_length):
        """
        Decompress RLE-compressed data
        
        Args:
            data (bytes): Compressed data
            original_length (int): Original length of the uncompressed data
            
        Returns:
            bytes: Decompressed data
        """
        if not data:
            return b''
        
        decompressed = bytearray()
        
        for i in range(0, len(data), 2):
            if i + 1 < len(data):  # Make sure we have a pair
                byte_val = data[i]
                count = data[i + 1]
                decompressed.extend([byte_val] * count)
        
        # Debug output
        print(f"RLE decompression: {len(data)} bytes -> {len(decompressed)} bytes")
        
        # Ensure we don't exceed the original length
        if len(decompressed) > original_length:
            print(f"Warning: RLE decompressed size ({len(decompressed)}) larger than original ({original_length})")
            decompressed = decompressed[:original_length]
        elif len(decompressed) < original_length:
            print(f"Warning: RLE decompressed size ({len(decompressed)}) smaller than original ({original_length})")
            # Pad with zeros if needed
            missing = original_length - len(decompressed)
            print(f"Padding with {missing} zero bytes")
            decompressed.extend([0] * missing)
        
        return bytes(decompressed)
    
    def should_use(self, data, threshold=0.9):
        """
        Determine if RLE should be used based on a quick analysis
        
        Args:
            data (bytes): Data to analyze
            threshold (float): Threshold for making decision
            
        Returns:
            bool: True if RLE should be used, False otherwise
        """
        if len(data) < 4:  # Too small for effective RLE
            return False
        
        # Quick sampling to estimate repetitions
        sample_size = min(1000, len(data))
        sample_step = max(1, len(data) // sample_size)
        
        repeats = 0
        for i in range(0, len(data) - 1, sample_step):
            if data[i] == data[i + 1]:
                repeats += 1
        
        repeat_ratio = repeats / (sample_size - 1)
        
        # If more than 30% of adjacent bytes are the same, RLE might be effective
        return repeat_ratio > 0.3


class DictionaryCompression(CompressionMethod):
    """
    Dictionary-based compression method (simplified LZ77)
    """
    def __init__(self, window_size=4096, lookahead_size=32):
        self.window_size = window_size
        self.lookahead_size = lookahead_size
    
    @property
    def type_id(self):
        return 2
    
    def compress(self, data):
        """
        Compress using a dictionary-based approach (simplified LZ77)
        
        Args:
            data (bytes): Data to compress
            
        Returns:
            bytes: Compressed data
        """
        if not data:
            return b''
        
        compressed = bytearray()
        pos = 0
        
        while pos < len(data):
            # Find the longest match in the sliding window
            match_pos, match_len = self._find_longest_match(data, pos)
            
            if match_len > 2:  # Only encode matches longer than 2 bytes
                # Encode as (flag, distance, length)
                compressed.append(1)  # Flag indicating a match
                
                # Convert distance to 2 bytes (little-endian)
                distance = pos - match_pos
                compressed.append(distance & 0xFF)
                compressed.append((distance >> 8) & 0xFF)
                
                # Length is a single byte
                compressed.append(match_len)
                
                pos += match_len
            else:
                # Encode as literal byte
                compressed.append(0)  # Flag indicating a literal
                compressed.append(data[pos])
                pos += 1
        
        return bytes(compressed)
    
    def decompress(self, data, original_length):
        """
        Decompress dictionary-compressed data
        
        Args:
            data (bytes): Compressed data
            original_length (int): Original length of the uncompressed data
            
        Returns:
            bytes: Decompressed data
        """
        if not data:
            return b''
        
        decompressed = bytearray()
        pos = 0
        
        while pos < len(data) and len(decompressed) < original_length:
            flag = data[pos]
            pos += 1
            
            if flag == 0:  # Literal
                if pos < len(data):
                    decompressed.append(data[pos])
                    pos += 1
            else:  # Match
                if pos + 2 < len(data):
                    # Read distance (little-endian)
                    distance = data[pos] | (data[pos + 1] << 8)
                    pos += 2
                    
                    # Read length
                    length = data[pos]
                    pos += 1
                    
                    # Copy from earlier in the output buffer
                    start = len(decompressed) - distance
                    for i in range(length):
                        if start + i < len(decompressed):
                            decompressed.append(decompressed[start + i])
                        else:
                            # Repeat the last byte for RLE-like patterns
                            decompressed.append(decompressed[-1])
        
        # Ensure we don't exceed the original length
        return bytes(decompressed[:original_length])
    
    def _find_longest_match(self, data, current_pos):
        """
        Find the longest matching sequence in the sliding window
        
        Args:
            data (bytes): Data to search in
            current_pos (int): Current position in the data
            
        Returns:
            tuple: (match_position, match_length)
        """
        start_pos = max(0, current_pos - self.window_size)
        lookahead = data[current_pos:current_pos + self.lookahead_size]
        
        best_match_pos = 0
        best_match_len = 0
        
        # Naive approach: check every position in the sliding window
        for i in range(start_pos, current_pos):
            # Find the length of the match
            match_len = 0
            while (match_len < len(lookahead) and 
                   current_pos + match_len < len(data) and 
                   data[i + match_len] == data[current_pos + match_len]):
                match_len += 1
            
            if match_len > best_match_len:
                best_match_pos = i
                best_match_len = match_len
        
        return best_match_pos, best_match_len
    
    def should_use(self, data, threshold=0.9):
        """
        Determine if dictionary compression should be used
        
        Args:
            data (bytes): Data to analyze
            threshold (float): Threshold for making decision
            
        Returns:
            bool: True if dictionary compression should be used
        """
        if len(data) < 100:  # Too small for effective dictionary compression
            return False
        
        # Quick sampling to estimate unique byte sequences
        sample_size = min(1000, len(data))
        sequence_length = 3
        
        sequences = set()
        for i in range(0, min(len(data) - sequence_length, sample_size)):
            seq = data[i:i + sequence_length]
            sequences.add(seq)
        
        # If there are many repeated sequences, dictionary compression might be effective
        # Calculate ratio of unique sequences to total sequences
        unique_ratio = len(sequences) / sample_size
        
        # If less than 80% of sequences are unique, dictionary compression might be effective
        return unique_ratio < 0.8


class HuffmanCompression(CompressionMethod):
    """
    Huffman coding compression method
    """
    @property
    def type_id(self):
        return 3
    
    def compress(self, data):
        """
        Compress using Huffman coding
        
        Args:
            data (bytes): Data to compress
            
        Returns:
            bytes: Compressed data
        """
        if not data:
            return b''
        
        # Count frequencies - ensure we're working with byte values (0-255)
        freq = Counter()
        for b in data:
            freq[b & 0xFF] += 1  # Ensure valid byte range with bitwise AND
        
        # Build Huffman tree
        tree = self._build_huffman_tree(freq)
        
        # Generate codes
        codes = {}
        self._generate_codes(tree, "", codes)
        
        # Store frequency table for decompression
        # Format: [num_entries][byte1][freq1][byte2][freq2]...
        compressed = bytearray()
        compressed.append(len(freq))
        
        for byte, count in freq.items():
            compressed.append(byte)
            # Store frequency as 4 bytes (little-endian)
            compressed.extend(count.to_bytes(4, byteorder='little'))
        
        # Encode the data
        encoded_bits = ""
        for byte in data:
            byte_val = byte & 0xFF  # Ensure valid byte
            encoded_bits += codes[byte_val]
        
        # Convert bits to bytes
        # First, store the number of bits
        num_bits = len(encoded_bits)
        compressed.extend(num_bits.to_bytes(4, byteorder='little'))
        
        # Then store the bits, padding to byte boundary
        for i in range(0, num_bits, 8):
            byte_bits = encoded_bits[i:i+8].ljust(8, '0')
            compressed.append(int(byte_bits, 2))
        
        return bytes(compressed)
    
    def decompress(self, data, original_length):
        """
        Decompress Huffman-compressed data
        
        Args:
            data (bytes): Compressed data
            original_length (int): Original length of the uncompressed data
            
        Returns:
            bytes: Decompressed data
        """
        if not data:
            return b''
        
        pos = 0
        
        # Read number of entries in frequency table
        num_entries = data[pos]
        pos += 1
        
        # Read frequency table
        freq = {}
        for _ in range(num_entries):
            byte = data[pos]
            pos += 1
            
            count = int.from_bytes(data[pos:pos+4], byteorder='little')
            pos += 4
            
            freq[byte] = count
        
        # Rebuild Huffman tree
        tree = self._build_huffman_tree(freq)
        
        # Read number of bits in encoded data
        num_bits = int.from_bytes(data[pos:pos+4], byteorder='little')
        pos += 4
        
        # Read encoded bits
        encoded_bits = ""
        for i in range(pos, len(data)):
            bits = bin(data[i])[2:].zfill(8)
            encoded_bits += bits
        
        # Truncate to the actual number of bits
        encoded_bits = encoded_bits[:num_bits]
        
        # Decode
        decompressed = bytearray()
        node = tree
        for bit in encoded_bits:
            if bit == '0':
                node = node[0]
            else:
                node = node[1]
            
            if isinstance(node, int):  # Leaf node
                decompressed.append(node)
                node = tree  # Reset to root
                
                if len(decompressed) >= original_length:
                    break
        
        return bytes(decompressed)
    
    def _build_huffman_tree(self, freq):
        """
        Build a Huffman tree from frequency table
        
        Args:
            freq (dict): Frequency table
            
        Returns:
            list/int: Huffman tree
        """
        heap = [[weight, [byte, ""]] for byte, weight in freq.items()]
        heapq.heapify(heap)
        
        while len(heap) > 1:
            lo = heapq.heappop(heap)
            hi = heapq.heappop(heap)
            
            for pair in lo[1:]:
                pair[1] = '0' + pair[1]
            for pair in hi[1:]:
                pair[1] = '1' + pair[1]
            
            heapq.heappush(heap, [lo[0] + hi[0]] + lo[1:] + hi[1:])
        
        # Convert heap representation to tree representation
        codes = sorted(heapq.heappop(heap)[1:], key=lambda p: (len(p[1]), p[1]))
        
        tree = self._build_tree_from_codes(codes)
        return tree
    
    def _build_tree_from_codes(self, codes):
        """
        Build a Huffman tree from codes
        
        Args:
            codes (list): List of [byte, code] pairs
            
        Returns:
            list/int: Huffman tree
        """
        tree = [None, None]  # [left, right]
        
        for byte, code in codes:
            node = tree
            for bit in code[:-1]:
                if bit == '0':
                    if node[0] is None:
                        node[0] = [None, None]
                    node = node[0]
                else:
                    if node[1] is None:
                        node[1] = [None, None]
                    node = node[1]
            
            # Last bit determines where to place the byte
            if code[-1] == '0':
                node[0] = byte
            else:
                node[1] = byte
        
        return tree
    
    def _generate_codes(self, node, code, codes):
        """
        Generate codes from Huffman tree
        
        Args:
            node: Current node in the tree
            code (str): Current code
            codes (dict): Dictionary to store the codes
        """
        if isinstance(node, int):  # Leaf node
            codes[node] = code
        else:
            if node[0] is not None:
                self._generate_codes(node[0], code + '0', codes)
            if node[1] is not None:
                self._generate_codes(node[1], code + '1', codes)
    
    def should_use(self, data, threshold=0.9):
        """
        Determine if Huffman coding should be used
        
        Args:
            data (bytes): Data to analyze
            threshold (float): Threshold for making decision
            
        Returns:
            bool: True if Huffman coding should be used
        """
        if len(data) < 100:  # Too small for effective Huffman coding
            return False
        
        # Calculate entropy
        freqs = Counter(data)
        entropy = 0
        for count in freqs.values():
            p = count / len(data)
            entropy -= p * np.log2(p)
        
        # If entropy is low, Huffman coding might be effective
        # Max entropy for bytes is 8 bits
        return entropy < 7.0  # Somewhat arbitrary threshold


class DeltaCompression(CompressionMethod):
    """
    Delta encoding compression method
    """
    @property
    def type_id(self):
        return 4
    
    def compress(self, data):
        """
        Compress using delta encoding
        
        Args:
            data (bytes): Data to compress
            
        Returns:
            bytes: Compressed data
        """
        if not data:
            return b''
        
        compressed = bytearray()
        
        # Store the first byte as-is
        compressed.append(data[0])
        
        # Store deltas for the rest
        for i in range(1, len(data)):
            delta = (data[i] - data[i-1]) & 0xFF  # Wrap around for byte arithmetic
            compressed.append(delta)
        
        return bytes(compressed)
    
    def decompress(self, data, original_length):
        """
        Decompress delta-encoded data
        
        Args:
            data (bytes): Compressed data
            original_length (int): Original length of the uncompressed data
            
        Returns:
            bytes: Decompressed data
        """
        if not data:
            return b''
        
        decompressed = bytearray()
        
        # First byte is stored as-is
        if data:
            decompressed.append(data[0])
        
        # Reconstruct from deltas
        prev = data[0] if data else 0
        for i in range(1, len(data)):
            value = (prev + data[i]) & 0xFF  # Wrap around for byte arithmetic
            decompressed.append(value)
            prev = value
        
        # Ensure we don't exceed the original length
        return bytes(decompressed[:original_length])
    
    def should_use(self, data, threshold=0.9):
        """
        Determine if delta encoding should be used
        
        Args:
            data (bytes): Data to analyze
            threshold (float): Threshold for making decision
            
        Returns:
            bool: True if delta encoding should be used
        """
        if len(data) < 4:  # Too small for effective delta encoding
            return False
        
        # Quick sampling to estimate if data has small variations
        sample_size = min(1000, len(data))
        sample_step = max(1, len(data) // sample_size)
        
        small_deltas = 0
        for i in range(0, len(data) - 1, sample_step):
            delta = abs(data[i] - data[i + 1])
            if delta < 32:  # Small delta
                small_deltas += 1
        
        small_delta_ratio = small_deltas / (sample_size - 1)
        
        # If more than 50% of deltas are small, delta encoding might be effective
        return small_delta_ratio > 0.5


class NoCompression(CompressionMethod):
    """
    No compression - used when other methods would not be effective
    """
    @property
    def type_id(self):
        return 255
    
    def compress(self, data):
        """
        No compression - return data as-is
        
        Args:
            data (bytes): Data to 'compress'
            
        Returns:
            bytes: Original data
        """
        # Make sure we return a copy of the data to avoid any reference issues
        return bytes(data)
    
    def decompress(self, data, original_length):
        """
        No decompression - return data as-is
        
        Args:
            data (bytes): Data to 'decompress'
            original_length (int): Original length (should be the same as data)
            
        Returns:
            bytes: Original data
        """
        # Check if data needs padding or truncation
        if len(data) != original_length:
            if len(data) < original_length:
                # Pad with zeros if needed
                print(f"Warning: NoCompression data is short ({len(data)} vs {original_length}), padding with zeros")
                return bytes(data) + b'\x00' * (original_length - len(data))
            else:
                # Truncate if needed
                print(f"Warning: NoCompression data is long ({len(data)} vs {original_length}), truncating")
                return bytes(data[:original_length])
        
        return bytes(data)


if __name__ == "__main__":
    # Simple test of the compression methods
    def test_compression_methods():
        test_data = b"AAAABBBCCDAAAABBBCCDA" * 20  # Simple test with repetitions
        
        # Test RLE compression
        rle = RLECompression()
        compressed = rle.compress(test_data)
        decompressed = rle.decompress(compressed, len(test_data))
        
        print(f"RLE: {len(test_data)} bytes -> {len(compressed)} bytes")
        print(f"RLE works correctly: {decompressed == test_data}")
        
        # Test dictionary compression
        dict_comp = DictionaryCompression()
        compressed = dict_comp.compress(test_data)
        decompressed = dict_comp.decompress(compressed, len(test_data))
        
        print(f"Dictionary: {len(test_data)} bytes -> {len(compressed)} bytes")
        print(f"Dictionary works correctly: {decompressed == test_data}")
        
        # Test Huffman compression
        huffman = HuffmanCompression()
        compressed = huffman.compress(test_data)
        decompressed = huffman.decompress(compressed, len(test_data))
        
        print(f"Huffman: {len(test_data)} bytes -> {len(compressed)} bytes")
        print(f"Huffman works correctly: {decompressed == test_data}")
        
        # Test Delta compression
        delta = DeltaCompression()
        compressed = delta.compress(test_data)
        decompressed = delta.decompress(compressed, len(test_data))
        
        print(f"Delta: {len(test_data)} bytes -> {len(compressed)} bytes")
        print(f"Delta works correctly: {decompressed == test_data}")
    
    # Run the test
    test_compression_methods()
