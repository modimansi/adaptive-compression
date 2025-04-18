import os
import time
from bitarray import bitarray
import numpy as np

class MarkerFinder:
    """
    Class for finding the shortest binary string that does not appear in the file.
    This is used as a marker for the compression algorithm.
    """
    
    def __init__(self, max_marker_length=32):
        """
        Initialize the MarkerFinder.
        
        Args:
            max_marker_length (int): Maximum marker length to consider (in bits)
        """
        self.max_marker_length = max_marker_length
        
    
    def find_marker(self, file_data, sample_size=None):
        """
        Find the shortest binary string that does not appear in the file data.
        
        Args:
            file_data (bytes): The binary file data
            sample_size (int, optional): Number of bytes to sample from the file.
                                       If None, use the entire file.
                                       
        Returns:
            bytes: The marker as bytes
            int: Length of the marker in bits
        """
        start_time = time.time()
        
        # If file is large, sample it
        if sample_size and len(file_data) > sample_size:
            # Sample evenly throughout the file
            step = len(file_data) // sample_size
            sampled_data = bytearray()
            sample_points = []
            
            for i in range(0, len(file_data), step):
                sampled_data.extend(file_data[i:i+1])
                sample_points.append(i)
            
            file_data = bytes(sampled_data[:sample_size])
            
            print(f"Finding marker using {len(file_data)} bytes sampled from a {len(file_data)} byte file")
            print(f"Sampling every {step} bytes at {len(sample_points)} points")
        else:
            print(f"Finding marker in entire file ({len(file_data)} bytes)")
        
        # Convert to bitarray for bit-level operations
        bits = bitarray()
        bits.frombytes(file_data)
        
        # Start with the smallest possible length
        marker_length = 1
        while marker_length <= self.max_marker_length:
            print(f"Checking markers of length {marker_length} bits ({2**marker_length} possibilities)")
            
            check_start_time = time.time()
            
            # For optimization, we'll use a sliding window approach
            # This is much faster than generating all possible markers
            possible_markers = 2**marker_length
            found = np.zeros(possible_markers, dtype=bool)
            
            # Check all bit windows of marker_length in the file
            for i in range(len(bits) - marker_length + 1):
                # Extract the window and convert to an integer
                window = bits[i:i+marker_length]
                # Important: convert the bit pattern to integer correctly
                window_str = window.to01()
                value = int(window_str, 2)
                
                # Mark this pattern as found
                if value < possible_markers:
                    found[value] = True
            
            # Calculate what percentage of possible patterns were found
            patterns_found = np.sum(found)
            coverage_percent = (patterns_found / possible_markers) * 100
            
            check_time = time.time() - check_start_time
            print(f"  Found {patterns_found} of {possible_markers} patterns ({coverage_percent:.2f}%) in {check_time:.4f} seconds")
            
            # Check if any markers weren't found
            for i in range(possible_markers):
                if not found[i]:
                    # Convert the integer to a bit string
                    marker_str = bin(i)[2:].zfill(marker_length)
                    
                    # Create marker bits
                    marker_bits = bitarray(marker_str)
                    
                    # For small markers, ensure they're in the most significant bits
                    if marker_length <= 8:
                        # Put the marker bits at the start of the first byte
                        # First pad to 8 bits to ensure alignment
                        while len(marker_bits) < 8:
                            marker_bits.append(0)
                        marker_bytes = marker_bits.tobytes()
                    else:
                        # For longer markers, pad to byte boundary
                        padding = 8 - (marker_length % 8) if marker_length % 8 else 0
                        padded_bits = marker_bits + bitarray('0' * padding)
                        marker_bytes = padded_bits.tobytes()
                    
                    elapsed_time = time.time() - start_time
                    print(f"Found marker of length {marker_length} bits in {elapsed_time:.4f} seconds")
                    print(f"Marker binary: {marker_str}")
                    print(f"Marker hex: {marker_bytes.hex()}")
                    
                    return marker_bytes, marker_length
            
            # If all patterns of current length are found, try longer patterns
            marker_length += 1
        
        # If we reach here, we couldn't find a marker
        raise ValueError(f"Could not find a marker of length <= {self.max_marker_length} bits")
    def find_marker_naive(self, file_data):
        """
        A simpler but less efficient implementation of find_marker.
        Useful for testing and understanding the algorithm.
        
        Args:
            file_data (bytes): The binary file data
                                       
        Returns:
            bytes: The marker as bytes
            int: Length of the marker in bits
        """
        # Convert to bitarray for bit-level operations
        bits = bitarray()
        bits.frombytes(file_data)
        bits_str = bits.to01()
        
        # Start with the smallest possible length
        marker_length = 1
        while marker_length <= self.max_marker_length:
            # Generate all possible binary strings of current length
            for i in range(2**marker_length):
                # Convert to binary string representation
                marker = bin(i)[2:].zfill(marker_length)
                
                # Check if marker appears in the file
                if marker not in bits_str:
                    # Convert to bitarray and then to bytes
                    marker_bits = bitarray(marker)
                    
                    # Pad to byte boundary for storage
                    padding = 8 - (len(marker_bits) % 8) if len(marker_bits) % 8 else 0
                    marker_bits = bitarray('0' * padding) + marker_bits
                    
                    return marker_bits.tobytes(), marker_length
            
            # If all patterns of current length are found, try longer patterns
            marker_length += 1
        
        # If we reach here, we couldn't find a marker
        raise ValueError(f"Could not find a marker of length <= {self.max_marker_length} bits")


if __name__ == "__main__":
    # Simple test of the MarkerFinder
    def test_marker_finder():
        # Create sample data with some patterns
        test_data = b"This is a test string. It contains various patterns."
        
        # Create the marker finder
        finder = MarkerFinder()
        
        # Find a marker
        marker, length = finder.find_marker(test_data)
        
        print(f"Found marker of length {length} bits: {marker.hex()}")
        
        # Verify the marker doesn't appear in the data
        bits = bitarray()
        bits.frombytes(test_data)
        bits_str = bits.to01()
        
        marker_bits = bitarray()
        marker_bits.frombytes(marker)
        marker_str = marker_bits.to01()[-length:]  # Only the last 'length' bits are the actual marker
        
        if marker_str in bits_str:
            print("ERROR: Marker appears in the data!")
        else:
            print("Verified: Marker does not appear in the data")
    
    # Run the test
    test_marker_finder()
