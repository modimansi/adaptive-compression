import zlib
import bz2
import lzma
import numpy as np
import sys
import os

from compression_methods import CompressionMethod

# Try to import zstandard (Zstd)
try:
    import zstandard as zstd
    HAS_ZSTD = True
except ImportError:
    HAS_ZSTD = False
    print("zstd library not available. Zstandard compression will be disabled.")

# Try to import lz4.frame (LZ4)
try:
    import lz4.frame
    HAS_LZ4 = True
except ImportError:
    HAS_LZ4 = False
    print("lz4 library not available. LZ4 compression will be disabled.")

# Try to import Brotli and LZHAM from an external file (brotli_lzham_compression.py)
try:
    if os.path.exists("brotli_lzham_compression.py"):
        from brotli_lzham_compression import BrotliCompression, HAS_BROTLI
        from brotli_lzham_compression import LZHAMCompression, HAS_LZHAM
        if HAS_BROTLI:
            print("Added Brotli compression")
        if HAS_LZHAM:
            print("Added LZHAM compression")
    else:
        HAS_BROTLI = False
        HAS_LZHAM = False
        print("brotli_lzham_compression.py not found in current directory")
except ImportError as e:
    print(f"Error importing from brotli_lzham_compression: {e}")
    HAS_BROTLI = False
    HAS_LZHAM = False

# -------------------------------------------------------------------
# Utility functions for choosing whether to compress
# -------------------------------------------------------------------

def calculate_entropy(data: bytes) -> float:
    """
    Compute approximate Shannon entropy for 'data'.
    """
    if not data:
        return 0.0
    counts = np.bincount(bytearray(data), minlength=256)
    probs = counts / len(data)
    probs = probs[probs > 0]
    return -np.sum(probs * np.log2(probs))

def calculate_text_ratio(data: bytes) -> float:
    """
    Estimate fraction of bytes that are typical text characters.
    """
    if not data:
        return 0.0
    text_chars = sum(1 for b in data if 32 <= b <= 127 or b in (9, 10, 13))
    return text_chars / len(data)

# -------------------------------------------------------------------
# DEFLATE
# -------------------------------------------------------------------
class DeflateCompression(CompressionMethod):
    @property
    def type_id(self):
        return 5
    
    def compress(self, data, level=9):
        if not data:
            return b''
        compressed = zlib.compress(data, level=level)
        print(f"DEFLATE compression (lvl={level}): {len(data)} -> {len(compressed)} bytes")
        return compressed
    
    def decompress(self, data, original_length):
        if not data:
            return b''
        try:
            decompressed = zlib.decompress(data)
            if len(decompressed) > original_length:
                decompressed = decompressed[:original_length]
            elif len(decompressed) < original_length:
                decompressed += bytes(original_length - len(decompressed))
            print(f"DEFLATE decompression: {len(data)} -> {len(decompressed)} bytes")
            return decompressed
        except Exception as e:
            print(f"DEFLATE decompress error: {e}")
            return bytes(original_length)
    
    def should_use(self, data: bytes, threshold=0.9) -> bool:
        """
        We'll accept data if it's at least 64 bytes 
        and the entropy is not extremely high (≅8.0).
        """
        if len(data) < 64:
            return False
        if calculate_entropy(data) >= 8.0:
            return False
        return True

# -------------------------------------------------------------------
# BZIP2
# -------------------------------------------------------------------
class Bzip2Compression(CompressionMethod):
    @property
    def type_id(self):
        return 6
    
    def compress(self, data, level=9):
        if not data:
            return b''
        compressed = bz2.compress(data, compresslevel=level)
        print(f"BZIP2 compression (lvl={level}): {len(data)} -> {len(compressed)} bytes")
        return compressed
    
    def decompress(self, data, original_length):
        if not data:
            return b''
        try:
            decompressed = bz2.decompress(data)
            if len(decompressed) > original_length:
                decompressed = decompressed[:original_length]
            elif len(decompressed) < original_length:
                decompressed += bytes(original_length - len(decompressed))
            print(f"BZIP2 decompression: {len(data)} -> {len(decompressed)} bytes")
            return decompressed
        except Exception as e:
            print(f"BZIP2 decompress error: {e}")
            return bytes(original_length)
    
    def should_use(self, data: bytes, threshold=0.9) -> bool:
        """
        BZIP2 has more overhead, so better for bigger chunks (≅ 1 KB).
        We'll skip if extremely high entropy (≅7.7).
        """
        if len(data) < 1024:
            return False
        
        ent = calculate_entropy(data)
        if ent >= 7.7:
            return False
        return True

# -------------------------------------------------------------------
# LZMA
# -------------------------------------------------------------------
class LZMACompression(CompressionMethod):
    """
    LZMA with a custom filter chain. We do NOT specify preset simultaneously.
    """
    @property
    def type_id(self):
        return 7
    
    def compress(self, data):
        if not data:
            return b''
        try:
            # Custom filter without specifying preset
            filters = [
                {
                    "id": lzma.FILTER_LZMA2,
                    "dict_size": 1 << 24,  # 16 MB dictionary
                }
            ]
            compressor = lzma.LZMACompressor(
                format=lzma.FORMAT_XZ,
                check=lzma.CHECK_CRC64,
                filters=filters
            )
            compressed = compressor.compress(data)
            compressed += compressor.flush()
            print(f"LZMA compression (dict=16MB): {len(data)} -> {len(compressed)} bytes")
            return compressed
        except Exception as e:
            print(f"LZMA compress error: {e}")
            return data
    
    def decompress(self, data, original_length):
        if not data:
            return b''
        try:
            decompressed = lzma.decompress(data)
            if len(decompressed) > original_length:
                decompressed = decompressed[:original_length]
            elif len(decompressed) < original_length:
                decompressed += bytes(original_length - len(decompressed))
            print(f"LZMA decompression: {len(data)} -> {len(decompressed)} bytes")
            return decompressed
        except Exception as e:
            print(f"LZMA decompress error: {e}")
            return bytes(original_length)
    
    def should_use(self, data: bytes, threshold=0.9) -> bool:
        """
        LZMA is best for bigger chunks (≅ 8 KB) 
        and moderate/low entropy (<8).
        """
        if len(data) < 8192:
            return False
        
        ent = calculate_entropy(data)
        if ent >= 8.0:
            return False
        return True

# -------------------------------------------------------------------
# Zstandard
# -------------------------------------------------------------------
if HAS_ZSTD:
    class ZstdCompression(CompressionMethod):
        @property
        def type_id(self):
            return 8
        
        def compress(self, data):
            if not data:
                return b''
            try:
                compressor = zstd.ZstdCompressor(level=19)  # near max
                compressed = compressor.compress(data)
                print(f"ZStd compression (lvl=19): {len(data)} -> {len(compressed)} bytes")
                return compressed
            except Exception as e:
                print(f"Zstd compress error: {e}")
                return data
        
        def decompress(self, data, original_length):
            if not data:
                return b''
            try:
                decompressor = zstd.ZstdDecompressor()
                decompressed = decompressor.decompress(data, max_output_size=original_length)
                if len(decompressed) > original_length:
                    decompressed = decompressed[:original_length]
                elif len(decompressed) < original_length:
                    decompressed += bytes(original_length - len(decompressed))
                print(f"ZStd decompression: {len(data)} -> {len(decompressed)} bytes")
                return decompressed
            except Exception as e:
                print(f"Zstd decompress error: {e}")
                return bytes(original_length)
        
        def should_use(self, data: bytes, threshold=0.9) -> bool:
            """
            Zstd is fairly good on many data types,
            but skip if chunk < 512 bytes or entropy > 8.2
            """
            if len(data) < 512:
                return False
            if calculate_entropy(data) > 8.2:
                return False
            return True

# -------------------------------------------------------------------
# LZ4
# -------------------------------------------------------------------
if HAS_LZ4:
    class LZ4Compression(CompressionMethod):
        @property
        def type_id(self):
            return 9
        
        def compress(self, data):
            if not data:
                return b''
            try:
                compressed = lz4.frame.compress(data, compression_level=9)
                print(f"LZ4 compression (lvl=9): {len(data)} -> {len(compressed)} bytes")
                return compressed
            except Exception as e:
                print(f"LZ4 compress error: {e}")
                return data
        
        def decompress(self, data, original_length):
            if not data:
                return b''
            try:
                decompressed = lz4.frame.decompress(data)
                if len(decompressed) > original_length:
                    decompressed = decompressed[:original_length]
                elif len(decompressed) < original_length:
                    decompressed += bytes(original_length - len(decompressed))
                print(f"LZ4 decompression: {len(data)} -> {len(decompressed)} bytes")
                return decompressed
            except Exception as e:
                print(f"LZ4 decompress error: {e}")
                return bytes(original_length)
        
        def should_use(self, data: bytes, threshold=0.9) -> bool:
            """
            LZ4 is very fast, best for bigger chunks (≅1 KB),
            skip if extremely high entropy (>8.1)
            """
            if len(data) < 1024:
                return False
            if calculate_entropy(data) > 8.1:
                return False
            return True

# -------------------------------------------------------------------
# If you want to define Brotli, LZHAM, etc. from external modules
# you'd do something similar with their compress/decompress logic,
# ensuring not to specify contradictory parameters.
#
# That's all for advanced_compression.py
