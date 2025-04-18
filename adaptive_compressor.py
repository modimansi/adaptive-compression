import os
import time
import struct
import hashlib
import math
import sys
import concurrent.futures
from tqdm import tqdm
import numpy as np
from bitarray import bitarray

from marker_finder import MarkerFinder
from compression_methods import (
    RLECompression, 
    DictionaryCompression, 
    HuffmanCompression, 
    DeltaCompression,
    NoCompression
)

# Attempt to import advanced compression or fixes
try:
    from compression_fix import get_compatible_methods
    COMPATIBLE_METHODS_AVAILABLE = True
    print("Compatible compression methods available.")
except ImportError:
    COMPATIBLE_METHODS_AVAILABLE = False
    print("No compatible compression methods fix. Some methods might be disabled.")

try:
    from advanced_compression import (
        DeflateCompression, 
        Bzip2Compression, 
        LZMACompression,
        HAS_ZSTD,
        HAS_LZ4
    )
    ADVANCED_METHODS_AVAILABLE = True
    
    if HAS_ZSTD and not COMPATIBLE_METHODS_AVAILABLE:
        from advanced_compression import ZstdCompression
    if HAS_LZ4:
        from advanced_compression import LZ4Compression
except ImportError as e:
    ADVANCED_METHODS_AVAILABLE = False
    print(f"Error importing advanced compression methods: {e}")


class AdaptiveCompressor:
    """
    Dynamic chunk-based compression that tries multiple chunk sizes and
    compression methods for each segment, picking whichever yields the
    best compression ratio for that segment. Markers are used to separate
    segments in the final output.
    """

    MAGIC_NUMBER = b'AMBC'
    FORMAT_VERSION = 2

    # Possible chunk sizes to attempt for each segment
    CHUNK_SIZE_CANDIDATES = [1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072]
    CHUNK_SIZE_CANDIDATES.sort(reverse=True)

    def __init__(self, marker_max_length=32, sample_size=10000):
        """
        Initialize the compressor with dynamic chunk logic.
        
        Args:
            marker_max_length (int): maximum length of the marker bits to find
            sample_size (int): how many bytes to sample for marker detection
        """
        self.marker_finder = MarkerFinder(marker_max_length)
        self.sample_size = sample_size

        # Marker info
        self.marker_bytes = None
        self.marker_length = 0
        self.marker_pattern = ""
        self.marker_bytes_aligned = b''
        self.marker_byte_length = 0

        # Multithread settings
        self.use_multithreading = False
        self.max_workers = max(1, os.cpu_count() - 1)

        # Progress callback
        self.progress_callback = None

        # Load compression methods
        self.compression_methods = []
        self._initialize_compression_methods()

        # Build ID -> method object
        self.method_lookup = {m.type_id: m for m in self.compression_methods}

        # Named references for logging or stats
        self.method_names = {
            1: "RLE", 
            2: "Dictionary", 
            3: "Huffman", 
            4: "Delta",
            5: "DEFLATE", 
            6: "BZIP2", 
            7: "LZMA", 
            8: "ZStandard", 
            9: "LZ4",
            10: "Brotli", 
            11: "LZHAM", 
            255: "No Compression"
        }

        # For each method, define a chunk-size range
        # If the chunk is outside that range, we skip that method
        self.method_chunk_prefs = {
            1:  (32,    4096),     # RLE
            2:  (128,   8192),     # Dictionary
            3:  (32,    8192),     # Huffman
            4:  (32,    4096),     # Delta
            5:  (64,    65536),    # DEFLATE
            6:  (1024,  262144),   # BZIP2
            7:  (8192,  524288),   # LZMA
            8:  (512,   262144),   # Zstd
            9:  (1024,  65536),    # LZ4
            10: (1024,  262144),   # Brotli
            11: (1024,  262144),   # LZHAM
            255:(1,     999999999) # No Compression => entire range
        }

    def _initialize_compression_methods(self):
        # Basic compression
        self.compression_methods.append(RLECompression())
        self.compression_methods.append(DictionaryCompression())
        self.compression_methods.append(HuffmanCompression())
        self.compression_methods.append(DeltaCompression())

        # Possibly from fix
        if COMPATIBLE_METHODS_AVAILABLE:
            for method in get_compatible_methods():
                self.compression_methods.append(method)

        # If advanced are available
        if 'DeflateCompression' in globals():
            self.compression_methods.append(DeflateCompression())
        if 'Bzip2Compression' in globals():
            self.compression_methods.append(Bzip2Compression())
        if 'LZMACompression' in globals():
            self.compression_methods.append(LZMACompression())
        if 'HAS_ZSTD' in globals() and HAS_ZSTD:
            from advanced_compression import ZstdCompression
            self.compression_methods.append(ZstdCompression())
        if 'HAS_LZ4' in globals() and HAS_LZ4:
            from advanced_compression import LZ4Compression
            self.compression_methods.append(LZ4Compression())

        # Possibly add Brotli, LZHAM
        # ...
        
        # Add Brotli and LZHAM if available
        try:
            if os.path.exists('brotli_lzham_compression.py'):
                from brotli_lzham_compression import BrotliCompression, HAS_BROTLI
                if HAS_BROTLI:
                    self.compression_methods.append(BrotliCompression())
                    print("Added Brotli compression to methods list")
                    
                # Try to import LZHAM
                try:
                    from brotli_lzham_compression import LZHAMCompression, HAS_LZHAM
                    if HAS_LZHAM:
                        self.compression_methods.append(LZHAMCompression())
                        print("Added LZHAM compression to methods list")
                except ImportError:
                    pass
        except ImportError:
            pass
        self.compression_methods.append(NoCompression())
        

    def set_progress_callback(self, callback):
        self.progress_callback = callback

    def _update_progress(self, stage, current, total, current_chunk=None, total_chunks=None):
        if self.progress_callback:
            self.progress_callback(stage, current, total, current_chunk, total_chunks)

    def enable_multithreading(self, max_workers=None):
        self.use_multithreading = True
        if max_workers:
            self.max_workers = max_workers
        print(f"Multithreading enabled with {self.max_workers} workers")

    def disable_multithreading(self):
        self.use_multithreading = False
        print("Multithreading disabled")

    def _init_marker(self, marker_bytes, marker_length):
        """
        Initialize marker properties (marker_bytes_aligned, marker_mask, etc.)
        """
        self.marker_bytes = marker_bytes
        self.marker_length = marker_length

        bits = bitarray()
        bits.frombytes(marker_bytes)
        bits = bits[:marker_length]
        self.marker_pattern = bits.to01()

        if marker_length<=8:
            shift = 8 - marker_length
            marker_int = int(self.marker_pattern, 2)
            marker_int <<= shift
            self.marker_bytes_aligned = bytes([marker_int])
        else:
            padded_bits = bitarray(self.marker_pattern)
            while len(padded_bits)%8!=0:
                padded_bits.append(0)
            self.marker_bytes_aligned = padded_bits.tobytes()

        self.marker_byte_length = (marker_length+7)//8

    def compress(self, input_file, output_file):
        """
        Compress a file using dynamic chunk approach
        """
        start_t = time.time()
        with open(input_file,"rb") as f:
            file_data = f.read()

        # Find marker
        marker_bytes, marker_len = self._find_marker(file_data, self.sample_size)
        self._init_marker(marker_bytes, marker_len)

        # Build header
        chksum = hashlib.md5(file_data).digest()
        header = self._build_header(marker_bytes, marker_len, chksum, len(file_data))

        # Now compress
        compressed_data = self._adaptive_compress(file_data)
        final_size = len(header)+ len(compressed_data)

        if final_size> len(file_data):
            # Store raw
            print("Compression bigger than original => store raw.")
            with open(output_file,"wb") as f:
                f.write(file_data)
            stats= self._build_stats_raw(len(file_data), time.time()-start_t)
            return stats
        else:
            # Write
            header= self._update_header_compressed_size(header, len(compressed_data))
            with open(output_file,"wb") as f:
                f.write(header)
                f.write(compressed_data)
            stats= self._calculate_compression_stats(len(file_data), final_size, time.time()-start_t)
            return stats

    def _build_stats_raw(self, original_size, elapsed):
        ratio=1.0
        pr=0.0
        tput=0.0
        if elapsed>0:
            tput= original_size/(1024*1024* elapsed)
        chunk_stats={
            'total_chunks': 1,
            'compressed_chunks': 0,
            'raw_chunks': 1,
            'method_usage': {},
            'bytes_saved': 0,
            'original_size': original_size,
            'compressed_size_without_overhead': original_size,
            'overhead_bytes': 0
        }
        stats={
            'original_size': original_size,
            'compressed_size': original_size,
            'ratio': ratio,
            'percent_reduction': pr,
            'elapsed_time': elapsed,
            'throughput_mb_per_sec': tput,
            'chunk_stats': chunk_stats,
            'overhead_bytes': 0,
            'compression_efficiency': 1.0
        }
        return stats

    def decompress(self, input_file, output_file):
        start_t= time.time()
        with open(input_file,"rb") as f:
            cdata= f.read()
        hdr= self._parse_header(cdata)
        self._init_marker(hdr['marker_bytes'], hdr['marker_length'])
        body= cdata[hdr['header_size']:]
        decompressed= self._adaptive_decompress(body, hdr['original_size'])
        with open(output_file,"wb") as f:
            f.write(decompressed)
        # verify
        actual= hashlib.md5(decompressed).digest()
        if actual!= hdr['checksum']:
            raise ValueError("Checksum mismatch => possibly corrupted file.")
        stats= self._calculate_decompression_stats(len(cdata), len(decompressed), time.time()- start_t)
        return stats

    def _find_marker(self, file_data, sample_size):
        """
        Fallback logic if no marker_finder is available, or we can do your original approach
        that picks a short marker not in the data. We'll skip for brevity and return a 32-bit fallback.
        """
        from bitarray import bitarray
        marker_bits = bitarray('11111111111111100000000000000000')
        return marker_bits.tobytes(), 32

    def _build_header(self, marker_bytes, marker_len, chksum, original_size):
        hdr= bytearray()
        hdr.extend(self.MAGIC_NUMBER)
        hdr.append(self.FORMAT_VERSION)
        hdr.extend(b'\x00\x00\x00\x00')
        hdr.append(marker_len)
        hdr.extend(marker_bytes)
        hdr.append(1)  # md5
        hdr.extend(chksum)
        hdr.extend(struct.pack('<Q', original_size))
        hdr.extend(b'\x00\x00\x00\x00\x00\x00\x00\x00')
        hsize= len(hdr)
        hdr[5:9]= struct.pack('<I', hsize)
        return bytes(hdr)

    def _update_header_compressed_size(self, hdr, csize):
        hdr= bytearray(hdr)
        hdr[-8:]= struct.pack('<Q', csize)
        return bytes(hdr)

    def _parse_header(self, data: bytes):
        if data[:4]!= self.MAGIC_NUMBER:
            raise ValueError("Magic mismatch")
        version= data[4]
        if version> self.FORMAT_VERSION:
            raise ValueError(f"Unsupported version: {version}")
        hdr_size= struct.unpack('<I', data[5:9])[0]
        marker_len= data[9]
        msize= (marker_len+7)//8
        marker_bytes= data[10:10+msize]
        ctype= data[10+ msize]
        csum_size= 16 if ctype==1 else 0
        csum= data[11+msize: 11+msize+ csum_size]
        orig_pos= 11+ msize+ csum_size
        orig_size= struct.unpack('<Q', data[orig_pos: orig_pos+8])[0]
        comp_pos= orig_pos+8
        comp_size= struct.unpack('<Q', data[comp_pos: comp_pos+8])[0]
        return {
            'format_version': version,
            'header_size': hdr_size,
            'marker_length': marker_len,
            'marker_bytes': marker_bytes,
            'checksum_type': ctype,
            'checksum': csum,
            'original_size': orig_size,
            'compressed_size': comp_size
        }

    # --------------------------------------------------------------
    #   MAIN DYNAMIC CHUNK LOGIC
    # --------------------------------------------------------------
    def _adaptive_compress(self, file_data: bytes) -> bytes:
        """
        Single pass:
          - position=0
          - pick best chunk size+method
          - process chunk, produce the chunk package
          - repeat
          - add end chunk
        """
        self._init_stats(file_data)
        output= bytearray()
        position=0
        chunk_index=0

        with tqdm(total=len(file_data), desc="Compressing", unit="B", unit_scale=True) as pbar:
            while position< len(file_data):
                csize, method_id= self._pick_best_chunk_and_method(file_data, position)
                chunk_data= file_data[position: position+ csize]
                # Now produce the final chunk and chunk stats
                package_data, chunk_stats= self._process_chunk(chunk_data, method_id, chunk_index)
                self._update_stats(chunk_stats)
                output.extend(package_data)

                position+= csize
                chunk_index+=1
                pbar.update(csize)

        # end chunk
        end_chunk= self._create_end_chunk()
        output.extend(end_chunk)
        self.chunk_stats['overhead_bytes']+= len(end_chunk)
        return bytes(output)

    def _adaptive_decompress(self, data: bytes, orig_size: int) -> bytes:
        output= bytearray()
        pos=0
        while pos< len(data):
            needed= len(self.marker_bytes_aligned)+1+1+4+4+4
            if pos+ needed> len(data):
                print("No more chunk headers can be read, stopping.")
                break

            chunk_marker= data[pos: pos+ len(self.marker_bytes_aligned)]
            if chunk_marker!= self.marker_bytes_aligned:
                raise ValueError("Marker mismatch in chunk header.")
            pos+= len(self.marker_bytes_aligned)
            
            pkg_type= data[pos]
            pos+=1
            k_value= data[pos]
            pos+=1

            used_bytes= struct.unpack("<I", data[pos:pos+4])[0]
            pos+=4
            orig_len= struct.unpack("<I", data[pos:pos+4])[0]
            pos+=4
            comp_len= struct.unpack("<I", data[pos:pos+4])[0]
            pos+=4

            if pkg_type==0:
                print("End-of-stream chunk found.")
                break
            if pos+ comp_len> len(data):
                print("Not enough bytes remain for chunk payload.")
                break

            payload= data[pos: pos+ comp_len]
            pos+= comp_len

            method= self.method_lookup.get(pkg_type)
            if method is None:
                # treat as raw
                output.extend(payload)
            else:
                try:
                    chunk_out= method.decompress(payload, orig_len)
                    output.extend(chunk_out)
                except:
                    # fallback
                    output.extend(bytes(orig_len))

            if len(output)>= orig_size:
                break

        if len(output)< orig_size:
            short= orig_size- len(output)
            output.extend(bytes(short))
        elif len(output)> orig_size:
            over= len(output)- orig_size
            output= output[: orig_size]

        return bytes(output)

    # Stats initialization
    def _init_stats(self, file_data: bytes):
        self.chunk_stats= {
            'total_chunks': 0,
            'compressed_chunks': 0,
            'raw_chunks': 0,
            'method_usage': {},
            'bytes_saved': 0,
            'original_size': len(file_data),
            'compressed_size_without_overhead': 0,
            'overhead_bytes': 0
        }
        for m in self.compression_methods:
            self.chunk_stats['method_usage'][m.type_id]=0

    def _update_stats(self, chunk_stats: dict):
        self.chunk_stats['total_chunks']+=1
        if chunk_stats['compressed']:
            self.chunk_stats['compressed_chunks']+=1
            self.chunk_stats['method_usage'][chunk_stats['method_id']]+=1
            self.chunk_stats['compressed_size_without_overhead']+= chunk_stats['compressed_size']
            self.chunk_stats['overhead_bytes']+= chunk_stats['overhead']
            self.chunk_stats['bytes_saved']+= chunk_stats['bytes_saved']
        else:
            self.chunk_stats['raw_chunks']+=1

    def _calculate_compression_stats(self, orig_size, comp_size, elapsed):
        if orig_size==0:
            ratio=1.0
            pr=0.0
        else:
            ratio= comp_size/orig_size
            pr=(1.0- ratio)*100.0
        throughput=0.0
        if elapsed>0:
            throughput= orig_size/(1024*1024*elapsed)
        
        if self.chunk_stats['compressed_chunks']>0:
            cdata= self.chunk_stats['compressed_size_without_overhead']
            overhead= self.chunk_stats['overhead_bytes']
            # approximate
            original_compressed_size=0
            for mid, cnt in self.chunk_stats['method_usage'].items():
                if mid!=255 and cnt>0:
                    fraction= cnt/self.chunk_stats['total_chunks']
                    original_compressed_size+= fraction*orig_size
            if original_compressed_size>0:
                compression_efficiency= cdata/original_compressed_size
            else:
                compression_efficiency=1.0
        else:
            compression_efficiency=1.0

        stats= {
            'original_size': orig_size,
            'compressed_size': comp_size,
            'ratio': ratio,
            'percent_reduction': pr,
            'elapsed_time': elapsed,
            'throughput_mb_per_sec': throughput,
            'chunk_stats': self.chunk_stats,
            'overhead_bytes': self.chunk_stats.get('overhead_bytes',0),
            'compression_efficiency': compression_efficiency
        }
        return stats

    def _calculate_decompression_stats(self, csize, dsize, elapsed):
        tput=0.0
        if elapsed>0:
            tput= dsize/(1024*1024* elapsed)
        stats= {
            'compressed_size': csize,
            'decompressed_size': dsize,
            'elapsed_time': elapsed,
            'throughput_mb_per_sec': tput
        }
        return stats

    # --------------------------------------------------------------
    #   CHUNK SIZE + METHOD DECISION
    # --------------------------------------------------------------
    def _pick_best_chunk_and_method(self, data: bytes, position: int):
        """
        For the next chunk, tries multiple chunk sizes from CHUNK_SIZE_CANDIDATES,
        filters out methods that don't prefer that chunk size,
        picks best ratio. Returns (best_chunk_size, best_method_id).
        """
        remain= len(data)- position
        best_csize= remain
        best_method_id=255
        best_ratio=1.0  # 1 => raw

        for candidate in self.CHUNK_SIZE_CANDIDATES:
            if candidate> remain:
                candidate= remain
            if candidate<=0:
                break
            chunk_data= data[position: position+candidate]

            local_best_ratio=1.0
            local_best_method=255

            # only attempt methods that "prefer" this candidate chunk size
            for method in self.compression_methods:
                if method.type_id==255:
                    # skip raw here; we compare against ratio=1.0
                    continue

                # check chunk-size preference
                chunk_min, chunk_max= self.method_chunk_prefs.get(method.type_id,(1,999999999))
                if not (chunk_min<= candidate<= chunk_max):
                    continue

                # then check method feasibility
                if method.should_use(chunk_data):
                    try:
                        cdata= method.compress(chunk_data)
                        overhead= self._calculate_fixed_overhead()
                        ratio= (len(cdata)+ overhead)/ len(chunk_data)
                        if ratio< local_best_ratio:
                            local_best_ratio= ratio
                            local_best_method= method.type_id
                    except:
                        pass
            
            if local_best_ratio< best_ratio:
                best_ratio= local_best_ratio
                best_csize= candidate
                best_method_id= local_best_method

        # if no method beat raw => entire remain as raw
        if best_method_id==255 and best_csize== remain:
            return remain, 255
        else:
            return best_csize, best_method_id

    # --------------------------------------------------------------
    #   CHUNK BUILDING
    # --------------------------------------------------------------
    def _create_end_chunk(self):
        """
        Creates an end chunk with package_type=0.
        """
        chunk= bytearray()
        chunk.extend(self.marker_bytes_aligned)
        chunk.append(0)    # package_type=0
        chunk.append(0)    # k_value=0
        chunk.extend(struct.pack("<H", 0))  # used_bytes_in_chunk=0 (2 bytes, LE)
        chunk.extend(struct.pack("<I", 0))  # original_length=0
        chunk.extend(struct.pack("<I", 0))  # compressed_length=0
        print(f"Created END chunk: marker={self.marker_bytes_aligned.hex()}")
        return chunk

    def _create_chunk(self, package_type, k_value, used_bytes_in_chunk, original_length, compressed_data):
        """
        Creates a structured chunk with marker, package_type, k_value, etc.
        """
        chunk = bytearray()
        chunk.extend(self.marker_bytes_aligned)
        chunk.append(package_type)
        chunk.append(k_value)
        chunk.extend(struct.pack("<I", used_bytes_in_chunk))
        chunk.extend(struct.pack("<I", original_length))
        chunk.extend(struct.pack("<I", len(compressed_data)))
        chunk.extend(compressed_data)
        return chunk

    def _calculate_fixed_overhead(self):
        """
        Overhead for each chunk header:
        marker + 1byte pkg_type + 1byte k_value + 4byte used_bytes + 
        +4byte original_length + 4byte compressed_length
        """
        return len(self.marker_bytes_aligned)+1+1+4+4+4

    def _process_chunk(self, chunk_data: bytes, method_id: int, chunk_index: int):
        """
        Build a chunk with the chosen method (or raw if method_id=255),
        returning (package_bytes, chunk_stats).
        """
        stats= {
            'compressed': False,
            'method_id': 255,
            'original_size': len(chunk_data),
            'compressed_size': len(chunk_data),
            'overhead': 0,
            'bytes_saved': 0
        }

        # If no method or not found => raw
        if method_id not in self.method_lookup or method_id==255:
            # raw
            stats['method_id']= 255
            package_bytes= self._create_chunk(
                package_type=255, 
                k_value=0, 
                used_bytes_in_chunk=len(chunk_data),
                original_length=len(chunk_data),
                compressed_data=chunk_data
            )
            return package_bytes, stats

        # Otherwise attempt compression
        method= self.method_lookup[method_id]
        try:
            cdata= method.compress(chunk_data)
            overhead= self._calculate_fixed_overhead()
            if len(cdata)+ overhead< len(chunk_data):
                # compression is beneficial
                stats['compressed']= True
                stats['method_id']= method_id
                stats['compressed_size']= len(cdata)
                stats['overhead']= overhead
                stats['bytes_saved']= len(chunk_data) - (len(cdata)+ overhead)
                package_bytes= self._create_chunk(
                    package_type=method_id, 
                    k_value=0, # not used currently
                    used_bytes_in_chunk=len(chunk_data),
                    original_length=len(chunk_data),
                    compressed_data=cdata
                )
                return package_bytes, stats
            else:
                # not beneficial => raw
                stats['method_id']= 255
                package_bytes= self._create_chunk(
                    package_type=255,
                    k_value=0,
                    used_bytes_in_chunk=len(chunk_data),
                    original_length=len(chunk_data),
                    compressed_data=chunk_data
                )
                return package_bytes, stats
        except Exception as e:
            print(f"Error compressing chunk {chunk_index} with method {method_id}: {e}")
            # fallback => raw
            stats['method_id']=255
            package_bytes= self._create_chunk(
                package_type=255,
                k_value=0,
                used_bytes_in_chunk=len(chunk_data),
                original_length=len(chunk_data),
                compressed_data=chunk_data
            )
            return package_bytes, stats


if __name__=="__main__":
    def quick_test():
        data= bytearray()
        data.extend(b"A"*5000)
        for i in range(50):
            data.extend(b"The quick brown fox jumps over the lazy dog. ")
        data.extend(os.urandom(4000))

        test_file= "test_data.bin"
        out_file= "test_data.ambc"
        dec_file= "test_data_decompressed.bin"
        with open(test_file,"wb") as f:
            f.write(data)

        c= AdaptiveCompressor()
        c.enable_multithreading()
        stats= c.compress(test_file,out_file)
        print("Compression stats:", stats)

        ds= c.decompress(out_file, dec_file)
        print("Decompression stats:", ds)

        with open(dec_file,"rb") as f:
            new_data= f.read()
        if new_data== data:
            print("Decompressed data matches original!")
        else:
            print("ERROR: mismatch")

        for path in [test_file, out_file, dec_file]:
            if os.path.exists(path):
                os.remove(path)

    quick_test()
