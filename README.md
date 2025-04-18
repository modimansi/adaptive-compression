
# Adaptive Compression Tool

An adaptive marker-based compression algorithm that dynamically selects optimal compression methods for different data patterns within files. This tool analyzes your data in chunks and applies the most efficient compression method to each chunk, resulting in better overall compression ratios.

## Features

- **Adaptive Compression**: Automatically selects the best compression method for different parts of a file
- **Multiple Compression Methods**: Leverages various algorithms including zlib, lzma, bz2, zstandard, lz4, and brotli
- **Marker-Based Approach**: Uses pattern recognition to identify optimal compression transitions
- **Graphical User Interface**: Simple Gradio-based interface for easy file compression/decompression
- **Command-Line Interface**: For batch processing and automated workflows
- **Compression Analysis**: Tools to analyze and visualize compression efficiency

## Installation

### Prerequisites

- Python 3.7 or higher
- pip package manager

### Basic Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/KalharPandya/adaptive-compression.git
   cd adaptive-compression
   ```

2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

### Optional Enhanced Installation

For better compression performance, install additional compression libraries:

```bash
pip install zstandard>=0.15.0 lz4>=3.0.0 Brotli>=1.0.9
```

Note: The LZHAM library (`pylzham`) requires manual installation if needed, as it's not available through pip by default.

## Running the Demo

### Graphical User Interface (GUI)

The simplest way to try the demo is through the GUI:

1. Launch the application:
   ```bash
   python main.py
   ```
   Or explicitly specify GUI mode:
   ```bash
   python main.py gui
   ```

2. If Gradio is not installed, you can add the `--install-gradio` flag:
   ```bash
   python main.py gui --install-gradio
   ```

3. The GUI will open in your browser with the following options:
   - **Compress**: Upload a file, configure chunk size, and compress
   - **Decompress**: Upload a compressed .ambc file and recover the original
   - **Analysis**: View compression statistics and performance metrics

### Command-Line Interface

For more advanced usage or batch processing:

#### Compression

```bash
python main.py compress <input_file> <output_file.ambc> [options]
```

Options:
- `--chunk-size`: Size of data chunks in bytes (default: 4096)
- `--methods`: Comma-separated list of compression methods to use
- `--disable-methods`: Comma-separated list of compression methods to disable
- `--show-progress`: Display a progress bar during compression

Example:
```bash
python main.py compress large_dataset.csv compressed_data.ambc --chunk-size 8192 --show-progress
```

#### Decompression

```bash
python main.py decompress <input_file.ambc> <output_file> [options]
```

Options:
- `--show-progress`: Display a progress bar during decompression

Example:
```bash
python main.py decompress compressed_data.ambc original_data.csv --show-progress
```

#### Compression Analysis

```bash
python main.py analyze [options]
```

Options:
- `--results-file`: Path to compression history file (default: compression_results/compression_history.json)
- `--output-dir`: Directory to save analysis charts and reports (default: analysis_output)

Example:
```bash
python main.py analyze --results-file my_results.json --output-dir my_analysis
```

## Customizing Chunk Size

The chunk size is a critical parameter that affects compression performance:

- **Smaller chunks** (e.g., 1024 bytes): More granular method selection but higher overhead
- **Larger chunks** (e.g., 16384 bytes): Less overhead but potentially lower compression efficiency
- **Default** (4096 bytes): Good balance for most use cases

You can adjust the chunk size in the GUI using the slider or in the CLI using the `--chunk-size` parameter.

## File Format

Files compressed with this tool use the `.ambc` extension (Adaptive Marker-Based Compression) and contain:

- File header with metadata
- Compression method table
- Compressed data chunks with method identifiers

## Troubleshooting

### Import Errors

If you encounter import errors related to the Gradio interface:

1. Ensure Gradio is installed and up to date:
   ```bash
   pip install --upgrade gradio>=3.0.0
   ```

2. Fix import cycles by making sure the Python path is correctly set:
   ```bash
   export PYTHONPATH=$PYTHONPATH:/path/to/adaptive-compression
   ```

3. If you see errors about missing modules like `Unable to import gradio_components`, check your Python path:
   ```bash
   python -c "import sys; print(sys.path)"
   ```

### Compression Method Errors

If you encounter errors related to compression methods:

1. Check which compression libraries are available:
   ```bash
   python -c "from compression_fix import check_compression_libraries; print(check_compression_libraries())"
   ```

2. Install any missing compression libraries as recommended above.

3. If an error occurs specifically with one compression method, you can still use the tool with the other methods.

### GUI Not Displaying

If the GUI fails to display:

1. Check if Gradio is properly installed:
   ```bash
   pip install gradio>=3.0.0
   ```

2. Make sure you have a browser available for the GUI to open in.

3. If you're in a headless environment, try using the command-line interface instead.

## Example Use Cases

1. **Large Dataset Compression**: Efficiently compress large CSV or JSON datasets with mixed content types
2. **Log File Archiving**: Compress log files that contain both structured and unstructured data
3. **Mixed Media Storage**: Optimize storage of files that contain both text and binary data
4. **Data Pipeline Integration**: Add to ETL processes to reduce data transfer sizes

## Performance Expectations

The adaptive compression typically achieves:
- 5-15% better compression ratios compared to using a single method
- Slightly longer compression time due to analysis overhead
- Similar decompression speed to standard methods

## Known Issues

For a list of known issues and limitations, please see the [KNOWN_ISSUES.md](KNOWN_ISSUES.md) file.

## Development and Contribution

For information on adding new compression methods or customizing the tool, please see the [SETUP.md](SETUP.md) file.
