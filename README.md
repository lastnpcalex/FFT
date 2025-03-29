# FFT Image Analysis Tool

A Python tool for analyzing and comparing images in the frequency domain using Fast Fourier Transform (FFT). This tool can analyze individual images, folders of images, and compare two sets of images, generating various visualizations of frequency domain features.

## Features

- **Single Image Analysis**: Compute FFT and visualize frequency domain characteristics
- **Batch Processing**: Analyze entire folders of images with automatic averaging
- **Comparison Analysis**: Compare frequency domain features between two image sets
- **Multiple Visualization Methods**:
  - Histogram of frequency magnitudes
  - Radial energy profiles (frequency vs. distance from center)
  - Angular energy distribution (frequency vs. angle)
  - High-energy density maps
  - Difference maps for comparative analysis

## Installation

1. Clone the repository:
   ```
   git clone https://github.com/lastnpcalex/fft.git
   cd fft
   ```

2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

## Usage

Run the script with Python:

```
python fft_analysis.py
```

The tool provides an interactive menu with two main modes:

### Mode 1: Single Folder Analysis

Analyzes all images in a folder, generating individual plots for each image and average plots for the entire folder.

1. Select option `1` at the menu prompt
2. Enter a target size for resizing all images (optional)
3. Provide the input folder path containing images
4. Provide the output folder path to save all generated plots

### Mode 2: Two Folder Comparison

Compares images between two folders by computing the average FFT for each folder and generating difference visualizations.

1. Select option `2` at the menu prompt
2. Enter a target size for resizing all images (optional)
3. Provide the first folder path
4. Provide the second folder path
5. Provide the output folder path to save comparison plots

### Image Resizing

For all modes, you can optionally specify a target size to resize all images before analysis. This ensures consistent dimensions for comparison, especially when comparing images of different sizes.

## Output Visualizations

The tool generates the following types of plots:

1. **Histogram**: Distribution of FFT log-magnitude values
2. **Radial Energy Profile**: Average log-magnitude vs. distance from the center
3. **Angular Energy Distribution**: Average log-magnitude vs. angle
4. **High-Energy Density Map**: Scatter plot of points with high energy (top 10% by default)
5. **Difference Maps** (for comparison mode): Visual and numerical differences between two sets of images

## Examples

### Single Folder Analysis
```
Select Mode:
[1] Single Folder Analysis (per-image + average plots)
[2] Two Folder Comparison (only average plots & differences)
Enter mode (1/2): 1
Enter target size for images as 'width,height' (or leave blank to use original sizes): 512,512
Enter the folder path containing images to analyze: ./dataset/portraits
Enter the output folder to save plots: ./results/portrait_analysis
```

### Two Folder Comparison
```
Select Mode:
[1] Single Folder Analysis (per-image + average plots)
[2] Two Folder Comparison (only average plots & differences)
Enter mode (1/2): 2
Enter target size for images as 'width,height' (or leave blank to use original sizes): 512,512
Enter the first folder path: ./dataset/real_photos
Enter the second folder path: ./dataset/ai_generated
Enter the output folder to save comparison plots: ./results/real_vs_ai_comparison
```

## Technical Details

### FFT Computation

The tool computes the 2D Fast Fourier Transform of grayscale images, shifts the zero-frequency component to the center, and visualizes the log-magnitude spectrum. The log scale (in decibels) is used for better visualization of the wide dynamic range of frequency components.

### Radial Profile

The radial profile shows how frequency magnitudes are distributed as a function of distance from the center (DC component). This reveals the general frequency characteristics of the image, such as the relative strength of high vs. low frequencies.

### Angular Profile

The angular profile shows how frequency magnitudes are distributed as a function of angle around the center. This can reveal directional biases in the image, such as predominant orientations or patterns.

## Applications

- Analyzing image texture characteristics
- Detecting patterns and periodicities in images
- Comparing different image sources or types
- Identifying distinctive frequency signatures
- Forensic analysis of potential image manipulations

## Requirements

- Python 3.6+
- NumPy
- Matplotlib
- Pillow (PIL)
- tqdm

## License

[MIT License](LICENSE)

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
