import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from pathlib import Path
import os
from tqdm import tqdm  # for progress bar

# --- FFT Computation with Optional Resizing ---
def compute_fft(image_path, target_size=None):
    """
    Load image in grayscale, optionally resize to target_size (tuple: (width, height)),
    compute its FFT, and return the log-magnitude spectrum.
    """
    img = Image.open(image_path).convert("L")
    if target_size is not None:
        img = img.resize(target_size, Image.BICUBIC)
    img_array = np.array(img)
    f = np.fft.fft2(img_array)
    fshift = np.fft.fftshift(f)
    magnitude = np.abs(fshift)
    # Compute log magnitude (in decibels)
    log_magnitude = 20 * np.log10(magnitude + 1e-8)
    return log_magnitude

# --- Plot Functions (with save support) ---
def plot_histogram(log_magnitude, image_label, save_path=None, bins=50):
    """
    Plot and save histogram of FFT log-magnitude values.
    """
    plt.figure()
    plt.hist(log_magnitude.ravel(), bins=bins, color='blue', alpha=0.7)
    plt.title(f"Histogram of FFT Log-Magnitudes for {image_label}")
    plt.xlabel("Log Magnitude (dB)")
    plt.ylabel("Frequency")
    plt.grid(True)
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()

def plot_radial_energy(log_magnitude, image_label, save_path=None):
    """
    Compute and plot the radial energy profile (average log-magnitude vs radius).
    """
    rows, cols = log_magnitude.shape
    center = np.array([rows // 2, cols // 2])
    Y, X = np.indices((rows, cols))
    # Compute Euclidean distance from the center
    R = np.sqrt((X - center[1])**2 + (Y - center[0])**2)
    # Bin the log_magnitude values by integer radius
    R_int = R.astype(np.int32)
    radial_sum = np.bincount(R_int.ravel(), weights=log_magnitude.ravel())
    radial_count = np.bincount(R_int.ravel())
    radial_profile = radial_sum / (radial_count + 1e-8)
    radii = np.arange(len(radial_profile))
    
    plt.figure()
    plt.plot(radii, radial_profile, 'r-')
    plt.title(f"Radial Energy Profile for {image_label}")
    plt.xlabel("Radius (pixels)")
    plt.ylabel("Average Log Magnitude (dB)")
    plt.grid(True)
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()

def plot_angular_energy(log_magnitude, image_label, save_path=None, num_bins=36):
    """
    Compute and plot the angular energy distribution (average log-magnitude vs angle).
    """
    rows, cols = log_magnitude.shape
    center = np.array([rows // 2, cols // 2])
    Y, X = np.indices(log_magnitude.shape)
    # Calculate angle (in degrees) relative to center
    angles = np.arctan2(Y - center[0], X - center[1])
    angles_deg = np.degrees(angles)
    
    angles_flat = angles_deg.ravel()
    energy_flat = log_magnitude.ravel()
    
    # Bin the energy values by angle
    bins = np.linspace(-180, 180, num_bins+1)
    bin_indices = np.digitize(angles_flat, bins) - 1
    angular_sum = np.zeros(num_bins)
    angular_count = np.zeros(num_bins)
    for i in range(num_bins):
        mask = (bin_indices == i)
        angular_sum[i] = energy_flat[mask].sum()
        angular_count[i] = np.sum(mask)
    angular_profile = angular_sum / (angular_count + 1e-8)
    bin_centers = (bins[:-1] + bins[1:]) / 2
    
    plt.figure()
    plt.plot(bin_centers, angular_profile, 'g-')
    plt.title(f"Angular Energy Distribution for {image_label}")
    plt.xlabel("Angle (degrees)")
    plt.ylabel("Average Log Magnitude (dB)")
    plt.grid(True)
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()

def plot_high_energy_density(log_magnitude, image_label, save_path=None, quantile=0.9):
    """
    Plot a 2D density map (scatter) of high-energy points (above a given quantile).
    """
    threshold = np.quantile(log_magnitude, quantile)
    high_energy_mask = log_magnitude >= threshold
    indices = np.argwhere(high_energy_mask)
    
    plt.figure()
    plt.scatter(indices[:, 1], indices[:, 0], s=1, alpha=0.5)
    plt.title(f"High-Energy Density Map for {image_label}\n(Top {int((1-quantile)*100)}% threshold)")
    plt.xlabel("Frequency X")
    plt.ylabel("Frequency Y")
    plt.gca().invert_yaxis()  # Match FFT coordinate convention
    plt.grid(True)
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()

# --- Helper Functions to Compute Distributions ---
def compute_histogram(log_magnitude, bins=50):
    counts, bin_edges = np.histogram(log_magnitude.ravel(), bins=bins)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    return counts, bin_centers

def compute_radial_profile(log_magnitude):
    rows, cols = log_magnitude.shape
    center = np.array([rows // 2, cols // 2])
    Y, X = np.indices((rows, cols))
    R = np.sqrt((X - center[1])**2 + (Y - center[0])**2)
    R_int = R.astype(np.int32)
    radial_sum = np.bincount(R_int.ravel(), weights=log_magnitude.ravel())
    radial_count = np.bincount(R_int.ravel())
    radial_profile = radial_sum / (radial_count + 1e-8)
    radii = np.arange(len(radial_profile))
    return radii, radial_profile

def compute_angular_profile(log_magnitude, num_bins=36):
    rows, cols = log_magnitude.shape
    center = np.array([rows // 2, cols // 2])
    Y, X = np.indices(log_magnitude.shape)
    angles = np.arctan2(Y - center[0], X - center[1])
    angles_deg = np.degrees(angles)
    angles_flat = angles_deg.ravel()
    energy_flat = log_magnitude.ravel()
    bins = np.linspace(-180, 180, num_bins+1)
    bin_indices = np.digitize(angles_flat, bins) - 1
    angular_sum = np.zeros(num_bins)
    angular_count = np.zeros(num_bins)
    for i in range(num_bins):
        mask = (bin_indices == i)
        angular_sum[i] = energy_flat[mask].sum()
        angular_count[i] = np.sum(mask)
    angular_profile = angular_sum / (angular_count + 1e-8)
    bin_centers = (bins[:-1] + bins[1:]) / 2
    return bin_centers, angular_profile

# --- Folder Averaging Functions ---
def analyze_folder_and_save(input_folder, output_folder, target_size=None):
    """
    Analyze each image in input_folder, saving four plots per image,
    and then compute and save average plots over the folder.
    If target_size is provided (tuple: (width, height)), all images are resized to that size.
    """
    input_folder = Path(input_folder)
    output_folder = Path(output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)
    folder_name = input_folder.stem

    # Supported image types
    image_files = sorted(
        list(input_folder.glob("*.png")) +
        list(input_folder.glob("*.jpg")) +
        list(input_folder.glob("*.jpeg")) +
        list(input_folder.glob("*.webp"))
    )
    
    avg_log_magnitude = None
    count = 0

    for i, img_file in enumerate(tqdm(image_files, desc="Processing images"), start=1):
        image_label = f"{folder_name}_{i}"
        log_magnitude = compute_fft(img_file, target_size=target_size)
        
        # Save individual plots
        plot_histogram(log_magnitude, image_label,
                       save_path=str(output_folder / f"{folder_name}_histogram_{i}.png"))
        plot_radial_energy(log_magnitude, image_label,
                           save_path=str(output_folder / f"{folder_name}_radial_{i}.png"))
        plot_angular_energy(log_magnitude, image_label,
                            save_path=str(output_folder / f"{folder_name}_angular_{i}.png"))
        plot_high_energy_density(log_magnitude, image_label,
                                 save_path=str(output_folder / f"{folder_name}_density_{i}.png"))
        
        # Accumulate for averaging
        if avg_log_magnitude is None:
            avg_log_magnitude = log_magnitude.copy()
        else:
            avg_log_magnitude += log_magnitude
        count += 1

    if count > 0:
        avg_log_magnitude /= count
        # Save average plots
        plot_histogram(avg_log_magnitude, f"{folder_name}_average",
                       save_path=str(output_folder / f"{folder_name}_histogram_average.png"))
        plot_radial_energy(avg_log_magnitude, f"{folder_name}_average",
                           save_path=str(output_folder / f"{folder_name}_radial_average.png"))
        plot_angular_energy(avg_log_magnitude, f"{folder_name}_average",
                            save_path=str(output_folder / f"{folder_name}_angular_average.png"))
        plot_high_energy_density(avg_log_magnitude, f"{folder_name}_average",
                                 save_path=str(output_folder / f"{folder_name}_density_average.png"))
        print(f"Processed {len(image_files)} images; average computed from {count} images.")
    else:
        print("No images processed.")

def get_average_fft(folder, target_size=None):
    """
    Compute and return the average FFT log-magnitude spectrum for all images in a folder.
    """
    folder = Path(folder)
    image_files = sorted(
        list(folder.glob("*.png")) +
        list(folder.glob("*.jpg")) +
        list(folder.glob("*.jpeg")) +
        list(folder.glob("*.webp"))
    )
    avg_log_magnitude = None
    count = 0
    for img_file in tqdm(image_files, desc=f"Processing {folder.stem}"):
        log_magnitude = compute_fft(img_file, target_size=target_size)
        if avg_log_magnitude is None:
            avg_log_magnitude = log_magnitude.copy()
        else:
            avg_log_magnitude += log_magnitude
        count += 1
    if count > 0:
        avg_log_magnitude /= count
    return avg_log_magnitude


# --- Two Folder Comparison ---
def compare_two_folders(folder1, folder2, output_folder, target_size=None):
    """
    Compute the average FFT log-magnitude spectrum for each of two folders,
    generate average plots for each, and then produce difference plots between them.
    """
    folder1 = Path(folder1)
    folder2 = Path(folder2)
    output_folder = Path(output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)
    
    avg1 = get_average_fft(folder1, target_size=target_size)
    avg2 = get_average_fft(folder2, target_size=target_size)
    
    folder1_name = folder1.stem
    folder2_name = folder2.stem

    # --- Difference Plots ---
    # Histogram Difference
    counts1, bin_centers = compute_histogram(avg1, bins=50)
    counts2, _ = compute_histogram(avg2, bins=50)
    diff_hist = counts1 - counts2
    
    plt.figure()
    plt.bar(bin_centers, diff_hist, width=(bin_centers[1]-bin_centers[0]))
    plt.title(f"Histogram Difference: {folder1_name} - {folder2_name}")
    plt.xlabel("Log Magnitude (dB)")
    plt.ylabel("Difference in Frequency")
    plt.grid(True)
    plt.savefig(output_folder / f"{folder1_name}_{folder2_name}_histogram_difference.png")
    plt.close()
    
    # Radial Energy Profile Difference
    radii1, radial_profile1 = compute_radial_profile(avg1)
    radii2, radial_profile2 = compute_radial_profile(avg2)
    # (Assuming the same dimensions, radii1 == radii2)
    radial_diff = radial_profile1 - radial_profile2
    
    plt.figure()
    plt.plot(radii1, radial_diff, 'r-')
    plt.title(f"Radial Energy Profile Difference: {folder1_name} - {folder2_name}")
    plt.xlabel("Radius (pixels)")
    plt.ylabel("Difference in Average Log Magnitude (dB)")
    plt.grid(True)
    plt.savefig(output_folder / f"{folder1_name}_{folder2_name}_radial_difference.png")
    plt.close()
    
    # Angular Energy Distribution Difference
    bin_centers_ang, angular_profile1 = compute_angular_profile(avg1, num_bins=36)
    _, angular_profile2 = compute_angular_profile(avg2, num_bins=36)
    angular_diff = angular_profile1 - angular_profile2
    
    plt.figure()
    plt.plot(bin_centers_ang, angular_diff, 'g-')
    plt.title(f"Angular Energy Distribution Difference: {folder1_name} - {folder2_name}")
    plt.xlabel("Angle (degrees)")
    plt.ylabel("Difference in Average Log Magnitude (dB)")
    plt.grid(True)
    plt.savefig(output_folder / f"{folder1_name}_{folder2_name}_angular_difference.png")
    plt.close()
    
    # Difference Map (Pixel-by-Pixel Difference)
    diff_map = avg1 - avg2
    plt.figure()
    plt.imshow(diff_map, cmap='seismic')
    plt.colorbar(label='Difference (dB)')
    plt.title(f"Difference Map: {folder1_name} - {folder2_name}")
    plt.axis("off")
    plt.savefig(output_folder / f"{folder1_name}_{folder2_name}_difference_map.png")
    plt.close()
    
    # --- Save the Average Plots for Each Folder ---
    # Folder 1 averages
    plot_histogram(avg1, f"{folder1_name}_average",
                   save_path=str(output_folder / f"{folder1_name}_histogram_average.png"))
    plot_radial_energy(avg1, f"{folder1_name}_average",
                       save_path=str(output_folder / f"{folder1_name}_radial_average.png"))
    plot_angular_energy(avg1, f"{folder1_name}_average",
                        save_path=str(output_folder / f"{folder1_name}_angular_average.png"))
    plot_high_energy_density(avg1, f"{folder1_name}_average",
                             save_path=str(output_folder / f"{folder1_name}_density_average.png"))
    
    # Folder 2 averages
    plot_histogram(avg2, f"{folder2_name}_average",
                   save_path=str(output_folder / f"{folder2_name}_histogram_average.png"))
    plot_radial_energy(avg2, f"{folder2_name}_average",
                       save_path=str(output_folder / f"{folder2_name}_radial_average.png"))
    plot_angular_energy(avg2, f"{folder2_name}_average",
                        save_path=str(output_folder / f"{folder2_name}_angular_average.png"))
    plot_high_energy_density(avg2, f"{folder2_name}_average",
                             save_path=str(output_folder / f"{folder2_name}_density_average.png"))
    
    print("Two folder comparison complete. Check the output folder for saved plots.")

# --- Main Interactive Menu ---
if __name__ == "__main__":
    print("Select Mode:")
    print("[1] Single Folder Analysis (per-image + average plots)")
    print("[2] Two Folder Comparison (only average plots & differences)")
    mode = input("Enter mode (1/2): ").strip()
    
    target_size = None
    target_input = input("Enter target size for images as 'width,height' (or leave blank to use original sizes): ").strip()
    if target_input:
        try:
            width, height = map(int, target_input.split(","))
            target_size = (width, height)
        except Exception as e:
            print("Invalid target size input. Using original image sizes.")
    
    if mode == "1":
        input_folder = input("Enter the folder path containing images to analyze: ").strip()
        output_folder = input("Enter the output folder to save plots: ").strip()
        analyze_folder_and_save(input_folder, output_folder, target_size=target_size)
        print("Analysis complete. Check the output folder for saved plots.")
    
    elif mode == "2":
        folder1 = input("Enter the first folder path: ").strip()
        folder2 = input("Enter the second folder path: ").strip()
        output_folder = input("Enter the output folder to save comparison plots: ").strip()
        compare_two_folders(folder1, folder2, output_folder, target_size=target_size)
    
    else:
        print("Invalid mode selected.")
