import numpy as np
import vedo
import glob
import os

def visualize_pointcloud_from_npz(npz_path, title="Point Cloud Visualization", sdf_range=(0, 0.1)):
    """
    Visualize point cloud from NPZ file using vedo with toggle functionality.
    
    Args:
        npz_path: Path to the NPZ file
        title: Title for the visualization window
        sdf_range: Tuple of (min, max) SDF values to filter points
    """
    # Load the NPZ file
    data = np.load(npz_path)
    
    # Extract point coordinates and SDF values
    points = data['poisson_grid_points']
    sdf_values = data['sdf_values']
    print(sdf_values.min(), sdf_values.max())
    
    print(f"Loaded {len(points)} points")
    print(f"Original SDF values range: {sdf_values.min():.3f} to {sdf_values.max():.3f}")
    
    # Filter points based on SDF range
    mask = (sdf_values >= sdf_range[0]) & (sdf_values <= sdf_range[1])
    filtered_points = points[mask]
    filtered_sdf_values = sdf_values[mask]
    
    print(f"Filtered to {len(filtered_points)} points with SDF values in range [{sdf_range[0]}, {sdf_range[1]}]")
    print(f"Filtered SDF values range: {filtered_sdf_values.min():.3f} to {filtered_sdf_values.max():.3f}")
    
    # Create vedo point clouds
    all_point_cloud = vedo.Points(points, r=2)
    all_point_cloud.cmap("coolwarm", sdf_values, vmin=sdf_values.min(), vmax=sdf_values.max())
    
    filtered_point_cloud = vedo.Points(filtered_points, r=3)
    filtered_point_cloud.cmap("coolwarm", filtered_sdf_values, vmin=sdf_range[0], vmax=sdf_range[1])
    
    # Create colorbars
    all_colorbar = vedo.ScalarBar(
        all_point_cloud, 
        title="All Points - SDF Values",
        pos=(0.8, 0.2),
        size=(100, 300)
    )
    
    filtered_colorbar = vedo.ScalarBar(
        filtered_point_cloud, 
        title=f"Filtered Points - SDF [{sdf_range[0]}, {sdf_range[1]}]",
        pos=(0.8, 0.2),
        size=(100, 300)
    )
    
    # Create the plotter
    plt = vedo.Plotter(title=title, size=(1200, 800))
    
    # Add the filtered point cloud and colorbar initially
    plt.add(filtered_point_cloud)
    plt.add(filtered_colorbar)
    
    # Store references for toggle
    plt.all_points = all_point_cloud
    plt.all_colorbar = all_colorbar
    plt.filtered_points = filtered_point_cloud
    plt.filtered_colorbar = filtered_colorbar
    plt.showing_filtered = True
    
    def toggle_view(*args):
        """Toggle between showing all points and filtered points"""
        if plt.showing_filtered:
            # Switch to all points
            plt.remove(plt.filtered_points)
            plt.remove(plt.filtered_colorbar)
            plt.add(plt.all_points)
            plt.add(plt.all_colorbar)
            plt.showing_filtered = False
            print("Switched to showing ALL points")
        else:
            # Switch to filtered points
            plt.remove(plt.all_points)
            plt.remove(plt.all_colorbar)
            plt.add(plt.filtered_points)
            plt.add(plt.filtered_colorbar)
            plt.showing_filtered = True
            print(f"Switched to showing FILTERED points (SDF [{sdf_range[0]}, {sdf_range[1]}])")
    
    # Add button for toggle
    plt.add_button(
        toggle_view,
        pos=(0.1, 0.9),
        states=["Show All Points", "Show Filtered Points"],
        size=16,
        bold=True,
        c=["red", "blue"],
        bc=["white", "white"]
    )
    
    # Add info text
    # Note: vedo doesn't have a simple add_text method, so we'll skip the info text for now
    # You can see the information in the console output above
    
    # Add keyboard shortcut
    plt.keyPressFunction = lambda key: toggle_view() if key == 't' or key == 'T' else None
    
    # Show the visualization
    plt.show()
    
    return filtered_point_cloud

def visualize_all_npz_files(npz_dir="data/npz"):
    """
    Visualize all NPZ files in the specified directory.
    
    Args:
        npz_dir: Directory containing NPZ files
    """
    npz_files = glob.glob(os.path.join(npz_dir, "*.npz"))
    npz_files.sort()
    
    print(f"Found {len(npz_files)} NPZ files:")
    for i, file in enumerate(npz_files):
        print(f"  {i+1}. {os.path.basename(file)}")
    
    if not npz_files:
        print("No NPZ files found!")
        return
    
    # Visualize the first file as an example
    print(f"\nVisualizing: {os.path.basename(npz_files[0])}")
    print("Showing only points with SDF values between 0 and 0.1 (near surface)")
    visualize_pointcloud_from_npz(npz_files[0], f"Point Cloud - {os.path.basename(npz_files[0])} (Near Surface)")

if __name__ == "__main__":
    # Check if the specific file exists
    # specific_file = "../data/npz-ffb/2.npz"
    # specific_file = "verify/current-5/2.npz"
    specific_file = "verify/now-300/2.npz"
    if os.path.exists(specific_file):
        print(f"Visualizing specific file: {specific_file}")
        print("Showing only points with SDF values between 0 and 0.1 (near surface)")
        visualize_pointcloud_from_npz(specific_file, "Point Cloud - 1.npz (Near Surface)", sdf_range=(-1, 1))
    else:
        print(f"File {specific_file} not found. Looking for all NPZ files...")
        visualize_all_npz_files()
