import numpy as np
import torch
from sklearn.cluster import DBSCAN, SpectralClustering, OPTICS
import logging


def read_data(file_path):
    """
    Reads the point cloud data from a text file and extracts the coordinates.
    
    Args:
    - file_path (str): Path to the input text file containing point cloud data.
    
    Returns:
    - np.ndarray: Coordinates of the point cloud (first 3 columns).
    """
    try:
        data = np.loadtxt(file_path)
        return data[:, :3]  # Extracts the first three columns (x, y, z)
    except Exception as e:
        logging.error(f"Error reading data from {file_path}: {e}")
        raise


def apply_clustering(coordinates, method="DBSCAN", **params):
    """
    Applies a specified clustering algorithm to the coordinates.
    
    Args:
    - coordinates (np.ndarray): The point cloud coordinates to cluster.
    - method (str): The clustering method to use. Supported methods: 'DBSCAN', 'Spectral', 'OPTICS'.
    - params (dict): Additional parameters for the selected clustering method.
    
    Returns:
    - np.ndarray: Labels assigned by the clustering algorithm (superpoint IDs).
    """
    if method == "DBSCAN":
        db = DBSCAN(eps=params.get("eps", 0.02), min_samples=params.get("min_samples", 2))
        labels = db.fit_predict(coordinates)
    
    elif method == "Spectral":
        spectral = SpectralClustering(n_clusters=params.get("n_clusters", 3), 
                                      affinity=params.get("affinity", "nearest_neighbors"))
        labels = spectral.fit_predict(coordinates)
    
    elif method == "OPTICS":
        optics = OPTICS(min_samples=params.get("min_samples", 2), 
                        xi=params.get("xi", 0.05), 
                        min_cluster_size=params.get("min_cluster_size", 0.05))
        labels = optics.fit_predict(coordinates)
    
    else:
        raise ValueError(f"Unsupported clustering method: {method}")
    
    return labels


def save_superpoint_ids(superpoint_ids, output_path):
    """
    Saves the superpoint IDs (labels) to a .pth file.
    
    Args:
    - superpoint_ids (list): The list of superpoint IDs to save.
    - output_path (str): The file path where the superpoint IDs will be saved.
    """
    try:
        torch.save(superpoint_ids, output_path)
        logging.info(f"Superpoint IDs saved to {output_path}")
    except Exception as e:
        logging.error(f"Error saving superpoint IDs to {output_path}: {e}")
        raise


def indexing(coordinates):
    """
    Assigning index values to the cordinates 
    
    Args:
    - coordinates (np.ndarray): The point cloud coordinates to cluster.
    Returns:
    - np.ndarray: Labels assigned by the clustering algorithm (superpoint IDs).
    """
    # Checking the length of the cordinates
    data_length = len(coordinates)
    print(f"Length of the data: {data_length}")

    return np.arange(data_length)

    
def main(input_file, output_file, clustering_method="DBSCAN"):
    """
    Main function to load data, apply clustering, and save results.
    
    Args:
    - input_file (str): Path to the input text file containing point cloud data.
    - output_file (str): Path to the output .pth file where the clustering result will be saved.
    - clustering_method (str): Clustering method to use (e.g., "DBSCAN", "Spectral", "OPTICS").
    """
    # Step 1: Read data
    coordinates = read_data(input_file)

    # Step 2: Checking if clustering is needed
    if clustering_method:
        superpoint_ids = apply_clustering(coordinates, method=clustering_method)

    superpoint_ids = indexing(coordinates)
    
    # Step 3: Convert the superpoint IDs to a list (not a tensor)
    superpoint_list = superpoint_ids.tolist()
    
    # Step 4: Save the superpoint IDs to a .pth file
    save_superpoint_ids(superpoint_list, output_file)


if __name__ == "__main__":
    # Set up logging configuration
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    # Define the paths for the input and output files
    input_file = r"C:\Users\Perception-Team\Downloads\Aug\ISBNet\dataset\s3dis\Stanford3dDataset_v1.2_Aligned_Version\Area_1\2-Air_weld_6\2-Air_weld_6.txt"
    output_file = r"C:\Users\Perception-Team\Downloads\Aug\ISBNet\dataset\s3dis\superpoints\Area_5_2-Air_weld_6.pth"
    
    # Choose the clustering method: DBSCAN, Spectral, or OPTICS
    clustering_method = None  # Default indexing method
    
    # Execute the main function
    main(input_file, output_file, clustering_method)
