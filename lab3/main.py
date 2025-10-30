import numpy as np
import matplotlib.pyplot as plt

class Model:
    def __init__(self, img_path: str) -> None:
        self.k_vals = [2,3,10,20,40] 
        self.img_data = self.read_image(img_path)
        self.img_H, self.img_W, _ = self.img_data.shape
        
        self.flat_img_data = self.reshape_image(self.img_data)
    
    @staticmethod    
    def read_image(path: str) -> np.ndarray:
        """Reads an image from the specified path and returns it as a numpy array."""
        image = plt.imread(path)
        if image.dtype == np.uint8:
            image = image / 255.0  # Normalize if in uint8 format
        return image
    
    @staticmethod
    def reshape_image(img_data) -> np.ndarray:
        """Reshapes the image data into a 2D array where each row is a pixel and each column is a color channel."""
        return img_data.reshape(-1, 3)  # Only take RGB data, ignore H and W values.
    
    @staticmethod
    def calculate_rmse(original: np.ndarray, reconstructed: np.ndarray) -> float:
        """Calculates the Root Mean Square Error between the original and reconstructed images."""
        diff = original - reconstructed
        mse = np.mean(np.square(diff))
        rmse = np.sqrt(mse)
        return rmse

    @staticmethod
    def initialize_centers_random(k: int, data: np.ndarray) -> np.ndarray:
        """Randomly initializes k cluster centers from the data points."""
        indices = np.random.choice(data.shape[0], k, replace=False)
        centers = data[indices]
        return centers

    @staticmethod
    def initialize_centers_spread(k:int, data: np.ndarray) -> np.ndarray:
        """Randomly initializes k cluster centers from the data points."""
        centers = [data[np.random.randint(data.shape[0])]]
        for _ in range(1, k):
            # Compute distance from each pixel to its nearest center
            distances = np.min(np.linalg.norm(data[:, None] - np.array(centers)[None, :], axis=2), axis=1)
            # Pick the pixel farthest from current centers
            next_center = data[np.argmax(distances)]
            centers.append(next_center)
        centers = np.array(centers)
        return centers
    
    @staticmethod
    def group_clusters(data: np.ndarray, centers: np.ndarray) -> np.ndarray:
        """Assigns each data point to the nearest cluster center."""
        # Compute squared Euclidean distance between each data point and each center
        dists_sq = np.sum((data[:, None, :] - centers[None, :, :]) ** 2, axis=2)

        # Find the index of the closest center for each point
        labels = np.argmin(dists_sq, axis=1)

        # Replace each point’s RGB value with its nearest center’s RGB value
        return labels
            
    @staticmethod
    def update_centers(data: np.ndarray, labels: np.ndarray, k: int) -> np.ndarray:
        """Updates the cluster centers based on the current assignments."""
        pass
    
    def reconstruct_image(self, data: np.ndarray, labels: np.ndarray, centers: np.ndarray, img_H: int, img_W: int) -> np.ndarray:
        """Reconstructs the image from the clustered data."""
        reconstructed_data = self.centers[labels]
        reconstructed_image = reconstructed_data.reshape(self.img_H, self.img_W, 3)
        return reconstructed_image  
    
    