import numpy as np
import matplotlib.pyplot as plt

class Model:
    def __init__(self, img_path: str, theta:float = 0.001) -> None:
        self.k_vals = [2,3,10,20,40] 
        self.img_data = self.read_image(img_path)
        self.img_H, self.img_W, _ = self.img_data.shape
        
        self.flat_img_data = self.reshape_image(self.img_data)
    
        self.centers = None
        self.labels = None
        self.theta = theta
        
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
        centers = np.zeros((k,3))
        
        for i in range(k):
            cluster_pixels = data[labels == i]
            if len(cluster_pixels) > 0:
                centers[i] = np.mean(cluster_pixels, axis=0)
            else:
                centers[i] = data[np.random.randint(data.shape[0])]
    
        return centers
    
    def reconstruct_image(self) -> np.ndarray:
        """Reconstructs the image from the clustered data."""
        reconstructed_data = self.centers[self.labels]
        reconstructed_image = reconstructed_data.reshape(self.img_H, self.img_W, 3)
        return reconstructed_image  
    
    def run_model(self, k: int, init_method: str = 'random', max_iters: int = 100) -> dict:
        """Runs the K-means clustering algorithm for a single k value."""
        # Initialize centers
        if init_method == 'random':
            self.centers = self.initialize_centers_random(k, self.flat_img_data)
        elif init_method == 'spread':
            self.centers = self.initialize_centers_spread(k, self.flat_img_data)
        else:
            raise ValueError("Invalid initialization method. Choose 'random' or 'spread'.")
        
        # Initialize for convergence check
        rmse_prev = float('inf')
        iterations = 0
        
        # K-means loop
        while iterations < max_iters:
            iterations += 1
            
            # Assignment step
            self.labels = self.group_clusters(self.flat_img_data, self.centers)
            
            # Update step
            self.centers = self.update_centers(self.flat_img_data, self.labels, k)
            
            # Calculate RMSE
            reconstructed_image = self.reconstruct_image()
            rmse_curr = self.calculate_rmse(self.img_data, reconstructed_image)
            
            # Check convergence (relative change)
            if iterations > 1:  # Skip first iteration
                relative_change = (rmse_prev - rmse_curr) / rmse_prev
                if relative_change < self.theta:
                    break
            
            rmse_prev = rmse_curr
        
        # Store results
        results = {
            'k': k,
            'init_method': init_method,
            'rmse': rmse_curr,
            'iterations': iterations,  
            'reconstructed_image': reconstructed_image,
            'converged': iterations < max_iters
        }
        
        return results
                
                
if __name__ == "__main__":
    img_path = 'Panda.jpg'
    img_name = img_path.split('/')[-1].split('.')[0]
    model = Model(img_path, theta=0.001)
    
    for k in model.k_vals:
        run_num = 0
        for init_method in ['random', 'random', 'spread']:
            run_num += 1
            results = model.run_model(k, init_method)
            print(f"K: {results['k']}, Init: {results['init_method']}, RMSE: {results['rmse']:.4f}, "
                  f"Iterations: {results['iterations']}, Converged: {results['converged']}")
            
            # Optionally display the reconstructed image
            plt.imshow(results['reconstructed_image'])
            plt.title(f'K={k}, Init={init_method}, RMSE={results["rmse"]:.4f}, Iter={results["iterations"]}')
            plt.axis('off')
            # plt.show()
            plt.savefig(f'results/{img_name}/{img_name}_k{k}_init{init_method}_run{run_num}.png')