import numpy as np
import matplotlib.pyplot as plt

class Model:
    def __init__(self, img_path: str) -> None:
        self.k_vals = [2,3,10,20,40] 
        self.img_data = self.read_image(img_path)
        self.img_H, self.img_W, _ = self.img_data.shape
    
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
        return img_data.reshape(-1, 3)  # Assuming the image has 3 color channels (RGB)
