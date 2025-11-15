from torchvision import datasets
import numpy as np

train_dataset = datasets.FashionMNIST ( root = './ data' , train =
        True , download = True )

test_dataset = datasets.FashionMNIST ( root = "./ data" , train =
        False , download = True )

def ReLU(z: np.ndarray) -> np.ndarray:
    return np.maximum(0, z)

def ReLU_derivative(z: np.ndarray) -> np.ndarray:
    return (z > 0).astype(float)

def softmax(z: np.ndarray) -> np.ndarray:
    # Subtract max for numerical stability for large z
    exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))
    return exp_z / np.sum(exp_z, axis=1, keepdims=True)


class NeuralNetwork:
    def __init__(self, layer_sizes: list[int], alpha: float, weight_decay: float) -> None:
        self.layer_sizes = layer_sizes
        self.alpha = alpha 
        self.weight_decay = weight_decay
        
        self.num_layers = len(layer_sizes)
        self.weights = []
        
        for i in range(self.num_layers - 1):
            # Initialize weights with 1 extra column for biases
            w = np.random.rand(layer_sizes[i] + 1, layer_sizes[i+1]) * np.sqrt(2 / layer_sizes[i])
            
            
            # Initialize biases to 0
            w[0, :] = 0
            
            self.weights.append(w)
        
    
    def forward_pass(self, x:np.ndarray) -> np.ndarray:
        pass
    
    def backward_pass(self, x:np.ndarray, y:np.ndarray) -> None:
        pass
    
    def update_weights(self) -> None:
        pass
    
    def train(self, 
              x_train:np.ndarray, 
              y_train:np.ndarray, 
              x_val:np.ndarray, 
              y_val:np.ndarray, 
              epochs: list[float], # IDK IF THIS IS RIGHT
              batch_size: int
              ) -> None:
        pass
    
    def predict(self, x:np.ndarray) -> np.ndarray:
        pass
    
    def evaluate(self, x:np.ndarray, y:np.ndarray) -> float:
        pass
        