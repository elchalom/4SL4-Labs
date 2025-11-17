from torchvision import datasets
import torch
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(4322)
torch.manual_seed(4322)

train_dataset = datasets.FashionMNIST (root='./data', train=True, download=True)
test_dataset = datasets.FashionMNIST (root="./data", train=False, download=True )

# Prepare data as numpy arrays
X_train = train_dataset.data.numpy().reshape(-1, 28 * 28).astype('float32') / 255.0
Y_train = train_dataset.targets.numpy()

X_test = test_dataset.data.numpy().reshape(-1, 28 * 28).astype('float32') / 255.0
Y_test = test_dataset.targets.numpy()

# Split training into train and validation (80/20)
validation_size = int(0.2 * X_train.shape[0])
X_validation, Y_validation = X_train[:validation_size], Y_train[:validation_size]
X_train, Y_train = X_train[validation_size:], Y_train[validation_size:]

# Save original labels
Y_train_orig = Y_train
Y_validation_orig = Y_validation
Y_test_orig = Y_test

# One-hot encode labels
def one_hot_encode(labels, num_classes=10):
    return np.eye(num_classes)[labels]

Y_train = one_hot_encode(Y_train)
Y_validation = one_hot_encode(Y_validation)
Y_test = one_hot_encode(Y_test)

# Standardize data
X_train_mean = X_train.mean(axis=0)
X_train_std = X_train.std(axis=0)
X_train_std[X_train_std == 0] = 1  # Avoid division by zero

X_train = (X_train - X_train_mean) / X_train_std
X_validation = (X_validation - X_train_mean) / X_train_std
X_test = (X_test - X_train_mean) / X_train_std
7

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
        
        self.gradients = [] # To store gradients during backpropagation
        self.activations = [] # To store activations during forward pass
        self.z_values = [] # To store z values during forward pass
        
        for i in range(self.num_layers - 1):
            # Initialize weights with 1 extra column for biases
            w = np.random.randn(layer_sizes[i] + 1, layer_sizes[i+1]) * np.sqrt(2 / layer_sizes[i])
            
            
            # Initialize biases to 0
            w[0, :] = 0
            
            self.weights.append(w)
        
    
    def forward_pass(self, X:np.ndarray) -> np.ndarray:
        self.activations = []   # Stores output of each layer
        self.z_values = []   # Stores input to activation function of each layer
        
        # Use (1 X) for first layer (1 for bias term)
        h = np.concatenate([np.ones((X.shape[0], 1)), X], axis=1) 
        self.activations.append(h)
        
        for i in range(self.num_layers - 2):
            # Compute weighted sum with bias
            # Z = (1 h) W 
            
            Z = h @ self.weights[i]
            self.z_values.append(Z)
            
            # Apply ReLU activation to obtain h without bias column
            h = ReLU(Z)
            
            # Add bias term to h for next layer
            h = np.concatenate([np.ones((h.shape[0], 1)), h], axis=1)
            
            self.activations.append(h)
        
        # Output layer
        Z = h @ self.weights[-1]
        self.z_values.append(Z)
        h = softmax(Z)
        self.activations.append(h)
        
        return h
            
    
    def backward_pass(self, X:np.ndarray, Y:np.ndarray) -> None:
        batch_size = X.shape[0]
        self.gradients = []  # Clear previous gradients
        
        # Output layer gradient (softmax + cross-entropy)
        dY = self.activations[-1] - Y  
        dZ = dY
        
        # Gradient for output layer weights
        # dW = dY (1 h(d-1))^T)
        # we divide by batch size to get average gradient over entire batch
        dW = self.activations[-2].T @ dY / batch_size
        
        # L2 regularization term (exclude bias weight)
        dW[1:, :] += (self.weight_decay / batch_size) * self.weights[-1][1:, :]
        self.gradients.insert(0, dW)
        
        # Backtrack though all hidden layers
        for i in range (self.num_layers -2, 0, -1):
            # Propagate gradient through weights (exclude bias weights)
            dZ = dZ @ self.weights[i][1:, :].T
            
            # Apply ReLU derivative
            dZ = dZ * ReLU_derivative(self.z_values[i-1])
            
            # Compute next dW
            dW = self.activations[i-1].T @ dZ / batch_size
            
            # Add L2 regularization term without bias weights
            dW[1:, :] += (self.weight_decay / batch_size) * self.weights[i-1][1:, :]
            
            self.gradients.insert(0, dW)
    
    def update_weights(self) -> None:
        # Loop through each layer and update weights
        for i in range (self.num_layers - 1):
            # Wnew = Wold - alpha * dW
            self.weights[i] -= self.alpha * self.gradients[i]
    
    def compute_loss(self, Y_pred:np.ndarray, Y_true:np.ndarray) -> float:
        m = Y_pred.shape[0]
        
        # Cross-entropy loss
        epsilon = 1e-8 # To avoid log(0)
        ce_loss = -np.sum(Y_true * np.log(Y_pred + epsilon)) / m
        
        # L2 regularization loss
        l2_loss = 0.0
        for w in self.weights:
            l2_loss += np.sum(w[1:, :] ** 2)  # Exclude bias weights
        l2_loss *= (self.weight_decay / (2 * m))
        
        return ce_loss + l2_loss
    
    def compute_error(self, Y_pred:np.ndarray, Y_true:np.ndarray) -> float:
        # Predicted classes
        pred_classes = np.argmax(Y_pred, axis=1)
        
        # True classes
        true_classes = np.argmax(Y_true, axis=1)
        
        # Compute error rate
        error_rate = np.mean(pred_classes != true_classes)
        
        return error_rate
    
    def train(self, 
              X_train:np.ndarray, 
              Y_train:np.ndarray, 
              X_val:np.ndarray, 
              Y_val:np.ndarray, 
              epochs: int,
              batch_size: int
              ) -> None:
        
        # Return variables for reporting
        results = {
            "train_loss": [],
            "val_loss": [],
            "train_error": [],
            "val_error": []
        }
        
        smallest_val_loss = float('inf')
        epochs_without_improvement = 0
        early_stopping_threshold = 5
        
        for epoch in range(epochs):
            # Shuffle training data
            indices = np.random.permutation(X_train.shape[0])
            X_shuffled = X_train[indices]
            Y_shuffled = Y_train[indices]
            
            # Mini-batch training
            num_batches = int(np.ceil(X_train.shape[0] / batch_size))
            
            for batch_idx in range(num_batches):
                # Only use data for current batch
                start_idx = batch_idx * batch_size
                end_indx = min(start_idx + batch_size, X_train.shape[0])
                
                X_batch = X_shuffled[start_idx:end_indx]
                Y_batch = Y_shuffled[start_idx:end_indx]
                
                # Forward pass
                predictions = self.forward_pass(X_batch)
                
                # Backward pass
                self.backward_pass(X_batch, Y_batch)
                
                # Update weights
                self.update_weights()
                
            # Evaluate after epoch
            train_predictions = self.forward_pass(X_train)
            train_loss = self.compute_loss(train_predictions, Y_train)
            train_error = self.compute_error(train_predictions, Y_train)
            
            val_predictions = self.forward_pass(X_val)
            val_loss = self.compute_loss(val_predictions, Y_val)
            val_error = self.compute_error(val_predictions, Y_val)
            
            # Store results
            results["train_loss"].append(train_loss)
            results["val_loss"].append(val_loss)
            results["train_error"].append(train_error)
            results["val_error"].append(val_error)
                
            print(f"Epoch {epoch+1}/{epochs} - "
                    f"Train Loss: {train_loss:.4f}, Train Error: {train_error:.4f} - "
                    f"Val Loss: {val_loss:.4f}, Val Error: {val_error:.4f}")
        
            # Early stopping check
            if val_loss < smallest_val_loss:
                smallest_val_loss = val_loss
                epochs_without_improvement = 0
                # Save the best weights
                self.best_weights = [w.copy() for w in self.weights]
                
            else:
                epochs_without_improvement += 1
                
            if epochs_without_improvement >= early_stopping_threshold:
                print("Early stopping triggered after {epoch+1} epochs.")
                # Restore best weights
                self.weights = self.best_weights
                break
            
        return results    
    
    def predict(self, X:np.ndarray) -> np.ndarray:
        predictions = self.forward_pass(X)
        return np.argmax(predictions, axis=1)
    
    def evaluate(self, X:np.ndarray, y:np.ndarray) -> float:
        predictions = self.forward_pass(X)
        loss = self.compute_loss(predictions, y)
        error = self.compute_error(predictions, y)
        
        return {
            'loss': loss,
            'error': error
        }
        
        
if __name__ == "__main__":
    # Train first model (initialization 1)
    print("=" * 50)
    print("Training Model 1")
    print("=" * 50)
    
    np.random.seed(3093)
    nn1 = NeuralNetwork(
        layer_sizes=[784, 156, 156, 10],
        alpha=0.01,
        weight_decay=0.0018738
    )
    
    history1 = nn1.train(
        X_train=X_train,
        Y_train=Y_train,
        X_val=X_validation,
        Y_val=Y_validation,
        epochs=100,
        batch_size=128
    )
    
    # Evaluate on test set
    test_results1 = nn1.evaluate(X_test, Y_test)
    print(f"\nModel 1 Test Results:")
    print(f"  Loss: {test_results1['loss']:.4f}")
    print(f"  Error: {test_results1['error']:.4f} ({test_results1['error']*100:.2f}%)")
    
    # Train second model (initialization 2)
    print("\n" + "=" * 50)
    print("Training Model 2")
    print("=" * 50)
    
    np.random.seed(9303)  # Different seed!
    nn2 = NeuralNetwork(
        layer_sizes=[784, 156, 156, 10],
        alpha=0.01,
        weight_decay=0.0018738
    )
    
    history2 = nn2.train(
        X_train=X_train,
        Y_train=Y_train,
        X_val=X_validation,
        Y_val=Y_validation,
        epochs=100,
        batch_size=128
    )
    
    # Evaluate on test set
    test_results2 = nn2.evaluate(X_test, Y_test)
    print(f"\nModel 2 Test Results:")
    print(f"  Loss: {test_results2['loss']:.4f}")
    print(f"  Error: {test_results2['error']:.4f} ({test_results2['error']*100:.2f}%)")
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Model 1 - Loss
    axes[0, 0].plot(history1['train_loss'], label='Train')
    axes[0, 0].plot(history1['val_loss'], label='Validation')
    axes[0, 0].set_title('Model 1 - Loss')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True)
    
    # Model 1 - Error
    axes[0, 1].plot(history1['train_error'], label='Train')
    axes[0, 1].plot(history1['val_error'], label='Validation')
    axes[0, 1].set_title('Model 1 - Error')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Error Rate')
    axes[0, 1].legend()
    axes[0, 1].grid(True)
    
    # Model 2 - Loss
    axes[1, 0].plot(history2['train_loss'], label='Train')
    axes[1, 0].plot(history2['val_loss'], label='Validation')
    axes[1, 0].set_title('Model 2 - Loss')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Loss')
    axes[1, 0].legend()
    axes[1, 0].grid(True)
    
    # Model 2 - Error
    axes[1, 1].plot(history2['train_error'], label='Train')
    axes[1, 1].plot(history2['val_error'], label='Validation')
    axes[1, 1].set_title('Model 2 - Error')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Error Rate')
    axes[1, 1].legend()
    axes[1, 1].grid(True)
    
    plt.tight_layout()
    plt.savefig('learning_curves.png', dpi=300)
    plt.show()
    
    print("\nTraining complete! Learning curves saved to 'learning_curves.png'")