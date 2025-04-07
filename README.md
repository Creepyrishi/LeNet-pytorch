# LeNet-5 PyTorch Implementation for MNIST Classification

[![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=flat&logo=pytorch&logoColor=white)](https://pytorch.org/)
A PyTorch implementation of Yann LeCun's LeNet-5 architecture for handwritten digit recognition on the MNIST dataset, achieving **98.85% test accuracy**.

![LeNet-5 Architecture](https://www.researchgate.net/profile/Adrian-Ulate-Caballero/publication/344420264/figure/fig2/AS:941642628784129@1601468831818/Architecture-of-LeNet-5.png)  
*LeNet-5 Architecture Diagram (Source: Yann LeCun)*

## Key Features
- **Original Architecture**: Faithful implementation using Tanh activations and Average Pooling
- **High Performance**: Achieves 98.85% accuracy on MNIST test set
- **Modular Design**: Clear separation of feature extraction and classification layers
- **GPU Support**: Automatic CUDA detection and utilization
- **Reproducibility**: Deterministic training with seed setting

## Model Architecture
```python
class LeNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 6, kernel_size=5),
            nn.Tanh(),
            nn.AvgPool2d(kernel_size=2, stride=2),
            nn.Conv2d(6, 16, kernel_size=5),
            nn.Tanh(),
            nn.AvgPool2d(kernel_size=2, stride=2)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(16*4*4, 120),
            nn.Tanh(),
            nn.Linear(120, 84),
            nn.Tanh(),
            nn.Linear(84, 10)
        )
```

### Layer Dimensions
| Layer                | Output Shape     | Parameters |
|----------------------|------------------|------------|
| Input                | 1×28×28          | -          |
| Conv1 + Tanh         | 6×24×24          | 156        |
| AvgPool1             | 6×12×12          | -          |
| Conv2 + Tanh         | 16×8×8           | 2,416      |
| AvgPool2             | 16×4×4           | -          |
| Flatten              | 256              | -          |
| FC1 + Tanh           | 120              | 30,840     |
| FC2 + Tanh           | 84               | 10,164     |
| Output               | 10               | 850        |

**Total Parameters**: 44,426

## Training Details
- **Optimizer**: Adam (lr=0.001)
- **Loss Function**: Cross Entropy
- **Batch Size**: 32
- **Epochs**: 100
- **Validation**: Accuracy check every 20 epochs

### Training Progress
```
Epoch 20 | Loss: 0.0093 | Validation Accuracy: 98.81%
Epoch 40 | Loss: 0.0070 | Validation Accuracy: 98.64%
Epoch 60 | Loss: 0.0061 | Validation Accuracy: 98.83%
Epoch 80 | Loss: 0.0078 | Validation Accuracy: 98.85%
Final | Test Accuracy: 98.85%
```
