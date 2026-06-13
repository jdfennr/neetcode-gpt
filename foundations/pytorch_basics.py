import torch
import torch.nn
from torchtyping import TensorType

# Round all answers to 4 decimal places: torch.round(tensor, decimals=4)
class Solution:
    def reshape(self, to_reshape: TensorType[float]) -> TensorType[float]:
        # Reshape (M, N) tensor to (M*N/2, 2)
        m, n = to_reshape.shape
        # Use integer division // to ensure the shape dimensions are integers
        new = torch.reshape(to_reshape, ((m * n) // 2, 2))
        return torch.round(new, decimals=4)

    def average(self, to_avg: TensorType[float]) -> TensorType[float]:
        # Compute column-wise mean (average across rows)
        avg_tensor = torch.mean(to_avg, dim=0)
        return torch.round(avg_tensor, decimals=4)

    def concatenate(self, cat_one: TensorType[float], cat_two: TensorType[float]) -> TensorType[float]:
        # Join two tensors side-by-side along dim=1
        cat_tensor = torch.cat((cat_one, cat_two), dim=1)
        return torch.round(cat_tensor, decimals=4)

    def get_loss(self, prediction: TensorType[float], target: TensorType[float]) -> TensorType[float]:
        # Compute Mean Squared Error between prediction and target
        # Fixed the typo here from 'prediciton' to 'prediction'
        loss = torch.nn.functional.mse_loss(prediction, target)
        return torch.round(loss, decimals=4)