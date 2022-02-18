import torch
import torch.nn as nn

class WeightingLayer(nn.Module):
    def __init__(self):
        super(WeightingLayer, self).__init__()
        self.fc1 = nn.Sequential(
            nn.Conv1d(32, 16,1,True),
            nn.ReLU()
        )
        self.fc2 = nn.Sequential(
            nn.Conv1d(16, 8,1,True),
            nn.ReLU()
        )
        self.fc3 = nn.Sequential(
            nn.Conv1d(8, 1,1,True),
            nn.Softplus()
        )
    
    def forward(self, X, K = 64):
        X = self.fc1(X)
        X = self.fc2(X)
        X = self.fc3(X)

        topk_indices = torch.topk(X, K, dim = -1).indices
        topk_indices = topk_indices.flatten()
        return topk_indices
