import torch.nn as nn
import torch

class PM_Model(nn.Module):
    def __init__(self, hidden_layers: int, drop_out: bool, drop_value:float, input_size: int, hidden_size: int, output_size: int = 1):
        super(PM_Model, self).__init__()

        layers = []

 
        layers.append(nn.Linear(input_size, hidden_size))
        layers.append(nn.ReLU())
        if drop_out:    
            layers.append(nn.Dropout(drop_value))

        for _ in range(hidden_layers):
            layers.append(nn.Linear(hidden_size, hidden_size))
            layers.append(nn.ReLU())
            if drop_out:
                layers.append(nn.Dropout(drop_value))

        layers.append(nn.Linear(hidden_size, output_size))

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)
    
