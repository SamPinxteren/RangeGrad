import torch.nn as nn
import minmax.mm as mm

class Dense(nn.Module):
    def __init__(self, inputs=1, hidden_size=200, hidden_layers=2, outputs=1):
        super(Dense, self).__init__()
        
        self.lin = mm.Linear(inputs, hidden_size)
        self.hls = nn.ModuleList([mm.Linear(hidden_size, hidden_size) for i in range(hidden_layers)])
        self.lout = mm.Linear(hidden_size, outputs)
        
        self.act = mm.ReLU()
    
    def forward(self, x):
        x = self.lin(x)
        x = self.act(x)
        for hl in self.hls:
            x = hl(x)
            x = self.act(x)
        x = self.lout(x)
        return x