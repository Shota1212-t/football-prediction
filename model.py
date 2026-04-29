import torch.nn as nn

class SoccerPredictor(nn.Module):
    def __init__(self, input_size):
        super(SoccerPredictor, self).__init__()
        self.fc1 = nn.Linear(input_size, 128)
        self.relu = nn.ReLU()
        self.dropout1 = nn.Dropout(0.4) 
        self.fc2 = nn.Linear(128, 64)
        self.dropout2 = nn.Dropout(0.4) 
        self.fc3 = nn.Linear(64, 32)    
        self.fc4 = nn.Linear(32, 3)     
        
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.dropout1(x)
        x = self.relu(self.fc2(x))
        x = self.dropout2(x)
        x = self.relu(self.fc3(x))
        x = self.fc4(x)
        return x