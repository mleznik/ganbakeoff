from tcn_lib import *
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader



class TCNClassifier(nn.Module):
    
    def __init__(self, seq_length):
        
        super().__init__()
        
        self.seq_length = seq_length
        
        self.device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")
        
        self.tcnNet = TCNDiscriminator(input_size=2, input_length=seq_length, channels=20, kernel_size=7, num_layers=8,dropout=0.2, num_classes=2).to(self.device)
        
        
        
        
        
            
            
    
    
    
    