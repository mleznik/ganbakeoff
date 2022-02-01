import torch
import torch.nn as nn
from torch.nn.utils import weight_norm

class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()
    

class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
        super(TemporalBlock, self).__init__()
        self.conv1 = weight_norm(nn.Conv1d(n_inputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = weight_norm(nn.Conv1d(n_outputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1, self.dropout1,
                                 self.conv2, self.chomp2, self.relu2, self.dropout2)
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()
        self.init_weights()

    def init_weights(self):
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class TemporalConvNet(nn.Module):
    def __init__(self, num_inputs, num_channels, kernel_size=2, dropout=0.2):
        super(TemporalConvNet, self).__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = num_inputs if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            layers += [TemporalBlock(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size,
                                     padding=(kernel_size-1) * dilation_size, dropout=dropout)]

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)
    

    
"""class TCNGenerator(nn.Module):
    def __init__(self, input_size=10, num_channels=[20]*8, output_size=1,kernel_size=2, dropout=0.2, input_length=32):
        super(TCNGenerator, self).__init__()
        
        self.input_length = input_length
        
        self.tcn = TemporalConvNet(input_size, num_channels, kernel_size=kernel_size, dropout=dropout)
        
        self.linear = nn.Linear(num_channels[-1], output_size)
        
        self.init_weights()
        

    def init_weights(self):
        self.linear.weight.data.normal_(0, 0.01)

    def forward(self, input):
        x, cond_input = input
        
        if cond_input is not None:
        
            cond_input_reshaped = torch.stack([cond_input]*self.input_length,axis=1)

            x = torch.cat((x,cond_input_reshaped),axis=2)
            

        
        x = x.permute(0,2,1)
        y1 = self.tcn(x)
        return self.linear(y1.transpose(1, 2))"""
    
class TCNGenerator(nn.Module):
    def __init__(self, input_size=10, channels=20, num_layers=8, output_size=1,kernel_size=2, dropout=0.2, input_length=32, multi_variate=1, output_function=None):
        super().__init__()
        
        self.input_length = input_length
        
        self.output_function=output_function
        
        num_channels = [channels]*num_layers
        
        self.tcn = TemporalConvNet(input_size, num_channels, kernel_size=kernel_size, dropout=dropout)
        
        self.linears = nn.ModuleList([nn.Linear(num_channels[-1], output_size) for _ in range(multi_variate)])
        
        
        #self.sigmoid = nn.Sigmoid()
        #self.sigmoid = nn.Tanh()
        
        
        
        self.init_weights()
        

    def init_weights(self):
        for linear in self.linears:
            linear.weight.data.normal_(0, 0.01)

    def forward(self, input):
        x, cond_input = input
        
       

        
        if cond_input is not None:
        
            cond_input_reshaped = torch.stack([cond_input]*self.input_length,axis=1)

            x = torch.cat((x,cond_input_reshaped),axis=2)
            

        
        x = x.permute(0,2,1)
        y1 = self.tcn(x)
        
    
        
        #print(f"Y1 Mean: {torch.mean(y1)}, Max: {torch.max(y1)}, Min: {torch.min(y1)}")
        
        
        output = torch.stack([l(y1[:,:,-1]) for l in self.linears], axis=2)
        
        #print(f"OUTPUT Mean: {torch.mean(output)}, Max: {torch.max(output)}, Min: {torch.min(output)}")
        
        if self.output_function:
            output = self.output_function(output)

        #print(f"After sigmoid Mean: {torch.mean(output)}, Max: {torch.max(output)}, Min: {torch.min(output)}")
        
        return output
        
        return self.linear(y1[:,:,-1]).unsqueeze(2)
        #return self.linear(y1.transpose(1, 2))
    

class TCNDiscriminator(nn.Module):
    def __init__(self, input_size=1, input_length=32,channels=20,num_layers=8, kernel_size=2, dropout=0.2, num_classes=1, softmax=False):
        super().__init__()
        
        self.input_length = input_length
        
        num_channels = [channels]*num_layers
    
        
        self.tcn = TemporalConvNet(input_size, num_channels, kernel_size=kernel_size, dropout=dropout)
        
        
        self.num_classes = num_classes
        
        self.linear1 = nn.Linear(num_channels[-1], num_classes)
        
        #self.linear2 = nn.Linear(input_length, 1)
        
        
        self.sigmoid = nn.Sigmoid()
        
        self.softmax = nn.Softmax(dim=1)
                
            
        
        self.init_weights()
        

    def init_weights(self):
        self.linear1.weight.data.normal_(0, 0.01)

    def forward(self, input):
        x, cond_input = input    
               
        if cond_input is not None:
            
           
                    
            cond_input_reshaped = torch.stack([cond_input]*self.input_length,axis=1)
    

            x = torch.cat((x,cond_input_reshaped),axis=2)
            
        x = x.transpose(1,2)
        y1 = self.tcn(x)
        
        if self.num_classes > 1: 
        
            return self.softmax(self.linear1(y1[:,:,-1]))
        
        return self.sigmoid(self.linear1(y1[:,:,-1]).squeeze(1))
