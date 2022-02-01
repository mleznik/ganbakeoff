import torch.nn as nn
import torch

class LSTMCell(nn.Module):
    def __init__(self, input_size=10, hidden_layer_size=100, batch_size=128, num_layers=1, bidirectional=False):
        super().__init__()
        self.hidden_layer_size = hidden_layer_size

        self.lstm = nn.LSTM(input_size, hidden_layer_size, num_layers=num_layers, bidirectional=bidirectional)

    def forward(self, input_seq):
        input_seq = input_seq.permute(1,0,2)
        
        lstm_out, _ = self.lstm(input_seq, self.hidden_cell)
        lstm_out = lstm_out.permute(1,0,2)
        return lstm_out
    
    
class LSTMGenerator(nn.Module):
    def __init__(self, batch_size=128,input_size=10, hidden_layer_size=100, output_size=32, seq_length=32, multi_variate=1, output_function=None, num_layers=1, flatten=True,bidirectional=False, device=None):
        super().__init__()
        
        self.seq_length = seq_length
        
        self.hidden_layer_size = hidden_layer_size
        
        self.output_function = output_function
        
        self.bidirectional = bidirectional
        
        self.flatten = flatten
        
        self.multi_variate = multi_variate
        
        self.num_layers = num_layers
        
        self.LSTM = LSTMCell(input_size, hidden_layer_size, batch_size, num_layers, bidirectional=bidirectional)
        
        if self.flatten:
            self.linears = nn.ModuleList([nn.Linear((2 if bidirectional else 1 )*hidden_layer_size*seq_length, output_size) for _ in range(multi_variate)])
        else:
            self.linears = nn.ModuleList([nn.Linear((2 if bidirectional else 1 )*hidden_layer_size, 1) for _ in range(multi_variate)])
        
        #self.sigmoid = nn.Sigmoid()
        
        self.device = device
        
        self.init_weights()
        
            
    def init_weights(self):
        for linear in self.linears:
            linear.weight.data.normal_(0, 0.01)
    

    def forward(self, input):
        
        sequence_input, cond_input = input
        
        if cond_input is not None:
        
            cond_input_reshaped = torch.stack([cond_input]*self.seq_length,axis=1)

            lstm_input = torch.cat((sequence_input,cond_input_reshaped),axis=2)
            
        else:
            lstm_input = sequence_input
        
        
        self.LSTM.hidden_cell = (torch.zeros((2 if self.bidirectional else 1 )*self.num_layers,lstm_input.shape[0],self.hidden_layer_size).to(self.device),
                            torch.zeros((2 if self.bidirectional else 1 )*self.num_layers,lstm_input.shape[0],self.hidden_layer_size).to(self.device))
        
        
        lstm_output = self.LSTM(lstm_input)
        
        
        if self.flatten:
            linear_input = lstm_output.contiguous().view(lstm_input.shape[0],-1)
            output = torch.stack([l(linear_input) for l in self.linears], axis=2)
            
        else: 
             output = torch.stack([l(lstm_output) for l in self.linears], axis=2).squeeze(3)
    
        

        
        if self.output_function:
            output = self.output_function(output)
            
        
        return output
        #return predictions.unsqueeze(2)

class TimeDistributed(nn.Module):
    ## Takes any module and stacks the time dimension with the batch dimenison of inputs before apply the module
    ## From: https://discuss.pytorch.org/t/any-pytorch-function-can-work-as-keras-timedistributed/1346/4
    def __init__(self, module, batch_first=False):
        super(TimeDistributed, self).__init__()
        self.module = module
        self.batch_first = batch_first

    def forward(self, x):

        if len(x.size()) <= 2:
            return self.module(x)

        # Squash samples and timesteps into a single axis
        x_reshape = x.contiguous().view(-1, x.size(-1))  # (samples * timesteps, input_size)

        y = self.module(x_reshape)

        # We have to reshape Y
        if self.batch_first:
            y = y.contiguous().view(x.size(0), -1, y.size(-1))  # (samples, timesteps, output_size)
        else:
            y = y.view(-1, x.size(1), y.size(-1))  # (timesteps, samples, output_size)

        return y
    
class GLU(nn.Module):
    #Gated Linear Unit
    def __init__(self, input_size):
        super(GLU, self).__init__()
        
        self.fc1 = nn.Linear(input_size,input_size)
        self.fc2 = nn.Linear(input_size, input_size)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        
        sig = self.sigmoid(self.fc1(x))
        x = self.fc2(x)
        return torch.mul(sig, x)


class LSTMDiscriminator(nn.Module):
    
    def __init__(self, config, multi_variates, batch_size, seq_length,device, output_size):
        super().__init__()
        self.device = device
        self.batch_size=batch_size
        self.hidden_size = config['hidden_layer_size']
        self.lstm_layers = config['num_layers']
        self.dropout = config['dropout']
        self.multi_variates = multi_variates
        self.seq_length = seq_length
        self.output_size = output_size
        
     
      
                                              
        self.lstm_encoder = nn.LSTM(input_size=self.multi_variates, 
                            hidden_size=self.hidden_size,
                           num_layers=self.lstm_layers,
                           dropout=self.dropout, batch_first=True)
        
    

        self.post_lstm_gate = TimeDistributed(GLU(self.hidden_size))
        self.post_lstm_norm = TimeDistributed(nn.BatchNorm1d(self.hidden_size + self.multi_variates), batch_first=True)

        self.output_layer = TimeDistributed(nn.Linear((self.hidden_size + self.multi_variates)*self.seq_length, self.output_size), batch_first=True)

        
        self.sigmoid = nn.Sigmoid()

        
        self.softmax = nn.Softmax(dim=1)
        
    def init_hidden(self, batch_size):
        return torch.zeros(self.lstm_layers, batch_size, self.hidden_size, device=self.device)
        
    def encode(self, x, hidden=None):
    
        if hidden is None:
            hidden = self.init_hidden(x.shape[0])
            
 
        output, (hidden, cell) = self.lstm_encoder(x, (hidden, hidden))
        
        return output, hidden
    

    def forward(self, input):

                    
        x, cond_input = input
        
            
        if cond_input is not None:
        
            cond_input_reshaped = torch.stack([cond_input]*self.seq_length,axis=1)

            x = torch.cat((x,cond_input_reshaped),axis=2)
        
        
        embedding_vectors = []
      


        ##LSTM
        encoder_output, hidden = self.encode(x)

        ## skip-connection
        encoder_output = torch.cat([encoder_output, x], dim=2)
    

        lstm_output = encoder_output

        ##skip connection over lstm
    
        attn_input = self.post_lstm_norm(lstm_output)
        
        
        output = attn_input
        
        #output = output.permute(1,0,2)
        
        
        output = self.output_layer(output.reshape(output.shape[0],-1))
        
        if self.output_size > 1:
            
            return self.softmax(output)
        
        return self.sigmoid(output)