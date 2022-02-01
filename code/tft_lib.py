"""
Implementation of Temporal Fusion Transformers: https://arxiv.org/abs/1912.09363
"""


from torch import nn
import math
import torch
import torch.nn.functional as F
#import ipdb


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


class GatedResidualNetwork(nn.Module):
    def __init__(self, input_size,hidden_state_size, output_size, dropout, hidden_context_size=None, batch_first=False):
        super(GatedResidualNetwork, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_context_size = hidden_context_size
        self.hidden_state_size=hidden_state_size
        self.dropout = dropout
        
        if self.input_size!=self.output_size:
            self.skip_layer = TimeDistributed(nn.Linear(self.input_size, self.output_size))

        self.fc1 = TimeDistributed(nn.Linear(self.input_size, self.hidden_state_size), batch_first=batch_first)
        self.elu1 = nn.ELU()
        
        if self.hidden_context_size is not None:
            self.context = TimeDistributed(nn.Linear(self.hidden_context_size, self.hidden_state_size),batch_first=batch_first)
            
        self.fc2 = TimeDistributed(nn.Linear(self.hidden_state_size,  self.output_size), batch_first=batch_first)
        self.elu2 = nn.ELU()
        
        self.dropout = nn.Dropout(self.dropout)
        self.bn = TimeDistributed(nn.BatchNorm1d(self.output_size),batch_first=batch_first)
        self.gate = TimeDistributed(GLU(self.output_size), batch_first=batch_first)

    def forward(self, x, context=None):

        if self.input_size!=self.output_size:
            residual = self.skip_layer(x)
        else:
            residual = x
        
        x = self.fc1(x)
        if context is not None:
            context = self.context(context)
            x = x+context
        x = self.elu1(x)
        
        x = self.fc2(x)
        x = self.dropout(x)
        x = self.gate(x)
        x = x+residual
        x = self.bn(x)
        
        return x

class PositionalEncoder(torch.nn.Module):
    def __init__(self, d_model, max_seq_len=160):
        super().__init__()
        self.d_model = d_model
        pe = torch.zeros(max_seq_len, d_model)
        for pos in range(max_seq_len):
            for i in range(0, d_model, 2):
                pe[pos, i] = \
                    math.sin(pos / (10000 ** ((2 * i) / d_model)))
                pe[pos, i + 1] = \
                    math.cos(pos / (10000 ** ((2 * (i + 1)) / d_model)))
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        with torch.no_grad():
            x = x * math.sqrt(self.d_model)
            seq_len = x.size(0)
            pe = self.pe[:, :seq_len].view(seq_len,1,self.d_model)
            x = x + pe
            return x

class VariableSelectionNetwork(nn.Module):
    def __init__(self, input_size, num_inputs, hidden_size, dropout, context=None):
        super(VariableSelectionNetwork, self).__init__()

        self.hidden_size = hidden_size
        self.input_size =input_size
        self.num_inputs = num_inputs
        self.dropout = dropout
        self.context=context

        if self.context is not None:
            self.flattened_grn = GatedResidualNetwork(self.num_inputs*self.input_size, self.hidden_size, self.num_inputs, self.dropout, self.context)
        else:
            self.flattened_grn = GatedResidualNetwork(self.num_inputs*self.input_size, self.hidden_size, self.num_inputs, self.dropout)


        self.single_variable_grns = nn.ModuleList()
        for i in range(self.num_inputs):
            self.single_variable_grns.append(GatedResidualNetwork(self.input_size, self.hidden_size, self.hidden_size, self.dropout))

        self.softmax = nn.Softmax(dim=2)

    def forward(self, embedding, context=None):
        if context is not None:
            sparse_weights = self.flattened_grn(embedding, context)
        else:
            sparse_weights = self.flattened_grn(embedding)
    

        sparse_weights = self.softmax(sparse_weights).unsqueeze(2)

        var_outputs = []
        for i in range(self.num_inputs):
            ##select slice of embedding belonging to a single input
            var_outputs.append(self.single_variable_grns[i](embedding[:,:, (i*self.input_size) : (i+1)*self.input_size]))

        var_outputs = torch.stack(var_outputs, axis=-1)

        outputs = var_outputs*sparse_weights
        
        outputs = outputs.sum(axis=-1)

        return outputs, sparse_weights



class TFTDiscriminator(nn.Module):
    def __init__(self, config, multi_variates, batch_size, seq_length,device, output_size):
        super().__init__()
        self.device = device
        self.batch_size=batch_size
        self.hidden_size = config['lstm_hidden_dimension']
        self.lstm_layers = config['lstm_layers']
        self.dropout = config['dropout']
        self.embedding_dim = config['embedding_dim']
        self.attn_heads = config['attn_heads']
        self.multi_variates = multi_variates
        self.seq_length = seq_length
        self.output_size = output_size
        
     
           
        self.time_varying_linear_layers = nn.ModuleList()
        for i in range(self.multi_variates):
            emb = TimeDistributed(nn.Linear(1, self.embedding_dim), batch_first=True).to(self.device)
            self.time_varying_linear_layers.append(emb)

        self.encoder_variable_selection = VariableSelectionNetwork(self.embedding_dim,
                                (self.multi_variates),
                                self.hidden_size,
                                self.dropout,
                                None)

        
        self.lstm_encoder_input_size = self.embedding_dim*(self.multi_variates)
       
                                      

        self.lstm_encoder = nn.LSTM(input_size=self.hidden_size, 
                            hidden_size=self.hidden_size,
                           num_layers=self.lstm_layers,
                           dropout=self.dropout)
        
    

        self.post_lstm_gate = TimeDistributed(GLU(self.hidden_size))
        self.post_lstm_norm = TimeDistributed(nn.BatchNorm1d(self.hidden_size))
        
        self.position_encoding = PositionalEncoder(self.hidden_size, self.seq_length)

        self.multihead_attn = nn.MultiheadAttention(self.hidden_size, self.attn_heads)
        self.post_attn_gate = TimeDistributed(GLU(self.hidden_size))

        self.post_attn_norm = TimeDistributed(nn.BatchNorm1d(self.hidden_size, self.hidden_size))
        self.pos_wise_ff = GatedResidualNetwork(self.hidden_size, self.hidden_size, self.hidden_size, self.dropout)

        self.pre_output_norm = TimeDistributed(nn.BatchNorm1d(self.hidden_size, self.hidden_size))
        self.pre_output_gate = TimeDistributed(GLU(self.hidden_size))

        self.output_layer = TimeDistributed(nn.Linear(self.hidden_size*self.seq_length, self.output_size), batch_first=True)
        
        self.sigmoid = nn.Sigmoid()
        
        self.softmax = nn.Softmax(dim=1)
        
    def init_hidden(self, batch_size):
        return torch.zeros(self.lstm_layers, batch_size, self.hidden_size, device=self.device)
        
    def apply_embedding(self, x, static_embedding, apply_masking):
        ###x should have dimensions (batch_size, timesteps, input_size)
        ## Apply masking is used to mask variables that should not be accessed after the encoding steps
        #Time-varying real embeddings 
        if apply_masking:
            time_varying_real_vectors = []
            """for i in range(self.time_varying_real_variables_decoder):
                emb = self.time_varying_linear_layers[i+self.num_input_series_to_mask](x[:,:,i+self.num_input_series_to_mask].view(x.size(0), -1, 1))
                time_varying_real_vectors.append(emb)"""
            time_varying_real_embedding = torch.cat(time_varying_real_vectors, dim=2)

        else: 
            time_varying_real_vectors = []
            for i in range(self.multi_variates):
                emb = self.time_varying_linear_layers[i](x[:,:,i].view(x.size(0), -1, 1))
                time_varying_real_vectors.append(emb)
            time_varying_real_embedding = torch.cat(time_varying_real_vectors, dim=2)
        
        
        
        embeddings = torch.cat([time_varying_real_embedding], dim=2)
        
        return embeddings.permute(1,0,2)
        
        return embeddings.view(-1,x.size(0),embeddings.size(2))
    
    def encode(self, x, hidden=None):
    
        if hidden is None:
            hidden = self.init_hidden(x.shape[1])
            
 
        output, (hidden, cell) = self.lstm_encoder(x, (hidden, hidden))
        
        return output, hidden
        
    def decode(self, x, hidden=None):
        
        if hidden is None:
            hidden = self.init_hidden()
            
        output, (hidden, cell) = self.lstm_decoder(x, (hidden,hidden))
        
        return output, hidden
    

    def forward(self, input):

        ##inputs should be in this order
            # static
            # time_varying_categorical
            # time_varying_real
                    
        x, cond_input = input
        
            
        if cond_input is not None:
        
            cond_input_reshaped = torch.stack([cond_input]*self.seq_length,axis=1)

            x = torch.cat((x,cond_input_reshaped),axis=2)
        
        
       # x = input

        embedding_vectors = []
                                                                
        ##Embedding and variable selection
        
        
        
        
        embeddings_encoder = self.apply_embedding(x[:,:self.seq_length,:].float().to(self.device), None, apply_masking=False)
        
        
        embeddings_encoder, encoder_sparse_weights = self.encoder_variable_selection(embeddings_encoder[:,:,:])
        

        
    

        
        pe = self.position_encoding(torch.zeros(self.seq_length, 1, embeddings_encoder.size(2)).to(self.device)).to(self.device)
        
        embeddings_encoder = embeddings_encoder+pe[:self.seq_length,:,:]
        


        ##LSTM
        lstm_input = torch.cat([embeddings_encoder], dim=0)
        encoder_output, hidden = self.encode(embeddings_encoder)

        lstm_output = encoder_output

        ##skip connection over lstm
        lstm_output = self.post_lstm_gate(lstm_output+lstm_input)

        ##skip connection over lstm
        attn_input = self.post_lstm_norm(lstm_output)

        #attn_input = self.position_encoding(attn_input)

        ##Attention
        attn_output, attn_output_weights = self.multihead_attn(attn_input[:self.seq_length,:,:], attn_input[:self.seq_length,:,:], attn_input[:self.seq_length,:,:])

        ##skip connection over attention
        attn_output = self.post_attn_gate(attn_output) + attn_input[:self.seq_length,:,:] 
        attn_output = self.post_attn_norm(attn_output)
        
        output = self.pos_wise_ff(attn_output) #[self.encode_length:,:,:])

        ##skip connection over Decoder
        
        

        output = self.pre_output_gate(output) + lstm_output[:self.seq_length,:,:]

        #Final output layers
        output = self.pre_output_norm(output)
        #print(output.shape)
        #print(self.output_layer)
        
    
        output = output.permute(1,0,2)
        
        
        output = self.output_layer(output.reshape(output.shape[0], -1))
        
        if self.output_size > 1:
            
            return self.softmax(output)
        
        return self.sigmoid(output)
        #return F.softmax(output, dim=1)
        #return F.log_softmax(output, dim=1)
        

class TFTGenerator(nn.Module):
    def __init__(self, config, z_latent_dim, multi_variates, batch_size, seq_length,output_function,device):
        super().__init__()
        self.device = device
        self.batch_size=batch_size
        self.hidden_size = config['lstm_hidden_dimension']
        self.lstm_layers = config['lstm_layers']
        self.dropout = config['dropout']
        self.embedding_dim = config['embedding_dim']
        self.attn_heads = config['attn_heads']
        self.z_latent_dim = z_latent_dim
        self.multi_variates = multi_variates
        self.seq_length = seq_length
        self.output_function = output_function
        
     
           
        self.time_varying_linear_layers = nn.ModuleList()
        for i in range(self.z_latent_dim):
            emb = TimeDistributed(nn.Linear(1, self.embedding_dim), batch_first=True).to(self.device)
            self.time_varying_linear_layers.append(emb)

        self.encoder_variable_selection = VariableSelectionNetwork(self.embedding_dim,
                                (self.z_latent_dim),
                                self.hidden_size,
                                self.dropout,
                                None)

        
        self.lstm_encoder_input_size = self.embedding_dim*(self.z_latent_dim)
       
                                      

        self.lstm_encoder = nn.LSTM(input_size=self.hidden_size, 
                            hidden_size=self.hidden_size,
                           num_layers=self.lstm_layers,
                           dropout=self.dropout)
        
    

        self.post_lstm_gate = TimeDistributed(GLU(self.hidden_size))
        self.post_lstm_norm = TimeDistributed(nn.BatchNorm1d(self.hidden_size))
        
        self.position_encoding = PositionalEncoder(self.hidden_size, self.seq_length)

        self.multihead_attn = nn.MultiheadAttention(self.hidden_size, self.attn_heads)
        self.post_attn_gate = TimeDistributed(GLU(self.hidden_size))

        self.post_attn_norm = TimeDistributed(nn.BatchNorm1d(self.hidden_size, self.hidden_size))
        self.pos_wise_ff = GatedResidualNetwork(self.hidden_size, self.hidden_size, self.hidden_size, self.dropout)

        self.pre_output_norm = TimeDistributed(nn.BatchNorm1d(self.hidden_size, self.hidden_size))
        self.pre_output_gate = TimeDistributed(GLU(self.hidden_size))

        self.output_layer = TimeDistributed(nn.Linear(self.hidden_size, self.multi_variates), batch_first=True)
        
        #self.sigmoid = nn.Sigmoid()
        #self.sigmoid = nn.Tanh()
        
    def init_hidden(self, batch_size):
        return torch.zeros(self.lstm_layers, batch_size, self.hidden_size, device=self.device)
        
    def apply_embedding(self, x, static_embedding, apply_masking):
        ###x should have dimensions (batch_size, timesteps, input_size)
        ## Apply masking is used to mask variables that should not be accessed after the encoding steps
        #Time-varying real embeddings 
        
        
        if apply_masking:
            time_varying_real_vectors = []
            """for i in range(self.time_varying_real_variables_decoder):
                emb = self.time_varying_linear_layers[i+self.num_input_series_to_mask](x[:,:,i+self.num_input_series_to_mask].view(x.size(0), -1, 1))
                time_varying_real_vectors.append(emb)"""
            time_varying_real_embedding = torch.cat(time_varying_real_vectors, dim=2)

        else: 
            time_varying_real_vectors = []
            for i in range(self.z_latent_dim):
                emb = self.time_varying_linear_layers[i](x[:,:,i].view(x.size(0), -1, 1))
                time_varying_real_vectors.append(emb)
            time_varying_real_embedding = torch.cat(time_varying_real_vectors, dim=2)
        
        
        
        embeddings = torch.cat([time_varying_real_embedding], dim=2)
        
        return embeddings.permute(1,0,2)
        
        return embeddings.view(-1,x.size(0),embeddings.size(2))
    
    def encode(self, x, hidden=None):
    
        if hidden is None:
            hidden = self.init_hidden(x.shape[1])
            
 
        output, (hidden, cell) = self.lstm_encoder(x, (hidden, hidden))
        
        return output, hidden
        
    def decode(self, x, hidden=None):
        
        if hidden is None:
            hidden = self.init_hidden()
            
        output, (hidden, cell) = self.lstm_decoder(x, (hidden,hidden))
        
        return output, hidden
    

    def forward(self, input):

        ##inputs should be in this order
            # static
            # time_varying_categorical
            # time_varying_real
            
        x, cond_input = input
        
            
        if cond_input is not None:
        
            cond_input_reshaped = torch.stack([cond_input]*self.seq_length,axis=1)

            x = torch.cat((x,cond_input_reshaped),axis=2)
        
       # x = input

        embedding_vectors = []
                                                                
        ##Embedding and variable selection
        
        
        
        
        embeddings_encoder = self.apply_embedding(x[:,:self.seq_length,:].float().to(self.device), None, apply_masking=False)
        
        embeddings_encoder, encoder_sparse_weights = self.encoder_variable_selection(embeddings_encoder[:,:,:])
        
    

        
        pe = self.position_encoding(torch.zeros(self.seq_length, 1, embeddings_encoder.size(2)).to(self.device)).to(self.device)
        
        embeddings_encoder = embeddings_encoder+pe[:self.seq_length,:,:]
        

        ##LSTM
        lstm_input = torch.cat([embeddings_encoder], dim=0)
        encoder_output, hidden = self.encode(embeddings_encoder)

        lstm_output = encoder_output

        ##skip connection over lstm
        lstm_output = self.post_lstm_gate(lstm_output+lstm_input)


        ##skip connection over lstm
        attn_input = self.post_lstm_norm(lstm_output)

        #attn_input = self.position_encoding(attn_input)

        ##Attention
        attn_output, attn_output_weights = self.multihead_attn(attn_input[:self.seq_length,:,:], attn_input[:self.seq_length,:,:], attn_input[:self.seq_length,:,:])

        ##skip connection over attention
        attn_output = self.post_attn_gate(attn_output) + attn_input[:self.seq_length,:,:] 
        attn_output = self.post_attn_norm(attn_output)
        
        output = self.pos_wise_ff(attn_output) #[self.encode_length:,:,:])

        ##skip connection over Decoder
        
        

        output = self.pre_output_gate(output) + lstm_output[:self.seq_length,:,:]

        #Final output layers
        output = self.pre_output_norm(output)
        #print(output.shape)
        #print(self.output_layer)
        
    
        output = output.permute(1,0,2)
        
        
        output = self.output_layer(output)
        
        if self.output_function:
            output = self.output_function(output)
        
        return output
        

        #return F.softmax(output, dim=1)
        #return F.log_softmax(output, dim=1)
        