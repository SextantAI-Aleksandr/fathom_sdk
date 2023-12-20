''' Neural networks come in many strange shapes and sizes
See https://www.youtube.com/watch?v=oJNHXPs0XDk
This provides a repository where they can live 
You can generate a model by importing architectures and using getattr
EXAMPLE:
import architectures
for model_name, args, kwargs in [
    ('Basic_4twlffd',  [len(input_tickers)*seq_length, len(output_tickers)],  {}),
    ('LSTM_w5ekck5',   [len(input_tickers), 32, 5, len(output_tickers)],      {}) 
]:
    model = getattr(architectures, model_name)(*args, **kwargs)
'''

#~~~IMPORT~LIBRARIES~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
import torch
import torch.nn as nn
from random import random, randrange
from torch.autograd import Variable 
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~



#~~~Fully~Connected~Models~~~~~~~~~~~~~~~~~~~~~~~~~~

# string constant for import 
MODEL_Linear_8gt1ab7 = 'Linear_8gt1ab7'

class Linear_8gt1ab7(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(Linear_8gt1ab7, self).__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(input_dim, output_dim),
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits


# string constant for import 
MODEL_Basic_4twlffd = 'Basic_4twlffd'

class Basic_4twlffd(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(Basic_4twlffd, self).__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(input_dim, 5),
            nn.Tanh(),
            nn.Linear(5, output_dim),
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits

# string constant for import 
MODEL_Basic_lc6kmf3 = 'Basic_lc6kmf3'

class Basic_lc6kmf3(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(Basic_lc6kmf3, self).__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(input_dim, 24),
            nn.Tanh(),
            nn.Linear(24, 12),
            nn.Tanh(),
            nn.Linear(12, 7),
            nn.Tanh(),
            nn.Linear(7, output_dim),
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits


# string constant for import 
MODEL_Basic_dr4pkX8 = 'Basic_dr4pkX8'

class Basic_dr4pkX8(nn.Module):
    # Like Basic_lc6kmf3 but with dropout
    # See https://blog.floydhub.com/long-short-term-memory-from-zero-to-hero-with-pytorch/
    def __init__(self, input_dim, output_dim, drop_prob=0.6):
        super(Basic_dr4pkX8, self).__init__()
        self.flatten = nn.Flatten()
        #self.dropout =  nn.Dropout(drop_prob)
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(input_dim, 24),
            nn.Dropout(drop_prob),
            nn.Tanh(),
            nn.Linear(24, 12),
            nn.Dropout(drop_prob),
            nn.Tanh(),
            nn.Linear(12, 7),
            nn.Dropout(drop_prob),
            nn.Tanh(),
            nn.Linear(7, output_dim),
            nn.Dropout(drop_prob)
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


#~~~LSTM~Models~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# string constant for import 
MODEL_LSTM_w5ekck5 = 'LSTM_w5ekck5'

class LSTM_w5ekck5(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim):
        super(LSTM_w5ekck5, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).requires_grad_()
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).requires_grad_()
        out, (hn, cn) = self.lstm(x, (h0.detach(), c0.detach()))
        out = self.fc(out[:, -1, :]) 
        return out

# string constant for import 
MODEL_LSTM_drX6pt6 = 'LSTM_drX6pt6'

class LSTM_drX6pt6(nn.Module):
    # This is like LSTM_w5ekck5 but with drouout
    # See https://blog.floydhub.com/long-short-term-memory-from-zero-to-hero-with-pytorch/
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim, drop_prob=0.6):
        super(LSTM_drX6pt6, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(drop_prob) # added this line....

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).requires_grad_()
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).requires_grad_()
        out, (hn, cn) = self.lstm(x, (h0.detach(), c0.detach()))
        out = self.dropout(out) # ... and this line...
        out = self.fc(out[:, -1, :]) 
        return out
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


#~~~ARCHITECTURE~SELECTION~~~~~~~~~~~~~~~~~~~~~~~~~
def list_possible_models(input_tickers, output_tickers, seq_length, prediction_deltas, max_lstm_layer_width, max_lstm_layer_count):
    # This function takes a few key parameters as input and stochastically selects and architecture
    possible_models = [
        (MODEL_Linear_8gt1ab7,  [len(input_tickers)*seq_length, len(output_tickers)*len(prediction_deltas)],  {}),
        (MODEL_Basic_4twlffd,   [len(input_tickers)*seq_length, len(output_tickers)*len(prediction_deltas)],  {}),
        (MODEL_Basic_lc6kmf3,   [len(input_tickers)*seq_length, len(output_tickers)*len(prediction_deltas)],  {}),
        (MODEL_Basic_dr4pkX8,   [len(input_tickers)*seq_length, len(output_tickers)*len(prediction_deltas)],  {'drop_prob':1-0.5*random()}),
        # The LSTM duplicates are because LSTMs work well
        (MODEL_LSTM_w5ekck5,    [len(input_tickers), 1+randrange(max_lstm_layer_width), 1+randrange(max_lstm_layer_count), len(output_tickers)*len(prediction_deltas)],      {}),
        (MODEL_LSTM_drX6pt6,    [len(input_tickers), 1+randrange(max_lstm_layer_width), 1+randrange(max_lstm_layer_count), len(output_tickers)*len(prediction_deltas)],      {'drop_prob':1-0.5*random()}),
        (MODEL_LSTM_drX6pt6,    [len(input_tickers), 1+randrange(max_lstm_layer_width), 1+randrange(max_lstm_layer_count), len(output_tickers)*len(prediction_deltas)],      {'drop_prob':1-0.5*random()}),
        (MODEL_LSTM_drX6pt6,    [len(input_tickers), 1+randrange(max_lstm_layer_width), 1+randrange(max_lstm_layer_count), len(output_tickers)*len(prediction_deltas)],      {'drop_prob':1-0.5*random()})
    ]
    return possible_models
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
