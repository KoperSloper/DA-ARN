import torch
from torch import nn
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class AttentionEncoder(nn.Module):
    def __init__(self, N, M, T, stateful=False):
        super(self.__class__, self).__init__()
        self.N = N
        self.M = M
        self.T = T

        self.encoder_lstm = nn.LSTMCell(input_size = self.N, hidden_size = self.M)
        
        self.W_e = nn.Linear(2*self.M, self.T)
        self.U_e = nn.Linear(self.T, self.T, bias=False)
        self.v_e = nn.Linear(self.T, 1, bias=False)

    def forward(self, inputs):
        encoded_inputs = torch.zeros((inputs.shape[0], self.T, self.M)).to(device)

        h_tm1 = torch.zeros((inputs.shape[0], self.M)).to(device)
        s_tm1 = torch.zeros((inputs.shape[0], self.M)).to(device)
        
        for t in range(self.T):
            # has dimension (batch_size, self.M*2)
            h_c_concat = torch.cat((h_tm1, s_tm1), dim=1)

            # W_e output of (batch_size, self.T)
            # unsqueeze adds extra dimension at index 1 so shape is (batch_size, 1, self.T)
            # repeat repeats dimension at index 1 self.N times so shape is (batch_size, self.N, self.T)
            x = self.W_e(h_c_concat).unsqueeze(1).repeat(1, self.N, 1)

            # input has shape (batch_size, self.T, self.N)
            # permute changes it to (batch_size, self.N, self.T)
            # output of y is shape (batch_size, self.N, self.T)
            y = self.U_e(inputs.permute(0, 2, 1))
            
            # tanh is applied elementwise and independent of other elements
            z = torch.tanh(x + y)

            # after v_e is applied, z has shape (batch_size, self.N, 1)
            # squeeze removes the last dimension so shape is (batch_size, self.N)
            e_k_t = torch.squeeze(self.v_e(z))
 
            # applies softmax along dimension 1 so shape is (batch_size, self.N)
            alpha_k_t = F.softmax(e_k_t, dim=1)

            # inputs[:, t, :] selects all batches, at time t, all features → shape (batch_size, self.N)
            weighted_inputs = alpha_k_t*inputs[:, t, :]

            # compute the new hidden state and cell state
            h_tm1, s_tm1 = self.encoder_lstm(weighted_inputs, (h_tm1, s_tm1))

            encoded_inputs[:, t, :] = h_tm1

        # encoded_inputs has shape (batch_size, self.T, self.M)
        return encoded_inputs
    
class TemporalAttentionDecoder(nn.Module):
    def __init__(self, M, P, T):
        super(self.__class__, self).__init__()
        # M is the number of hidden units in the encoder
        self.M = M
        # P is the number of hidden units in the decoder
        self.P = P
        # T is the number of time steps
        self.T = T

        self.decoder_lstm = nn.LSTMCell(input_size = 1, hidden_size = self.P)

        self.W_d = nn.Linear(2*self.P, self.M)
        self.U_d = nn.Linear(self.M, self.M, bias=False)
        self.v_d = nn.Linear(self.M, 1, bias=False)

        self.W_tilda = nn.Linear(self.M+1, 1)

        self.W_y = nn.Linear(self.P+self.M, self.P)
        self.v_y = nn.Linear(self.P, 1)

    def forward(self, encoded_inputs, y):
        d_tm1 = torch.zeros(encoded_inputs.shape[0], self.P).to(device)
        c_tm1 = torch.zeros(encoded_inputs.shape[0], self.P).to(device)

        for t in range(self.T):
            # has dimension (batch_size, 2*self.P)
            d_s_concat = torch.cat((d_tm1, c_tm1), dim=1)

            # W_d output of (batch_size, self.M)
            # unsqueeze adds extra dimension at index 1 so shape is (batch_size, 1, self.M)
            # repeat repeats dimension at index 1 self.T times so shape is (batch_size, self.T, self.M)
            x1 = self.W_d(d_s_concat).unsqueeze(1).repeat(1, self.T, 1)

            # y1 has shape (batch_size, self.T, self.M)
            y1 = self.U_d(encoded_inputs)
            z1 = torch.tanh(x1 + y1)

            # after v_d is applied, z1 has shape (batch_size, self.T, 1)
            l_i_t = self.v_d(z1)

            # applies softmax along dimension 1 so shape is (batch_size, self.T, 1)
            b_i_t = F.softmax(l_i_t, dim=1)

            # c_t has shape (batch_size, self.M)
            c_t = torch.sum(b_i_t*encoded_inputs, dim=1)

            # y_c_concat has shape of (batch_size, self.M+1)
            y_c_concat = torch.concat((c_t, y[:,t,:]), dim=1)

            # y_tilda has shape of (batch_size, 1)
            y_tilda = self.W_tilda(y_c_concat)

            d_tm1, c_tm1 = self.decoder_lstm(y_tilda, (d_tm1, c_tm1))

        d_c_concat = torch.cat((d_tm1, c_t), dim=1)
        y_hat = self.v_y(self.W_y(d_c_concat))
        return y_hat

class DARNN(nn.Module):
    def __init__(self, N, M, P, T):
        """
        param: N: int
            number of time series
        param: M: int
            number of encoder LSTM units
        param: P:
            number of deocder LSTM units
        param: T:
            number of timesteps
        """
        super(self.__class__, self).__init__()
        self.encoder = AttentionEncoder(N, M, T).to(device)
        self.decoder = TemporalAttentionDecoder(M, P, T).to(device)
        
    def forward(self, X_history, y_history):
        out = self.decoder(self.encoder(X_history), y_history)
        return out
    
def initialize_model(time_series, hidden_dim1, hidden_dim2, timesteps):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DARNN(time_series, hidden_dim1, hidden_dim2, timesteps).to(device)
    return model