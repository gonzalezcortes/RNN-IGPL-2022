from pandas.core.accessor import PandasDelegate
import torch #pytorch
import torch.nn as nn
from torch.autograd import Variable
import time
torch.manual_seed(2)

class model_LSTM_1(nn.Module):
    def __init__(self,input_size, n_class, hidden_n, n_layers, dropout_prob):
        super(model_LSTM_1, self).__init__()
        self.hidden_n = hidden_n 
        self.n_layers = n_layers

        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_n, num_layers=n_layers, batch_first=True)
        self.fc =  nn.Linear(hidden_n, n_class)

    def forward(self,x):
        h0 = torch.zeros(self.n_layers, x.size(0), self.hidden_n).requires_grad_()
        c0 = torch.zeros(self.n_layers, x.size(0), self.hidden_n).requires_grad_()
        output, _ = self.lstm(x, (h0.detach(), c0.detach())) #output, (hn, cn)
        
        out = output[:, -1, :]
        out = self.fc(out)
        return out

class model_LSTM_2(nn.Module):
    def __init__(self,input_size, n_class, hidden_n, n_layers, dropout_prob):
        super(model_LSTM_2, self).__init__()
        self.hidden_n = hidden_n 
        self.n_layers = n_layers

        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_n, num_layers=n_layers, batch_first=True)
        self.fc_1 =  nn.Linear(hidden_n, 128)
        self.relu = nn.ReLU()
        self.fc_2 = nn.Linear(128, n_class)

    def forward(self,x):
        h0 = torch.zeros(self.n_layers, x.size(0), self.hidden_n).requires_grad_()
        c0 = torch.zeros(self.n_layers, x.size(0), self.hidden_n).requires_grad_()
        output, _ = self.lstm(x, (h0.detach(), c0.detach())) 
        
        out = output[:, -1, :]
        out = self.fc_1(out)
        out = self.relu(out)
        out = self.fc_2(out)
        return out

class model_GRU_1(nn.Module):
    def __init__(self,input_size, n_class, hidden_n, n_layers, dropout_prob):
        super(model_GRU_1, self).__init__()
        self.hidden_n = hidden_n
        self.n_layers = n_layers

        self.gru = nn.GRU(input_size=input_size, hidden_size=hidden_n, num_layers=n_layers, batch_first=True)
        self.fc = nn.Linear(hidden_n, n_class)

    def forward(self, x):
        h0 = torch.zeros(self.n_layers, x.size(0), self.hidden_n).requires_grad_()
        output, _ = self.gru(x, h0.detach())
        out = output[:, -1, :]
        out = self.fc(out)
        return out

class model_RNN_1(nn.Module):
    def __init__(self,input_size, n_class, hidden_n, n_layers, dropout_prob):
        super(model_RNN_1, self).__init__()
        self.hidden_n = hidden_n
        self.n_layers = n_layers

        self.rnn = nn.RNN(input_size=input_size, hidden_size=hidden_n, num_layers=n_layers, batch_first=True)
        self.fc = nn.Linear(hidden_n, n_class)

    def forward(self, x):
        h0 = torch.zeros(self.n_layers, x.size(0), self.hidden_n).requires_grad_()
        output, _ = self.rnn(x, h0.detach())
        out = output[:, -1, :]
        out = self.fc(out)
        return out

class training:
    def convert_tensor(a,b,c,d):
        return Variable(torch.Tensor(a)), Variable(torch.Tensor(b)), Variable(torch.Tensor(c)), Variable(torch.Tensor(d))

    def reshape_3d(data, dimension_0, dimension_1, dimension_2):
        return torch.reshape(data, (dimension_0, dimension_1, dimension_2))

    def training_iteration(num_epochs, learning_rate, model_x, X_train_tensors, X_test_tensor, y_train_tensors, silent_mode):
        t0 = time.time()
        criterion = torch.nn.MSELoss()
        optimizer = torch.optim.Adam(model_x.parameters(), lr=learning_rate) 

        for epoch in range(num_epochs):
            outputs = model_x.forward(X_train_tensors)
            optimizer.zero_grad()

            loss = criterion(outputs, y_train_tensors)
            loss.backward()
            optimizer.step()
            if silent_mode != True:
                if epoch % 100 == 0:
                    print("Epoch: %d, loss: %1.5f" % (epoch, loss.item())) 

        prediction = model_x(X_test_tensor)
        running_time = time.time()-t0
        return prediction.data.numpy(), running_time