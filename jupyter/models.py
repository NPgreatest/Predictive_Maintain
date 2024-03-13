import torch
import torch.nn as nn


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

class LSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, layer_dim, output_dim):
        super(LSTMModel, self).__init__()
        self.hidden_dim = hidden_dim
        self.layer_dim = layer_dim
        self.lstm = nn.LSTM(input_dim, hidden_dim, layer_dim, batch_first=True)
        self.ln = nn.LayerNorm(hidden_dim)  # Layer normalization for LSTM
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        h0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).to(x.device)
        c0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).to(x.device)
        out, (hn, cn) = self.lstm(x, (h0, c0))
        out = self.ln(out[:, -1, :])  # Apply layer normalization to the output
        out = self.fc(out)
        return out

class BiLSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, layer_dim, output_dim):
        super(BiLSTMModel, self).__init__()
        self.hidden_dim = hidden_dim
        self.layer_dim = layer_dim
        self.lstm = nn.LSTM(input_dim, hidden_dim, layer_dim, batch_first=True, bidirectional=True)
        self.ln = nn.LayerNorm(hidden_dim * 2)  # Layer normalization for BiLSTM
        self.fc = nn.Linear(hidden_dim * 2, output_dim)

    def forward(self, x):
        h0 = torch.zeros(self.layer_dim * 2, x.size(0), self.hidden_dim).to(x.device)
        c0 = torch.zeros(self.layer_dim * 2, x.size(0), self.hidden_dim).to(x.device)
        out, (hn, cn) = self.lstm(x, (h0, c0))
        out = self.ln(out[:, -1, :])  # Apply layer normalization to the output
        out = self.fc(out)
        return out

class GRUModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, layer_dim, output_dim):
        super(GRUModel, self).__init__()
        self.hidden_dim = hidden_dim
        self.layer_dim = layer_dim
        self.gru = nn.GRU(input_dim, hidden_dim, layer_dim, batch_first=True)
        self.ln = nn.LayerNorm(hidden_dim)  # Layer normalization for GRU
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        h0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim).to(x.device)
        out, hn = self.gru(x, h0)
        out = self.ln(out[:, -1, :])  # Apply layer normalization to the output
        out = self.fc(out)
        return out

class BiGRUModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, layer_dim, output_dim):
        super(BiGRUModel, self).__init__()
        self.hidden_dim = hidden_dim
        self.layer_dim = layer_dim
        # 设置双向GRU
        self.gru = nn.GRU(input_dim, hidden_dim, layer_dim, batch_first=True, bidirectional=True)
        # 因为是双向的，所以隐藏层维度是hidden_dim的两倍
        self.fc = nn.Linear(hidden_dim * 2, output_dim)

    def forward(self, x):
        # 初始化隐藏状态
        h0 = torch.zeros(self.layer_dim * 2, x.size(0), self.hidden_dim).to(device)  # *2 是因为双向
        out, hn = self.gru(x, h0)
        # 因为是双向的，所以要调整最后一层的输出
        out = self.fc(out[:, -1, :])  # 取最后时刻的输出
        return out
