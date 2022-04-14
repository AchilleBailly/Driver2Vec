import torch
from torch import nn
from pytorch_wavelets import DWT1DForward

class HaarWavelet(nn.Module):
    def __init__(self, input_channels, input_length, output_length):
        super(HaarWavelet, self).__init__()
        self.input_channels = input_channels
        self.input_length = input_length
        self.wavelet_size = input_channels * input_length // 2
        self.dwt = DWT1DForward(J=1, wave="haar", mode="zero")
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(self.wavelet_size, output_length, bias=True)
        self.fc2 = nn.Linear(self.wavelet_size, output_length, bias=True)
        nn.init.kaiming_uniform_(self.fc1.weight)
        nn.init.kaiming_uniform_(self.fc2.weight)
    
    def forward(self, x):
        # Get wavelet transform
        haar_approx, [haar_detail] = self.dwt(x)
        haar_approx, haar_detail = self.flatten(haar_approx), self.flatten(haar_detail)
        # print(f"HaarA: {haar_approx.shape} | HaarD: {haar_detail.shape}")
        # Pass through fully connected layer
        out1, out2 = self.fc1(haar_approx), self.fc2(haar_detail)
        return torch.cat([out1, out2], dim=1)