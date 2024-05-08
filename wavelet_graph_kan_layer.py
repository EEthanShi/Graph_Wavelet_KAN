import torch
import torch.nn as nn

import numpy as np

    
class NaiveWaveletKANLayer(nn.Module):
    def __init__(self, inputdim, outdim, gridsize=300, addbias=True, num_scales=3):
        super(NaiveWaveletKANLayer, self).__init__()
        self.gridsize = gridsize
        self.addbias = addbias
        self.inputdim = inputdim
        self.outdim = outdim
        self.num_scales = num_scales  # Number of different sigma values

        # Initialize Fourier coefficients
        self.fouriercoeffs = nn.Parameter(torch.randn(num_scales, outdim, inputdim, gridsize) / 
                                          (np.sqrt(inputdim) * np.sqrt(self.gridsize)))
        
        # Initialize sigma parameters for each scale
        self.sigmas = nn.Parameter(torch.ones(num_scales, 1, 1, gridsize))

        if self.addbias:
            self.bias = nn.Parameter(torch.zeros(1, outdim))

    def mexican_hat_wavelet(self, x, sigma):
        # Mexican hat wavelet function
        sigma_x = x / sigma
        return (2 / (torch.sqrt(torch.tensor(3)) * (torch.pi**0.25))) * (1 - sigma_x**2) * torch.exp(-sigma_x**2 / 2)
    
    def forward(self, x):
        x = x.view(-1, self.inputdim)  #
        xrshp = x.view(x.shape[0], 1, self.inputdim, 1)  

        results = []
        for i in range(self.num_scales):
            wavelet_transform = self.mexican_hat_wavelet(xrshp, self.sigmas[i])  # Apply wavelet function and sum
            results.append(wavelet_transform)
        
        results_reshape = []
        for i in results: 
            results_re = torch.reshape(i, (1, x.shape[0], x.shape[1], self.gridsize))
            results_reshape.append(results_re)
        
        concatenated_results = torch.cat(results_reshape, dim=0)
        
        y = torch.einsum("dbik,djik->bj", concatenated_results, self.fouriercoeffs)

        if self.addbias:
            y += self.bias
        
        y = y.view(-1, self.outdim)  # Reshape to original output shape
        return y
    
    
    
