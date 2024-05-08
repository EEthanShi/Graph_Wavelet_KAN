class NaiveWaveletKANLayer(nn.Module):
    def __init__(self, inputdim, outdim, gridsize=300, addbias=True):
        super(NaiveWaveletKANLayer, self).__init__()
        self.gridsize = gridsize
        self.addbias = addbias
        self.inputdim = inputdim
        self.outdim = outdim
        #self.num_scales = num_scales  # Number of different sigma values

        # Initialize Fourier coefficients
        self.fouriercoeffs = nn.Parameter(torch.randn(outdim, inputdim, gridsize) /
                                          (np.sqrt(inputdim) * np.sqrt(self.gridsize)))
        
        # Initialize sigma parameters for each scale
        self.sigmas = nn.Parameter(torch.ones(1, inputdim, gridsize))
        self.locs = nn.Parameter(torch.randn(1, inputdim, gridsize))

        if self.addbias:
            self.bias = nn.Parameter(torch.zeros(1, outdim))

    def mexican_hat_wavelet(self, x, loc, sigma):
        # Mexican hat wavelet function
        sigma_x = (x - loc) / sigma
        return (2 / (torch.sqrt(torch.tensor(3)) * (torch.pi**0.25))) * (1 - sigma_x**2) * torch.exp(-sigma_x**2 / 2)
    
    def forward(self, x):
        x = x.view(-1, self.inputdim)  #     N x d
        xrshp = x.view(x.shape[0], self.inputdim, 1)   # N x d x 1

        #for i in range(self.num_scales):
        wavelet_transform = self.mexican_hat_wavelet(xrshp, self.locs, self.sigmas)  # Apply wavelet function in shape N x d x G  (G grid number)
        
        
        y = torch.einsum("bdk,odk->bo", wavelet_transform, self.fouriercoeffs)   # N x o

        if self.addbias:
            y += self.bias
        
        y = y.view(-1, self.outdim)  # Reshape to original output shape
        return y
    
    
    
    
