import torch
import torch.nn as nn


class SpectralConv1d(nn.Module):
    """
    1D spectral convolution layer used in FNO.
    """
    def __init__(self, in_channels, out_channels, modes):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes = modes
        scale = 1.0 / (in_channels * out_channels)
        self.weights = nn.Parameter(
            scale * torch.randn(in_channels, out_channels, modes, dtype=torch.cfloat)
        )

    def _compl_mul1d(self, a, b):
        # (batch, in_channels, modes) * (in_channels, out_channels, modes)
        return torch.einsum("bim,iom->bom", a, b)

    def forward(self, x):
        # x: (batch, channels, x)
        batch, _, n = x.shape
        x_ft = torch.fft.rfft(x, dim=-1)

        out_ft = torch.zeros(
            batch,
            self.out_channels,
            x_ft.size(-1),
            device=x.device,
            dtype=torch.cfloat,
        )
        out_ft[:, :, : self.modes] = self._compl_mul1d(
            x_ft[:, :, : self.modes], self.weights
        )

        x = torch.fft.irfft(out_ft, n=n, dim=-1)
        return x


class FNO1D(nn.Module):
    """
    Standard 1D Fourier Neural Operator for next-step prediction.
    Input shape: (batch, x, channels)
    Output shape: (batch, x, channels)
    """
    def __init__(
        self,
        modes,
        width,
        depth,
        input_channels=2,
        output_channels=1,
        fc_dim=128,
    ):
        super().__init__()
        self.modes = modes
        self.width = width
        self.depth = depth
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.fc_dim = fc_dim

        self.fc0 = nn.Linear(input_channels, width)
        self.convs = nn.ModuleList(
            [SpectralConv1d(width, width, modes) for _ in range(depth)]
        )
        self.ws = nn.ModuleList(
            [nn.Conv1d(width, width, kernel_size=1) for _ in range(depth)]
        )
        self.fc1 = nn.Linear(width, fc_dim)
        self.fc2 = nn.Linear(fc_dim, output_channels)
        self.act = nn.GELU()

    def forward(self, x):
        # x: (batch, x, channels)
        x = self.fc0(x)
        x = x.permute(0, 2, 1)
        for conv, w in zip(self.convs, self.ws):
            x = self.act(conv(x) + w(x))
        x = x.permute(0, 2, 1)
        x = self.act(self.fc1(x))
        x = self.fc2(x)
        return x


class ConditionalFNO1D(nn.Module):
    """
    1D Fourier Neural Operator with FiLM-style conditioning.
    """
    def __init__(
        self,
        modes,
        width,
        depth,
        input_channels=2,
        output_channels=1,
        fc_dim=128,
        cond_dim=0,
        cond_hidden=128,
    ):
        super().__init__()
        self.modes = modes
        self.width = width
        self.depth = depth
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.fc_dim = fc_dim
        self.cond_dim = int(cond_dim)

        self.fc0 = nn.Linear(input_channels, width)
        self.convs = nn.ModuleList(
            [SpectralConv1d(width, width, modes) for _ in range(depth)]
        )
        self.ws = nn.ModuleList(
            [nn.Conv1d(width, width, kernel_size=1) for _ in range(depth)]
        )
        self.fc1 = nn.Linear(width, fc_dim)
        self.fc2 = nn.Linear(fc_dim, output_channels)
        self.act = nn.GELU()

        self.cond_mlp = None
        if self.cond_dim > 0:
            self.cond_mlp = nn.Sequential(
                nn.Linear(self.cond_dim, cond_hidden),
                nn.GELU(),
                nn.Linear(cond_hidden, depth * 2 * width),
            )
            nn.init.zeros_(self.cond_mlp[-1].weight)
            nn.init.zeros_(self.cond_mlp[-1].bias)

    def forward(self, x, cond=None):
        # x: (batch, x, channels)
        batch = x.size(0)
        film = None
        if self.cond_mlp is not None:
            if cond is None:
                cond = torch.zeros(batch, self.cond_dim, device=x.device, dtype=x.dtype)
            film = self.cond_mlp(cond).view(batch, self.depth, 2, self.width)

        x = self.fc0(x)
        x = x.permute(0, 2, 1)
        for idx, (conv, w) in enumerate(zip(self.convs, self.ws)):
            h = conv(x) + w(x)
            if film is not None:
                gamma = film[:, idx, 0].unsqueeze(-1)
                beta = film[:, idx, 1].unsqueeze(-1)
                h = h * (1.0 + gamma) + beta
            x = self.act(h)
        x = x.permute(0, 2, 1)
        x = self.act(self.fc1(x))
        x = self.fc2(x)
        return x
