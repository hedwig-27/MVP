import torch
import torch.nn as nn

from models.fno1d import FNO1D, ConditionalFNO1D


class PrimitiveOperator(nn.Module):
    """
    Anonymous primitive operator: predicts delta_u for one-step update.
    """
    def __init__(
        self,
        modes,
        width,
        depth,
        input_channels,
        output_channels,
        fc_dim=128,
        primitive_type="fno",
        kernel_size=5,
        cond_dim=0,
    ):
        super().__init__()
        self.output_channels = output_channels
        self.primitive_type = primitive_type
        self.cond_dim = int(cond_dim)

        if primitive_type == "fno":
            if self.cond_dim > 0:
                self.net = ConditionalFNO1D(
                    modes=modes,
                    width=width,
                    depth=depth,
                    input_channels=input_channels,
                    output_channels=output_channels,
                    fc_dim=fc_dim,
                    cond_dim=self.cond_dim,
                )
            else:
                self.net = FNO1D(
                    modes=modes,
                    width=width,
                    depth=depth,
                    input_channels=input_channels,
                    output_channels=output_channels,
                    fc_dim=fc_dim,
                )
        elif primitive_type == "cnn":
            padding = kernel_size // 2
            layers = []
            in_ch = input_channels
            for _ in range(max(depth, 1)):
                layers.append(nn.Conv1d(in_ch, width, kernel_size=kernel_size, padding=padding))
                layers.append(nn.GELU())
                in_ch = width
            layers.append(nn.Conv1d(in_ch, output_channels, kernel_size=kernel_size, padding=padding))
            self.net = nn.Sequential(*layers)
        else:
            raise ValueError(f"Unknown primitive_type: {primitive_type}")

    def forward(self, u_t, cond=None):
        if self.primitive_type == "cnn":
            x = u_t.permute(0, 2, 1)
            pred_next = self.net(x).permute(0, 2, 1)
        else:
            if self.cond_dim > 0:
                if cond is None:
                    cond = torch.zeros(u_t.size(0), self.cond_dim, device=u_t.device, dtype=u_t.dtype)
                pred_next = self.net(u_t, cond)
            else:
                pred_next = self.net(u_t)
        u_state = u_t[..., : self.output_channels]
        return pred_next - u_state
