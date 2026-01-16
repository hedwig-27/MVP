import torch.nn as nn

from models.fno1d import FNO1D


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
    ):
        super().__init__()
        self.output_channels = output_channels
        self.primitive_type = primitive_type

        if primitive_type == "fno":
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

    def forward(self, u_t):
        if self.primitive_type == "cnn":
            x = u_t.permute(0, 2, 1)
            pred_next = self.net(x).permute(0, 2, 1)
        else:
            pred_next = self.net(u_t)
        u_state = u_t[..., : self.output_channels]
        return pred_next - u_state
