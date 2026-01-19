import torch
import torch.nn as nn


class Router(nn.Module):
    """
    Sparse router that selects top-k primitives based on low-dimensional stats of u_t.
    """
    def __init__(
        self,
        num_primitives,
        state_channels,
        hidden_dim=64,
        top_k=2,
        stats=("mean", "std", "min", "max"),
        local_segments=0,
        local_stats=("mean", "std"),
        fft_bins=0,
        equation_dim=0,
        pde_param_dim=0,
        num_datasets=1,
        dataset_embed_dim=8,
    ):
        super().__init__()
        self.num_primitives = num_primitives
        self.state_channels = state_channels
        self.hidden_dim = hidden_dim
        self.top_k = top_k
        self.stats = stats
        self.local_segments = int(local_segments)
        self.local_stats = local_stats
        self.fft_bins = fft_bins
        self.equation_dim = int(equation_dim)
        self.pde_param_dim = pde_param_dim
        self.num_datasets = num_datasets
        self.dataset_embed_dim = dataset_embed_dim

        input_dim = 0
        input_dim += sum(1 for s in stats if s in ("mean", "std", "min", "max", "grad_norm"))
        if self.local_segments > 0 and self.local_stats:
            input_dim += self.local_segments * sum(1 for s in self.local_stats if s in ("mean", "std", "min", "max"))
        if fft_bins > 0:
            input_dim += fft_bins
        if self.equation_dim > 0:
            input_dim += self.equation_dim
        if pde_param_dim > 0:
            input_dim += pde_param_dim
        if num_datasets > 1:
            input_dim += dataset_embed_dim

        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_primitives),
        )
        self.dataset_embed = None
        if num_datasets > 1:
            self.dataset_embed = nn.Embedding(num_datasets, dataset_embed_dim)

    def _compute_stats(self, u_state):
        # u_state: (batch, x, channels)
        feats = []
        if "mean" in self.stats:
            feats.append(u_state.mean(dim=(1, 2)))
        if "std" in self.stats:
            feats.append(u_state.std(dim=(1, 2)))
        if "min" in self.stats:
            feats.append(u_state.amin(dim=(1, 2)))
        if "max" in self.stats:
            feats.append(u_state.amax(dim=(1, 2)))
        if "grad_norm" in self.stats:
            grad = u_state[:, 1:, :] - u_state[:, :-1, :]
            feats.append(grad.abs().mean(dim=(1, 2)))
        if self.local_segments > 0 and self.local_stats:
            segments = torch.chunk(u_state, self.local_segments, dim=1)
            for stat in self.local_stats:
                if stat not in ("mean", "std", "min", "max"):
                    continue
                local_vals = []
                for seg in segments:
                    if stat == "mean":
                        local_vals.append(seg.mean(dim=(1, 2)))
                    elif stat == "std":
                        local_vals.append(seg.std(dim=(1, 2)))
                    elif stat == "min":
                        local_vals.append(seg.amin(dim=(1, 2)))
                    elif stat == "max":
                        local_vals.append(seg.amax(dim=(1, 2)))
                if local_vals:
                    feats.append(torch.stack(local_vals, dim=-1))
        if self.fft_bins > 0:
            fft = torch.fft.rfft(u_state, dim=1)
            mag = torch.abs(fft).mean(dim=2)
            feats.append(mag[:, : self.fft_bins])
        return torch.cat([f.unsqueeze(-1) if f.dim() == 1 else f for f in feats], dim=-1)

    def forward(self, u_state, pde_params=None, dataset_id=None, equation=None, return_topk=False):
        stats_vec = self._compute_stats(u_state)
        feats = [stats_vec]
        if self.equation_dim > 0:
            if equation is None:
                equation = torch.zeros(
                    stats_vec.size(0),
                    self.equation_dim,
                    device=stats_vec.device,
                    dtype=stats_vec.dtype,
                )
            elif equation.dim() == 1:
                equation = equation.unsqueeze(0)
            feats.append(equation)
        if self.pde_param_dim > 0:
            if pde_params is None:
                pde_params = torch.zeros(
                    stats_vec.size(0),
                    self.pde_param_dim,
                    device=stats_vec.device,
                    dtype=stats_vec.dtype,
                )
            feats.append(pde_params)
        if self.dataset_embed is not None:
            if dataset_id is None:
                feats.append(
                    torch.zeros(
                        stats_vec.size(0),
                        self.dataset_embed_dim,
                        device=stats_vec.device,
                        dtype=stats_vec.dtype,
                    )
                )
            else:
                if dataset_id.dim() > 1:
                    dataset_id = dataset_id.view(-1)
                feats.append(self.dataset_embed(dataset_id))
        stats_vec = torch.cat(feats, dim=-1)

        logits = self.mlp(stats_vec)
        if self.top_k < self.num_primitives:
            topk_vals, topk_idx = torch.topk(logits, self.top_k, dim=-1)
            masked = torch.full_like(logits, -1e9)
            masked.scatter_(1, topk_idx, topk_vals)
            weights = torch.softmax(masked, dim=-1)
        else:
            topk_idx = torch.argsort(logits, dim=-1, descending=True)
            weights = torch.softmax(logits, dim=-1)

        if return_topk:
            return weights, topk_idx
        return weights


class PrimitiveAggregator(nn.Module):
    def __init__(self, num_primitives, agg_type="linear", hidden_dim=32):
        super().__init__()
        self.num_primitives = num_primitives
        self.agg_type = agg_type
        self.hidden_dim = hidden_dim
        if agg_type == "mlp":
            self.mlp = nn.Sequential(
                nn.Linear(num_primitives, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, 1),
            )
        elif agg_type != "linear":
            raise ValueError(f"Unknown aggregator type: {agg_type}")

    @property
    def needs_delta_stack(self):
        return self.agg_type == "mlp"

    def forward(self, delta_stack, weights):
        # delta_stack: (B, X, C, P), weights: (B, P)
        weighted = delta_stack * weights.unsqueeze(1).unsqueeze(2)
        if self.agg_type == "linear":
            return weighted.sum(dim=-1)
        b, x, c, p = weighted.shape
        flat = weighted.reshape(b * x * c, p)
        out = self.mlp(flat).reshape(b, x, c)
        return out


class PrimitiveCNO(nn.Module):
    """
    Automatic primitive discovery model.
    u_{t+1} = u_t + sum_i alpha_i * PrimitiveOperator_i(u_t)
    """
    def __init__(self, primitives, router, output_channels, aggregator=None):
        super().__init__()
        self.primitives = nn.ModuleList(primitives)
        self.router = router
        self.output_channels = output_channels
        self.num_primitives = len(primitives)
        self.aggregator = aggregator

    def forward(
        self,
        u_t,
        pde_params=None,
        dataset_id=None,
        equation=None,
        return_weights=False,
        return_deltas=False,
        sparse_primitives=False,
    ):
        u_state = u_t[..., : self.output_channels]
        weights, topk_idx = self.router(u_state, pde_params, dataset_id, equation, return_topk=True)
        need_delta_stack = return_deltas or (self.aggregator and self.aggregator.needs_delta_stack)

        if sparse_primitives and self.router.top_k < self.num_primitives:
            delta_sum = torch.zeros_like(u_state)
            delta_stack = None
            if need_delta_stack:
                delta_stack = torch.zeros(
                    u_state.size(0),
                    u_state.size(1),
                    u_state.size(2),
                    self.num_primitives,
                    device=u_state.device,
                    dtype=u_state.dtype,
                )
            used = torch.unique(topk_idx).tolist()
            for idx in used:
                mask = (topk_idx == idx).any(dim=1)
                if not mask.any():
                    continue
                delta = self.primitives[idx](u_t[mask])
                w = weights[mask, idx].view(-1, 1, 1)
                delta_sum[mask] = delta_sum[mask] + delta * w
                if need_delta_stack:
                    delta_stack[mask, :, :, idx] = delta
        else:
            deltas = [p(u_t) for p in self.primitives]
            delta_stack = torch.stack(deltas, dim=-1)  # (batch, x, channels, k)
            weighted = delta_stack * weights.unsqueeze(1).unsqueeze(2)
            delta_sum = weighted.sum(dim=-1)

        if self.aggregator is None:
            delta_out = delta_sum
        elif self.aggregator.needs_delta_stack:
            delta_out = self.aggregator(delta_stack, weights)
        else:
            delta_out = self.aggregator(delta_stack, weights)

        u_next = u_state + delta_out

        if return_weights and return_deltas:
            return u_next, weights, topk_idx, delta_stack
        if return_weights:
            return u_next, weights, topk_idx
        if return_deltas:
            return u_next, delta_stack
        return u_next

    def route(self, u_t, pde_params=None, dataset_id=None, equation=None):
        u_state = u_t[..., : self.output_channels]
        return self.router(u_state, pde_params, dataset_id, equation, return_topk=True)
