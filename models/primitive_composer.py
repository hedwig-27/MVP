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
        fft_bins=0,
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
        self.fft_bins = fft_bins
        self.pde_param_dim = pde_param_dim
        self.num_datasets = num_datasets
        self.dataset_embed_dim = dataset_embed_dim

        input_dim = len(stats)
        if fft_bins > 0:
            input_dim += fft_bins
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
        if self.fft_bins > 0:
            fft = torch.fft.rfft(u_state, dim=1)
            mag = torch.abs(fft).mean(dim=2)
            feats.append(mag[:, : self.fft_bins])
        return torch.cat([f.unsqueeze(-1) if f.dim() == 1 else f for f in feats], dim=-1)

    def forward(self, u_state, pde_params=None, dataset_id=None, return_topk=False):
        stats_vec = self._compute_stats(u_state)
        feats = [stats_vec]
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
                dataset_id = torch.zeros(stats_vec.size(0), device=stats_vec.device, dtype=torch.long)
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


class PrimitiveCNO(nn.Module):
    """
    Automatic primitive discovery model.
    u_{t+1} = u_t + sum_i alpha_i * PrimitiveOperator_i(u_t)
    """
    def __init__(self, primitives, router, output_channels):
        super().__init__()
        self.primitives = nn.ModuleList(primitives)
        self.router = router
        self.output_channels = output_channels
        self.num_primitives = len(primitives)

    def forward(self, u_t, pde_params=None, dataset_id=None, return_weights=False, return_deltas=False):
        deltas = [p(u_t) for p in self.primitives]
        u_state = u_t[..., : self.output_channels]
        weights, topk_idx = self.router(u_state, pde_params, dataset_id, return_topk=True)

        delta_stack = torch.stack(deltas, dim=-1)  # (batch, x, channels, k)
        weighted = delta_stack * weights.unsqueeze(1).unsqueeze(2)
        delta_sum = weighted.sum(dim=-1)
        u_next = u_state + delta_sum

        if return_weights and return_deltas:
            return u_next, weights, topk_idx, delta_stack
        if return_weights:
            return u_next, weights, topk_idx
        if return_deltas:
            return u_next, delta_stack
        return u_next

    def route(self, u_t, pde_params=None, dataset_id=None):
        u_state = u_t[..., : self.output_channels]
        return self.router(u_state, pde_params, dataset_id, return_topk=True)
