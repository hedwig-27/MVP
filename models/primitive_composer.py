import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class Router(nn.Module):
    """
    Sparse router that selects top-k primitives based on low-dimensional stats of u_t.
    Supports optional spatial routing (per-position weights) via a Conv1d path.
    Can emit per-primitive vector codes for richer aggregation.
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
        use_dataset_embed=True,
        code_dim=0,
        code_as_weight=True,
        spatial=False,
        spatial_kernel=3,
        spatial_downsample=1,
        spatial_use_state=True,
        spatial_hidden_dim=None,
        always_on_index=None,
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
        self.use_dataset_embed = bool(use_dataset_embed)
        self.code_dim = int(code_dim)
        self.code_as_weight = bool(code_as_weight)
        self.spatial = bool(spatial)
        self.spatial_kernel = int(spatial_kernel)
        self.spatial_downsample = int(spatial_downsample)
        self.spatial_use_state = bool(spatial_use_state)
        self.spatial_hidden_dim = int(spatial_hidden_dim) if spatial_hidden_dim is not None else hidden_dim
        self.always_on_index = always_on_index if always_on_index is None else int(always_on_index)

        stats_dim = 0
        stats_dim += sum(1 for s in stats if s in ("mean", "std", "min", "max", "grad_norm"))
        if self.local_segments > 0 and self.local_stats:
            stats_dim += self.local_segments * sum(
                1 for s in self.local_stats if s in ("mean", "std", "min", "max")
            )
        if fft_bins > 0:
            stats_dim += fft_bins
        self.stats_dim = stats_dim

        input_dim = stats_dim
        if self.equation_dim > 0:
            input_dim += self.equation_dim
        if pde_param_dim > 0:
            input_dim += pde_param_dim
        if self.use_dataset_embed and num_datasets > 1:
            input_dim += dataset_embed_dim
        if self.spatial and self.spatial_use_state:
            input_dim += self.state_channels

        self.trunk = None
        self.logits_head = None
        self.code_head = None
        self.spatial_conv = None
        if self.spatial:
            kernel = max(1, self.spatial_kernel)
            padding = kernel // 2
            self.spatial_conv = nn.Sequential(
                nn.Conv1d(input_dim, self.spatial_hidden_dim, kernel_size=kernel, padding=padding),
                nn.ReLU(),
                nn.Conv1d(self.spatial_hidden_dim, num_primitives, kernel_size=1),
            )
        else:
            self.trunk = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
            )
            self.logits_head = nn.Linear(hidden_dim, num_primitives)
            if self.code_dim > 0:
                self.code_head = nn.Linear(hidden_dim, num_primitives * self.code_dim)
        self.dataset_embed = None
        if self.use_dataset_embed and num_datasets > 1:
            self.dataset_embed = nn.Embedding(num_datasets, dataset_embed_dim)

    def _force_always_on(self, topk_idx):
        if self.always_on_index is None:
            return topk_idx
        idx = self.always_on_index
        if idx < 0 or idx >= self.num_primitives:
            return topk_idx
        if topk_idx.dim() == 2:
            has = (topk_idx == idx).any(dim=1)
            if has.all():
                return topk_idx
            topk_idx = topk_idx.clone()
            replace_rows = (~has).nonzero(as_tuple=False).view(-1)
            if replace_rows.numel() > 0:
                topk_idx[replace_rows, -1] = idx
            return topk_idx
        if topk_idx.dim() == 3:
            bsz, xsz, ksz = topk_idx.shape
            has = (topk_idx == idx).any(dim=2)
            if has.all():
                return topk_idx
            topk_idx = topk_idx.reshape(bsz * xsz, ksz)
            missing = (~has).reshape(bsz * xsz)
            if missing.any():
                topk_idx[missing, -1] = idx
            return topk_idx.reshape(bsz, xsz, ksz)
        return topk_idx

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
        batch = u_state.size(0)
        spatial = getattr(self, "spatial", False)
        stats_dim = getattr(self, "stats_dim", 0)
        spatial_use_state = getattr(self, "spatial_use_state", False)
        spatial_downsample = getattr(self, "spatial_downsample", 1)

        stats_vec = None
        if stats_dim > 0:
            stats_vec = self._compute_stats(u_state)

        codes = None
        if spatial:
            x_len = u_state.size(1)
            feats = []
            if spatial_use_state:
                feats.append(u_state)
            if stats_vec is not None:
                feats.append(stats_vec.unsqueeze(1).expand(-1, x_len, -1))

            if self.equation_dim > 0:
                if equation is None:
                    equation = torch.zeros(
                        batch, self.equation_dim, device=u_state.device, dtype=u_state.dtype
                    )
                elif equation.dim() == 1:
                    equation = equation.unsqueeze(0)
                feats.append(equation.unsqueeze(1).expand(-1, x_len, -1))
            if self.pde_param_dim > 0:
                if pde_params is None:
                    pde_params = torch.zeros(
                        batch, self.pde_param_dim, device=u_state.device, dtype=u_state.dtype
                    )
                feats.append(pde_params.unsqueeze(1).expand(-1, x_len, -1))
            if self.dataset_embed is not None:
                if dataset_id is None:
                    embed = torch.zeros(
                        batch, self.dataset_embed_dim, device=u_state.device, dtype=u_state.dtype
                    )
                else:
                    if dataset_id.dim() > 1:
                        dataset_id = dataset_id.view(-1)
                    embed = self.dataset_embed(dataset_id)
                feats.append(embed.unsqueeze(1).expand(-1, x_len, -1))

            if not feats:
                raise ValueError("Spatial router has no input features configured.")
            feat = torch.cat(feats, dim=-1)
            feat = feat.permute(0, 2, 1)
            did_downsample = False
            if spatial_downsample > 1 and feat.size(-1) >= spatial_downsample:
                feat = F.avg_pool1d(
                    feat,
                    kernel_size=spatial_downsample,
                    stride=spatial_downsample,
                )
                did_downsample = True
            logits = self.spatial_conv(feat)
            if did_downsample:
                logits = F.interpolate(
                    logits,
                    size=x_len,
                    mode="linear",
                    align_corners=False,
                )
            logits = logits.permute(0, 2, 1)
            if self.top_k < self.num_primitives:
                topk_idx = torch.topk(logits, self.top_k, dim=-1).indices
                topk_idx = self._force_always_on(topk_idx)
                masked = torch.full_like(logits, -1e9)
                gathered = torch.gather(logits, -1, topk_idx)
                masked.scatter_(-1, topk_idx, gathered)
                weights = torch.softmax(masked, dim=-1)
            else:
                topk_idx = torch.argsort(logits, dim=-1, descending=True)
                weights = torch.softmax(logits, dim=-1)
        else:
            feats = []
            if stats_vec is not None:
                feats.append(stats_vec)
            if self.equation_dim > 0:
                if equation is None:
                    equation = torch.zeros(
                        batch, self.equation_dim, device=u_state.device, dtype=u_state.dtype
                    )
                elif equation.dim() == 1:
                    equation = equation.unsqueeze(0)
                feats.append(equation)
            if self.pde_param_dim > 0:
                if pde_params is None:
                    pde_params = torch.zeros(
                        batch, self.pde_param_dim, device=u_state.device, dtype=u_state.dtype
                    )
                feats.append(pde_params)
            if self.dataset_embed is not None:
                if dataset_id is None:
                    feats.append(
                        torch.zeros(
                            batch, self.dataset_embed_dim, device=u_state.device, dtype=u_state.dtype
                        )
                    )
                else:
                    if dataset_id.dim() > 1:
                        dataset_id = dataset_id.view(-1)
                    feats.append(self.dataset_embed(dataset_id))
            if not feats:
                raise ValueError("Router has no input features configured.")
            stats_vec = torch.cat(feats, dim=-1)

            hidden = self.trunk(stats_vec)
            codes_raw = None
            if self.code_head is not None:
                codes_raw = self.code_head(hidden).view(batch, self.num_primitives, self.code_dim)
            if self.code_as_weight and codes_raw is not None:
                weights = torch.softmax(codes_raw, dim=1)
                scores = torch.linalg.norm(weights, dim=-1)
                if self.top_k < self.num_primitives:
                    topk_idx = torch.topk(scores, self.top_k, dim=-1).indices
                    topk_idx = self._force_always_on(topk_idx)
                    mask = torch.zeros_like(scores)
                    mask.scatter_(1, topk_idx, 1.0)
                    weights = weights * mask.unsqueeze(-1)
                    denom = weights.sum(dim=1, keepdim=True).clamp_min(1e-8)
                    weights = weights / denom
                else:
                    topk_idx = torch.argsort(scores, dim=-1, descending=True)
                codes = codes_raw
            else:
                logits = self.logits_head(hidden)
                if self.top_k < self.num_primitives:
                    topk_idx = torch.topk(logits, self.top_k, dim=-1).indices
                    topk_idx = self._force_always_on(topk_idx)
                    masked = torch.full_like(logits, -1e9)
                    gathered = torch.gather(logits, 1, topk_idx)
                    masked.scatter_(1, topk_idx, gathered)
                    weights = torch.softmax(masked, dim=-1)
                else:
                    topk_idx = torch.argsort(logits, dim=-1, descending=True)
                    weights = torch.softmax(logits, dim=-1)
                codes = codes_raw

        if return_topk:
            return weights, topk_idx, codes
        return weights


class _ConditionalTransformerBlock(nn.Module):
    def __init__(self, hidden_dim, n_heads, ff_dim, dropout):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.attn = nn.MultiheadAttention(
            hidden_dim, n_heads, dropout=dropout, batch_first=True
        )
        self.drop1 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.ff = nn.Sequential(
            nn.Linear(hidden_dim, ff_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, hidden_dim),
        )
        self.drop2 = nn.Dropout(dropout)

    def forward(self, x, cond_attn, cond_ff):
        # cond_attn/cond_ff: (gamma, beta) with shape (B, D), or None
        y = self.norm1(x)
        if cond_attn is not None:
            gamma, beta = cond_attn
            y = y * (1.0 + gamma.unsqueeze(1)) + beta.unsqueeze(1)
        attn_out, _ = self.attn(y, y, y, need_weights=False)
        x = x + self.drop1(attn_out)

        y = self.norm2(x)
        if cond_ff is not None:
            gamma, beta = cond_ff
            y = y * (1.0 + gamma.unsqueeze(1)) + beta.unsqueeze(1)
        ff_out = self.ff(y)
        x = x + self.drop2(ff_out)
        return x


class TransformerRouter(nn.Module):
    """
    Transformer-based router that attends over function tokens and condition tokens.
    Outputs top-k sparse weights (global, not spatial) compatible with PrimitiveCNO.
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
        use_dataset_embed=True,
        code_dim=0,
        code_as_weight=True,
        use_state_tokens=True,
        state_downsample=4,
        use_stats_token=True,
        n_layers=2,
        n_heads=4,
        ff_dim=None,
        dropout=0.0,
        use_primitive_queries=True,
        cond_hidden_dim=None,
        always_on_index=None,
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
        self.use_dataset_embed = bool(use_dataset_embed)
        self.code_dim = int(code_dim)
        self.code_as_weight = bool(code_as_weight)
        self.use_state_tokens = bool(use_state_tokens)
        self.state_downsample = int(state_downsample)
        self.use_stats_token = bool(use_stats_token)
        self.n_layers = int(n_layers)
        self.n_heads = int(n_heads)
        self.ff_dim = int(ff_dim) if ff_dim is not None else int(hidden_dim * 4)
        self.dropout = float(dropout)
        self.use_primitive_queries = bool(use_primitive_queries)
        self.cond_hidden_dim = int(cond_hidden_dim) if cond_hidden_dim is not None else hidden_dim
        self.always_on_index = always_on_index if always_on_index is None else int(always_on_index)

        stats_dim = 0
        stats_dim += sum(1 for s in stats if s in ("mean", "std", "min", "max", "grad_norm"))
        if self.local_segments > 0 and self.local_stats:
            stats_dim += self.local_segments * sum(
                1 for s in self.local_stats if s in ("mean", "std", "min", "max")
            )
        if fft_bins > 0:
            stats_dim += fft_bins
        self.stats_dim = stats_dim

        self.state_proj = None
        if self.use_state_tokens:
            self.state_proj = nn.Linear(state_channels, hidden_dim)

        self.stats_proj = None
        if self.use_stats_token and stats_dim > 0:
            self.stats_proj = nn.Linear(stats_dim, hidden_dim)

        self.dataset_embed = None
        if self.use_dataset_embed and num_datasets > 1:
            self.dataset_embed = nn.Embedding(num_datasets, dataset_embed_dim)

        self.cls_token = nn.Parameter(torch.zeros(1, 1, hidden_dim))

        self.blocks = nn.ModuleList(
            [
                _ConditionalTransformerBlock(
                    hidden_dim=hidden_dim,
                    n_heads=self.n_heads,
                    ff_dim=self.ff_dim,
                    dropout=self.dropout,
                )
                for _ in range(self.n_layers)
            ]
        )

        cond_dim = 0
        if self.equation_dim > 0:
            cond_dim += self.equation_dim
        if self.pde_param_dim > 0:
            cond_dim += self.pde_param_dim
        if self.dataset_embed is not None:
            cond_dim += self.dataset_embed_dim
        self.cond_dim = cond_dim
        self.cond_mlp = None
        if cond_dim > 0:
            self.cond_mlp = nn.Sequential(
                nn.Linear(cond_dim, self.cond_hidden_dim),
                nn.ReLU(),
                nn.Linear(self.cond_hidden_dim, self.n_layers * 4 * hidden_dim),
            )
            nn.init.zeros_(self.cond_mlp[-1].weight)
            nn.init.zeros_(self.cond_mlp[-1].bias)

        self.query_attn = None
        self.primitive_queries = None
        self.query_norm = None
        if self.use_primitive_queries:
            self.primitive_queries = nn.Parameter(
                torch.randn(num_primitives, hidden_dim) * 0.02
            )
            self.query_attn = nn.MultiheadAttention(
                hidden_dim, self.n_heads, dropout=self.dropout, batch_first=True
            )
            self.query_norm = nn.LayerNorm(hidden_dim)

        self.logits_head = None
        self.code_head = None
        if self.use_primitive_queries:
            self.logits_head = nn.Linear(hidden_dim, 1)
            if self.code_dim > 0:
                self.code_head = nn.Linear(hidden_dim, self.code_dim)
        else:
            self.logits_head = nn.Linear(hidden_dim, num_primitives)
            if self.code_dim > 0:
                self.code_head = nn.Linear(hidden_dim, num_primitives * self.code_dim)

    def _force_always_on(self, topk_idx):
        if self.always_on_index is None:
            return topk_idx
        idx = self.always_on_index
        if idx < 0 or idx >= self.num_primitives:
            return topk_idx
        if topk_idx.dim() == 2:
            has = (topk_idx == idx).any(dim=1)
            if has.all():
                return topk_idx
            topk_idx = topk_idx.clone()
            replace_rows = (~has).nonzero(as_tuple=False).view(-1)
            if replace_rows.numel() > 0:
                topk_idx[replace_rows, -1] = idx
            return topk_idx
        if topk_idx.dim() == 3:
            bsz, xsz, ksz = topk_idx.shape
            has = (topk_idx == idx).any(dim=2)
            if has.all():
                return topk_idx
            topk_idx = topk_idx.reshape(bsz * xsz, ksz)
            missing = (~has).reshape(bsz * xsz)
            if missing.any():
                topk_idx[missing, -1] = idx
            return topk_idx.reshape(bsz, xsz, ksz)
        return topk_idx

    def _compute_stats(self, u_state):
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

    def _sinusoidal_pos_emb(self, length, dim, device, dtype):
        position = torch.arange(length, device=device, dtype=dtype).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, dim, 2, device=device, dtype=dtype) * (-math.log(10000.0) / dim)
        )
        pe = torch.zeros(length, dim, device=device, dtype=dtype)
        pe[:, 0::2] = torch.sin(position * div_term)
        if dim % 2 == 1:
            pe[:, 1::2] = torch.cos(position * div_term[:-1])
        else:
            pe[:, 1::2] = torch.cos(position * div_term)
        return pe

    def forward(self, u_state, pde_params=None, dataset_id=None, equation=None, return_topk=False):
        batch, x_len, _ = u_state.shape

        tokens = [self.cls_token.expand(batch, -1, -1)]
        if self.use_state_tokens:
            state = u_state
            if self.state_downsample > 1 and x_len >= self.state_downsample:
                pooled = F.avg_pool1d(
                    state.permute(0, 2, 1),
                    kernel_size=self.state_downsample,
                    stride=self.state_downsample,
                )
                state = pooled.permute(0, 2, 1)
            state_tokens = self.state_proj(state)
            pos = self._sinusoidal_pos_emb(
                state_tokens.size(1), state_tokens.size(2), state_tokens.device, state_tokens.dtype
            )
            state_tokens = state_tokens + pos.unsqueeze(0)
            tokens.append(state_tokens)

        if self.stats_proj is not None:
            stats_vec = self._compute_stats(u_state)
            tokens.append(self.stats_proj(stats_vec).unsqueeze(1))

        if not tokens:
            raise ValueError("TransformerRouter has no input tokens configured.")

        feat = torch.cat(tokens, dim=1)
        cond_vec = None
        if self.cond_dim > 0:
            cond_parts = []
            if self.equation_dim > 0:
                if equation is None:
                    equation = torch.zeros(
                        batch, self.equation_dim, device=u_state.device, dtype=u_state.dtype
                    )
                elif equation.dim() == 1:
                    equation = equation.unsqueeze(0)
                cond_parts.append(equation)
            if self.pde_param_dim > 0:
                if pde_params is None:
                    pde_params = torch.zeros(
                        batch, self.pde_param_dim, device=u_state.device, dtype=u_state.dtype
                    )
                cond_parts.append(pde_params)
            if self.dataset_embed is not None:
                if dataset_id is None:
                    embed = torch.zeros(
                        batch, self.dataset_embed_dim, device=u_state.device, dtype=u_state.dtype
                    )
                else:
                    if dataset_id.dim() > 1:
                        dataset_id = dataset_id.view(-1)
                    embed = self.dataset_embed(dataset_id)
                cond_parts.append(embed)
            cond_vec = torch.cat(cond_parts, dim=-1) if cond_parts else None

        cond_params = None
        if self.cond_mlp is not None and cond_vec is not None:
            cond_params = self.cond_mlp(cond_vec).view(batch, self.n_layers, 4, self.hidden_dim)

        encoded = feat
        for idx, block in enumerate(self.blocks):
            if cond_params is not None:
                gamma_attn = cond_params[:, idx, 0]
                beta_attn = cond_params[:, idx, 1]
                gamma_ff = cond_params[:, idx, 2]
                beta_ff = cond_params[:, idx, 3]
                encoded = block(encoded, (gamma_attn, beta_attn), (gamma_ff, beta_ff))
            else:
                encoded = block(encoded, None, None)

        codes_raw = None
        if self.use_primitive_queries:
            queries = self.primitive_queries.unsqueeze(0).expand(batch, -1, -1)
            attn_out, _ = self.query_attn(queries, encoded, encoded, need_weights=False)
            if self.query_norm is not None:
                attn_out = self.query_norm(attn_out)
            if self.code_head is not None:
                codes_raw = self.code_head(attn_out)
            logits = self.logits_head(attn_out).squeeze(-1)
        else:
            cls = encoded[:, 0]
            if self.code_head is not None:
                codes_raw = self.code_head(cls).view(batch, self.num_primitives, self.code_dim)
            logits = self.logits_head(cls)

        if self.code_as_weight and codes_raw is not None:
            weights = torch.softmax(codes_raw, dim=1)
            scores = torch.linalg.norm(weights, dim=-1)
            if self.top_k < self.num_primitives:
                topk_idx = torch.topk(scores, self.top_k, dim=-1).indices
                topk_idx = self._force_always_on(topk_idx)
                mask = torch.zeros_like(scores)
                mask.scatter_(1, topk_idx, 1.0)
                weights = weights * mask.unsqueeze(-1)
                denom = weights.sum(dim=1, keepdim=True).clamp_min(1e-8)
                weights = weights / denom
            else:
                topk_idx = torch.argsort(scores, dim=-1, descending=True)
            codes = codes_raw
        else:
            if self.top_k < self.num_primitives:
                topk_idx = torch.topk(logits, self.top_k, dim=-1).indices
                topk_idx = self._force_always_on(topk_idx)
                masked = torch.full_like(logits, -1e9)
                gathered = torch.gather(logits, 1, topk_idx)
                masked.scatter_(1, topk_idx, gathered)
                weights = torch.softmax(masked, dim=-1)
            else:
                topk_idx = torch.argsort(logits, dim=-1, descending=True)
                weights = torch.softmax(logits, dim=-1)
            codes = codes_raw

        if return_topk:
            return weights, topk_idx, codes
        return weights


class PrimitiveAggregator(nn.Module):
    def __init__(
        self,
        num_primitives,
        agg_type="linear",
        hidden_dim=32,
        output_channels=1,
        code_dim=0,
    ):
        super().__init__()
        self.num_primitives = num_primitives
        self.agg_type = agg_type
        self.hidden_dim = hidden_dim
        self.output_channels = int(output_channels)
        self.code_dim = int(code_dim)
        if agg_type == "mlp":
            self.mlp = nn.Sequential(
                nn.Linear(num_primitives, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, 1),
            )
        elif agg_type == "mlp_joint":
            in_dim = self.num_primitives * (self.output_channels + 1 + self.code_dim)
            self.mlp = nn.Sequential(
                nn.Linear(in_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, self.output_channels),
            )
        elif agg_type != "linear":
            raise ValueError(f"Unknown aggregator type: {agg_type}")

    @property
    def needs_delta_stack(self):
        return self.agg_type in ("mlp", "mlp_joint")

    def forward(self, delta_stack, weights, codes=None, topk_idx=None):
        # delta_stack: (B, X, C, P), weights: (B, P), (B, X, P), or (B, P, C)
        if weights.dim() == 3 and weights.size(1) == delta_stack.size(-1):
            # vector weights (B, P, C)
            weighted = delta_stack * weights.permute(0, 2, 1).unsqueeze(1)
        elif weights.dim() == 3:
            # spatial weights (B, X, P)
            weighted = delta_stack * weights.unsqueeze(2)
        else:
            weighted = delta_stack * weights.unsqueeze(1).unsqueeze(2)
        if self.agg_type == "linear":
            return weighted.sum(dim=-1)
        if self.agg_type == "mlp":
            b, x, c, p = weighted.shape
            flat = weighted.reshape(b * x * c, p)
            out = self.mlp(flat).reshape(b, x, c)
            return out

        if weights.dim() != 2:
            raise ValueError("mlp_joint expects weights with shape (B, P).")
        b, x, c, p = delta_stack.shape
        if topk_idx is not None:
            mask = torch.zeros(b, p, device=delta_stack.device, dtype=delta_stack.dtype)
            mask.scatter_(1, topk_idx, 1.0)
        else:
            mask = (weights > 0).to(delta_stack.dtype)
        masked_delta = delta_stack * mask[:, None, None, :]
        delta_flat = masked_delta.permute(0, 1, 3, 2).reshape(b, x, p * c)

        weight_flat = (weights * mask).unsqueeze(1).expand(-1, x, -1)
        parts = [delta_flat, weight_flat]

        if self.code_dim > 0:
            if codes is None:
                codes = torch.zeros(b, p, self.code_dim, device=delta_stack.device, dtype=delta_stack.dtype)
            if codes.dim() != 3 or codes.size(0) != b or codes.size(1) != p:
                raise ValueError("codes must have shape (B, P, code_dim) for mlp_joint.")
            code_masked = codes * mask.unsqueeze(-1)
            code_flat = code_masked.reshape(b, p * self.code_dim).unsqueeze(1).expand(-1, x, -1)
            parts.append(code_flat)

        feat = torch.cat(parts, dim=-1)
        out = self.mlp(feat.reshape(b * x, -1)).reshape(b, x, self.output_channels)
        return out


class PrimitiveCNO(nn.Module):
    """
    Automatic primitive discovery model.
    u_{t+1} = u_t + sum_i alpha_i * PrimitiveOperator_i(u_t)
    """
    def __init__(self, primitives, router, output_channels, aggregator=None, base_operator=None):
        super().__init__()
        self.primitives = nn.ModuleList(primitives)
        self.router = router
        self.output_channels = output_channels
        self.num_primitives = len(primitives)
        self.aggregator = aggregator
        self.base_operator = base_operator
        self.base_index = self.num_primitives if base_operator is not None else None
        self.total_primitives = self.num_primitives + (1 if base_operator is not None else 0)

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
        weights, topk_idx, codes = self.router(
            u_state, pde_params, dataset_id, equation, return_topk=True
        )
        need_delta_stack = return_deltas or (self.aggregator and self.aggregator.needs_delta_stack)

        use_sparse = sparse_primitives and self.router.top_k < self.total_primitives and weights.dim() in (2, 3)

        if use_sparse:
            delta_sum = torch.zeros_like(u_state)
            delta_stack = None
            if need_delta_stack:
                delta_stack = torch.zeros(
                    u_state.size(0),
                    u_state.size(1),
                    u_state.size(2),
                    self.total_primitives,
                    device=u_state.device,
                    dtype=u_state.dtype,
                )
            used = torch.unique(topk_idx).tolist()
            for idx in used:
                mask = (topk_idx == idx).any(dim=1)
                if not mask.any():
                    continue
                if self.base_index is not None and idx == self.base_index:
                    delta = self.base_operator(u_t[mask])
                else:
                    delta = self.primitives[idx](u_t[mask])
                if weights.dim() == 3 and weights.size(1) == self.total_primitives:
                    w = weights[mask, idx].view(-1, 1, self.output_channels)
                else:
                    w = weights[mask, idx].view(-1, 1, 1)
                delta_sum[mask] = delta_sum[mask] + delta * w
                if need_delta_stack:
                    delta_stack[mask, :, :, idx] = delta
        else:
            deltas = [p(u_t) for p in self.primitives]
            if self.base_operator is not None:
                deltas.append(self.base_operator(u_t))
            delta_stack = torch.stack(deltas, dim=-1)  # (batch, x, channels, k)
            if weights.dim() == 3 and weights.size(1) == self.total_primitives:
                if weights.size(-1) != self.output_channels:
                    raise ValueError("Vector weights require code_dim == output_channels.")
                weighted = delta_stack * weights.permute(0, 2, 1).unsqueeze(1)
            elif weights.dim() == 3:
                weighted = delta_stack * weights.unsqueeze(2)
            else:
                weighted = delta_stack * weights.unsqueeze(1).unsqueeze(2)
            delta_sum = weighted.sum(dim=-1)

        if self.aggregator is None:
            delta_out = delta_sum
        elif self.aggregator.needs_delta_stack:
            delta_out = self.aggregator(delta_stack, weights, codes=codes, topk_idx=topk_idx)
        else:
            delta_out = self.aggregator(delta_stack, weights, codes=codes, topk_idx=topk_idx)

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
        weights, topk_idx, _ = self.router(u_state, pde_params, dataset_id, equation, return_topk=True)
        return weights, topk_idx


class TermNet(nn.Module):
    def __init__(self, in_dim, out_channels, hidden_dim=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(in_dim, hidden_dim, kernel_size=1),
            nn.GELU(),
            nn.Conv1d(hidden_dim, out_channels, kernel_size=1),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, feats):
        # feats: (B, X, F)
        x = feats.permute(0, 2, 1)
        out = self.net(x).permute(0, 2, 1)
        return out


class TermLibrary(nn.Module):
    def __init__(
        self,
        term_names,
        output_channels,
        hidden_dim=32,
        cond_dim=0,
        use_scale_mlp=True,
    ):
        super().__init__()
        self.term_names = [str(t) for t in term_names] if term_names else []
        self.output_channels = int(output_channels)
        self.cond_dim = int(cond_dim)
        self.use_scale_mlp = bool(use_scale_mlp)

        feature_counts = {
            "advection": 1,
            "nonlinear_advection": 1,
            "diffusion": 1,
            "reaction": 2,
            "sorption": 2,
            "cns": 3,
        }

        self.term_nets = nn.ModuleDict()
        for term in self.term_names:
            feat_count = feature_counts.get(term, 1)
            in_dim = self.output_channels * feat_count
            self.term_nets[term] = TermNet(in_dim, self.output_channels, hidden_dim=hidden_dim)

        self.scale_mlp = None
        if self.use_scale_mlp and self.cond_dim > 0 and self.term_names:
            self.scale_mlp = nn.Sequential(
                nn.Linear(self.cond_dim, max(hidden_dim, 16)),
                nn.GELU(),
                nn.Linear(max(hidden_dim, 16), len(self.term_names)),
            )
            nn.init.zeros_(self.scale_mlp[-1].weight)
            nn.init.zeros_(self.scale_mlp[-1].bias)

    def _central_diff(self, u, dx):
        return (torch.roll(u, shifts=-1, dims=1) - torch.roll(u, shifts=1, dims=1)) / (2.0 * dx)

    def _second_diff(self, u, dx):
        return (torch.roll(u, shifts=-1, dims=1) - 2.0 * u + torch.roll(u, shifts=1, dims=1)) / (dx * dx)

    def forward(self, u_state, coeffs=None, cond=None, grid=None, return_terms=False):
        if not self.term_names:
            delta = torch.zeros_like(u_state)
            return (delta, {}, {}) if return_terms else delta

        batch, x_len, _ = u_state.shape
        if coeffs is None:
            coeffs = torch.zeros(batch, len(self.term_names), device=u_state.device, dtype=u_state.dtype)
        if coeffs.dim() == 1:
            coeffs = coeffs.unsqueeze(0)

        if grid is not None and grid.size(-1) > 0:
            dx = grid[:, 1, 0] - grid[:, 0, 0]
            dx = dx.abs().clamp_min(1e-6).view(batch, 1, 1)
        else:
            dx = torch.ones(batch, 1, 1, device=u_state.device, dtype=u_state.dtype)

        u = u_state
        u_x = self._central_diff(u, dx)
        u_xx = self._second_diff(u, dx)
        u2 = u * u
        u2_x = self._central_diff(u2, dx)

        term_outputs = {}
        term_features = {}
        delta = torch.zeros_like(u_state)

        scale = None
        if self.scale_mlp is not None:
            if cond is None:
                cond = torch.zeros(batch, self.cond_dim, device=u_state.device, dtype=u_state.dtype)
            scale = 1.0 + self.scale_mlp(cond)

        for idx, term in enumerate(self.term_names):
            if term == "advection":
                feats = [u_x]
            elif term == "nonlinear_advection":
                feats = [u2_x]
            elif term == "diffusion":
                feats = [u_xx]
            elif term == "reaction":
                feats = [u, u2]
            elif term == "sorption":
                feats = [u, u_xx]
            elif term == "cns":
                feats = [u, u_x, u_xx]
            else:
                feats = [u]

            feat_tensor = torch.cat(feats, dim=-1)
            term_out = self.term_nets[term](feat_tensor)
            coeff = coeffs[:, idx].view(batch, 1, 1)
            if scale is not None:
                coeff = coeff * scale[:, idx].view(batch, 1, 1)
            delta = delta + coeff * term_out

            if return_terms:
                term_outputs[term] = term_out
                term_features[term] = feat_tensor

        if return_terms:
            return delta, term_outputs, term_features
        return delta


class CompositePDEModel(nn.Module):
    """
    Base + term library + residual experts composite model.
    """
    def __init__(
        self,
        base_operator,
        term_library,
        residual_experts,
        router,
        output_channels,
        term_count=0,
        cond_dim=0,
        use_dataset_embed=False,
        dataset_embed_dim=8,
        num_datasets=1,
    ):
        super().__init__()
        self.base_operator = base_operator
        self.term_library = term_library
        self.residual_experts = nn.ModuleList(residual_experts) if residual_experts else nn.ModuleList()
        self.router = router
        self.output_channels = int(output_channels)
        self.term_count = int(term_count)
        self.cond_dim = int(cond_dim)
        self.use_dataset_embed = bool(use_dataset_embed)
        self.dataset_embed = None
        if self.use_dataset_embed and num_datasets > 1:
            self.dataset_embed = nn.Embedding(num_datasets, dataset_embed_dim)

    def _build_cond(self, pde_params, equation, dataset_id):
        parts = []
        if pde_params is not None:
            parts.append(pde_params)
        if equation is not None:
            parts.append(equation)
        if self.dataset_embed is not None:
            if dataset_id is None:
                embed = torch.zeros(
                    pde_params.size(0) if pde_params is not None else equation.size(0),
                    self.dataset_embed.embedding_dim,
                    device=pde_params.device if pde_params is not None else equation.device,
                    dtype=pde_params.dtype if pde_params is not None else equation.dtype,
                )
            else:
                if dataset_id.dim() > 1:
                    dataset_id = dataset_id.view(-1)
                embed = self.dataset_embed(dataset_id)
            parts.append(embed)
        if not parts:
            return None
        return torch.cat(parts, dim=-1)

    def forward(
        self,
        u_t,
        pde_params=None,
        dataset_id=None,
        equation=None,
        return_weights=False,
        return_parts=False,
        return_terms=False,
        sparse_primitives=False,
        use_base=True,
        use_terms=True,
        use_residual=True,
    ):
        u_state = u_t[..., : self.output_channels]
        grid = u_t[..., self.output_channels :] if u_t.size(-1) > self.output_channels else None
        cond = self._build_cond(pde_params, equation, dataset_id)

        delta_base = torch.zeros_like(u_state)
        if use_base and self.base_operator is not None:
            delta_base = self.base_operator(u_t, cond=cond)

        coeffs = None
        if equation is not None and self.term_count > 0:
            coeffs = equation[:, : self.term_count]
        delta_terms = torch.zeros_like(u_state)
        term_outputs = {}
        term_features = {}
        if use_terms and self.term_library is not None:
            if return_terms:
                delta_terms, term_outputs, term_features = self.term_library(
                    u_state, coeffs=coeffs, cond=cond, grid=grid, return_terms=True
                )
            else:
                delta_terms = self.term_library(u_state, coeffs=coeffs, cond=cond, grid=grid)

        delta_res = torch.zeros_like(u_state)
        weights = None
        topk_idx = None
        if use_residual and self.residual_experts and self.router is not None:
            weights, topk_idx, _ = self.router(
                u_state, pde_params, dataset_id, equation, return_topk=True
            )
            deltas = [expert(u_t, cond=cond) for expert in self.residual_experts]
            delta_stack = torch.stack(deltas, dim=-1)
            if weights.dim() == 3 and weights.size(1) == delta_stack.size(-1):
                weighted = delta_stack * weights.permute(0, 2, 1).unsqueeze(1)
            elif weights.dim() == 3:
                weighted = delta_stack * weights.unsqueeze(2)
            else:
                weighted = delta_stack * weights.unsqueeze(1).unsqueeze(2)
            delta_res = weighted.sum(dim=-1)

        delta = delta_base + delta_terms + delta_res
        u_next = u_state + delta

        if return_parts:
            parts = {
                "delta_base": delta_base,
                "delta_terms": delta_terms,
                "delta_res": delta_res,
                "residual_weights": weights,
                "term_outputs": term_outputs,
                "term_features": term_features,
            }
            return u_next, parts
        if return_weights:
            return u_next, weights, topk_idx
        return u_next

    def route(self, u_t, pde_params=None, dataset_id=None, equation=None):
        if self.router is None:
            return None, None
        u_state = u_t[..., : self.output_channels]
        weights, topk_idx, _ = self.router(u_state, pde_params, dataset_id, equation, return_topk=True)
        return weights, topk_idx
