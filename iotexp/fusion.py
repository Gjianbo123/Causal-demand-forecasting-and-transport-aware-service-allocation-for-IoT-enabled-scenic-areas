"""Drop-in torch modules. Input forecasts [batch,node,horizon], output [B,K,N,D]."""
import math
import torch
from torch import nn
from torch.distributions import Independent, Normal

METHODS = ("F0", "F1", "F2", "F3", "F4")


class HorizonFusion(nn.Module):
    def __init__(self, method, resources, horizons=4, embed_dim=16):
        super().__init__()
        if method not in METHODS or min(resources, horizons, embed_dim) < 1:
            raise ValueError("Invalid fusion specification")
        self.method, self.resources, self.horizons, self.embed_dim = method, resources, horizons, embed_dim
        if method == "F0":
            return
        # Shared within a variant across nodes, resources and lead times.
        # Parameters are trainable separately for each policy; checkpoint initialization is paired.
        self.embedding = nn.Sequential(nn.Linear(1, embed_dim), nn.Tanh())
        self.lead_embedding = nn.Parameter(torch.empty(horizons, embed_dim))
        nn.init.normal_(self.lead_embedding, std=.02)
        if method == "F1":
            self.concat_projection = nn.Linear(horizons * embed_dim, embed_dim)
        elif method in ("F3", "F4"):
            self.key = nn.Linear(embed_dim, embed_dim, bias=False)
            self.query = nn.Parameter(torch.empty(1 if method == "F3" else resources, embed_dim))
            nn.init.normal_(self.query, std=.1)

    def forward(self, forecast):
        if forecast.ndim != 3 or forecast.shape[-1] != self.horizons:
            raise ValueError("forecast must be [B,N,H]")
        b, n, h = forecast.shape
        if self.method == "F0":
            return forecast.new_zeros((b, self.resources, n, self.embed_dim)), None
        embedded = self.embedding(forecast.unsqueeze(-1)) + self.lead_embedding
        if self.method == "F1":
            fused = self.concat_projection(embedded.flatten(-2))
            return fused[:, None].expand(-1, self.resources, -1, -1), None
        if self.method == "F2":
            weights = forecast.new_full((b, self.resources, n, h), 1.0 / h)
        else:
            query = self.query.expand(self.resources, -1)
            scores = torch.einsum("bnhd,kd->bknh", torch.tanh(self.key(embedded)), query)
            weights = torch.softmax(scores / math.sqrt(self.embed_dim), dim=-1)
        return torch.einsum("bknh,bnhd->bknd", weights, embedded), weights


class ActorCritic(nn.Module):
    """Reference Gaussian raw-score policy; replace with original action head if needed.

    PPO densities refer to raw [K,N] scores BEFORE softmax/rounding/repair.
    F0 receives a zero forecast tensor and has no forecast parameters.
    """
    def __init__(self, method, state_dim, nodes, resources, hidden=256, embed_dim=16, horizons=4, seed=0,
                 action_shape=None):
        super().__init__()
        self.method, self.nodes, self.resources = method, nodes, resources
        self.action_shape = tuple(action_shape) if action_shape is not None else (resources, nodes)
        if not self.action_shape or any(not isinstance(v, int) or v < 1 for v in self.action_shape):
            raise ValueError("action_shape must contain positive integers")
        # Separate initialization streams avoid a changed fusion module shifting backbone RNG.
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            self.state_net = nn.Sequential(nn.Linear(state_dim, hidden), nn.Tanh(),
                                           nn.Linear(hidden, hidden), nn.Tanh())
            self.fusion_projection = nn.Linear(resources * nodes * embed_dim, hidden, bias=False)
            self.actor = nn.Linear(hidden, math.prod(self.action_shape))
            self.critic = nn.Linear(hidden, 1)
            nn.init.normal_(self.actor.weight, std=.01)
            nn.init.zeros_(self.actor.bias)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed + 100003)
            self.fusion = HorizonFusion(method, resources, horizons, embed_dim)
        self.log_std = nn.Parameter(torch.full(self.action_shape, -.5))
        if method == "F0":
            self.fusion_projection.requires_grad_(False)

    def forward(self, state, forecast):
        fusion, weights = self.fusion(forecast)
        hidden = torch.tanh(self.state_net(state) + self.fusion_projection(fusion.flatten(1)))
        mean = self.actor(hidden).reshape(-1, *self.action_shape)
        std = self.log_std.clamp(-5, 2).exp().expand_as(mean)
        return Independent(Normal(mean, std), len(self.action_shape)), self.critic(hidden).squeeze(-1), weights

    @torch.no_grad()
    def act(self, state, forecast, generator=None, deterministic=False):
        dist, value, weights = self(state, forecast)
        if deterministic:
            raw = dist.base_dist.loc
        else:
            eps = torch.randn(dist.base_dist.loc.shape, generator=generator, device=state.device)
            raw = dist.base_dist.loc + dist.base_dist.scale * eps
        return raw, dist.log_prob(raw), value, weights

    def parameter_count(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
