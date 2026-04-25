"""
CentralizedCriticModel — MAPPO Centralized Critic per RLlib 2.10.x
====================================================================
Paradigma CTDE (Centralized Training, Decentralized Execution):
  - Actor: vede solo obs locale (25D veicolo / 19D pedone)
  - Critic: vede global_obs = fixed-slot concat di TUTTE le obs degli agenti

Componenti:
  1. CentralizedCriticModel  — TorchModelV2 con actor/critic separati
  2. CentralizedCriticCallbacks — inietta global_obs nel SampleBatch
     via on_postprocess_trajectory(), ricalcola VF e GAE
  3. on_episode_start/step/end — Custom metrics per TensorBoard
  4. PopArtLayer — Adaptive value normalization (Block 4.2)

Fixed-slot global_obs layout (Block 4.1):
  [v0_25D | v1_25D | v2_25D | p0_19D | p1_19D | p2_19D | alive_mask_6D]
  Total: 3*25 + 3*19 + 6 = 138D (for 3V+3P config)

PopArt (Block 4.2):
  Wraps the critic's output layer with running mean/std normalization.
  Enabled via custom_model_config["use_popart"] = True.
  Reference: van Hasselt et al. 2016 "Learning values across many orders
  of magnitude", Hessel et al. 2019 "Multi-task with PopArt".

Struttura reti:
  Actor MLP:  obs_dim → 256 → 256 → action_logits
  Critic MLP: global_obs_dim → 256 → 256 → 1 (value, PopArt-normalized)
  Critic Attention (Block 4.4, use_attention=True):
    global_obs → split per-agent slots → per-type linear projection (→ embed_dim)
    → MultiHeadAttention (alive_mask as key_padding_mask) → masked mean-pool
    → hidden → 1 (value, PopArt-normalized)
    Reference: Iqbal & Sha 2019 (MAAC, ICML) — attention only on critic (CTDE).
"""

import logging
from typing import Dict, List

import numpy as np
import torch
import torch.nn as nn
from ray.rllib.algorithms.callbacks import DefaultCallbacks
from ray.rllib.evaluation.postprocessing import Postprocessing, compute_advantages
from ray.rllib.models.modelv2 import ModelV2
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.utils.annotations import override

logger = logging.getLogger(__name__)

# Key injected into SampleBatch for centralized critic
GLOBAL_OBS = "global_obs"

# Obs dimensions per agent type (must match env constants)
_VEHICLE_OBS_DIM = 25
_PEDESTRIAN_OBS_DIM = 19


def _raise_on_nonfinite_np(name: str, arr: np.ndarray):
    arr = np.asarray(arr)
    if np.isfinite(arr).all():
        return
    bad = np.argwhere(~np.isfinite(arr))
    first = tuple(bad[0].tolist()) if bad.size else ()
    value = arr[first] if first else arr
    raise ValueError(f"{name} contains non-finite values at {first}: {value}")


def _raise_on_nonfinite_torch(name: str, tensor: torch.Tensor):
    if torch.isfinite(tensor).all():
        return
    bad = (~torch.isfinite(tensor)).nonzero(as_tuple=False)
    first = tuple(bad[0].tolist()) if bad.numel() else ()
    value = tensor[first].detach().cpu().item() if first else float("nan")
    raise ValueError(f"{name} contains non-finite values at {first}: {value}")


def _agent_obs_dim(agent_id: str, fallback: int = 0) -> int:
    """
    Return expected obs dim for an agent based on its ID prefix.
    For non-CARLA agents (e.g. MPE test), returns fallback if provided,
    or raises ValueError.
    """
    if agent_id.startswith("vehicle"):
        return _VEHICLE_OBS_DIM
    elif agent_id.startswith("pedestrian"):
        return _PEDESTRIAN_OBS_DIM
    if fallback > 0:
        return fallback
    raise ValueError(f"Unknown agent type for: {agent_id}")


def _build_slot_order(agent_ids) -> List[str]:
    """
    Build canonical slot order: vehicles sorted, then pedestrians sorted.
    Agents that don't match vehicle/pedestrian prefix are appended last
    (sorted), for compatibility with non-CARLA envs (e.g. MPE test).
    """
    vehicles = sorted(a for a in agent_ids if a.startswith("vehicle"))
    pedestrians = sorted(a for a in agent_ids if a.startswith("pedestrian"))
    others = sorted(a for a in agent_ids
                    if not a.startswith("vehicle") and not a.startswith("pedestrian"))
    return vehicles + pedestrians + others


def _slot_obs_dim_for_agent(agent_id: str, model, fallback: int = 0) -> int:
    """Resolve slot obs dim using model-provided slot dims when available."""
    slot_dims = getattr(model, "_slot_obs_dims", {}) or {}
    if agent_id.startswith("vehicle") and "vehicle" in slot_dims:
        return int(slot_dims["vehicle"])
    if agent_id.startswith("pedestrian") and "pedestrian" in slot_dims:
        return int(slot_dims["pedestrian"])
    return _agent_obs_dim(agent_id, fallback=fallback)


def compute_global_obs_dim_with_mask(n_vehicles: int, n_pedestrians: int) -> int:
    """global_obs = fixed slots + alive_mask.

    Layout: [v0|v1|...|vN|p0|p1|...|pM|alive_mask]
    alive_mask has one entry per agent (n_vehicles + n_pedestrians).
    """
    n_agents = n_vehicles + n_pedestrians
    return n_vehicles * _VEHICLE_OBS_DIM + n_pedestrians * _PEDESTRIAN_OBS_DIM + n_agents


def _resolve_custom_model_config(model_config: dict, kwargs: dict) -> dict:
    """Merge RLlib custom model options from legacy config and modern kwargs.

    Precedence is explicit kwargs over model_config["custom_model_config"] to
    support both calling conventions without changing current trainer wiring.
    """
    cfg = dict(model_config.get("custom_model_config", {}) or {})
    cfg.update(kwargs or {})
    return cfg


# ---------------------------------------------------------------------------
# PopArt — Adaptive Value Normalization (Block 4.2)
# ---------------------------------------------------------------------------

class PopArtLayer(nn.Module):
    """Linear layer with PopArt normalization for value function output.

    Maintains running mean (mu) and std (sigma) of value targets.
    Output is in normalized space; denormalize() maps back to original scale.

    On each update(targets):
      1. Compute new mu, sigma from targets
      2. Adjust weight & bias to preserve the output mapping (the "Art" step)
      3. Store new mu, sigma

    Reference: van Hasselt et al. 2016, Eq. 7-9.

    Args:
        in_features: input dimension (last hidden layer size)
        beta: EMA decay for running statistics (default 3e-4)
    """

    def __init__(self, in_features: int, beta: float = 3e-4):
        super().__init__()
        self.beta = beta
        self.linear = nn.Linear(in_features, 1)

        # Running statistics (not model parameters — buffers)
        self.register_buffer("mu", torch.zeros(1))
        self.register_buffer("sigma", torch.ones(1))
        self.register_buffer("count", torch.zeros(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass: returns NORMALIZED value prediction."""
        return self.linear(x)

    def denormalize(self, normalized_value: torch.Tensor) -> torch.Tensor:
        """Map from normalized space back to original value scale."""
        return normalized_value * self.sigma + self.mu

    def normalize_targets(self, targets: torch.Tensor) -> torch.Tensor:
        """Normalize value targets for loss computation."""
        return (targets - self.mu) / self.sigma.clamp(min=1e-6)

    @torch.no_grad()
    def update(self, targets: torch.Tensor):
        """Update running statistics and adjust weights (Art step).

        Args:
            targets: raw (unnormalized) value targets from GAE computation.
        """
        targets = targets.detach().float().reshape(-1)
        finite_mask = torch.isfinite(targets)
        if not finite_mask.all():
            n_bad = (~finite_mask).sum().item()
            logger.warning("PopArt.update skipping %d non-finite targets", n_bad)
            targets = targets[finite_mask]
        if targets.numel() == 0:
            return

        new_mu = targets.mean()
        new_sigma = targets.std(unbiased=False).clamp(min=1e-6)
        if not torch.isfinite(new_mu).item() or not torch.isfinite(new_sigma).item():
            logger.warning("PopArt.update produced non-finite stats; skipping update")
            return

        if self.count.item() == 0:
            # First update: initialize directly
            self.mu.copy_(new_mu)
            self.sigma.copy_(new_sigma)
            self.count.add_(1)
            return

        old_mu = self.mu.clone()
        old_sigma = self.sigma.clone()

        updated_mu = old_mu * (1 - self.beta) + new_mu * self.beta
        updated_sigma = old_sigma * (1 - self.beta) + new_sigma * self.beta
        if not torch.isfinite(updated_mu).item() or not torch.isfinite(updated_sigma).item():
            logger.warning("PopArt.update produced non-finite EMA stats; skipping update")
            return

        # Art step: adjust W, b so that output is preserved
        # old: y = W @ x + b  → denorm: y * old_sigma + old_mu
        # new: y' = W' @ x + b' → denorm: y' * new_sigma + new_mu
        # We want: W' @ x * new_sigma + new_mu = W @ x * old_sigma + old_mu
        # => W' = W * (old_sigma / new_sigma)
        # => b' = (b * old_sigma + old_mu - new_mu) / new_sigma
        ratio = old_sigma / updated_sigma.clamp(min=1e-6)
        self.linear.weight.data.mul_(ratio)
        self.linear.bias.data.mul_(ratio).add_(
            (old_mu - updated_mu) / updated_sigma.clamp(min=1e-6)
        )
        self.mu.copy_(updated_mu)
        self.sigma.copy_(updated_sigma)
        self.count.add_(1)

# ---------------------------------------------------------------------------
# Attention Critic Encoder (Block 4.4)
# ---------------------------------------------------------------------------

class AttentionCriticEncoder(nn.Module):
    """Encode global_obs via per-agent slot attention for centralized critic.

    Splits the flat global_obs into per-agent slots using known dims,
    projects each slot to a shared embedding space, applies multi-head
    self-attention with alive_mask, and mean-pools attended tokens.

    Reference: Iqbal & Sha 2019 (MAAC, ICML).

    Args:
        agent_order: canonical list of agent IDs (e.g. [v0, v1, v2, p0, p1, p2])
        slot_obs_dims: dict mapping agent type prefix to obs dim
        embed_dim: per-agent embedding dimension (default 64)
        num_heads: number of attention heads (default 4)
    """

    def __init__(
        self,
        agent_order: List[str],
        slot_obs_dims: dict,
        embed_dim: int = 64,
        num_heads: int = 4,
    ):
        super().__init__()
        self._agent_order = list(agent_order)
        self._slot_obs_dims = dict(slot_obs_dims)
        self._embed_dim = embed_dim
        self._n_agents = len(agent_order)

        self._type_projectors = nn.ModuleDict()
        for agent_id in agent_order:
            type_key = self._type_key(agent_id)
            if type_key not in self._type_projectors:
                obs_dim = self._resolve_slot_dim(agent_id)
                self._type_projectors[type_key] = nn.Linear(obs_dim, embed_dim)

        self._slot_dims = [self._resolve_slot_dim(aid) for aid in agent_order]
        self._split_sizes = self._slot_dims + [self._n_agents]

        self.attention = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            batch_first=True,
        )

        total_expected = sum(self._split_sizes)
        logger.info(
            f"AttentionCriticEncoder: {self._n_agents} agents, "
            f"embed_dim={embed_dim}, heads={num_heads}, "
            f"split_sizes={self._split_sizes}, total_input={total_expected}D"
        )

    @staticmethod
    def _type_key(agent_id: str) -> str:
        if agent_id.startswith("vehicle"):
            return "vehicle"
        elif agent_id.startswith("pedestrian"):
            return "pedestrian"
        return "other"

    def _resolve_slot_dim(self, agent_id: str) -> int:
        type_key = self._type_key(agent_id)
        if type_key in self._slot_obs_dims:
            return int(self._slot_obs_dims[type_key])
        raise ValueError(f"No slot dim for agent type '{type_key}' (agent_id={agent_id})")

    def forward(self, global_obs: torch.Tensor) -> torch.Tensor:
        """(batch, global_obs_dim) → (batch, embed_dim) attended + mean-pooled."""
        parts = global_obs.split(self._split_sizes, dim=-1)
        slot_tensors = parts[:self._n_agents]
        alive_mask_raw = parts[self._n_agents]

        embeddings = []
        for i, agent_id in enumerate(self._agent_order):
            type_key = self._type_key(agent_id)
            proj = self._type_projectors[type_key](slot_tensors[i])
            embeddings.append(proj)

        tokens = torch.stack(embeddings, dim=1)

        key_padding_mask = (alive_mask_raw < 0.5)
        all_masked = key_padding_mask.all(dim=1)
        if all_masked.any():
            key_padding_mask[all_masked] = False

        attended, _ = self.attention(
            tokens, tokens, tokens,
            key_padding_mask=key_padding_mask,
        )

        alive_weights = (~key_padding_mask).float().unsqueeze(-1)
        pooled = (attended * alive_weights).sum(dim=1) / alive_weights.sum(dim=1).clamp(min=1.0)

        return pooled

# ---------------------------------------------------------------------------
# GNN Critic Encoder (Block 4.5)
# ---------------------------------------------------------------------------

class GraphConvLayer(nn.Module):
    """GraphSAGE-style mean aggregation with separate self/neighbor weights.

    h_i' = W_self h_i + W_neigh * mean_{j alive, j != i} h_j

    Reference: Hamilton et al. 2017 (GraphSAGE, NeurIPS).
    """

    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.w_self = nn.Linear(in_dim, out_dim)
        self.w_neigh = nn.Linear(in_dim, out_dim)

    def forward(self, x: torch.Tensor, alive: torch.Tensor) -> torch.Tensor:
        # x: (B, N, D_in); alive: (B, N) with 1.0=alive, 0.0=dead
        alive_exp = alive.unsqueeze(-1)
        total = (x * alive_exp).sum(dim=1, keepdim=True)
        neigh_sum = total - x * alive_exp
        alive_count = alive_exp.sum(dim=1, keepdim=True) - alive_exp
        neigh_mean = neigh_sum / alive_count.clamp(min=1.0)
        return self.w_self(x) + self.w_neigh(neigh_mean)


class GATLayer(nn.Module):
    """Graph Attention layer (Veličković et al. 2018, ICLR), multi-head,
    fully-connected graph with alive-mask over keys.

    e_ij = LeakyReLU(a_src^T Wh_i + a_dst^T Wh_j)
    alpha_ij = softmax_j(e_ij) over alive keys
    h_i' = sum_j alpha_ij * W h_j  (concat heads when concat=True)
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        num_heads: int = 4,
        concat: bool = True,
    ):
        super().__init__()
        if concat and out_dim % num_heads != 0:
            raise ValueError(
                f"out_dim ({out_dim}) must be divisible by num_heads "
                f"({num_heads}) when concat=True"
            )
        self.num_heads = num_heads
        self.concat = concat
        self.head_dim = out_dim // num_heads if concat else out_dim
        self.w = nn.Linear(in_dim, num_heads * self.head_dim, bias=False)
        self.a_src = nn.Parameter(torch.empty(num_heads, self.head_dim))
        self.a_dst = nn.Parameter(torch.empty(num_heads, self.head_dim))
        nn.init.xavier_uniform_(self.a_src)
        nn.init.xavier_uniform_(self.a_dst)
        self.leaky_relu = nn.LeakyReLU(0.2)

    def forward(self, x: torch.Tensor, alive: torch.Tensor) -> torch.Tensor:
        # x: (B, N, D_in); alive: (B, N)
        B, N, _ = x.shape
        h = self.w(x).view(B, N, self.num_heads, self.head_dim)
        a_src = self.a_src.view(1, 1, self.num_heads, self.head_dim)
        a_dst = self.a_dst.view(1, 1, self.num_heads, self.head_dim)
        alpha_src = (h * a_src).sum(dim=-1)
        alpha_dst = (h * a_dst).sum(dim=-1)
        # e[b, i, j, head] = alpha_src[b, i, head] + alpha_dst[b, j, head]
        e = alpha_src.unsqueeze(2) + alpha_dst.unsqueeze(1)
        e = self.leaky_relu(e)
        key_mask = alive.unsqueeze(1).unsqueeze(-1)
        e = e.masked_fill(key_mask < 0.5, float("-inf"))
        attn = torch.softmax(e, dim=2)
        # Safety: all-masked rows produce NaN after softmax(-inf) -> zero them out.
        attn = torch.nan_to_num(attn, nan=0.0)
        # out[b, i, head, d] = sum_j attn[b, i, j, head] * h[b, j, head, d]
        out = (attn.unsqueeze(-1) * h.unsqueeze(1)).sum(dim=2)
        if self.concat:
            return out.reshape(B, N, self.num_heads * self.head_dim)
        return out.mean(dim=2)


class GNNCriticEncoder(nn.Module):
    """GNN encoder for centralized critic, parallel to AttentionCriticEncoder.

    - use_attention=False → stacked GraphConvLayer (uniform mean aggregation)
    - use_attention=True  → stacked GATLayer (learned edge attention)

    Pipeline: split slots → per-type Linear(obs→embed) → L × {GraphConv|GAT}(ELU)
              → masked mean-pool → (B, embed_dim)
    """

    def __init__(
        self,
        agent_order: List[str],
        slot_obs_dims: dict,
        embed_dim: int = 64,
        num_heads: int = 4,
        num_layers: int = 2,
        use_attention: bool = False,
    ):
        super().__init__()
        self._agent_order = list(agent_order)
        self._slot_obs_dims = dict(slot_obs_dims)
        self._embed_dim = embed_dim
        self._n_agents = len(agent_order)
        self._use_attention = use_attention

        self._type_projectors = nn.ModuleDict()
        for agent_id in agent_order:
            type_key = AttentionCriticEncoder._type_key(agent_id)
            if type_key not in self._type_projectors:
                obs_dim = self._resolve_slot_dim(agent_id)
                self._type_projectors[type_key] = nn.Linear(obs_dim, embed_dim)

        self._slot_dims = [self._resolve_slot_dim(aid) for aid in agent_order]
        self._split_sizes = self._slot_dims + [self._n_agents]

        if use_attention:
            self.layers = nn.ModuleList([
                GATLayer(embed_dim, embed_dim, num_heads=num_heads, concat=True)
                for _ in range(num_layers)
            ])
        else:
            self.layers = nn.ModuleList([
                GraphConvLayer(embed_dim, embed_dim)
                for _ in range(num_layers)
            ])
        self.activation = nn.ELU()

        kind = "GAT" if use_attention else "GraphConv"
        heads_info = num_heads if use_attention else "-"
        logger.info(
            f"GNNCriticEncoder: {self._n_agents} agents, embed_dim={embed_dim}, "
            f"layers={num_layers}, kind={kind}, heads={heads_info}, "
            f"split_sizes={self._split_sizes}, total_input={sum(self._split_sizes)}D"
        )

    def _resolve_slot_dim(self, agent_id: str) -> int:
        type_key = AttentionCriticEncoder._type_key(agent_id)
        if type_key in self._slot_obs_dims:
            return int(self._slot_obs_dims[type_key])
        raise ValueError(f"No slot dim for agent type '{type_key}' (agent_id={agent_id})")

    def forward(self, global_obs: torch.Tensor) -> torch.Tensor:
        """(batch, global_obs_dim) → (batch, embed_dim) message-passed + mean-pooled."""
        parts = global_obs.split(self._split_sizes, dim=-1)
        slot_tensors = parts[:self._n_agents]
        alive_raw = parts[self._n_agents]

        embeddings = []
        for i, agent_id in enumerate(self._agent_order):
            type_key = AttentionCriticEncoder._type_key(agent_id)
            embeddings.append(self._type_projectors[type_key](slot_tensors[i]))
        x = torch.stack(embeddings, dim=1)

        alive = (alive_raw >= 0.5).float()
        all_dead = (alive.sum(dim=1, keepdim=True) < 0.5)
        alive = torch.where(all_dead.expand_as(alive), torch.ones_like(alive), alive)

        for layer in self.layers:
            x = self.activation(layer(x, alive))

        alive_w = alive.unsqueeze(-1)
        pooled = (x * alive_w).sum(dim=1) / alive_w.sum(dim=1).clamp(min=1.0)
        return pooled

# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class CentralizedCriticModel(TorchModelV2, nn.Module):
    """
    MAPPO model: separate actor (local obs) + centralized critic (global obs).

    custom_model_config keys:
        hidden_size (int): hidden layer size (default 256)
        n_hidden_layers (int): number of hidden layers (default 2)
        global_obs_dim (int): dimension of concatenated global observation
        agent_order (list[str]): canonical fixed slot order for all agents
        slot_obs_dims (dict): expected obs dim by agent type/prefix
        use_popart (bool): enable PopArt value normalization (default False)
        popart_beta (float): EMA decay for PopArt stats (default 3e-4)
    """

    def __init__(
        self,
        obs_space,
        action_space,
        num_outputs,
        model_config,
        name,
        **kwargs,
    ):
        TorchModelV2.__init__(self, obs_space, action_space, num_outputs,
                              model_config, name)
        nn.Module.__init__(self)

        custom = _resolve_custom_model_config(model_config, kwargs)
        hidden = int(custom.get("hidden_size", 256))
        n_layers = int(custom.get("n_hidden_layers", 2))
        self._global_obs_dim = int(custom.get("global_obs_dim", 42))
        self._agent_order = list(custom.get("agent_order", []))
        self._slot_obs_dims = dict(custom.get("slot_obs_dims", {}))
        self._use_popart = bool(custom.get("use_popart", False))
        popart_beta = float(custom.get("popart_beta", 3e-4))
        self._use_attention = bool(custom.get("use_attention", False))
        attention_embed_dim = int(custom.get("attention_embed_dim", 64))
        attention_heads = int(custom.get("attention_heads", 4))
        self._use_gnn = bool(custom.get("use_gnn", False))
        gnn_embed_dim = int(custom.get("gnn_embed_dim", 64))
        gnn_heads = int(custom.get("gnn_heads", 4))
        gnn_layers = int(custom.get("gnn_layers", 2))

        local_obs_dim = int(np.prod(obs_space.shape))

        # Actor: local obs → action logits
        actor_layers = []
        in_dim = local_obs_dim
        for _ in range(n_layers):
            actor_layers.extend([nn.Linear(in_dim, hidden), nn.Tanh()])
            in_dim = hidden
        actor_layers.append(nn.Linear(in_dim, num_outputs))
        self.actor = nn.Sequential(*actor_layers)

        # Critic encoder dispatch: GNN > Attention > MLP (Blocks 4.4 + 4.5)
        if self._use_gnn:
            if not self._agent_order or not self._slot_obs_dims:
                raise ValueError(
                    "use_gnn=True requires agent_order and slot_obs_dims"
                )
            self.critic_attention = None
            self.critic_gnn = GNNCriticEncoder(
                agent_order=self._agent_order,
                slot_obs_dims=self._slot_obs_dims,
                embed_dim=gnn_embed_dim,
                num_heads=gnn_heads,
                num_layers=gnn_layers,
                use_attention=self._use_attention,
            )
            self.critic_hidden = nn.Sequential(
                nn.Linear(gnn_embed_dim, hidden),
                nn.Tanh(),
            )
        elif self._use_attention:
            if not self._agent_order or not self._slot_obs_dims:
                raise ValueError(
                    "use_attention=True requires agent_order and slot_obs_dims"
                )
            self.critic_attention = AttentionCriticEncoder(
                agent_order=self._agent_order,
                slot_obs_dims=self._slot_obs_dims,
                embed_dim=attention_embed_dim,
                num_heads=attention_heads,
            )
            self.critic_gnn = None
            self.critic_hidden = nn.Sequential(
                nn.Linear(attention_embed_dim, hidden),
                nn.Tanh(),
            )
        else:
            self.critic_attention = None
            self.critic_gnn = None
            critic_hidden_layers = []
            in_dim = self._global_obs_dim
            for _ in range(n_layers):
                critic_hidden_layers.extend([nn.Linear(in_dim, hidden), nn.Tanh()])
                in_dim = hidden
            self.critic_hidden = nn.Sequential(*critic_hidden_layers)

        # Critic output: either PopArt or plain Linear
        if self._use_popart:
            self.critic_head = PopArtLayer(hidden, beta=popart_beta)
            logger.info(f"PopArt enabled (beta={popart_beta})")
        else:
            self.critic_head = nn.Linear(hidden, 1)

        self._cur_value = None

        if self._use_gnn:
            encoder_name = "GAT" if self._use_attention else "GNN"
        elif self._use_attention:
            encoder_name = "MLP+Attn"
        else:
            encoder_name = "MLP"
        logger.info(
            f"CentralizedCriticModel '{name}': "
            f"actor {local_obs_dim}->{num_outputs}, "
            f"critic {self._global_obs_dim}->1 [{encoder_name}], "
            f"hidden={hidden}x{n_layers}, "
            f"popart={self._use_popart}, "
            f"agent_order={len(self._agent_order) if self._agent_order else 'dynamic'}"
        )

    @override(ModelV2)
    def forward(self, input_dict, state, seq_lens):
        obs = input_dict["obs_flat"].float()
        _raise_on_nonfinite_torch("obs_flat", obs)
        action_logits = self.actor(obs)

        # Clamp NaN logits to zero (uniform action) instead of crashing
        if not torch.isfinite(action_logits).all():
            n_bad = (~torch.isfinite(action_logits)).sum().item()
            logger.warning(
                "action_logits has %d non-finite values — clamping to 0",
                n_bad,
            )
            action_logits = torch.nan_to_num(action_logits, nan=0.0, posinf=0.0, neginf=0.0)

        # During training: global_obs injected by callbacks
        if GLOBAL_OBS in input_dict:
            global_obs = input_dict[GLOBAL_OBS].float()
        else:
            # Rollout: critic uses local obs zero-padded
            batch_sz = obs.shape[0]
            global_obs = torch.zeros(
                batch_sz, self._global_obs_dim, device=obs.device
            )
            global_obs[:, :obs.shape[-1]] = obs

        _raise_on_nonfinite_torch("global_obs", global_obs)

        if self.critic_gnn is not None:
            encoded = self.critic_gnn(global_obs)
            critic_features = self.critic_hidden(encoded)
        elif self.critic_attention is not None:
            attended = self.critic_attention(global_obs)
            critic_features = self.critic_hidden(attended)
        else:
            critic_features = self.critic_hidden(global_obs)
        normalized_value = self.critic_head(critic_features).squeeze(-1)

        # Denormalize if PopArt is enabled
        if self._use_popart and isinstance(self.critic_head, PopArtLayer):
            self._cur_value = self.critic_head.denormalize(
                normalized_value.unsqueeze(-1)
            ).squeeze(-1)
        else:
            self._cur_value = normalized_value

        # Clamp NaN values instead of crashing
        if not torch.isfinite(self._cur_value).all():
            n_bad = (~torch.isfinite(self._cur_value)).sum().item()
            logger.warning(
                "value_function has %d non-finite values — clamping to 0",
                n_bad,
            )
            self._cur_value = torch.nan_to_num(self._cur_value, nan=0.0, posinf=0.0, neginf=0.0)
        return action_logits, state

    @override(ModelV2)
    def value_function(self):
        assert self._cur_value is not None, "forward() must be called first"
        return self._cur_value

    def critic_forward_raw(self, global_obs: torch.Tensor) -> torch.Tensor:
        """Raw critic inference: global_obs → denormalized value.

        Used by callbacks for VF recomputation (bypasses actor).
        """
        
        if self.critic_gnn is not None:
            encoded = self.critic_gnn(global_obs)
            features = self.critic_hidden(encoded)
        elif self.critic_attention is not None:
            attended = self.critic_attention(global_obs)
            features = self.critic_hidden(attended)
        else:
            features = self.critic_hidden(global_obs)
        normalized = self.critic_head(features).squeeze(-1)
        if self._use_popart and isinstance(self.critic_head, PopArtLayer):
            return self.critic_head.denormalize(normalized.unsqueeze(-1)).squeeze(-1)
        return normalized


# ---------------------------------------------------------------------------
# Callbacks: inject global_obs + recompute GAE + custom metrics
# ---------------------------------------------------------------------------

class CentralizedCriticCallbacks(DefaultCallbacks):
    """
    RLlib callbacks for centralized critic MAPPO.

    on_episode_start():  init per-episode outcome tracking
    on_episode_step():   capture termination_reason when each agent terminates
    on_episode_end():    aggregate per-policy metrics → TensorBoard custom_metrics
    on_postprocess_trajectory():
      1. Build fixed-slot global_obs with alive_mask (Block 4.1)
      2. Recompute VF predictions with global_obs
      3. Update PopArt statistics if enabled (Block 4.2)
      4. Recompute GAE advantages
    """

    # ------------------------------------------------------------------
    # Episode-level metrics
    # ------------------------------------------------------------------

    def on_episode_start(
        self, *, worker, base_env, policies, episode,
        env_index=None, **kwargs,
    ):
        """Initialize per-episode agent outcome tracking."""
        episode.user_data["agent_outcomes"] = {}

    def on_episode_step(
        self, *, worker, base_env, policies, episode,
        env_index=None, **kwargs,
    ):
        """Capture each agent's termination info at the step they terminate.

        Reads from two sources:
          1. episode.last_info_for() — for agents still in the obs dict
          2. env._terminated_agent_infos — side-channel for agents whose
             info can't be in the RLlib infos dict (keys must ⊆ obs keys)
        """
        # --- Source 1: standard RLlib info path ---
        for agent_id in episode.get_agents():
            if agent_id in episode.user_data["agent_outcomes"]:
                continue
            info = episode.last_info_for(agent_id)
            if not info:
                continue
            tr = info.get("termination_reason")
            if tr and tr != "alive":
                episode.user_data["agent_outcomes"][agent_id] = {
                    "termination_reason": tr,
                    "route_completion": info.get("route_completion", 0.0),
                    "path_efficiency": info.get("path_efficiency", 0.0),
                }

        # --- Source 2: side-channel for terminated agents ---
        try:
            raw_env = base_env.get_sub_environments()[0]
            inner = getattr(raw_env, "par_env", None) or getattr(raw_env, "env", None)
            if inner is None:
                inner = raw_env
            term_infos = getattr(inner, "_terminated_agent_infos", {})
            for agent_id, info in term_infos.items():
                if agent_id in episode.user_data["agent_outcomes"]:
                    continue
                tr = info.get("termination_reason")
                if tr and tr != "alive":
                    episode.user_data["agent_outcomes"][agent_id] = {
                        "termination_reason": tr,
                        "route_completion": info.get("route_completion", 0.0),
                        "path_efficiency": info.get("path_efficiency", 0.0),
                    }
        except (AttributeError, IndexError):
            pass

    def on_episode_end(
        self, *, worker, base_env, policies, episode,
        env_index=None, **kwargs,
    ):
        """Aggregate per-policy metrics → TensorBoard custom_metrics."""
        outcomes = episode.user_data.get("agent_outcomes", {})

        for policy_id in ("vehicle_policy", "pedestrian_policy"):
            prefix = "vehicle" if policy_id == "vehicle_policy" else "pedestrian"
            agent_data = {
                aid: out for aid, out in outcomes.items()
                if aid.startswith(prefix)
            }
            n = len(agent_data)
            if n == 0:
                continue

            reasons = [d["termination_reason"] for d in agent_data.values()]
            episode.custom_metrics[f"{policy_id}/success_rate"] = (
                reasons.count("route_complete") / n
            )
            episode.custom_metrics[f"{policy_id}/collision_rate"] = (
                reasons.count("collision") / n
            )
            episode.custom_metrics[f"{policy_id}/offroad_rate"] = (
                reasons.count("offroad") / n
            )
            episode.custom_metrics[f"{policy_id}/stuck_rate"] = (
                reasons.count("stuck") / n
            )
            episode.custom_metrics[f"{policy_id}/timeout_rate"] = (
                reasons.count("timeout") / n
            )
            episode.custom_metrics[f"{policy_id}/route_completion"] = float(
                np.mean([d["route_completion"] for d in agent_data.values()])
            )
            episode.custom_metrics[f"{policy_id}/path_efficiency"] = float(
                np.mean([d["path_efficiency"] for d in agent_data.values()])
            )

        # Debug: warn about agents with no captured outcome
        all_agents = set(episode.get_agents())
        missing = all_agents - set(outcomes.keys())
        if missing:
            logger.warning(
                f"Episode {episode.episode_id}: no termination_reason for: {missing}"
            )

    # ------------------------------------------------------------------
    # Trajectory postprocessing — fixed-slot global_obs (Block 4.1)
    # ------------------------------------------------------------------

    def on_postprocess_trajectory(
        self,
        *,
        worker,
        episode,
        agent_id,
        policy_id,
        policies,
        postprocessed_batch: SampleBatch,
        original_batches: Dict,
        **kwargs,
    ):
        own_obs = postprocessed_batch[SampleBatch.CUR_OBS]
        _raise_on_nonfinite_np(f"{agent_id}.own_obs", own_obs)
        batch_size = own_obs.shape[0]

        policy = policies[policy_id]
        model = policy.model
        expected_dim = model._global_obs_dim

        # --- Build canonical slot order from config (preferred) or observed agents ---
        if getattr(model, "_agent_order", None):
            slot_order = list(model._agent_order)
        else:
            all_agent_ids = set(original_batches.keys())
            all_agent_ids.add(agent_id)
            slot_order = _build_slot_order(all_agent_ids)
        n_agents = len(slot_order)

        # --- Extract obs per agent, aligned to batch_size ---
        agent_obs_map = {}
        agent_alive_map = {}
        for aid in slot_order:
            obs_dim = _slot_obs_dim_for_agent(aid, model, fallback=own_obs.shape[-1])
            if aid == agent_id:
                agent_obs_map[aid] = own_obs.astype(np.float32, copy=False)
                agent_alive_map[aid] = np.ones((batch_size,), dtype=np.float32)
                continue

            if aid not in original_batches:
                agent_obs_map[aid] = None
                agent_alive_map[aid] = np.zeros((batch_size,), dtype=np.float32)
                continue

            other_data = original_batches[aid]
            if isinstance(other_data, tuple):
                other_batch = other_data[-1]
            else:
                other_batch = other_data
            opp_obs = other_batch[SampleBatch.CUR_OBS]
            _raise_on_nonfinite_np(f"{agent_id}.opp_obs[{aid}]", opp_obs)

            if opp_obs.shape[-1] != obs_dim:
                raise ValueError(
                    f"{agent_id}: opp_obs[{aid}] dim {opp_obs.shape[-1]} "
                    f"!= expected slot dim {obs_dim}"
                )

            alive = np.zeros((batch_size,), dtype=np.float32)
            if len(opp_obs) > batch_size:
                opp_obs = opp_obs[:batch_size]
                alive[:] = 1.0
            elif len(opp_obs) < batch_size:
                logger.debug(
                    f"{agent_id}: padding opp_obs[{aid}] "
                    f"from {len(opp_obs)} to {batch_size} (zero-fill missing rows)"
                )
                pad_n = batch_size - len(opp_obs)
                pad = np.zeros((pad_n, obs_dim), dtype=np.float32)
                opp_obs = np.concatenate([opp_obs.astype(np.float32), pad], axis=0)
                alive[:len(other_batch[SampleBatch.CUR_OBS])] = 1.0
            else:
                alive[:] = 1.0
            agent_obs_map[aid] = opp_obs.astype(np.float32, copy=False)
            agent_alive_map[aid] = alive

        # --- Assemble fixed-slot global_obs ---
        slots = []
        alive_mask = np.zeros((batch_size, n_agents), dtype=np.float32)

        for i, aid in enumerate(slot_order):
            obs_dim = _slot_obs_dim_for_agent(aid, model, fallback=own_obs.shape[-1])
            obs_data = agent_obs_map.get(aid)

            if obs_data is not None and obs_data.shape[-1] == obs_dim:
                slots.append(obs_data.astype(np.float32, copy=False))
                alive_mask[:, i] = agent_alive_map.get(
                    aid, np.zeros((batch_size,), dtype=np.float32)
                )
            else:
                if obs_data is not None:
                    logger.warning(
                        f"{agent_id}: slot {aid} obs dim {obs_data.shape[-1]} != "
                        f"expected {obs_dim}, zero-filling slot"
                    )
                slots.append(np.zeros((batch_size, obs_dim), dtype=np.float32))

        slots.append(alive_mask)
        global_obs = np.concatenate(slots, axis=-1)

        if global_obs.shape[-1] != expected_dim:
            raise ValueError(
                f"{agent_id}: assembled global_obs dim {global_obs.shape[-1]} != "
                f"expected {expected_dim}"
            )

        _raise_on_nonfinite_np(f"{agent_id}.global_obs", global_obs)
        postprocessed_batch[GLOBAL_OBS] = global_obs

        # --- Recompute VF predictions with global_obs ---
        device = next(model.parameters()).device
        with torch.no_grad():
            global_obs_t = torch.as_tensor(global_obs, dtype=torch.float32, device=device)
            vf_preds = model.critic_forward_raw(global_obs_t).cpu().numpy()

        _raise_on_nonfinite_np(f"{agent_id}.vf_preds", vf_preds)
        postprocessed_batch[SampleBatch.VF_PREDS] = vf_preds

        # --- Bootstrap value for incomplete trajectories ---
        last_r = 0.0
        if not postprocessed_batch[SampleBatch.TERMINATEDS][-1]:
            last_slots = []
            last_mask = np.zeros((1, n_agents), dtype=np.float32)

            for i, aid in enumerate(slot_order):
                obs_dim = _slot_obs_dim_for_agent(aid, model, fallback=own_obs.shape[-1])
                obs_data = agent_obs_map.get(aid)
                if obs_data is not None and obs_data.shape[-1] == obs_dim:
                    last_slots.append(obs_data[-1:].astype(np.float32))
                    last_mask[:, i] = agent_alive_map.get(
                        aid, np.zeros((batch_size,), dtype=np.float32)
                    )[-1]
                else:
                    last_slots.append(np.zeros((1, obs_dim), dtype=np.float32))

            last_slots.append(last_mask)
            last_global = np.concatenate(last_slots, axis=-1)

            if last_global.shape[-1] != expected_dim:
                raise ValueError(
                    f"{agent_id}: bootstrap last_global dim {last_global.shape[-1]} "
                    f"!= expected {expected_dim}"
                )

            _raise_on_nonfinite_np(f"{agent_id}.last_global", last_global)
            with torch.no_grad():
                t = torch.as_tensor(last_global, dtype=torch.float32, device=device)
                last_r = model.critic_forward_raw(t).item()
            if not np.isfinite(last_r):
                raise ValueError(f"{agent_id}.last_r is non-finite: {last_r}")

        # --- Recompute GAE ---
        train_batch = compute_advantages(
            postprocessed_batch,
            last_r,
            policy.config["gamma"],
            policy.config["lambda"],
            use_gae=True,
            use_critic=True,
        )

        # --- PopArt: update statistics with value targets (Block 4.2) ---
        if model._use_popart and isinstance(model.critic_head, PopArtLayer):
            value_targets = train_batch[Postprocessing.VALUE_TARGETS]
            targets_t = torch.as_tensor(
                value_targets, dtype=torch.float32, device=device
            )
            model.critic_head.update(targets_t)

        return train_batch
