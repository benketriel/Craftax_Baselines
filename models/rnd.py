import jax.numpy as jnp
import flax.linen as nn
import numpy as np
from flax.linen.initializers import constant, orthogonal
from typing import Sequence

import distrax

from models.alt_activations import get_activation


class RNDNetwork(nn.Module):
    layer_size: int
    output_dim: int
    num_layers: int

    @nn.compact
    def __call__(self, x):
        activation = nn.relu

        emb = x
        for _ in range(self.num_layers):
            emb = nn.Dense(
                self.layer_size,
            )(emb)
            emb = activation(emb)

        emb = nn.Dense(self.output_dim)(emb)

        return emb


class ActorCriticRND(nn.Module):
    action_dim: Sequence[int]
    layer_width: int
    activation: str = "tanh"

    @nn.compact
    def __call__(self, x):
        if self.activation == "relu":
            activation = nn.relu
        else:
            activation = nn.tanh

        actor_mean = nn.Dense(
            self.layer_width,
            kernel_init=orthogonal(np.sqrt(2)),
            bias_init=constant(0.0),
        )(x)
        actor_mean = activation(actor_mean)

        actor_mean = nn.Dense(
            self.layer_width,
            kernel_init=orthogonal(np.sqrt(2)),
            bias_init=constant(0.0),
        )(actor_mean)
        actor_mean = activation(actor_mean)

        actor_mean = nn.Dense(
            self.layer_width,
            kernel_init=orthogonal(np.sqrt(2)),
            bias_init=constant(0.0),
        )(actor_mean)
        actor_mean = activation(actor_mean)

        actor_mean = nn.Dense(
            self.action_dim, kernel_init=orthogonal(0.01), bias_init=constant(0.0)
        )(actor_mean)
        pi = distrax.Categorical(logits=actor_mean)

        # Extrinsic reward
        critic_e = nn.Dense(
            self.layer_width,
            kernel_init=orthogonal(np.sqrt(2)),
            bias_init=constant(0.0),
        )(x)
        critic_e = activation(critic_e)

        critic_e = nn.Dense(
            self.layer_width,
            kernel_init=orthogonal(np.sqrt(2)),
            bias_init=constant(0.0),
        )(critic_e)
        critic_e = activation(critic_e)

        critic_e = nn.Dense(
            self.layer_width,
            kernel_init=orthogonal(np.sqrt(2)),
            bias_init=constant(0.0),
        )(critic_e)
        critic_e = activation(critic_e)

        critic_e = nn.Dense(1, kernel_init=orthogonal(1.0), bias_init=constant(0.0))(
            critic_e
        )

        # Intrinsic reward
        critic_i = nn.Dense(
            self.layer_width,
            kernel_init=orthogonal(np.sqrt(2)),
            bias_init=constant(0.0),
        )(x)
        critic_i = activation(critic_i)

        critic_i = nn.Dense(
            self.layer_width,
            kernel_init=orthogonal(np.sqrt(2)),
            bias_init=constant(0.0),
        )(critic_i)
        critic_i = activation(critic_i)

        critic_i = nn.Dense(
            self.layer_width,
            kernel_init=orthogonal(np.sqrt(2)),
            bias_init=constant(0.0),
        )(critic_i)
        critic_i = activation(critic_i)

        critic_i = nn.Dense(1, kernel_init=orthogonal(1.0), bias_init=constant(0.0))(
            critic_i
        )

        return pi, jnp.squeeze(critic_e, axis=-1), jnp.squeeze(critic_i, axis=-1)


class MLP(nn.Module):
    hidden_size: int
    num_layers: int
    out_dim: int
    activation: str
    final_activation: bool = False

    @nn.compact
    def __call__(self, x):
        act = get_activation(self.activation)
        for _ in range(self.num_layers):
            x = nn.Dense(self.hidden_size)(x)
            x = act(x)
        x = nn.Dense(self.out_dim)(x)
        if self.final_activation:
            x = act(x)
        return x



class ActorCriticRNDTiled(nn.Module):
    # keep original interface knobs
    action_dim: int
    layer_width: int
    activation: str = "tanh"

    # tiled preprocessing
    grid_h: int = 7
    grid_w: int = 9
    tile_in: int = 21
    tile_out: int = 10
    rest_out: int = 20
    tile_num_layers: int = 2           # per-tile MLP depth
    tile_hidden_size: int = 128        # per-tile MLP width
    tile_activation: str = "relog"     # per-tile MLP activation

    @nn.compact
    def __call__(self, obs: jnp.ndarray):
        """
        obs: (..., grid_h*grid_w*tile_in + rest_dim)

        Returns:
          pi: distrax.Categorical
          value_e: (...,) extrinsic value
          value_i: (...,) intrinsic value
        """
        # --------- 1) Tiled preprocessing (shared MLP over each tile) ----------
        *batch, feat = obs.shape
        grid_feat = self.grid_h * self.grid_w * self.tile_in
        if feat < grid_feat:
            raise ValueError(f"Input feature dim {feat} < required grid features {grid_feat}")

        tiles_flat = obs[..., :grid_feat]
        rest = obs[..., grid_feat:]

        tiles = tiles_flat.reshape(*batch, self.grid_h, self.grid_w, self.tile_in)

        # shared per-tile MLP
        tile_mlp = MLP(
            hidden_size=self.tile_hidden_size,
            num_layers=self.tile_num_layers,
            out_dim=self.tile_out,
            activation=self.tile_activation,
            final_activation=True,
        )
        tiles_enc = tile_mlp(tiles.reshape(-1, self.tile_in))
        tiles_enc = tiles_enc.reshape(*batch, self.grid_h * self.grid_w * self.tile_out)

        # rest encoder
        rest_mlp = MLP(
            hidden_size=self.tile_hidden_size,
            num_layers=self.tile_num_layers,
            out_dim=self.rest_out,
            activation=self.tile_activation,
            final_activation=True,
        )
        rest_enc = rest_mlp(rest)

        # latent that replaces "x" in the original heads
        x = jnp.concatenate([tiles_enc, rest_enc], axis=-1)

        # choose head activation like the original
        if self.activation == "relu":
            act = nn.relu
        else:
            act = nn.tanh

        # -------------------- 2) Actor head (3 layers) ------------------------
        actor = nn.Dense(
            self.layer_width,
            kernel_init=orthogonal(jnp.sqrt(2.0)),
            bias_init=constant(0.0),
        )(x)
        actor = act(actor)
        actor = nn.Dense(
            self.layer_width,
            kernel_init=orthogonal(jnp.sqrt(2.0)),
            bias_init=constant(0.0),
        )(actor)
        actor = act(actor)
        actor = nn.Dense(
            self.layer_width,
            kernel_init=orthogonal(jnp.sqrt(2.0)),
            bias_init=constant(0.0),
        )(actor)
        actor = act(actor)
        logits = nn.Dense(
            self.action_dim,
            kernel_init=orthogonal(0.01),
            bias_init=constant(0.0),
        )(actor)
        pi = distrax.Categorical(logits=logits)

        # ---------------- Extrinsic critic head (3 layers) --------------------
        ce = nn.Dense(
            self.layer_width,
            kernel_init=orthogonal(jnp.sqrt(2.0)),
            bias_init=constant(0.0),
        )(x)
        ce = act(ce)
        ce = nn.Dense(
            self.layer_width,
            kernel_init=orthogonal(jnp.sqrt(2.0)),
            bias_init=constant(0.0),
        )(ce)
        ce = act(ce)
        ce = nn.Dense(
            self.layer_width,
            kernel_init=orthogonal(jnp.sqrt(2.0)),
            bias_init=constant(0.0),
        )(ce)
        ce = act(ce)
        value_e = nn.Dense(
            1,
            kernel_init=orthogonal(1.0),
            bias_init=constant(0.0),
        )(ce)

        # ---------------- Intrinsic critic (RND) head (3 layers) --------------
        ci = nn.Dense(
            self.layer_width,
            kernel_init=orthogonal(jnp.sqrt(2.0)),
            bias_init=constant(0.0),
        )(x)
        ci = act(ci)
        ci = nn.Dense(
            self.layer_width,
            kernel_init=orthogonal(jnp.sqrt(2.0)),
            bias_init=constant(0.0),
        )(ci)
        ci = act(ci)
        ci = nn.Dense(
            self.layer_width,
            kernel_init=orthogonal(jnp.sqrt(2.0)),
            bias_init=constant(0.0),
        )(ci)
        ci = act(ci)
        value_i = nn.Dense(
            1,
            kernel_init=orthogonal(1.0),
            bias_init=constant(0.0),
        )(ci)

        return pi, jnp.squeeze(value_e, axis=-1), jnp.squeeze(value_i, axis=-1)
