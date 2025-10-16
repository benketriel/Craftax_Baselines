import jax.numpy as jnp
import flax.linen as nn

from models.alt_activations import get_activation


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


class ENTEncoderTiled(nn.Module):
    layer_size: int
    num_layers: int
    activation: str = "relog"

    grid_h: int = 7
    grid_w: int = 9
    tile_in: int = 21
    tile_out: int = 10
    rest_out: int = 20

    @nn.compact
    def __call__(self, obs: jnp.ndarray) -> jnp.ndarray:
        """
        obs shape: (..., grid_h*grid_w*tile_in + rest_dim)
        returns latent of shape (..., grid_h*grid_w*tile_out + rest_out)
        """
        *batch, feat = obs.shape
        grid_feat = self.grid_h * self.grid_w * self.tile_in
        if feat < grid_feat:
            raise ValueError(f"Input feature dim {feat} < required grid features {grid_feat}")

        tiles_flat = obs[..., :grid_feat]
        rest = obs[..., grid_feat:]

        tiles = tiles_flat.reshape(*batch, self.grid_h, self.grid_w, self.tile_in)
        tile_mlp = MLP(self.layer_size, self.num_layers, self.tile_out, self.activation, final_activation=True)
        tiles_enc = tile_mlp(tiles.reshape(-1, self.tile_in)).reshape(*batch, self.grid_h * self.grid_w * self.tile_out)

        rest_mlp = MLP(self.layer_size, self.num_layers, self.rest_out, self.activation, final_activation=True)
        rest_enc = rest_mlp(rest)

        latent = jnp.concatenate([tiles_enc, rest_enc], axis=-1)
        return latent


class ENTDecoderTiled(nn.Module):
    layer_size: int
    num_layers: int
    output_dim: int
    activation: str = "relog"

    grid_h: int = 7
    grid_w: int = 9
    tile_in: int = 21
    tile_out: int = 10
    rest_out: int = 20

    @nn.compact
    def __call__(self, latent: jnp.ndarray) -> jnp.ndarray:
        """
        latent shape: (..., grid_h*grid_w*tile_out + rest_out)
        returns obs shaped (..., grid_h*grid_w*tile_in + rest_dim)
        """
        *batch, feat = latent.shape
        expected_tile_lat = self.grid_h * self.grid_w * self.tile_out
        if feat < expected_tile_lat + self.rest_out:
            raise ValueError(
                f"Latent feature dim {feat} < required {expected_tile_lat + self.rest_out} "
                f"(tiles {expected_tile_lat} + rest {self.rest_out})"
            )

        tiles_lat = latent[..., :expected_tile_lat]
        rest_lat = latent[..., expected_tile_lat:]

        expected_tile_out = self.grid_h * self.grid_w * self.tile_in

        tile_dec = MLP(self.layer_size, self.num_layers, self.tile_in, self.activation, final_activation=False)
        tiles_dec = tile_dec(tiles_lat.reshape(-1, self.tile_out)).reshape(*batch, expected_tile_out)

        rest_dec_mlp = MLP(self.layer_size, self.num_layers, self.output_dim - expected_tile_out, self.activation, final_activation=False)
        rest_dec = rest_dec_mlp(rest_lat)

        obs = jnp.concatenate([tiles_dec, rest_dec],axis=-1)
        return obs


class ENTEncoder(nn.Module):
    layer_size: int
    output_dim: int
    num_layers: int
    activation: str = "relog"

    @nn.compact
    def __call__(self, obs):
        act_fn = get_activation(self.activation)

        emb = obs
        for _ in range(self.num_layers):
            emb = nn.Dense(self.layer_size,)(emb)
            emb = act_fn(emb)

        emb = nn.Dense(self.output_dim)(emb)

        emb = act_fn(emb)

        return emb


class ENTDynamics(nn.Module):
    layer_size: int
    output_dim: int
    num_layers: int
    num_actions: int
    activation: str = "relog"

    @nn.compact
    def __call__(self, obs, act):
        act_fn = get_activation(self.activation)

        # act_oh = jax.nn.one_hot(act, self.num_actions, dtype=obs.dtype)  # (B, num_actions)
        act_emb = nn.Embed(num_embeddings=self.num_actions, features=self.num_actions,)(act)  # (B, num_actions)

        emb = jnp.concatenate([obs, act_emb], axis=-1)

        emb = obs
        for _ in range(self.num_layers):
            emb = nn.Dense(self.layer_size,)(emb)
            emb = act_fn(emb)

        emb = nn.Dense(self.output_dim)(emb)

        emb = act_fn(emb)

        return emb


class ENTDecoder(nn.Module):
    layer_size: int
    output_dim: int
    num_layers: int
    activation: str = "relog"

    @nn.compact
    def __call__(self, latent):
        act_fn = get_activation(self.activation)

        emb = latent
        for _ in range(self.num_layers):
            emb = nn.Dense(self.layer_size,)(emb)
            emb = act_fn(emb)

        obs = nn.Dense(self.output_dim)(emb)

        return obs
