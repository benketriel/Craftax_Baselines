import math
import argparse
import os
import sys
import time
from PIL import Image

import jax
import jax.numpy as jnp
import numpy as np
import optax
import matplotlib.pyplot as plt

from craftax.craftax_env import make_craftax_env_from_name

import wandb
from typing import NamedTuple

from flax.training import orbax_utils
from flax.training.train_state import TrainState
from orbax.checkpoint import (PyTreeCheckpointer, CheckpointManagerOptions, CheckpointManager)

from logz.batch_logging import batch_log, create_log_dict
from models.actor_critic import (ActorCritic, ActorCriticConv)
# from models.ent import ENTEncoder, ENTDynamics, ENTDecoder
from models.ent import ENTEncoderTiled, ENTDecoderTiled
from models.ewma import ewma_denormalize, ewma_init, ewma_normalize, ewma_update
from wrappers import (LogWrapper, OptimisticResetVecEnvWrapper, BatchEnvWrapper, AutoResetEnvWrapper)

# jax.config.update("jax_disable_jit", True)


class Transition(NamedTuple):
    done: jnp.ndarray
    action: jnp.ndarray
    value_env: jnp.ndarray      # denormalized (for GAE)
    value_norm: jnp.ndarray     # normalized (for PPO value clipping)
    reward_e: jnp.ndarray
    reward_i: jnp.ndarray
    reward: jnp.ndarray
    log_prob: jnp.ndarray
    obs: jnp.ndarray
    next_obs: jnp.ndarray
    info: jnp.ndarray


def _save_ent_histograms_host(counts_np, step_int, seed, out_dir, title="ENT Histograms"):
    L, K = counts_np.shape
    cols = int(np.ceil(np.sqrt(L)))
    rows = int(np.ceil(L / cols))

    fig, axs = plt.subplots(rows, cols, figsize=(cols * 3.2, rows * 2.8))
    if rows == 1 and cols == 1:
        axs = np.array([[axs]])
    elif rows == 1:
        axs = np.array([axs])
    elif cols == 1:
        axs = axs[:, None]

    for idx in range(rows * cols):
        r, c = divmod(idx, cols)
        ax = axs[r, c]
        if idx < L:
            ax.bar(np.arange(K), counts_np[idx])
            ax.set_title(f"z{idx}", fontsize=9)
            ax.set_xlabel("bin", fontsize=8)
            ax.set_ylabel("count", fontsize=8)
        else:
            ax.axis("off")

    fig.suptitle(title, fontsize=12)
    fig.tight_layout(pad=1.0)
    fig.subplots_adjust(top=0.90, bottom=0.08, left=0.08, right=0.98)

    run_dir = os.path.join(out_dir, str(seed))
    os.makedirs(run_dir, exist_ok=True)
    path = os.path.join(run_dir, f"{int(step_int)}.ent_histograms.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _save_reconstruction_strips_host(
    obs_np,
    recon_np,
    step_int,
    seed,
    out_dir,
    pixels_per_row=9*7,
    upscale=2,
    title_prefix="ENT Recon Strips",
):
    """
    Save one mosaic image visualizing up to 10 pairs of [original vs reconstruction]
    as RGB "barcode" rows. Each pixel encodes 3 consecutive values (R,G,B).
    Layout per sample:
        [ original row of 50 pixels ]
        [ reconstructed row of 50 pixels ]
        [ 1 white spacer row ]
    repeated until all values are shown (with padding).
    """
    N = min(10, obs_np.shape[0])
    assert obs_np.shape == recon_np.shape and obs_np.ndim == 2, "Expect (N, D)"

    def vector_to_strip(orig_vec, recon_vec):
        # Normalize both using shared min/max (so colors are comparable)
        combo = np.concatenate([orig_vec, recon_vec], axis=0)
        vmin, vmax = float(np.min(combo)), float(np.max(combo))
        if vmax <= vmin + 1e-12:
            vmax = vmin + 1e-12
        o = (orig_vec - vmin) / (vmax - vmin)
        r = (recon_vec - vmin) / (vmax - vmin)

        # After normalization, pad so length is a multiple of 3 (RGB)
        def pad_to_multiple_of(x, m):
            pad = (-len(x)) % m
            if pad:
                # pad with 1.0 => white pixels
                x = np.pad(x, (0, pad), constant_values=1.0)
            return x

        o = pad_to_multiple_of(o, 3)
        r = pad_to_multiple_of(r, 3)

        # Pack into pixels (num_pixels, 3)
        o_rgb = (o.reshape(-1, 3) * 255.0).clip(0, 255).astype(np.uint8)
        r_rgb = (r.reshape(-1, 3) * 255.0).clip(0, 255).astype(np.uint8)

        num_pixels = o_rgb.shape[0]
        rows_needed = math.ceil(num_pixels / pixels_per_row)
        total_pixels = rows_needed * pixels_per_row

        # Pad to full rows (with white pixels)
        white_px = np.array([[255, 255, 255]], dtype=np.uint8)
        if total_pixels > num_pixels:
            pad_n = total_pixels - num_pixels
            o_rgb = np.vstack([o_rgb, np.repeat(white_px, pad_n, axis=0)])
            r_rgb = np.vstack([r_rgb, np.repeat(white_px, pad_n, axis=0)])

        # Reshape into row blocks
        o_rows = o_rgb.reshape(rows_needed, pixels_per_row, 3)
        r_rows = r_rgb.reshape(rows_needed, pixels_per_row, 3)

        # Build the strip with 3 scanlines per block: orig, recon, spacer
        spacer = np.full((1, pixels_per_row, 3), 255, dtype=np.uint8)
        blocks = []
        for i in range(rows_needed):
            blocks.append(o_rows[i:i+1])  # 1 x W x 3
            blocks.append(r_rows[i:i+1])  # 1 x W x 3
            blocks.append(spacer)         # 1 x W x 3
        strip = np.vstack(blocks)  # (rows_needed*3, pixels_per_row, 3)
        return strip

    strips = [vector_to_strip(obs_np[i], recon_np[i]) for i in range(N)]

    # Stack samples vertically with a thicker spacer between samples
    max_w = max(s.shape[1] for s in strips)
    # Pad widths to match (should already match, but be safe)
    padded = []
    for s in strips:
        if s.shape[1] < max_w:
            pad_w = max_w - s.shape[1]
            right_pad = np.full((s.shape[0], pad_w, 3), 255, dtype=np.uint8)
            s = np.concatenate([s, right_pad], axis=1)
        padded.append(s)

    inter_sample_spacer = np.full((6, max_w, 3), 255, dtype=np.uint8)
    mosaic_rows = []
    for i, s in enumerate(padded):
        mosaic_rows.append(s)
        if i != len(padded) - 1:
            mosaic_rows.append(inter_sample_spacer)
    mosaic = np.vstack(mosaic_rows)  # H x W x 3

    # Optional small black header with step text (kept simple: no text render to avoid font deps)
    # Instead, put info in filename.
    img = Image.fromarray(mosaic)
    if upscale and upscale > 1:
        img = img.resize((img.width * upscale, img.height * upscale), resample=Image.NEAREST)

    run_dir = os.path.join(out_dir, str(seed))
    os.makedirs(run_dir, exist_ok=True)
    path = os.path.join(run_dir, f"{int(step_int)}.ent_recon_strips.png")
    img.save(path, format="PNG")


def make_train(config):
    config["NUM_UPDATES"] = (config["TOTAL_TIMESTEPS"] // config["NUM_STEPS"] // config["NUM_ENVS"])
    config["MINIBATCH_SIZE"] = (config["NUM_ENVS"] * config["NUM_STEPS"] // config["NUM_MINIBATCHES"])

    env = make_craftax_env_from_name(config["ENV_NAME"], not config["USE_OPTIMISTIC_RESETS"])
    env_params = env.default_params

    env = LogWrapper(env)
    if config["USE_OPTIMISTIC_RESETS"]:
        env = OptimisticResetVecEnvWrapper(env, num_envs=config["NUM_ENVS"], reset_ratio=min(config["OPTIMISTIC_RESET_RATIO"], config["NUM_ENVS"]))
    else:
        env = AutoResetEnvWrapper(env)
        env = BatchEnvWrapper(env, num_envs=config["NUM_ENVS"])

    def linear_schedule(count):
        frac = (1.0 - (count // (config["NUM_MINIBATCHES"] * config["UPDATE_EPOCHS"])) / config["NUM_UPDATES"] )
        return config["LR"] * frac

    def train(rng):
        num_actions = env.action_space(env_params).n
        # INIT NETWORK
        if "Symbolic" in config["ENV_NAME"]:
            network = ActorCritic(num_actions, config["LAYER_SIZE"])
        else:
            network = ActorCriticConv(num_actions, config["LAYER_SIZE"])

        rng, _rng = jax.random.split(rng)
        init_x = jnp.zeros((1, *env.observation_space(env_params).shape))
        network_params = network.init(_rng, init_x)
        if config["ANNEAL_LR"]:
            tx = optax.chain(optax.clip_by_global_norm(config["MAX_GRAD_NORM"]), optax.adam(learning_rate=linear_schedule, eps=1e-5))
        else:
            tx = optax.chain(optax.clip_by_global_norm(config["MAX_GRAD_NORM"]), optax.adam(config["LR"], eps=1e-5))
        train_state = TrainState.create(apply_fn=network.apply, params=network_params, tx=tx)
        
        # Exploration state
        ex_state = {
            "ent_encoder": None,
            "ent_dynamics": None,
            "ent_decoder": None,
            "latent_histogram": None,
            "ewma_return_norm": None,
        }

        if config["TRAIN_ENT"]:
            obs_shape = env.observation_space(env_params).shape
            assert len(obs_shape) == 1, "Only configured for 1D observations"
            obs_shape = obs_shape[0]

            # # Encoder
            # icm_encoder_network = ENTEncoder(num_layers=config["ENT_NUM_LAYERS"], output_dim=config["ENT_LATENT_SIZE"], layer_size=config["ENT_LAYER_SIZE"], )
            # rng, _rng = jax.random.split(rng)
            # icm_encoder_network_params = icm_encoder_network.init(_rng, jnp.zeros((1, obs_shape)))
            # tx = optax.chain(optax.clip_by_global_norm(config["MAX_GRAD_NORM"]), optax.adam(config["ENT_LR"], eps=1e-5),)
            # ex_state["ent_encoder"] = TrainState.create(apply_fn=icm_encoder_network.apply, params=icm_encoder_network_params, tx=tx,)

            # # Dynamics
            # icm_dynamics_network = ENTDynamics(num_layers=config["ENT_NUM_LAYERS"], output_dim=config["ENT_LATENT_SIZE"], layer_size=config["ENT_LAYER_SIZE"], num_actions=num_actions)
            # rng, _rng = jax.random.split(rng)
            # icm_dynamics_network_params = icm_dynamics_network.init(_rng, jnp.zeros((1, obs_shape), dtype=jnp.float32), jnp.zeros((1,), dtype=jnp.int32))
            # tx = optax.chain(optax.clip_by_global_norm(config["MAX_GRAD_NORM"]), optax.adam(config["ENT_LR"], eps=1e-5),)
            # ex_state["ent_dynamics"] = TrainState.create(apply_fn=icm_dynamics_network.apply, params=icm_dynamics_network_params, tx=tx,)

            # # Decoder
            # icm_forward_network = ENTDecoder(num_layers=config["ENT_NUM_LAYERS"], output_dim=obs_shape, layer_size=config["ENT_LAYER_SIZE"])
            # rng, _rng = jax.random.split(rng)
            # icm_forward_network_params = icm_forward_network.init(_rng, jnp.zeros((1, config["ENT_LATENT_SIZE"])))
            # tx = optax.chain(optax.clip_by_global_norm(config["MAX_GRAD_NORM"]), optax.adam(config["ENT_LR"], eps=1e-5), )
            # ex_state["ent_decoder"] = TrainState.create(apply_fn=icm_forward_network.apply, params=icm_forward_network_params, tx=tx, )

            # ex_state["latent_histogram"] = jnp.zeros((config["ENT_LATENT_SIZE"], config["ENT_HISTOGRAM_BINS"]), dtype=jnp.float32)

            # Encoder tiled
            icm_encoder_network = ENTEncoderTiled(num_layers=config["ENT_NUM_LAYERS"], tile_out=config["ENT_TILE_OUT"], rest_out=config["ENT_REST_OUT"], layer_size=config["ENT_LAYER_SIZE"], )
            rng, _rng = jax.random.split(rng)
            icm_encoder_network_params = icm_encoder_network.init(_rng, jnp.zeros((1, obs_shape)))
            tx = optax.chain(optax.clip_by_global_norm(config["MAX_GRAD_NORM"]), optax.adam(config["ENT_LR"], eps=1e-5),)
            ex_state["ent_encoder"] = TrainState.create(apply_fn=icm_encoder_network.apply, params=icm_encoder_network_params, tx=tx,)

            latent_size = config["ENT_TILE_OUT"] * 7 * 9 + config["ENT_REST_OUT"]

            # Decoder tiled
            icm_forward_network = ENTDecoderTiled(num_layers=config["ENT_NUM_LAYERS"], tile_out=config["ENT_TILE_OUT"], rest_out=config["ENT_REST_OUT"], output_dim=obs_shape, layer_size=config["ENT_LAYER_SIZE"])
            rng, _rng = jax.random.split(rng)
            icm_forward_network_params = icm_forward_network.init(_rng, jnp.zeros((1, latent_size)))
            tx = optax.chain(optax.clip_by_global_norm(config["MAX_GRAD_NORM"]), optax.adam(config["ENT_LR"], eps=1e-5), )
            ex_state["ent_decoder"] = TrainState.create(apply_fn=icm_forward_network.apply, params=icm_forward_network_params, tx=tx, )

            ex_state["latent_histogram"] = jnp.zeros((config["ENT_TILE_OUT"] + config["ENT_REST_OUT"], config["ENT_HISTOGRAM_BINS"]), dtype=jnp.float32)
            
        ex_state["ewma_return_norm"] = ewma_init()

        # INIT ENV
        rng, _rng = jax.random.split(rng)
        obsv, env_state = env.reset(_rng, env_params)

        # TRAIN LOOP
        def _update_step(runner_state, unused):
            # COLLECT TRAJECTORIES
            def _env_step(runner_state, unused):
                (train_state, env_state, last_obs, ex_state, rng, update_step) = runner_state

                # SELECT ACTION
                rng, _rng = jax.random.split(rng)
                pi, value_norm = network.apply(train_state.params, last_obs)

                # Force fully random actions:
                # import distrax
                # pi = distrax.Categorical(logits=pi.logits * 0 + (1 / pi.logits.shape[-1]))

                action = pi.sample(seed=_rng)
                log_prob = pi.log_prob(action)
                value_env = ewma_denormalize(ex_state["ewma_return_norm"], value_norm)

                # STEP ENV
                rng, _rng = jax.random.split(rng)
                obsv, env_state, reward_e, done, info = env.step(_rng, env_state, action, env_params)
                if not config["SUPERVISED"]:
                    reward_e = reward_e * 0

                reward_i = jnp.zeros(config["NUM_ENVS"])

                if config["TRAIN_ENT"]:
                    latent_obs = ex_state["ent_encoder"].apply_fn(ex_state["ent_encoder"].params, last_obs)
                    nbins = config["ENT_HISTOGRAM_BINS"]
                    bin_width = config["ENT_BIN_WIDTH"]
                    min_count = config["ENT_MIN_COUNT"]
                    decay = config["ENT_DECAY"]
                    ent_scale = config["ENT_REWARD_COEFF"]


                    # Fast, per neuron histogram
                    # # latent_obs: (B, L)
                    # # Map activations to bin indices
                    # z = jnp.round(latent_obs / bin_width).astype(jnp.int32) + nbins // 2
                    # z = jnp.clip(z, 0, nbins - 1)  # (B, L)

                    # # Shared counts (L, K)
                    # counts = ex_state["latent_histogram"]  # (L, K)

                    # # ----- log-density BEFORE updating (avoid bias) -----
                    # # log_p[l, k] = log(count + min) - log(sum_k count + min*nbins)
                    # log_p = jnp.log(counts + min_count) - jnp.log(jnp.sum(counts, axis=-1, keepdims=True) + min_count * nbins)  # (L, K)

                    # # Gather log-prob at bins z for each env & latent:
                    # # Broadcast (L, K) -> (B, L, K), then take along bin axis
                    # log_p_b = jnp.broadcast_to(log_p, (z.shape[0],) + log_p.shape)  # (B, L, K)
                    # log_p_sel = jnp.take_along_axis(log_p_b, z[..., None], axis=-1)[..., 0]  # (B, L)

                    # # Intrinsic reward per env (B,)
                    # reward_i = -jnp.sum(log_p_sel, axis=-1) * ent_scale

                    # # ----- build batch increments (B, L, K) -----
                    # incr_b = jax.nn.one_hot(z, nbins, dtype=jnp.float32)  # (B, L, K)

                    # # Do not add to edge bins - let outliers be lowest possible density
                    # edge_mask = jnp.ones((nbins,), dtype=jnp.float32).at[0].set(0.0).at[nbins - 1].set(0.0)
                    # incr_b = incr_b * edge_mask  # broadcast over (B, L, K)

                    # # Aggregate over batch and apply decay once to the shared histogram
                    # incr = jnp.sum(incr_b, axis=0)  # (L, K)
                    # new_counts = counts * decay + incr  # (L, K)

                    # # Write back
                    # ex_state["latent_histogram"] = new_counts




                    # Slow, convolutional/tiled aggregation histogram

                    # Shapes:
                    # latent_obs: (B, L) where L = tile_out * 63 + rest_out
                    # counts/ex_state["latent_histogram"]: (L2, K) where L2 = tile_out + rest_out

                    # Constants (make sure these are Python ints for good XLA specialization)
                    tile_mult = 7 * 9
                    tile_out  = config["ENT_TILE_OUT"]
                    rest_out  = config["ENT_REST_OUT"]
                    B         = latent_obs.shape[0]

                    # --- map activations to bin indices (B, L) ---
                    z = jnp.round(latent_obs / bin_width).astype(jnp.int32) + nbins // 10
                    z = jnp.clip(z, 0, nbins - 1)  # (B, L)

                    # --- split into tiles and rest ---
                    z_tile = z[:, :tile_out * tile_mult].reshape(B, tile_mult, tile_out)   # (B, 63, T)
                    z_rest = z[:, tile_out * tile_mult:]                                   # (B, R)

                    # --- current shared histogram (L2, K) ---
                    counts = ex_state["latent_histogram"]          # (tile_out + rest_out, K)
                    counts_tile = counts[:tile_out, :]             # (T, K)
                    counts_rest = counts[tile_out:, :]             # (R, K)

                    # --- log-density BEFORE updating (avoid bias) ---
                    def _logp(c):  # c: (*, K)
                        return jnp.log(c + min_count) - jnp.log(jnp.sum(c, axis=-1, keepdims=True) + min_count * nbins)

                    log_p_tile = _logp(counts_tile)  # (T, K)
                    log_p_rest = _logp(counts_rest)  # (R, K)

                    # ===== Efficient gather for reward =====
                    # Tiles: want log_p_tile[t, z_tile[b, s, t]] -> (B, 63, T)
                    def gather_tile_per_t(logp_t, z_t):           # (K,), (B, 63)
                        return jnp.take(logp_t, z_t, axis=0)      # -> (B, 63)

                    log_p_sel_tile = jax.vmap(gather_tile_per_t, in_axes=(0, 2), out_axes=2)(
                        log_p_tile, z_tile
                    )  # (B, 63, T)

                    # Rest: want log_p_rest[r, z_rest[b, r]] -> (B, R)
                    z_rest_T = jnp.swapaxes(z_rest, 0, 1)  # (R, B)

                    def gather_rest_per_r(logp_r, z_r):    # (K,), (B,)
                        return jnp.take(logp_r, z_r, axis=0)  # -> (B,)

                    log_p_sel_rest = jax.vmap(gather_rest_per_r, in_axes=(0, 0), out_axes=1)(
                        log_p_rest, z_rest_T
                    )  # (B, R)

                    # Intrinsic reward (B,)
                    sum_tile = jnp.sum(log_p_sel_tile, axis=(1, 2))  # sum over 63 and T
                    sum_rest = jnp.sum(log_p_sel_rest, axis=1)       # sum over R
                    reward_i = -(sum_tile + sum_rest) * ent_scale

                    # ===== One-shot bincount updates (no vmapped small bincounts) =====
                    # Edge suppression mask applied after reshape to (., K)
                    # Tiles: for each feature t, aggregate counts over all (B * 63) indices
                    # z_tile: (B, 63, T) -> flatten first two dims
                    z_tile_flat = z_tile.reshape(-1, z_tile.shape[-1])    # (B*63, T)
                    # Offsets so each feature t accumulates into its own block of K bins
                    t_idx = jnp.arange(tile_out, dtype=jnp.int32)         # (T,)
                    tile_offsets = t_idx * nbins                           # (T,)
                    # Broadcast-add offsets across last axis
                    z_tile_off = z_tile_flat + tile_offsets                # (B*63, T)
                    # Single bincount over all features at once
                    incr_tile_all = jnp.bincount(
                        z_tile_off.reshape(-1),
                        length=tile_out * nbins
                    ).astype(jnp.float32)                                  # (T*K,)
                    incr_tile = incr_tile_all.reshape(tile_out, nbins)     # (T, K)
                    # Edge suppression
                    incr_tile = incr_tile.at[:, 0].set(0.0).at[:, nbins - 1].set(0.0)

                    # Rest: for each rest feature r, aggregate over batch
                    if rest_out:
                        r_idx = jnp.arange(rest_out, dtype=jnp.int32)      # (R,)
                        rest_offsets = r_idx * nbins
                        z_rest_off = z_rest + rest_offsets                 # (B, R)
                        incr_rest_all = jnp.bincount(
                            z_rest_off.reshape(-1),
                            length=rest_out * nbins
                        ).astype(jnp.float32)                               # (R*K,)
                        incr_rest = incr_rest_all.reshape(rest_out, nbins)  # (R, K)
                        incr_rest = incr_rest.at[:, 0].set(0.0).at[:, nbins - 1].set(0.0)
                    else:
                        incr_rest = counts_rest * 0.0  # keep shape (0, K) if R=0

                    # Apply decay once to the shared histogram
                    new_counts_tile = counts_tile * decay + incr_tile      # (T, K)
                    new_counts_rest = counts_rest * decay + incr_rest      # (R, K)
                    new_counts = jnp.concatenate([new_counts_tile, new_counts_rest], axis=0)  # (T+R, K)

                    # Write back
                    ex_state["latent_histogram"] = new_counts




                reward = reward_e + reward_i

                transition = Transition(done=done, action=action, value_env=value_env, value_norm=value_norm, reward=reward, reward_i=reward_i, reward_e=reward_e, log_prob=log_prob,
                    obs=last_obs, next_obs=obsv, info=info)
                runner_state = (train_state, env_state, obsv, ex_state, rng, update_step)
                return runner_state, transition

            runner_state, traj_batch = jax.lax.scan(_env_step, runner_state, None, config["NUM_STEPS"])

            # CALCULATE ADVANTAGE
            ( train_state, env_state, last_obs, ex_state, rng, update_step ) = runner_state
            _, last_val_norm = network.apply(train_state.params, last_obs)
            last_val_env = ewma_denormalize(ex_state["ewma_return_norm"], last_val_norm)

            def _calculate_gae(traj_batch, last_val_env):
                def _get_advantages(gae_and_next_value, transition):
                    gae, next_value_env = gae_and_next_value
                    done, value_env, reward = (transition.done, transition.value_env, transition.reward)
                    delta = reward + config["GAMMA"] * next_value_env * (1 - done) - value_env
                    gae = (delta + config["GAMMA"] * config["GAE_LAMBDA"] * (1 - done) * gae)
                    return (gae, value_env), gae

                _, advantages = jax.lax.scan(_get_advantages, (jnp.zeros_like(last_val_env), last_val_env), traj_batch, reverse=True, unroll=16, )
                targets_env = advantages + traj_batch.value_env
                return advantages, targets_env

            advantages, targets_env = _calculate_gae(traj_batch, last_val_env)

            ex_state["ewma_return_norm"] = ewma_update(ex_state["ewma_return_norm"], targets_env)
            targets_norm = ewma_normalize(ex_state["ewma_return_norm"], targets_env)

            # UPDATE NETWORK
            def _update_epoch(update_state, unused):
                def _update_minbatch(train_state, batch_info):
                    traj_batch, advantages, targets_norm = batch_info

                    # Policy/value network
                    def _loss_fn(params, traj_batch, gae, targets_norm):
                        # RERUN NETWORK
                        pi, value_norm = network.apply(params, traj_batch.obs)
                        log_prob = pi.log_prob(traj_batch.action)

                        # CALCULATE VALUE LOSS
                        value_pred_clipped = traj_batch.value_norm + (value_norm - traj_batch.value_norm).clip(-config["CLIP_EPS"], config["CLIP_EPS"])
                        value_losses = jnp.square(value_norm - targets_norm)
                        value_losses_clipped = jnp.square(value_pred_clipped - targets_norm)
                        value_loss = (0.5 * jnp.maximum(value_losses, value_losses_clipped).mean())

                        # CALCULATE ACTOR LOSS
                        ratio = jnp.exp(log_prob - traj_batch.log_prob)
                        gae = (gae - gae.mean()) / (gae.std() + 1e-8)
                        loss_actor1 = ratio * gae
                        loss_actor2 = (jnp.clip(ratio, 1.0 - config["CLIP_EPS"], 1.0 + config["CLIP_EPS"], ) * gae)
                        loss_actor = -jnp.minimum(loss_actor1, loss_actor2)
                        loss_actor = loss_actor.mean()
                        entropy = pi.entropy().mean()

                        total_loss = (loss_actor + config["VF_COEF"] * value_loss - config["ENT_COEF"] * entropy )
                        return total_loss, (value_loss, loss_actor, entropy)

                    grad_fn = jax.value_and_grad(_loss_fn, has_aux=True)
                    total_loss, grads = grad_fn(train_state.params, traj_batch, advantages, targets_norm)
                    train_state = train_state.apply_gradients(grads=grads)

                    losses = (total_loss, 0)
                    return train_state, losses

                (train_state, traj_batch, advantages, targets_norm, rng,) = update_state
                rng, _rng = jax.random.split(rng)
                batch_size = config["MINIBATCH_SIZE"] * config["NUM_MINIBATCHES"]
                assert (batch_size == config["NUM_STEPS"] * config["NUM_ENVS"] ), "batch size must be equal to number of steps * number of envs"
                permutation = jax.random.permutation(_rng, batch_size)
                batch = (traj_batch, advantages, targets_norm)
                batch = jax.tree.map(lambda x: x.reshape((batch_size,) + x.shape[2:]), batch)
                shuffled_batch = jax.tree.map(lambda x: jnp.take(x, permutation, axis=0), batch)
                minibatches = jax.tree.map(lambda x: jnp.reshape(x, [config["NUM_MINIBATCHES"], -1] + list(x.shape[1:])), shuffled_batch,)
                train_state, losses = jax.lax.scan(_update_minbatch, train_state, minibatches)
                update_state = (train_state, traj_batch, advantages, targets_norm, rng,)
                return update_state, losses

            update_state = (train_state, traj_batch, advantages, targets_norm, rng, )
            update_state, loss_info = jax.lax.scan(_update_epoch, update_state, None, config["UPDATE_EPOCHS"])

            train_state = update_state[0]
            metric = jax.tree.map(lambda x: (x * traj_batch.info["returned_episode"]).sum() / traj_batch.info["returned_episode"].sum(), traj_batch.info)

            rng = update_state[-1]

            # UPDATE EXPLORATION STATE
            def _update_ex_epoch(update_state, unused):
                def _update_ex_minbatch(ex_state, traj_batch):
                    def _recon_loss_fn(ent_encoder_params, ent_decoder_params, traj_batch):
                        latent_obs = ex_state["ent_encoder"].apply_fn(ent_encoder_params, last_obs)
                        reconstruct_obs = ex_state["ent_decoder"].apply_fn(ent_decoder_params, latent_obs)

                        error = (last_obs - reconstruct_obs)
                        mse = jnp.square(error).mean(axis=-1)
                        return jnp.mean(mse)

                    recon_grad_fn = jax.value_and_grad(_recon_loss_fn, has_aux=False, argnums=(0, 1, ),)
                    recon_loss, grads = recon_grad_fn(ex_state["ent_encoder"].params, ex_state["ent_decoder"].params, traj_batch,)
                    ent_encoder_grad, ent_decoder_grad = grads
                    ex_state["ent_encoder"] = ex_state["ent_encoder"].apply_gradients(grads=ent_encoder_grad)
                    ex_state["ent_decoder"] = ex_state["ent_decoder"].apply_gradients(grads=ent_decoder_grad)


                    losses = (recon_loss,)
                    return ex_state, losses

                (ex_state, traj_batch, rng) = update_state
                rng, _rng = jax.random.split(rng)
                batch_size = config["MINIBATCH_SIZE"] * config["NUM_MINIBATCHES"]
                assert (batch_size == config["NUM_STEPS"] * config["NUM_ENVS"]), "batch size must be equal to number of steps * number of envs"
                permutation = jax.random.permutation(_rng, batch_size)
                batch = jax.tree.map(lambda x: x.reshape((batch_size,) + x.shape[2:]), traj_batch)
                shuffled_batch = jax.tree.map(lambda x: jnp.take(x, permutation, axis=0), batch)
                minibatches = jax.tree.map(lambda x: jnp.reshape(x, [config["NUM_MINIBATCHES"], -1] + list(x.shape[1:])), shuffled_batch, )
                ex_state, losses = jax.lax.scan(_update_ex_minbatch, ex_state, minibatches)
                update_state = (ex_state, traj_batch, rng)
                return update_state, losses

            if config["TRAIN_ENT"]:
                ex_update_state = (ex_state, traj_batch, rng)
                ex_update_state, ex_loss = jax.lax.scan(_update_ex_epoch, ex_update_state, None, config["EXPLORATION_UPDATE_EPOCHS"],)
                metric["ent_recon_loss"] = ex_loss[0].mean()
                metric["reward_i"] = traj_batch.reward_i.mean()
                metric["reward_e"] = traj_batch.reward_e.mean()

                (ex_state, traj_batch, rng) = ex_update_state

            # wandb logging
            if config["DEBUG"] and config["USE_WANDB"]:

                def callback(metric, update_step):
                    to_log = create_log_dict(metric, config)
                    batch_log(update_step, to_log, config)

                jax.debug.callback(callback, metric, update_step, )

            # Optional file dump of ENT histograms every N updates
            if config["TRAIN_ENT"]:
                def _maybe_dump(counts, step_scalar):
                    # This runs on the host (Python) side.
                    step_val = int(step_scalar)
                    if (config["ENT_DUMP_EVERY"] > 0) and (step_val % config["ENT_DUMP_EVERY"] == 0):
                        counts_np = np.array(counts)
                        _save_ent_histograms_host(
                            counts_np=counts_np,
                            step_int=step_val,
                            seed=config["SEED"],
                            out_dir=config["ENT_DUMP_DIR"],
                            title=f"ENT histograms @ update {step_val}"
                        )

                # Note: arguments to callback must be JAX arrays / scalars
                jax.debug.callback(
                    _maybe_dump,
                    ex_state["latent_histogram"],     # (L, K) float32
                    update_step,                      # scalar int
                )

                latest_obs = traj_batch.obs[-1]  # shape: (NUM_ENVS, obs_dim)

                # Use a STATIC M to avoid dynamic slicing issues under jit
                M = min(10, config["NUM_ENVS"])
                rng, _rng2 = jax.random.split(rng)
                perm = jax.random.permutation(_rng2, config["NUM_ENVS"])
                idx = perm[:M]  # static slice size
                obs_examples = jnp.take(latest_obs, idx, axis=0)  # (M, obs_dim)

                # encode -> decode with current ENT params
                latent_examples = ex_state["ent_encoder"].apply_fn(
                    ex_state["ent_encoder"].params, obs_examples
                )
                recon_examples = ex_state["ent_decoder"].apply_fn(
                    ex_state["ent_decoder"].params, latent_examples
                )

                def _maybe_dump_recon_strips(obs_samples, recon_samples, step_scalar):
                    step_val = int(step_scalar)
                    if (config["ENT_DUMP_EVERY"] > 0) and (step_val % config["ENT_DUMP_EVERY"] == 0):
                        _save_reconstruction_strips_host(
                            obs_np=np.array(obs_samples),
                            recon_np=np.array(recon_samples),
                            step_int=step_val,
                            seed=config["SEED"],
                            out_dir=config["ENT_DUMP_DIR"],
                            title_prefix=f"ENT recon strips @ update {step_val}",
                        )

                jax.debug.callback(
                    _maybe_dump_recon_strips,
                    obs_examples,          # (M, D)
                    recon_examples,        # (M, D)
                    update_step,           # scalar int
                )                

            runner_state = (train_state, env_state, last_obs, ex_state, rng, update_step + 1, )
            return runner_state, metric

        rng, _rng = jax.random.split(rng)
        runner_state = (train_state, env_state, obsv, ex_state, _rng, 0, )
        runner_state, metric = jax.lax.scan(_update_step, runner_state, None, config["NUM_UPDATES"] )
        return {"runner_state": runner_state} #, "info": metric}

    return train


def run_ppo(config):
    config = {k.upper(): v for k, v in config.__dict__.items()}

    if config["USE_WANDB"]:
        wandb.init(
            project=config["WANDB_PROJECT"],
            entity=config["WANDB_ENTITY"],
            config=config,
            name=config["ENV_NAME"]
            + "-"
            + str(int(config["TOTAL_TIMESTEPS"] // 1e6))
            + "M"
            + "-"
            + config["WANDB_SUFFIX"],
        )

    rng = jax.random.PRNGKey(config["SEED"])
    rngs = jax.random.split(rng, config["NUM_REPEATS"])

    train_jit = jax.jit(make_train(config))
    train_vmap = jax.vmap(train_jit)

    t0 = time.time()
    out = train_vmap(rngs)
    t1 = time.time()
    print("Time to run experiment", t1 - t0)
    print("SPS: ", config["TOTAL_TIMESTEPS"] / (t1 - t0))
    # print(out["info"])

    if config["USE_WANDB"]:

        def _save_network(rs_index, dir_name):
            train_states = out["runner_state"][rs_index]
            train_state = jax.tree.map(lambda x: x[0], train_states)
            orbax_checkpointer = PyTreeCheckpointer()
            options = CheckpointManagerOptions(max_to_keep=1, create=True)
            path = os.path.join(wandb.run.dir, dir_name)
            checkpoint_manager = CheckpointManager(path, orbax_checkpointer, options)
            print(f"saved runner state to {path}")
            save_args = orbax_utils.save_args_from_target(train_state)
            checkpoint_manager.save(
                config["TOTAL_TIMESTEPS"],
                train_state,
                save_kwargs={"save_args": save_args},
            )

        if config["SAVE_POLICY"]:
            _save_network(0, "policies")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--use_wandb", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--wandb_suffix", type=str, default="sup-ent")

    # Their originals:
    # parser.add_argument("--env_name", type=str, default="Craftax-Symbolic-v1")
    parser.add_argument("--total_timesteps", type=lambda x: int(float(x)), default=1e9)  # Allow scientific notation
    parser.add_argument("--num_envs", type=int, default=1024)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--anneal_lr", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--num_steps", type=int, default=64)
    parser.add_argument("--update_epochs", type=int, default=4)
    # parser.add_argument("--supervised", action=argparse.BooleanOptionalAction, default=True)
    # parser.add_argument("--train_ent", action=argparse.BooleanOptionalAction, default=False)

    # Crafter hyperparams:
    parser.add_argument("--env_name", type=str, default="Craftax-Classic-Symbolic-v1")
    # parser.add_argument("--total_timesteps", type=lambda x: int(float(x)), default=1e6)  # Allow scientific notation
    # parser.add_argument("--num_envs", type=int, default=8)
    # parser.add_argument("--lr", type=float, default=0.2e-4)
    # parser.add_argument("--anneal_lr", action=argparse.BooleanOptionalAction, default=False)
    # parser.add_argument("--num_steps", type=int, default=512)
    # parser.add_argument("--update_epochs", type=int, default=3)
    parser.add_argument("--supervised", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--train_ent", action=argparse.BooleanOptionalAction, default=True)

    parser.add_argument("--num_minibatches", type=int, default=8)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae_lambda", type=float, default=0.8)
    parser.add_argument("--clip_eps", type=float, default=0.2)
    parser.add_argument("--ent_coef", type=float, default=0.01)
    parser.add_argument("--vf_coef", type=float, default=0.5)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--activation", type=str, default="tanh")
    parser.add_argument("--debug", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--jit", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--save_policy", action="store_true")
    parser.add_argument("--num_repeats", type=int, default=1)
    parser.add_argument("--layer_size", type=int, default=512)
    parser.add_argument("--wandb_project", type=str)
    parser.add_argument("--wandb_entity", type=str)
    parser.add_argument("--use_optimistic_resets", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--optimistic_reset_ratio", type=int, default=16)

    # EXPLORATION
    parser.add_argument("--exploration_update_epochs", type=int, default=4)
    # ENT
    parser.add_argument("--ent_reward_coeff", type=float, default=0.00001)
    parser.add_argument("--ent_lr", type=float, default=1.5e-3)
    parser.add_argument("--ent_layer_size", type=int, default=128)  # Might need to use 256 for better results
    parser.add_argument("--ent_num_layers", type=int, default=3)  # Might need 6 for better results
    parser.add_argument("--ent_tile_out", type=int, default=16)
    parser.add_argument("--ent_rest_out", type=int, default=32)
    parser.add_argument("--ent_histogram_bins", type=int, default=151)
    parser.add_argument("--ent_bin_width", type=float, default=0.01)
    parser.add_argument("--ent_min_count", type=float, default=0.01)
    parser.add_argument("--ent_decay", type=float, default=0.99)

    parser.add_argument("--ent_dump_every", type=int, default=1000)   # save every N PPO updates
    parser.add_argument("--ent_dump_dir", type=str, default="tmp")   # output root dir

    args, rest_args = parser.parse_known_args(sys.argv[1:])
    if rest_args:
        raise ValueError(f"Unknown args {rest_args}")

    if args.seed is None:
        args.seed = np.random.randint(2**31)
        os.makedirs('tmp', exist_ok=True)
        with open('tmp/last_seed', 'w', encoding='utf-8') as f:
            f.write(str(args.seed))

    if args.jit:
        run_ppo(args)
    else:
        with jax.disable_jit():
            run_ppo(args)
