import copy
import io
import os
import time
from flask import Flask, send_file, request, jsonify, render_template
import hydra
import jax
from jax import numpy as jnp
import numpy as np
from PIL import Image
import distrax
from conf.config import EnjoyConfig
from envs.pcgrl_env import PCGRLEnv, gen_dummy_queued_state
from envs.probs.problem import Problem
from eval import init_config_for_eval
from purejaxrl.experimental.s5.wrappers import LossLogWrapper
from train import init_checkpointer

from utils import gymnax_pcgrl_make, init_config, init_network


class PCGRLWebApp():
    def __init__(self, enjoy_config):

        self.enjoy_config = init_config(enjoy_config)
        exp_dir = enjoy_config.exp_dir
        if not enjoy_config.random_agent:
            print(f'Loading checkpoint from {exp_dir}')
            checkpoint_manager, restored_ckpt = init_checkpointer(enjoy_config)
            runner_state = restored_ckpt['runner_state']
            self.network_params = runner_state.train_state.params
            steps_prev_complete = restored_ckpt['steps_prev_complete']
        elif not os.path.exists(exp_dir):
            os.makedirs(exp_dir)
            steps_prev_complete = 0


        if enjoy_config.render_ims:
            frames_dir = os.path.join(exp_dir, 'frames')
            os.makedirs(frames_dir, exist_ok=True)

        best_frames_dir = os.path.join(exp_dir, 'best_frames')
        os.makedirs(best_frames_dir, exist_ok=True)

        env: PCGRLEnv

        # Preserve config as it was during training, for future reference (i.e. naming output of enjoy/eval)
        train_config = copy.deepcopy(enjoy_config)

        enjoy_config = init_config_for_eval(enjoy_config)
        env, self.env_params = gymnax_pcgrl_make(enjoy_config.env_name, config=enjoy_config)
        env = LossLogWrapper(env)
        self.env = env
        self.tile_enum = env.prob.tile_enum
        self.ints_to_tiles = {t.value: t.name.lower() for t in self.tile_enum}
        self.tiles_to_ints = {v: k for k, v in self.ints_to_tiles.items()}
        # env.prob.init_graphics()
        self.network = init_network(env, self.env_params, enjoy_config)

        self.rng = jax.random.PRNGKey(enjoy_config.eval_seed)

        # Can manually define frozen tiles here, e.g. to set an OOD task
        # frz_map = jnp.zeros(env.map_shape, dtype=bool)
        # frz_map = frz_map.at[7, 3:-3].set(1)
        # queued_state = gen_dummy_queued_state(env)
        # queued_state = env.queue_frz_map(queued_state, frz_map)

        self.obs, self.env_state = env.reset(self.rng, self.env_params)
        self.latent = jnp.zeros(self.obs.map_obs.shape[:-1] + (enjoy_config.nca_latent_dim,), dtype=jnp.float32) if enjoy_config.model == "nca" else jnp.zeros((1,), dtype=jnp.float32)
        self.use_mask = enjoy_config.model == "nca" and enjoy_config.representation == "nca" and enjoy_config.nca_mask_keep_prob < 1.0
        # obs, env_state = jax.vmap(env.reset, in_axes=(0, None, None))(
        #     rng_reset, env_params, queued_state
        # )

    def reset(self):
        self.rng = jax.random.split(self.rng)[0]
        self.obs, self.env_state = self.env.reset(self.rng, self.env_params)
        self.latent = jnp.zeros(self.obs.map_obs.shape[:-1] + (self.enjoy_config.nca_latent_dim,), dtype=jnp.float32) if self.enjoy_config.model == "nca" else jnp.zeros((1,), dtype=jnp.float32)

    def tick(self):
        self.rng, rng_apply, rng_action, rng_step = jax.random.split(self.rng, 4)
        if enjoy_config.random_agent:
            action = self.env.action_space(self.env_params).sample(rng_action)[None, None, None]
        else:
            obs = jax.tree_util.tree_map(lambda x: x[None], self.obs)
            if self.enjoy_config.model == "nca":
                if self.use_mask:
                    logits, _, self.latent = self.network.apply(
                        self.network_params, obs, self.latent, rngs={'nca_mask': rng_apply}
                    )
                else:
                    logits, _, self.latent = self.network.apply(
                        self.network_params, obs, self.latent
                    )
                pi = distrax.Categorical(logits=logits)
                action = pi.sample(seed=rng_action)[0]
            else:
                action = self.network.apply(self.network_params, obs)[0].sample(seed=rng_action)[0]

        self.obs, self.env_state, reward, done, info = self.env.step(rng_step, self.env_state, action, self.env_params)
        if self.enjoy_config.model == "nca":
            reset_mask = jnp.reshape(jnp.asarray(done), (1,) + (1,) * (self.latent.ndim - 1))
            self.latent = jnp.where(reset_mask, jnp.zeros_like(self.latent), self.latent)

    def update_stats(self):
        env_map = self.env_state.log_env_state.env_state.env_map
        prob_state = self.env.prob.get_curr_stats(env_map)
        self.env_state = self.env_state.replace(
            log_env_state=self.env_state.log_env_state.replace(
                env_state=self.env_state.log_env_state.env_state.replace(
                    prob_state=prob_state.replace(
                        ctrl_trgs=self.env.prob.stat_trgs
                    )
                )
            )
        )


app = Flask(__name__)

# Placeholder for the image storage, initialized with a dummy image
# current_image = np.zeros((100, 100, 3), dtype=np.uint8)

pcgrl: PCGRLWebApp = None
enjoy_config: EnjoyConfig = None

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/tile_click', methods=['POST'])
def handle_tile_click():
    data = request.json
    x = data['x']
    y = data['y']
    tileName = data['tileName']
    # Process the click event here, for example, updating the map
    print(f"Tile clicked at ({x}, {y}) with tile type: {tileName}")
    # Return a response, could be a confirmation or result of the operation
    return jsonify({"status": "success", "message": "Tile click processed"})

# @app.route('/image')
# def serve_image():
#     global current_image
#     img = Image.fromarray(current_image)
#     byte_io = io.BytesIO()
#     img.save(byte_io, 'PNG')
#     byte_io.seek(0)
#     return send_file(byte_io, mimetype='image/png')

@app.route('/apply_edits', methods=['POST'])
def apply_edits():
    response = request.get_json()
    edits = response['edits']
    env_map = pcgrl.env_state.log_env_state.env_state.env_map
    for coords, tile_type in edits.items():
        xy = coords.split(',')
        if len(xy) != 2:
            breakpoint()
        x, y = xy
        x, y = int(x), int(y)
        tile_int = pcgrl.tiles_to_ints[tile_type]
        env_map = env_map.at[y, x].set(tile_int)

    pcgrl.env_state = pcgrl.env_state.replace(
        log_env_state=pcgrl.env_state.log_env_state.replace(
            env_state=pcgrl.env_state.log_env_state.env_state.replace(
                env_map=env_map
            )
        )
    )
    pcgrl.update_stats()
    return get_pcgrl_state()

@app.route('/update_tile', methods=['POST'])
def update_tile():
    data = request.get_json()
    coords = data['coords']
    x, y = coords.split(',')
    x, y = int(x), int(y)
    tile_type = data['tileType']
    global pcgrl
    if pcgrl is None:
        return jsonify(success=False, message='PCGRLWebApp not initialized')
    tile_int = pcgrl.tiles_to_ints[tile_type]
    env_map = pcgrl.env_state.log_env_state.env_state.env_map
    env_map = env_map.at[y, x].set(tile_int)
    pcgrl.env_state = pcgrl.env_state.replace(
        log_env_state=pcgrl.env_state.log_env_state.replace(
            env_state=pcgrl.env_state.log_env_state.env_state.replace(
                env_map=env_map
            )
        )
    )
    pcgrl.update_stats()
    return get_pcgrl_state()

@app.route('/get_tile_strs')
def get_tile_strs():
    global pcgrl
    if pcgrl is None:
        return []
    tile_strs = [t.name.lower() for t in pcgrl.tile_enum]
    return tile_strs

def get_map():
    global pcgrl
    env_map = pcgrl.env_state.log_env_state.env_state.env_map
    env_map = np.array(env_map)
    env_map_str = np.vectorize(pcgrl.ints_to_tiles.get)(env_map).tolist()
    return env_map_str

def get_path():
    global pcgrl
    path_coords = pcgrl.env.prob.get_path_coords(pcgrl.env_state.log_env_state.env_state.env_map,
                                                 prob_state=pcgrl.env_state.log_env_state.env_state.prob_state)
    path_coords = [np.array(pc).tolist() for pc in path_coords]
    return path_coords


@app.route('/init_env', methods=['POST'])
def init_env():
    global pcgrl
    if pcgrl is None:
        print('PCGRLWebApp not initialized, initializing...')
        pcgrl = PCGRLWebApp(enjoy_config)
    print('Environment initialized')
    return get_pcgrl_state()

@app.route('/get_folders')
def get_folders():
    base_path = '../saves'
    folders = [f.name for f in os.scandir(base_path) if f.is_dir()]
    folders = sorted(folders, reverse=False)
    return jsonify(folders)

@app.route('/tick', methods=['GET'])
def tick_pcgrl():
    global pcgrl
    if pcgrl is None:
        print('PCGRLWebApp not initialized, initializing...')
        pcgrl = PCGRLWebApp(enjoy_config)
    pcgrl.tick()
    return get_pcgrl_state()

def get_pcgrl_state():
    global pcgrl
    state = jsonify(
        map=get_map(),
        paths=get_path(),
    )
    return state

@app.route('/clear_map', methods=['POST'])
def clear_map():
    global pcgrl
    if pcgrl is None:
        return jsonify(success=False, message='PCGRLWebApp not initialized')
    env_map = jnp.full(pcgrl.env_params.map_shape, dtype=int, fill_value=pcgrl.tiles_to_ints['empty'])
    pcgrl.env_state = pcgrl.env_state.replace(
        log_env_state=pcgrl.env_state.log_env_state.replace(
            env_state=pcgrl.env_state.log_env_state.env_state.replace(
                env_map=env_map
            )
        )
    )
    pcgrl.update_stats()
    return get_pcgrl_state() 
    

@app.route('/reset_env', methods=['POST'])
def reset_pcgrl():
    global pcgrl
    if pcgrl is None:
        print('PCGRLWebApp not initialized, initializing...')
        pcgrl = PCGRLWebApp(enjoy_config)
    pcgrl.reset()
    return get_pcgrl_state()


@hydra.main(config_path='./', config_name='enjoy_pcgrl')
def main(cfg: EnjoyConfig):
    global enjoy_config
    enjoy_config = init_config(cfg)





if __name__ == '__main__':

    # Convenienve HACK so that we can render progress without stopping training. Uncomment this or 
    # set JAX_PLATFORM_NAME=cpu in your terminal environment before running this script to run it on cpu.
    # WARNING: Be sure to set it back to gpu before training again!
    main()
    app.run(debug=True)
