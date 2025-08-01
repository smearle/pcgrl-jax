############ baseline experiments (?) ############
# # (-3) sweeping over action observation window without control
# hypers = {
#     'arf_size': [3, 5, 8, 16, 32],
#     'seed': [0, 1, 2],
#     'model': ['conv', 'seqnca'],
#     'n_envs': [600],
#     'total_timesteps': [200_000_000],
# }

# # (-2) sweeping over value branch observation window without control
# hypers = {
#     'vrf_size': [3, 5, 8, 16, 32],
#     'seed': [0, 1, 2],
#     'model': ['conv', 'seqnca'],
#     'n_envs': [600],
#     'total_timesteps': [200_000_000],
# }

# (-1) sweeping over observation window without control
# hypers = {
#     'obs_size': [3, 5, 8, 16],
#     'seed': [0, 1, 2],
#     'model': ['conv', 'seqnca'],
#     'n_envs': [600],
#     'total_timesteps': [200_000_000],
# }

############ experiments for controllability of different rf sizes ############
# # (2) sweeping over action observation window (formerly "patch width")
# hypers = {
#     'ctrl_metrics': [['diameter']],
#     'arf_size': [3, 5, 8, 16, 32],
#     'seed': [0, 1, 2],
#     'model': ['conv', 'seqnca'],
#     'n_envs': [600],
#     'total_timesteps': [200_000_000],
# }

# # (1) woops, this is actually what we meant at (0)
# hypers = {
#     'ctrl_metrics': [['diameter']],
#     'obs_size': [3, 5, 8, 16],
#     'seed': [0, 1, 2],
#     'model': ['conv', 'seqnca'],
#     'n_envs': [600],
#     'total_timesteps': [200_000_000],
# }

# # (0) first sweep for ICLR
# hypers = {
#     'ctrl_metrics': [['diameter']],
#     'vrf_size': [3, 5, 8, 16, 32],
#     'seed': [0, 1, 2],
#     'model': ['conv', 'seqnca'],
#     'n_envs': [600],
#     'total_timesteps': [200_000_000],
# }



### Jan. 2024 experiments ###

# hypers = [
#     {
#         'NAME': 'cp_binary',
#         'change_pct': [0.2, 0.4, 0.6, 0.8, 1.0],
#         'seed': list(range(6)),
#         'n_envs': [600],
#         'max_board_scans': [3.0],
#         'total_timesteps': [1_000_000_000],
#     },
# ]

# hypers = [
#     {
#         'NAME': 'cp_binary_conv2',
#         'model': ['conv2'],
#         'change_pct': [0.2, 0.4, 0.6, 0.8, 1.0],
#         'seed': list(range(6)),
#         'n_envs': [600],
#         'max_board_scans': [3.0],
#         'total_timesteps': [1_000_000_000],
#     },
# ]

# hypers = [
#     {
#         'NAME': 'cp_binary_seqnca',
#         'model': ['seqnca'],
#         'change_pct': [0.2, 0.4, 0.6, 0.8, 1.0],
#         'seed': list(range(3, 6)),
#         'n_envs': [600],
#         'max_board_scans': [3.0],
#         # 'total_timesteps': [200_000_000],
#         'total_timesteps': [1_000_000_000],
#     },
# ]

# hypers = [
#     {
#         'NAME': 'binary_conv2_rf',
#         'model': ['conv2'],
#         'obs_size': [5, 10, 15, 20, 25, -1],
#         'seed': [3, 4, 5],
#         'n_envs': [600],
#         'max_board_scans': [5.0],
#         'total_timesteps': [1_000_000_000],
#     },
# ]

# hypers = [
#     {
#         'NAME': 'bs_binary',
#         'max_board_scans': [1, 5, 10],
#         'change_pct': [-1.0],
#         # 'seed': list(range(6)),
#         'seed': list(range(3)),
#         'n_envs': [600],
#         'total_timesteps': [1_000_000_000],
#     }
# ]

# hypers = [
#     {
#         'NAME': 'arf_conv_binary',
#         'arf_size': [3, 5, 8, 16, -1],
#         'change_pct': [-1.0],
#         'seed': [3, 4, 5],
#         # 'seed': list(range(6)),
#         'n_envs': [600],
#         'max_board_scans': [4],
#         'total_timesteps': [200_000_000],
#     },
# ]

# hypers = [
#     {
#         'NAME': 'fixed_arf_dungeon',
#         'problem': ['dungeon'],
#         'model': ['seqnca'],
#         'arf_size': [8],
#         'vrf_size': [8, 12, 16, 23, -1],
#         'change_pct': [-1.0],
#         # 'seed': list(range(3, 6)),
#         'seed': list(range(0, 3)),
#         'n_envs': [600],
#         'max_board_scans': [5],
#         'total_timesteps': [1_000_000_000],
#     },
# ]


# hypers = [
#     {   
#         'NAME': 'fixed_arf_binary',
#         'model': ['seqnca'],
#         'arf_size': [8],
#         'vrf_size': [8, 12, 16, 23, 31],
#         'change_pct': [-1.0],
#         'seed': list(range(3, 6)),
#         'n_envs': [600],
#         'max_board_scans': [5],
#         'total_timesteps': [1_000_000_000],
#     },
# ]

# hypers = [
#     {
#         'NAME': 'dungeon',
#         'problem': ['dungeon'],
#         'change_pct': [-1.0],
#         'seed': list(range(3, 6)),
#         'n_envs': [600],
#         'max_board_scans': [4],
#         'total_timesteps': [200_000_000],
#     },
# ]

# hypers = [
#     {
#         'NAME': 'dungeon_conv2',
#         'problem': ['dungeon'],
#         'model': ['conv2'],
#         'change_pct': [-1.0],
#         'seed': list(range(6)),
#         'n_envs': [600],
#         'max_board_scans': [5],
#         'total_timesteps': [1_000_000_000],
#     },
# ]

# hypers = [
#     {
#         'NAME': 'obss_dungeon_conv2_ctrl_path',
#         'problem': ['dungeon'],
#         'ctrl_metrics': [['path_length']],
#         'obs_size': [3, 5, 8, 16, -1],
#         'model': ['conv2'],
#         'change_pct': [-1.0],
#         'seed': list(range(3)),
#         'n_envs': [600],
#         'max_board_scans': [5],
#         'total_timesteps': [1_000_000_000],
#     },
# ]


# hypers = [
#     {
#         'NAME': 'obss_hiddims_dungeon',
#         'problem': ['dungeon'],
#         'obs_size_hid_dims': [3, 5, 8, 16, -1],
#         'model': ['conv2'],
#         'change_pct': [-1.0],
#         'seed': list(range(5)),
#         'n_envs': [600],
#         'max_board_scans': [5],
#         'total_timesteps': [1_000_000_000],
#     },
# ]


# hypers = [
#     {
#         'NAME': 'obss_hiddims_dungeon2',
#         'problem': ['dungeon2'],
#         # 'obs_size_hid_dims': [3, 5, 8, 16, -1],
#         'obs_size_hid_dims': [3, 5, 8, 16],
#         # 'obs_size_hid_dims': [-1],
#         'model': ['conv2'],
#         'change_pct': [-1.0],
#         'seed': list(range(5)),
#         'n_envs': [600],
#         'max_board_scans': [5],
#         'total_timesteps': [1_000_000_000],
#     },
# ]


# hypers = [
#     {
#         'NAME': 'obss_hiddims_dungeon_conv2_ctrl_path',
#         'problem': ['dungeon'],
#         'ctrl_metrics': [['path_length']],
#         # 'obs_size': [16, -1],
#         'obs_size_hid_dims': [3, 5, 8, 16, -1],
#         'model': ['conv2'],
#         'change_pct': [-1.0],
#         'seed': list(range(3,6)),
#         'n_envs': [100],
#         'max_board_scans': [5],
#         'total_timesteps': [1_000_000_000],
#     },
# ]


# hypers = [
#     {
#         'NAME': 'obss_cp_dungeon_conv2_ctrl_path',
#         'problem': ['dungeon'],
#         'ctrl_metrics': [['path_length']],
#         'obs_size': [16, -1],
#         # 'obs_size': [-1],
#         'change_pct': [0.6, 0.8, 1.0],
#         'model': ['conv2'],
#         'seed': [0, 1, 2],
#         # 'seed': list(range(6)),
#         'n_envs': [600],
#         'max_board_scans': [5],
#         'total_timesteps': [1_000_000_000],
#     },
# ]

# hypers = [
#     {
#         'NAME': 'wide_nca_binary_cp',
#         'model': ['nca'],
#         'representation': ['wide'],
#         'problem': ['binary'],
#         'change_pct': [0.6, 0.8, 1.0, -1],
#         'seed': [0, 1, 2],
#         'n_envs': [100],
#         'max_board_scans': [5],
#         'total_timesteps': [1_000_000_000],
#     },
# ]

# hypers = [
#     {
#         'NAME': 'arf_seqnca_binary',
#         'model': ['seqnca'],
#         'arf_size': [3, 5, 8, 16, -1],
#         'change_pct': [-1.0],
#         'seed': list(range(6)),
#         'n_envs': [200],
#         'max_board_scans': [5],
#         'total_timesteps': [1_000_000_000],
#     },
# ]

# hypers = [
#     {
#         'NAME': 'act_shape_seqnca_dungeon',
#         'model': ['seqnca'],
#         'problem': ['dungeon'],
#         'act_shape': [(1,1), (2,2), (3,3), (4,4), (5,5), (6,6)],
#         'arf_size': [-1],
#         'change_pct': [-1.0],
#         'seed': list(range(6)),
#         'n_envs': [200],
#         'max_board_scans': [5],
#         'total_timesteps': [1_000_000_000],
#      #  'eval_map_width': [16, 18, 20],
#     },
# ]

# hypers = [
#     {
#         'NAME': 'act_shape_conv2_dungeon',
#         'model': ['conv2'],
#         'problem': ['dungeon'],
#         'act_shape': [(1, 1), (2,2), (3,3), (4,4), (5,5), (6,6)],
#         'arf_size': [-1],
#         'change_pct': [-1.0],
#         'seed': [0, 1, 2],
#         # 'seed': [3, 4, 5],
#         'n_envs': [200],
#         'max_board_scans': [5],
#         'total_timesteps': [1_000_000_000],
#       # 'eval_map_width': [16, 18, 20],
#     },
# ]

# hypers = [
#     {
#         'NAME': 'arf_seqnca_dungeon',    # FIXME looks like we don't have arf conv dungeon??
#         'model': ['seqnca'],
#         'problem': ['dungeon'],
#         'arf_size': [3, 5, 8, 16, -1],
#         # 'arf_size': [31],
#         'change_pct': [-1.0],
#         'seed': [0, 1, 2],
#         # 'seed': [3, 4, 5],
#         'n_envs': [200],
#         'max_board_scans': [5],
#         # 'eval_map_width': [16, 18, 20],
#         'total_timesteps': [1_000_000_000],
#     },
# ]


# hypers = [
#     {
#         'NAME': 'obss_conv2_dungeon',
#         'model': ['conv2'],
#         'problem': ['dungeon'],
#         'obs_size': [15, 20, 25, -1],
#         'change_pct': [-1.0],
#         'seed': [0, 1, 2],
#         'n_envs': [400],
#         'max_board_scans': [5],
#         'total_timesteps': [1_000_000_000],
#     },
# ]


###### add some experiment to train on smaller/bigger map size for comparison #######
# hypers = [
#     {
#         'NAME': 'diff_size_dungeon',
#         # 'model': ['conv2', 'seqnca'],
#         'model': ['conv2'],
#         'problem': ['dungeon'],
#         # 'map_width': [8, 10, 18, 20],
#         'map_width': [24, 28, 32],
#         'obs_size': [31],
#         'change_pct': [-1.0],
#         # 'seed': [3, 4, 5],
#         'seed': [0, 1, 2],
#         'n_envs': [200],
#         'max_board_scans': [5],
#         'total_timesteps': [20_000_000],
#     },
# ]

# hypers = [
#     {
#         'NAME': 'diff_size_binary',
#         'model': ['conv2', 'seqnca'],
#         'map_width': [8, 10, 18, 20],
#         'arf_size': [-1],
#         # 'arf_size': [3, 5, 8, 16, 31, -1],
#         'change_pct': [-1.0],
#         'seed': [3, 4, 5],
#         # 'seed': [0, 1, 2],
#         'n_envs': [600],
#         'max_board_scans': [5],
#         'total_timesteps': [1_000_000_000],
#     },
# ]

# hypers = [
#     {
#         'NAME': 'diffshape_pinpoint_maze_emptystart',
#         'model': ['conv2'],
#         'problem': ['maze'],
#         'map_width': [16],
#         'obs_size': [8, 16, 24, -1],
#         'randomize_map_shape': [True, False],
#         # 'randomize_map_shape': [True],
#         'change_pct': [-1.0],
#         # 'seed': [3, 4, 5],
#         'seed': [0, 1, 2],
#         'n_envs': [600],
#         'empty_start': [True],
#         # 'pinpoints': [True, False],
#         'pinpoints': [True],
#         'max_board_scans': [5],
#         'total_timesteps': [1_000_000_000],
#     },
# ]

# hypers = [
#     {
#         'NAME': 'diffshape_pinpoint_ctrl_dungeon',
#         'model': ['conv2'],
#         'problem': ['dungeon'],
#         'map_width': [16],
#         'ctrl_metrics': [['path_length', 'nearest_enemy']],
#         'obs_size': [8, 16, 24, -1],
#         'change_pct': [-1.0],
#         'seed': [0, 1, 2],
#         'n_envs': [600],
#         'randomize_map_shape': [True],
#         'empty_start': [False],
#         'pinpoints': [True],
#         'max_board_scans': [5],
#         'total_timesteps': [1_000_000_000],
#     },
# ]

# hypers = [
#     {
#         'NAME': 'bigdungeon',
#         'model': ['conv2'],
#         'problem': ['dungeon'],
#         'map_width': [32],
#         'change_pct': [0.8, -1.0],
#         'seed': [0, 1, 2],
#         'n_envs': [200],
#         'randomize_map_shape': [True],
#         'ctrl_metrics': [['path_length', 'nearest_enemy']],
#         'empty_start': [False],
#         'pinpoints': [True],
#         'max_board_scans': [5],
#         'total_timesteps': [1_000_000_000],
#         'timestep_chunk_size': [10_000_000],
#         # 'obs_size_hid_dims': [8, 16, 24, 31],
#     },
# ]

# hypers = [
#     {
#         'NAME': 'bigdungeon2',
#         'model': ['conv2'],
#         'problem': ['dungeon2'],
#         'map_width': [32],
#         'change_pct': [0.8, -1.0],
#         'seed': [0, 1, 2],
#         'n_envs': [200],
#         'randomize_map_shape': [True],
#         'ctrl_metrics': [['path_length', 'nearest_enemy']],
#         'empty_start': [False],
#         'pinpoints': [True],
#         'max_board_scans': [5],
#         'total_timesteps': [1_000_000_000],
#         'timestep_chunk_size': [10_000_000],
#         # 'obs_size_hid_dims': [8, 16, 24, 31],
#     },
# ]
 
# hypers = [
#     {
#         'NAME': 'everything_dungeon',
#         'model': ['conv2'],
#         'problem': ['dungeon'],
#         'seed': [0, 1, 2],
#         'n_envs': [600],
#         'randomize_map_shape': [True, False],
#         'ctrl_metrics': [['path_length', 'nearest_enemy']],
#         'empty_start': [True],
#         'pinpoints': [True, False],
#         'max_board_scans': [5],
#         'total_timesteps': [1_000_000_000],
#         'timestep_chunk_size': [10_000_000],
#         'obs_size_hid_dims': [8, 16, 24, 31],
#     },
# ]


# hypers = [
#     {
#         'NAME': 'ma_obs_size',
#         'multiagent': True,

#         'problem': ['binary'],
#         'map_width': [16],
#         'n_agents': [2],
#         'representation': ['turtle'],
#         'model': ['rnn'],
#         'obs_size_hid_dims': [3, 5, 8, 16, -1],
#         'n_envs': [400],
#         'empty_start': [False],
#         'total_timesteps': [1_000_000_000],
#         'ckpt_freq': [100],
#         'render_freq': [100],
#         'seed': [0, 1, 2],
#     },
# ]


# hypers = [
#     {
#         'NAME': 'ma_board_scans',
#         'multiagent': True,

#         'problem': ['binary'],
#         'map_width': [16],
#         'n_agents': [1,2,3],
#         'representation': ['turtle'],
#         'model': ['conv2'],
#         'obs_size_hid_dims': [3, 16, 31],
#         'max_board_scans': [0.75, 1.0, 1.5],
#         # 'max_board_scans': [3.0],
#         # 'randomize_map_shape': [True, False],
#         # 'randomize_map_shape': [True],
#         # 'change_pct': [-1.0],
#         # 'seed': [3, 4, 5],
#         # 'seed': [0, 1, 2],
#         'n_envs': [400],
#         'empty_start': [False],
#         # 'pinpoints': [True, False],
#         'total_timesteps': [1_000_000_000],
#         'ckpt_freq': [100],
#         'render_freq': [100],
#         'seed': [0, 1, 2],
#     },
# ]

# hypers = [
#     {
#         'NAME': 'ma_n_agents',
#         'multiagent': True,

#         'problem': ['binary'],
#         'map_width': [16],
#         'n_agents': [1,2,3],
#         'representation': ['turtle'],
#         'model': ['rnn'],
#         'obs_size_hid_dims': [3],
#         'max_board_scans': [1.0],
#         # 'max_board_scans': [3.0],
#         # 'randomize_map_shape': [True, False],
#         # 'randomize_map_shape': [True],
#         # 'change_pct': [-1.0],
#         # 'seed': [3, 4, 5],
#         # 'seed': [0, 1, 2],
#         'n_envs': [400],
#         'empty_start': [False],
#         # 'pinpoints': [True, False],
#         'total_timesteps': [1_000_000_000],
#         'ckpt_freq': [100],
#         'render_freq': [100],
#         'seed': [0, 1, 2],
#     },
# ]

# hypers = [
#     {
#         'NAME': 'ma_obs_size',
#         'multiagent': True,

#         'problem': ['binary'],
#         'map_width': [16],
#         'n_agents': [3],
#         'representation': ['turtle'],
#         'model': ['rnn'],
#         'obs_size_hid_dims': [3, 16, 31],
#         'max_board_scans': [1.0],
#         # 'max_board_scans': [3.0],
#         # 'randomize_map_shape': [True, False],
#         # 'randomize_map_shape': [True],
#         # 'change_pct': [-1.0],
#         # 'seed': [3, 4, 5],
#         # 'seed': [0, 1, 2],
#         'n_envs': [400],
#         'empty_start': [False],
#         # 'pinpoints': [True, False],
#         'total_timesteps': [1_000_000_000],
#         'ckpt_freq': [100],
#         'render_freq': [100],
#         'seed': [0, 1, 2],
#     },
# ]

# hypers = [
#     {
#         'NAME': 'ma_n_steps',
#         'multiagent': True,

#         'problem': ['binary'],
#         'map_width': [16],
#         'n_agents': [3],
#         'representation': ['turtle'],
#         'model': ['rnn'],
#         'obs_size_hid_dims': [3],
#         'max_board_scans': [0.75, 1.0, 1.5],
#         # 'max_board_scans': [3.0],
#         # 'randomize_map_shape': [True, False],
#         # 'randomize_map_shape': [True],
#         # 'change_pct': [-1.0],
#         # 'seed': [3, 4, 5],
#         # 'seed': [0, 1, 2],
#         'n_envs': [400],
#         'empty_start': [False],
#         # 'pinpoints': [True, False],
#         'total_timesteps': [1_000_000_000],
#         'ckpt_freq': [100],
#         'render_freq': [100],
#         'seed': [0, 1, 2],
#     },
# ]

# hypers = [
#     {
#         'NAME': 'ma_5_agents',
#         'multiagent': True,

#         'problem': ['binary'],
#         'map_width': [16],
#         'n_agents': [5],
#         'representation': ['turtle'],
#         'model': ['rnn'],
#         'obs_size_hid_dims': [3],
#         'max_board_scans': [0.75, 1.0, 1.5],
#         # 'max_board_scans': [3.0],
#         # 'randomize_map_shape': [True, False],
#         # 'randomize_map_shape': [True],
#         # 'change_pct': [-1.0],
#         # 'seed': [3, 4, 5],
#         # 'seed': [0, 1, 2],
#         'n_envs': [200],
#         'empty_start': [False],
#         # 'pinpoints': [True, False],
#         'total_timesteps': [1_000_000_000],
#         'ckpt_freq': [100],
#         'render_freq': [100],
#         'seed': [0, 1, 2],
#     },
# ]

# hypers = [
#     {
#         'NAME': 'ma_board_scans_conv2',
#         'multiagent': True,

#         'problem': ['binary'],
#         'map_width': [16],
#         'n_agents': [1,2,3],
#         'representation': ['turtle'],
#         'model': ['conv2'],
#         'obs_size': [3, 16, 31],
#         'max_board_scans': [0.75, 1.0, 1.5],
#         'n_envs': [400],
#         'empty_start': [False],
#         'total_timesteps': [1_000_000_000],
#         'ckpt_freq': [100],
#         'render_freq': [100],
#         'seed': [0, 1, 2],
#     },
# ]

# hypers = [
#     {
#         'NAME': 'ma_freezer',
#         'multiagent': True,

#         'problem': ['binary'],
#         'a_freezer': [True],
#         'map_width': [16],
#         'n_agents': [2,3],
#         'representation': ['turtle'],
#         'model': ['rnn'],
#         'obs_size_hid_dims': [3],
#         'max_board_scans': [0.5, 0.75, 1.0, 1.5],
#         # 'randomize_map_shape': [True, False],
#         # 'randomize_map_shape': [True],
#         # 'change_pct': [-1.0],
#         # 'seed': [3, 4, 5],
#         # 'seed': [0, 1, 2],
#         'n_envs': [400],
#         'empty_start': [False],
#         # 'pinpoints': [True, False],
#         'total_timesteps': [1_000_000_000],
#         'ckpt_freq': [100],
#         'render_freq': [100],
#         'seed': [0, 1, 2],
#     },
# ]



# hypers = [
#     {
#         'NAME': 'ma_board_scans_0.5_rebuttal',
#         'multiagent': True,

#         'problem': ['binary'],
#         'map_width': [16],
#         'n_agents': [1,2,3],
#         'representation': ['turtle'],
#         'model': ['rnn'],
#         'obs_size_hid_dims': [3, 16, 31],
#         'max_board_scans': [0.5],
#         'randomize_map_shape': [False],
#         # 'change_pct': [-1.0],
#         'n_envs': [400],
#         'empty_start': [False],
#         # 'pinpoints': [True, False],
#         'total_timesteps': [300_000_000],
#         'ckpt_freq': [100],
#         'render_freq': [100],
#         'seed': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
#     },
# ]

# hypers = [
#     {
#         'NAME': 'ma_board_scans_binary_rebuttal_more_seeds',
#         'multiagent': True,

#         'problem': ['binary'],
#         'map_width': [16],
#         'n_agents': [1,2,3],
#         'representation': ['turtle'],
#         'model': ['rnn'],
#         'obs_size_hid_dims': [3, 16, 31],
#         'max_board_scans': [0.5],
#         # 'max_board_scans': [3.0],
#         'randomize_map_shape': [True, False],
#         # 'randomize_map_shape': [True],
#         # 'change_pct': [-1.0],
#         # 'seed': [3, 4, 5],
#         # 'seed': [0, 1, 2],
#         'n_envs': [400],
#         'empty_start': [False],
#         # 'pinpoints': [True, False],
#         'total_timesteps': [1_000_000_000],
#         'ckpt_freq': [100],
#         'render_freq': [100],
#         'seed': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
#     },
# ]


# hypers = [
#     {
#         'NAME': 'ma_board_scans_maze_obs_3',
#         'multiagent': True,

#         'problem': ['maze'],
#         'map_width': [16],
#         'n_agents': [1, 2, 3],
#         'representation': ['turtle'],
#         'model': ['rnn'],
#         'obs_size_hid_dims': [3],
#         'max_board_scans': [1.0],
#         'randomize_map_shape': [True, False],
#         # 'change_pct': [-1.0],
#         'n_envs': [400],
#         'empty_start': [False],
#         'total_timesteps': [300_000_000],
#         'ckpt_freq': [100],
#         'render_freq': [100],
#         'seed': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
#     },
# ]

# hypers = [
#     {
#         'NAME': 'ma_board_scans_maze_obs_31',
#         'multiagent': True,

#         'problem': ['maze'],
#         'map_width': [16],
#         'n_agents': [1, 2, 3],
#         'representation': ['turtle'],
#         'model': ['rnn'],
#         'obs_size_hid_dims': [31],
#         'max_board_scans': [1.0],
#         'randomize_map_shape': [True, False],
#         # 'change_pct': [-1.0],
#         'n_envs': [400],
#         'empty_start': [False],
#         'total_timesteps': [300_000_000],
#         'ckpt_freq': [100],
#         'render_freq': [100],
#         'seed': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
#     },
# ]

# hypers = [
#     {
#         'NAME': 'ma_board_scans_dungeon_obs_3',
#         'multiagent': True,

#         'problem': ['dungeon'],
#         'map_width': [16],
#         'n_agents': [1, 2, 3],
#         'representation': ['turtle'],
#         'model': ['rnn'],
#         'obs_size_hid_dims': [3],
#         'max_board_scans': [1.0],
#         'randomize_map_shape': [True, False],
#         # 'change_pct': [-1.0],
#         'n_envs': [400],
#         'empty_start': [False],
#         'total_timesteps': [300_000_000],
#         'ckpt_freq': [100],
#         'render_freq': [100],
#         'seed': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
#     },
# ]

# hypers = [
#     {
#         'NAME': 'ma_n_agents_dungeon_conv2_obs_3',
#         'multiagent': True,

#         'problem': ['dungeon'],
#         'map_width': [16],
#         'n_agents': [1, 2, 3],
#         # 'n_agents': [3],
#         'representation': ['turtle'],
#         'model': ['conv2'],
#         'obs_size_hid_dims': [3],
#         # 'obs_size_hid_dims': [3, 5, 8],
#         'max_board_scans': [1.0],
#         # 'max_board_scans': [3.0],
#         'randomize_map_shape': [False],
#         # 'change_pct': [-1.0],
#         'n_envs': [400],
#         'empty_start': [False],
#         'total_timesteps': [900_000_000],
#         'ckpt_freq': [100],
#         'render_freq': [100],
#         'seed': [0, 1, 2, 3, 4],
#     },
# ]

# hypers = [
#     {
#         'NAME': 'ma_board_scans_dungeon_obs_31',
#         'multiagent': True,

#         'problem': ['dungeon'],
#         'map_width': [16],
#         'n_agents': [1, 2, 3],
#         'representation': ['turtle'],
#         'model': ['rnn'],
#         'obs_size_hid_dims': [31],
#         'max_board_scans': [1.0],
#         'randomize_map_shape': [True, False],
#         # 'change_pct': [-1.0],
#         'n_envs': [400],
#         'empty_start': [False],
#         'total_timesteps': [300_000_000],
#         'ckpt_freq': [100],
#         'render_freq': [100],
#         'seed': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
#     },
# ]

# hypers = [
#     {
#         'NAME': 'ma_board_scans_maze_and_dungeon_obs_3_empty_start',
#         'multiagent': True,

#         'problem': ['maze','dungeon'],
#         'map_width': [16],
#         'n_agents': [1, 2, 3],
#         'representation': ['turtle'],
#         'model': ['rnn'],
#         'obs_size_hid_dims': [3],
#         'max_board_scans': [1.0],
#         'randomize_map_shape': [True, False],
#         # 'change_pct': [-1.0],
#         'n_envs': [400],
#         'empty_start': [True],
#         'total_timesteps': [300_000_000],
#         'ckpt_freq': [100],
#         'render_freq': [100],
#         'seed': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
#     },
# ]

# hypers = [
#     {
#         'NAME': 'ma_sparse_reward_n_agents',
#         'multiagent': True,

#         'problem': ['binary'],
#         'a_freezer': [True],
#         'map_width': [16],
#         'n_agents': [1],
#         'reward_freq': [1, 2, 3, 5, 10],
#         'representation': ['turtle'],
#         'model': ['rnn'],
#         'obs_size_hid_dims': [3],
#         'max_board_scans': [1.0, 3.0],
#         # 'max_board_scans': [1.0, 3.0],
#         'n_envs': [400],
#         'empty_start': [False],
#         'total_timesteps': [1_000_000_000],
#     },
# ]
# hypers = [
#     {
#         'NAME': 'ma_board_scans_dungeon_obs_31',
#         'multiagent': True,

#         'problem': ['dungeon'],
#         'map_width': [16],
#         'n_agents': [1, 2, 3],
#         'representation': ['turtle'],
#         'model': ['rnn'],
#         'obs_size_hid_dims': [31],
#         'max_board_scans': [1.0],
#         'randomize_map_shape': [True, False],
#         # 'change_pct': [-1.0],
#         'n_envs': [400],
#         'empty_start': [False],
#         'total_timesteps': [300_000_000],
#         'ckpt_freq': [100],
#         'render_freq': [100],
#         'seed': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
#     },
# ]

# hypers = [
#     {
#         'NAME': 'ma_board_scans_n_agents',
#         'multiagent': True,

#         'problem': ['binary'],
#         'map_width': [16],
#         'n_agents': [1,2,3],
#         'representation': ['turtle'],
#         'model': ['rnn'],
#         'obs_size_hid_dims': [3],
#         'max_board_scans': [0.75, 1.0, 1.5],
#         'n_envs': [400],
#         'empty_start': [False],
#         'total_timesteps': [1_000_000_000],
#         'ckpt_freq': [100],
#         'render_freq': [100],
#         'seed': [0, 1, 2],
#     },
# ]

# hypers = [
#     {
#         'NAME': 'ma_n_agents',
#         'multiagent': True,

#         'problem': ['binary'],
#         'map_width': [16],
#         'n_agents': [1,2,3],
#         'representation': ['turtle'],
#         'model': ['rnn'],
#         'obs_size_hid_dims': [3],
#         'max_board_scans': [1.0],
#         'n_envs': [400],
#         'empty_start': [False],
#         'total_timesteps': [1_000_000_000],
#         'ckpt_freq': [100],
#         'render_freq': [100],
#         'seed': [0, 1, 2],
#     },
# ]

# hypers = [
#     {
#         'NAME': 'ma_n_envs',
#         'multiagent': True,

#         'problem': ['binary'],
#         'map_width': [16],
#         'n_agents': [1],
#         'representation': ['turtle'],
#         'model': ['rnn'],
#         'obs_size_hid_dims': [3],
#         'max_board_scans': [1.0],
#         'n_envs': [400, 800, 1200],
#         'empty_start': [False],
#         'total_timesteps': [1_000_000_000],
#         'ckpt_freq': [100],
#         'render_freq': [100],
#         'seed': [0, 1, 2],
#     },
# ]


# hypers = [
#     {
#         'NAME': 'ma_obs_size',
#         'multiagent': True,

#         'problem': ['binary'],
#         'map_width': [16],
#         'n_agents': [3],
#         'representation': ['turtle'],
#         'model': ['rnn'],
#         'obs_size_hid_dims': [3, 16, 31],
#         'max_board_scans': [1.0],
#         'n_envs': [400],
#         'empty_start': [False],
#         'total_timesteps': [1_000_000_000],
#         'ckpt_freq': [100],
#         'render_freq': [100],
#         'seed': [0, 1, 2],
#     },
# ]

# hypers = [
#     {
#         'NAME': 'ma_n_steps',
#         'multiagent': True,

#         'problem': ['binary'],
#         'map_width': [16],
#         'n_agents': [3],
#         'representation': ['turtle'],
#         'model': ['rnn'],
#         'obs_size_hid_dims': [3],
#         'max_board_scans': [0.75, 1.0, 1.5],
#         'n_envs': [400],
#         'empty_start': [False],
#         # 'pinpoints': [True, False],
#         'total_timesteps': [1_000_000_000],
#         'ckpt_freq': [100],
#         'render_freq': [100],
#         'seed': [0, 1, 2],
#     },
# ]

# hypers = [
#     {
#         'NAME': 'ma_5_agents',
#         'multiagent': True,

#         'problem': ['binary'],
#         'map_width': [16],
#         'n_agents': [4,5],
#         'representation': ['turtle'],
#         'model': ['rnn'],
#         'obs_size_hid_dims': [3],
#         'max_board_scans': [0.75, 1.0, 1.5],
#         'n_envs': [200],
#         'empty_start': [False],
#         # 'pinpoints': [True, False],
#         'total_timesteps': [1_000_000_000],
#         'ckpt_freq': [100],
#         'render_freq': [100],
#         'seed': [0, 1, 2],
#     },
# ]

# hypers = [
#     {
#         'NAME': 'ma_board_scans_conv2',
#         'multiagent': True,

#         'problem': ['binary'],
#         'map_width': [16],
#         'n_agents': [1,2,3],
#         'representation': ['turtle'],
#         'model': ['conv2'],
#         'obs_size': [3, 16, 31],
#         'max_board_scans': [0.75, 1.0, 1.5],
#         'n_envs': [400],
#         'empty_start': [False],
#         'total_timesteps': [1_000_000_000],
#         'ckpt_freq': [100],
#         'render_freq': [100],
#         'seed': [0, 1, 2],
#     },
# ]

# hypers = [
#     {
#         'NAME': 'ma_board_scans_n_agents_conv2',
#         'multiagent': True,

#         # 'problem': ['binary', 'dungeon'],
#         'problem': ['dungeon'],
#         'map_width': [16],
#         # 'n_agents': [1,2,3],
#         'n_agents': [3],
#         'representation': ['turtle'],
#         'model': ['conv2'],
#         # 'obs_size_hid_dims': [3],
#         # 'max_board_scans': [0.75, 1.0, 1.5],
#         'max_board_scans': [1.5],
#         # 'randomize_map_shape': [True, False],
#         'randomize_map_shape': [True],
#         'n_envs': [400],
#         'empty_start': [False],
#         'total_timesteps': [1_000_000_000],
#         'ckpt_freq': [100],
#         'render_freq': [100],
#         # 'seed': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
#         'seed': [0],
#     },
# ]

# hypers = [
#     {
#         'NAME': 'ma_4_agents',
#         'multiagent': True,

#         'problem': ['binary'],
#         'map_width': [16],
#         'n_agents': [4],
#         'representation': ['turtle'],
#         'model': ['rnn'],
#         'obs_size_hid_dims': [3],
#         'max_board_scans': [0.75, 1.0, 1.5],
#         'n_envs': [200],
#         'empty_start': [False],
#         'total_timesteps': [1_000_000_000],
#         'ckpt_freq': [100],
#         'render_freq': [100],
#         'seed': [0, 1, 2],
#     },
# ]


# hypers = [
#     {
#         'NAME': 'ma_n_agents_full_start',
#         'multiagent': True,

#         'problem': ['binary'],
#         'map_width': [16],
#         'n_agents': [1,2,3],
#         'representation': ['turtle'],
#         'model': ['rnn'],
#         'obs_size_hid_dims': [3],
#         # 'max_board_scans': [1.0],
#         'max_board_scans': [3.0],
#         'n_envs': [400],
#         'full_start': [True],
#         'total_timesteps': [1_000_000_000],
#         'ckpt_freq': [100],
#         'render_freq': [100],
#         'seed': [0, 1, 2],
#     },
# ]

# hypers = [
#     {
#         'NAME': 'ma_freezer',
#         'multiagent': True,

#         'problem': ['binary'],
#         'a_freezer': [True],
#         'map_width': [16],
#         'n_agents': [2,3],
#         'representation': ['turtle'],
#         'model': ['rnn'],
#         'obs_size_hid_dims': [3],
# #       'max_board_scans': [0.5, 0.75, 1.0, 1.5],
#         'max_board_scans': [0.75, 1.0, 1.5],
#         'n_envs': [400],
#         'empty_start': [False],
#         'total_timesteps': [1_000_000_000],
#         'ckpt_freq': [100],
#         'render_freq': [100],
#         'seed': [0, 1, 2],
#     },
# ]


# hypers = [
#     {
#         'NAME': 'ma_sparse_reward_n_agents',
#         'multiagent': True,

#         'problem': ['binary'],
#         'a_freezer': [True],
#         'map_width': [16],
#         'n_agents': [1],
#         'reward_freq': [1, 2, 3, 5, 10],
#         'representation': ['turtle'],
#         'model': ['rnn'],
#         'obs_size_hid_dims': [3],
#         'max_board_scans': [1.0, 3.0],
#         # 'max_board_scans': [1.0, 3.0],
#         'n_envs': [400],
#         'empty_start': [False],
#         'total_timesteps': [1_000_000_000],
#         'ckpt_freq': [100],
#         'render_freq': [100],
#         'seed': [0, 1, 2],
#     },
# ]

# hypers = [
#     {
#         'NAME': 'for_some_reasoning_conv2',
#         'multiagent': False,

#         'problem': ['binary', 'maze', 'dungeon'],
#         'map_width': [16],
#         'n_agents': [1],
#         'representation': ['narrow','turtle'],
#         'model': ['conv2'],
#         'max_board_scans': [1.0, 3.0],
#         'n_envs': [400],
#         'empty_start': [False, True],
#         'total_timesteps': [1_000_000_000],
#         'ckpt_freq': [100],
#         'render_freq': [100],
#         'seed': [0, 1, 2, 3, 4, 5],
#     },
# ]

# hypers = [
#     {   
#         'NAME': 'for_some_reasoning_seqnca',
#         'multiagent': False,

#         'problem': ['binary', 'maze', 'dungeon'],
#         'model': ['seqnca'],
#         'representation': ['narrow'],
#         'arf_size': [8, 5, 3],
#         'change_pct': [-1.0],
#         'seed': list(range(0, 6)),
#         'n_envs': [400],
#         'max_board_scans': [1, 3],
#         'total_timesteps': [1_000_000_000],
#     },
# ]


hypers = [
    {
        'NAME': 'ma_n_agents_dungeon_conv2_obs_3',
        'multiagent': True,

        'problem': ['dungeon'],
        'map_width': [16],
        'n_agents': [1, 2, 3],
        # 'n_agents': [3],
        'representation': ['turtle'],
        'model': ['conv2'],
        'obs_size_hid_dims': [5],
        # 'obs_size_hid_dims': [3, 5, 8],
        'max_board_scans': [1.0, 3.0],
        # 'max_board_scans': [3.0],
        'randomize_map_shape': [False],
        # 'change_pct': [-1.0],
        'n_envs': [400],
        'empty_start': [False],
        'total_timesteps': [2_000_000_000],
        'ckpt_freq': [100],
        'render_freq': [100],
        'seed': [0, 1, 2, 3, 4],
    },
]

# hypers = [
#     {
#         'NAME': 'ma_board_scans_binary_aiide',
#         'multiagent': True,
# 
#         'problem': ['binary'],
#         'map_width': [16],
#         'n_agents': [1,2,3],
#         'representation': ['turtle'],
#         'model': ['conv2'],
#         'obs_size_hid_dims': [3, 31],
#         'max_board_scans': [1.0],
#         # 'max_board_scans': [3.0],
#         'randomize_map_shape': [False],
#         'n_envs': [400],
#         'empty_start': [False],
#         # 'pinpoints': [True, False],
#         'total_timesteps': [900_000_000],
#         'ckpt_freq': [100],
#         'render_freq': [100],
#         'seed': [0, 1, 2, 3, 4],
#     },
# ]

hypers = [
    {
        'NAME': 'ma_board_scans_binary_aiide',
        'multiagent': True,

        'problem': ['binary'],
        'map_width': [16],
        'n_agents': [1,2,3],
        'representation': ['turtle'],
        'model': ['conv2'],
        'obs_size_hid_dims': [3, 31],
        'max_board_scans': [1.0],
        # 'max_board_scans': [3.0],
        'randomize_map_shape': [False],
        'n_envs': [400],
        'empty_start': [False],
        # 'pinpoints': [True, False],
        'total_timesteps': [900_000_000],
        'ckpt_freq': [100],
        'render_freq': [100],
        'seed': [0, 1, 2, 3, 4],
    },
]

eval_hypers = {
    'eval_randomize_map_shape': [True, False],
    # 'eval_randomize_map_shape': [True],
    # 'eval_map_width': [8, 16, 24, 32],
    'eval_map_width': [16],
    # 'eval_max_board_scans': [10],
}

########################
