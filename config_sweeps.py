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
#         'seed': [0, 1, 2],
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
#         'seed': [0, 1, 2],
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
#         # 'seed': [3, 4, 5],
#         'seed': [0, 1, 2],
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
#         'obs_size': [5, 10, 15, 20, 25, 31],
#         'seed': [0, 1, 2],
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
#         'seed': [0, 1, 2],
#         'n_envs': [600],
#         'total_timesteps': [1_000_000_000],
#     }
# ]

# hypers = [
#     {
#         'NAME': 'arf_conv_binary',
#         'arf_size': [3, 5, 8, 16, 31],
#         'change_pct': [-1.0],
#         'seed': [0, 1, 2],
#         # 'seed': [3, 4, 5],
#         'n_envs': [600],
#         'max_board_scans': [4],
#         'total_timesteps': [200_000_000],
#      #  'eval_map_width': [16, 18, 20],
#     },
# ]

# hypers = [
#     {
#         'NAME': 'fixed_arf_dungeon',
#         'problem': ['dungeon'],
#         'model': ['seqnca'],
#         'arf_size': [8],
#         'vrf_size': [8, 12, 16, 23, 31],
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
#         'NAME': 'fixed_arf_binary',
#         'model': ['seqnca'],
#         'arf_size': [8],
#         'vrf_size': [8, 12, 16, 23, 31],
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
#         'problem': ['dungeon'],
#         'change_pct': [-1.0],
#         'seed': [0, 1, 2],
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
#         'seed': [0, 1, 2],
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
#         'obs_size': [3, 5, 8, 16, 31],
#         'model': ['conv2'],
#         'change_pct': [-1.0],
#         'seed': [0, 1, 2],
#         'n_envs': [600],
#         'max_board_scans': [5],
#         'total_timesteps': [1_000_000_000],
#     },
# ]

# hypers = [
#     {
#         'NAME': 'obss_cp_dungeon_conv2_ctrl_path',
#         'problem': ['dungeon'],
#         'ctrl_metrics': [['path_length']],
#         # 'obs_size': [16, 31],
#         'obs_size': [31],
#         'change_pct': [0.6, 0.8, 1.0],
#         'model': ['conv2'],
#         'seed': [0, 1, 2],
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
#         'arf_size': [3, 5, 8, 16, 31],
#         'change_pct': [-1.0],
#         # 'seed': [3, 4, 5],
#         'seed': [0, 1, 2],
#         'n_envs': [200],
#         'max_board_scans': [5],
#         'total_timesteps': [1_000_000_000],
#      #  'eval_map_width': [16, 18, 20],
#     },
# ]

# hypers = [
#     {
#         'NAME': 'act_shape_seqnca_dungeon',
#         'model': ['seqnca'],
#         'problem': ['dungeon'],
#         'act_shape': [(1,1), (2,2), (3,3), (4,4), (5,5), (6,6)],
#         'arf_size': [31],
#         'change_pct': [-1.0],
#         'seed': [3, 4, 5],
#         # 'seed': [0, 1, 2],
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
#         'arf_size': [31],
#         'change_pct': [-1.0],
#         # 'seed': [0, 1, 2],
#         'seed': [3, 4, 5],
#         'n_envs': [200],
#         'max_board_scans': [5],
#         'total_timesteps': [1_000_000_000],
#       # 'eval_map_width': [16, 18, 20],
#     },
# ]

# hypers = [
#     {
#         'NAME': 'arf_seqnca_dungeon',
#         'model': ['seqnca'],
#         'problem': ['dungeon'],
#         'arf_size': [3, 5, 8, 16, 31],
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
#         'obs_size': [15, 20, 25],
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
#         'model': ['conv2', 'seqnca'],
#         'problem': ['dungeon'],
#         'map_width': [8, 10, 18, 20],
#         'arf_size': [3, 5, 8, 16, 31, -1],
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
#         'NAME': 'diff_size_binary',
#         'model': ['conv2', 'seqnca'],
#         'map_width': [8, 10, 18, 20],
#         'arf_size': [3, 5, 8, 16, 31, -1],
#         'change_pct': [-1.0],
#         'seed': [3, 4, 5],
#         # 'seed': [0, 1, 2],
#         'n_envs': [600],
#         'max_board_scans': [5],
#         'total_timesteps': [1_000_000_000],
#     },
# ]

hypers = [
    {
        'NAME': 'diffshape_pinpoint_maze_emptystart',
        'model': ['conv2'],
        'problem': ['maze'],
        'map_width': [16],
        'obs_size': [8, 16, 24, -1],
        'change_pct': [-1.0],
        'seed': [0, 1, 2],
        'n_envs': [600],
        'randomize_map_shape': [True, False],
        'empty_start': [True],
        'pinpoints': [True, False],
        'max_board_scans': [5],
        'total_timesteps': [1_000_000_000],
    },
]

########################