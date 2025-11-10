
base1_setting = {
    'spatial_encoder_param': {
        'last_channel': 128,
        'interverted_residual_setting': [
            # t, c, n, s
            [1, 16, 1, 1],
            [6, 24, 2, 2],
            [6, 32, 3, 2],
            [6, 64, 4, 2],
        ],
        'upsample_scale_factor': 10,
    },

    'keypoints_encoder_param': {
        'interverted_residual_setting': [
            # t, c, n, s
            [1, 4, 1, 1],
            [6, 8, 2, 1],
            [6, 16, 3, 2],
            [6, 21, 4, 1],
        ],
        'upsample_scale_factor': 1,
    },

    'cross_keypoints_fusion_param':{
        'trainable': True,
        'init_adjacent_matrix': True,
    },

    'temporal_keypoints_fusion_param':{
        'num_history': 9,
        'num_blocks': 4,
    },

    'handpose_encoder_param': {
        'num_head': 4,
        'dim_feedforward': 1024,
        'num_layers': 4,
    },
}



base1_setting_varaint1 = {# initialize with structure kernel without training for base 1 model
    'spatial_encoder_param': {
        'last_channel': 128,
        'interverted_residual_setting': [
            # t, c, n, s
            [1, 16, 1, 1],
            [6, 24, 2, 2],
            [6, 32, 3, 2],
            [6, 64, 4, 2],
        ],
        'upsample_scale_factor': 10,
    },

    'keypoints_encoder_param': {
        'interverted_residual_setting': [
            # t, c, n, s
            [1, 4, 1, 1],
            [6, 8, 2, 1],
            [6, 16, 3, 2],
            [6, 21, 4, 1],
        ],
        'upsample_scale_factor': 1,
    },

    'cross_keypoints_fusion_param':{
        'trainable': False,
        'init_adjacent_matrix': True,
    },

    'temporal_keypoints_fusion_param':{
        'num_history': 9,
        'num_blocks': 4,
    },

    'handpose_encoder_param': {
        'num_head': 4,
        'dim_feedforward': 1024,
        'num_layers': 4,
    },
}



base1_setting_varaint2 = {# random initializatign with training for base 1 model
    'spatial_encoder_param': {
        'last_channel': 128,
        'interverted_residual_setting': [
            # t, c, n, s
            [1, 16, 1, 1],
            [6, 24, 2, 2],
            [6, 32, 3, 2],
            [6, 64, 4, 2],
        ],
        'upsample_scale_factor': 10,
    },

    'keypoints_encoder_param': {
        'interverted_residual_setting': [
            # t, c, n, s
            [1, 4, 1, 1],
            [6, 8, 2, 1],
            [6, 16, 3, 2],
            [6, 21, 4, 1],
        ],
        'upsample_scale_factor': 1,
    },

    'cross_keypoints_fusion_param':{
        'trainable': True,
        'init_adjacent_matrix': False,
    },

    'temporal_keypoints_fusion_param':{
        'num_history': 9,
        'num_blocks': 4,
    },

    'handpose_encoder_param': {
        'num_head': 4,
        'dim_feedforward': 1024,
        'num_layers': 4,
    },
}



base2_setting = {
    'spatial_encoder_param' : {
        'last_channel': 128,
        'interverted_residual_setting': [
            # t, c, n, s
            [1, 16, 1, 1],
            [6, 24, 2, 2],
            [6, 32, 3, 2],
            [6, 64, 4, 1],
        ],
        'upsample_scale_factor': 10,

    },

    'keypoints_encoder_param' : {
        'interverted_residual_setting': [
            # t, c, n, s
            [1, 4, 1, 1],
            [6, 8, 2, 1],
            [6, 16, 3, 1],
            [6, 21, 4, 1],
        ],
        'upsample_scale_factor': 1,
    },

    'cross_keypoints_fusion_param' : {
        'trainable': True,
        'init_adjacent_matrix': True,
    },

    'temporal_keypoints_fusion_param' :{
        'num_history': 9,
        'num_blocks': 4,
    },

    'handpose_encoder_param' : {
        'num_head': 4,
        'dim_feedforward': 1024,
        'num_layers': 4,
    },
}

large1_setting = {
    'spatial_encoder_param' : {
        'last_channel': 128,
        'interverted_residual_setting': [
            # t, c, n, s
            [1, 16, 1, 1],
            [6, 24, 2, 1],
            [6, 32, 3, 1],
            [6, 64, 4, 1],
            [6, 96, 3, 1],
        ],
        'upsample_scale_factor': 10,
    },

    'keypoints_encoder_param' : {
        'interverted_residual_setting': [
            # t, c, n, s
            [1, 4, 1, 1],
            [6, 8, 2, 1],
            [6, 16, 3, 1],
            [6, 21, 4, 1],
            [6, 96, 3, 1],
        ],
        'upsample_scale_factor': 1,
    },

    'cross_keypoints_fusion_param' : {
        'trainable': True,
        'init_adjacent_matrix': True,
    },

    'temporal_keypoints_fusion_param' :{
        'num_history': 9,
        'num_blocks': 4,
    },

    'handpose_encoder_param' : {
        'num_head': 4,
        'dim_feedforward': 1024,
        'num_layers': 4,
    }
}

large2_setting = {
    'spatial_encoder_param' : {
        'last_channel': 128,
        'interverted_residual_setting': [
            # t, c, n, s
            [1, 16, 1, 1],
            [6, 24, 2, 2],
            [6, 32, 3, 2],
            [6, 64, 4, 1],
            [6, 96, 3, 1],
            [6, 160, 3, 2],
            [6, 320, 1, 1],
        ],
        'upsample_scale_factor': 10,
    },

    'keypoints_encoder_param' : {
        'interverted_residual_setting': [
            # t, c, n, s
            [1, 16, 1, 1],
            [6, 24, 2, 1],
            [6, 32, 3, 1],
            [6, 64, 4, 1],
            [6, 96, 3, 1],
            [6, 160, 3, 1],
            [6, 320, 1, 1],
        ],
        'upsample_scale_factor': 1,
    },

    'cross_keypoints_fusion_param' : {
        'trainable': True,
        'init_adjacent_matrix': True,
    },

    'temporal_keypoints_fusion_param' :{
        'num_history': 9,
        'num_blocks': 4,
    },

    'handpose_encoder_param' : {
        'num_head': 4,
        'dim_feedforward': 1024,
        'num_layers': 4,
    }
    
}

small_setting = {
    'spatial_encoder_param': {
        'last_channel': 128,
        'interverted_residual_setting': [
            # t, c, n, s
            [1, 16, 1, 1],
            [6, 24, 2, 2],
            [6, 32, 3, 1],
        ],
        'upsample_scale_factor': 1,
    },

    'keypoints_encoder_param' : {
        'interverted_residual_setting': [
            # t, c, n, s
            [1, 4, 1, 2],
            [6, 8, 2, 2],
            [6, 16, 3, 1],
        ],
        'upsample_scale_factor': 1,
    },

    'cross_keypoints_fusion_param' : {
        'trainable': True,
        'init_adjacent_matrix': True,
    },

    'temporal_keypoints_fusion_param' :{
        'num_history': 9,
        'num_blocks': 4,
    },

    'handpose_encoder_param' : {
        'num_head': 2,
        'dim_feedforward': 512,
        'num_layers': 3,
    }
}