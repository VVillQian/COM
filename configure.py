def get_default_config(data_name):
    if data_name in ['animal']:
        """The default configs."""
        return dict(
            name        = 'animal',
            dims        = [4096, 4096],
            view_num    = 2,
            hidden_size = 364,
            class_num   = 50,
            path        = './data/animal.mat',
            splitrate   = 0.8,
            norm        = 1,
            training=dict(
                missing_rate=0.5,
                epoch=300,
                batch_size=4096,
                lr=1.0e-4,#COM
                lambda1=0.0001,
                lambda2=0.0001,
                lambda3=0.0001,
            ),
        )
    elif data_name in ['cub']:
        """The default configs."""
        return dict(
            name        = 'cub',
            dims        = [1024, 300],
            view_num    = 2,
            hidden_size = 128,
            class_num   = 10,
            path        = './data/cub_googlenet_doc2vec_c10.mat',
            splitrate   = 0.8,
            norm        = 1,
            training=dict(
                missing_rate=0.5,
                epoch=300,
                batch_size=128,
                lr=1e-4,
                lambda1=0.0001,
                lambda2=0.0001,
                lambda3=0.0001,
            ),
        )
    elif data_name in ['ORL']:
        """The default configs."""
        return dict(
            name        = 'orl',
            dims        = [4096, 3304, 6750],
            view_num    = 3,
            hidden_size = 256,
            class_num   = 40,
            path        = './data/ORL_mtv.mat',
            splitrate   = 0.8,
            norm        = 1,
            training=dict(
                missing_rate=0.5,
                epoch=300,
                batch_size=64,
                lr=1.0e-3,
                lambda1=0.01,
                lambda2=0.01,
                lambda3=0.01,
            ),
        )
    elif data_name in ['PIE']:
        """The default configs."""
        return dict(
            name        = 'pie',
            dims        = [484, 256, 279],
            view_num    = 3,
            hidden_size = 256,
            class_num   = 68,
            path        = './data/PIE_face_10.mat',
            splitrate   = 0.8,
            norm        = 1,
            training=dict(
                missing_rate=0.5,
                epoch=300,
                batch_size=64,
                lr=1.0e-4,
                lambda1=0.1,
                lambda2=0.1,
                lambda3=0.1,
            ),
        )
    elif data_name in ['yaleB']:
        """The default configs."""
        return dict(
            name        = 'yale',
            dims        = [2500,3304,6750],
            view_num    = 3,
            hidden_size = 128,
            class_num   = 10,
            path        = './data/yaleB_mtv.mat',
            splitrate   = 0.8,
            norm        = 1,
            training=dict(
                missing_rate=0.5,
                epoch=300,
                batch_size=64,
                lr=1e-5,
                lambda1=0.1,#0.01
                lambda2=0.0001,
                lambda3=0.01,#0.0001
            ),
        )

    elif data_name in ['KS']:
        """The default configs."""
        return dict(
            name        = 'Kinetics-Sounds',
            dims        = [1024, 768],
            view_num    = 2,
            hidden_size = 600,
            class_num   = 31,
            path        = 'Kinetics-Sounds',
            splitrate   = 0.8,
            norm        = 1,
            training=dict(
                missing_rate=0.5,
                epoch=500,
                batch_size=4096,
                lr=1.0e-4,
                lambda1=0.0001,
                lambda2=0.0001,
                lambda3=0.0001,
            ),
        )

    else:
        raise Exception('Undefined data_name')
