Layer_name_list = ['conv',
                   'conv_3d',
                   'time_distr_conv',
                   'time_distr_conv_3d',
                   'lstm',
                   'hidden_dense',
                   'dense',
                   'Flatten',
                   'Dropout']
experiments = dict()

experiments['exp_fuzzy1'] = {'row_all': [('dense', {1})],
                             'output': [
                                 ('concatenate', {1}),
                                 ('dense', {1})
                             ],
                             }

experiments['cnn1'] = {
    # 'images': [('conv', [3, 10]),
    #            ('Flatten', []),
    #            ('dense', {0.25, 0.125}),
    #            ('dense', {0.25, 0.125}),
    #            ],
    'nwp': [('conv', [1, 3]),
            ('Flatten', []),
            ('dense', {1, 2, 3}),
            ('dense', {0.25, 0.125}),
            ],

    # 'row_obs': [('dense', {5, 0.5, 1, 2}),
    #             ('dense', {64, }),
    #             ],
    'row_calendar': [('dense', {5, 128}),
                     ('dense', {64}),
                     ],
    'output': [('concatenate', {1}),
               ('dense', {0.5, 1}),
               ('dense', {32, 64, 128})
               ],
}
experiments['cnn2'] = {

    'nwp': [('conv', [1, 3]),
            ('Flatten', []),
            ('dense', {1, 2, 3}),
            ('dense', {0.25, 0.125}),
            ],
    # 'row_obs_calendar': [
    #     ('dense', {0.25, 0.125}),
    #     ('dense', {0.25, 0.125}),
    # ],
    'output': [('concatenate', {1}),
               ('dense', {0.5, 1}),
               ('dense', {32, 64, 128})
               ],
}
experiments['cnn3'] = {
    # 'images': [('conv', [3, 10]),
    #            ('Flatten', []),
    #            ('dense', {0.25, 0.125}),
    #            ('dense', {0.25, 0.125}),
    #            ],
    'nwp': [('conv', [1, 3]),
            ('conv', [1, 2]),
            ('Flatten', []),
            ('dense', {1, 2, 3}),
            ('dense', {0.25, 0.125}),
            ],
    'row_calendar': [('dense', {5, 128}),
                     ('dense', {64}),
                     ],
    #
    # 'row_obs': [('dense', {5, 0.5, 1, 2}),
    #             ('dense', {64, }),
    #             ],

    'output': [('concatenate', {1}),
               ('dense', {0.5, 0.25}),
               ('dense', {32, 64, 128})
               ],
}
experiments['mlp1'] = {
    # 'row_nwp': [('dense', {0.25, 0.125}),
    #             ('dense', {0.25, 0.125}),
    #             ],
    'row_nwp': [('dense', {5, 7, 4, 1024}),
                ('dense', {256, 64}),
                ],
    'row_calendar': [('dense', {5, 10, 20}),
                     ('dense', {64}),
                     ],
    'output': [('concatenate', {1}),
               ('dense', {0.5, 1.5}),
               ('dense', {32, 64})
               ],
}
experiments['mlp2'] = {
    'row_all': [('dense', {4, 8}),
                ('dense', {1, 0.5}),
                ],
    # 'row_obs_nwp': [('dense', {512, 7, 4096, 1024}),
    #             ('dense', {256, 'linear'}),
    #             ('dense', {256, 512, 64}),
    #             ],
    # 'row_calendar': [('dense', {5, 100, 20}),
    #              ('dense', {64}),
    #              ],
    'output': [
        ('concatenate', {1}),
        ('dense', {32, 64, 128})
    ],
}
experiments['mlp3'] = {
    # 'row_nwp': [('dense', {0.25, 0.125}),
    #             ('dense', {0.25, 0.125}),
    #             ],
    'row_all': [('dense', {512, 7, 4096, 1024}),
                ('dense', {256, 'linear'}),
                ('dense', {256, 512, 64}),
                ],
    # 'row_calendar': [('dense', {5, 10, 20}),
    #              ('dense', {64}),
    #              ],
    'output': [('concatenate', {1}),
               ('dense', {32, 64, 128})
               ],
}

experiments['distributed_cnn1'] = {
    'nwp': [('conv', [3, 10]),
            ('Flatten', []),
            ('dense', {0.25, 0.125}),
            ('dense', {0.25, 0.125}),
            ],
    # 'row_obs': [('dense', {5, 0.5, 1, 2}),
    #             ('dense', {64, }),
    #             ],
    'row_calendar': [('dense', {5, 10, 20}),
                     ('dense', {64}),
                     ],
    'output': [('concatenate', {1}),
               ('dense', {0.5, 0.25}),
               ('dense', {0.5, 'linear'}),
               ('dense', {32, 64, 128})
               ],
}

experiments['distributed_mlp1'] = {
    'row_nwp': [('dense', {4, 8}),
                ('dense', {0.25, 0.5}),
                ],
    # 'row_obs': [('dense', {5, 0.5, 1, 2}),
    #             ('dense', {64, }),
    #             ],
    'row_calendar': [('dense', {5, 10, 20}),
                     ('dense', {64}),
                     ],
    'output': [('concatenate', {1}),
               ('dense', {0.5, 0.25}),
               ('dense', {0.5, 'linear'}),
               ('dense', {32, 64, 128})
               ],
}
experiments['distributed_mlp2'] = {
    'row_nwp': [('dense', {4, 8}),
                ('dense', {0.25, 0.5}),
                ],
    # 'row_obs': [('dense', {5, 0.5, 1, 2}),
    #             ('dense', {64, }),
    #             ],
    'row_calendar': [('dense', {5, 10, 20}),
                     ('dense', {64}),
                     ],
    'output': [('concatenate', {1}),
               ('dense', {0.5, 0.25}),
               ('dense', {0.5, 'linear'}),
               ('dense', {32, 64, 128})
               ],
}
experiments['mlp_for_combine_data'] = {
    'row_all': [('dense', {4, 8}),
                ('dense', {0.5, 0.75}),
                ],
    'output': [('concatenate', {1}),
               ('dense', {0.5, 0.75}),
               ('dense', {32, 64, 128})
               ],
}
experiments['output_combine'] = {
    'output': [('concatenate', {1}),
               ('dense', {32})
               ]
}
mlp_for_combine_simple = {
    'row_all': [('dense', {4, 8}),
                ],
    'output': [('concatenate', {1}),
               ('dense', {0.5, 0.75}),
               ('dense', {32, 64, 128})
               ],
}
experiments['distributed_mlp3'] = {
    'row_all': [('dense', {0.25, 0.125}),
                ],
    'output': [('concatenate', {1}),
               ('dense', {0.5, 0.25}),
               ('dense', {0.5, 'linear'}),
               ('dense', {32, 64, 128})
               ],
}

experiments['lstm1'] = {
    'lstm': [('lstm', {1, 2}),
             ('lstm', {2, 1}),
             ('Flatten', []),
             ('dense', {1, 0.5, 2})],
    'output': [('concatenate', {1}),
               ('dense', {1, 0.5}),
               ('dense', {12, 64, 128})]
}
experiments['lstm2'] = {
    'lstm': [('lstm', {1, 2}),
             ('Flatten', []),
             ('dense', {1, 3, 2}),
             ('Reshape', []),  #: [32, 32]}
             ('lstm', {1, 2}),
             ('Flatten', []),
             ('dense', {1, 0.5, 2})
             ],
    'output': [('concatenate', {1}),
               ('dense', {12, 64, 128})
               ]
}
experiments['lstm3'] = {
    'lstm': [('lstm', {1, 2}),
             ('Flatten', []),
             ('dense', {1, 3, 2}),
             ('Reshape', []),
             ('lstm', {1, 2}),
             ('Flatten', []),
             ('dense', {1, 3, 2}),
             ('Reshape', []),
             ('lstm', {1}),
             ('Flatten', []),
             ('dense', {1, 2, 0.5})
             ],
    'output': [('concatenate', {1}),
               ('dense', {1, 0.5, 0.25}),
               ('dense', {12, 64, 128}),
               ]
}
experiments['lstm4'] = {
    'lstm': [('lstm', {0.5, 1, 2}),
             ('Flatten', []),
             ('dense', {1, 0.5}),
             ('dense', {1, 0.5, 0.25})],
    'output': [('concatenate', {1}),
               ('dense', {12, 64, 128})
               ]
}
experiments['transformer'] = {
    'lstm': [('transformer', 256),
             ('Flatten', [])],
    # ('dense', {1, 0.5}),
    # ('dense', {1, 0.5, 0.25})],
    'output': [('concatenate', {1}),
               ('dense', 256),
               ('dense', 64)
               ]
}
experiments['timm_net'] = {
    'images': [('timm_net', 1),
               ('Flatten', []),
               ('dense', 1024),
               ('dense', 256)
               ],
    'row_calendar': [('dense', 64),
                     ],
    # 'row_obs_nwp': [('dense', 4),
    #              ('dense', 720),
    #              ('dense', 128)
    #              ],
    'output': [('concatenate', {1}),
               ('dense', 64),
               ]
}
experiments['CrossViVit_net'] = {
    'images': [('vit_net', 1),
               ('Flatten', []),
               ('dense', 1024),
               ('dense', 128)
               ],

    'lstm': [('transformer', 128),
             ('Flatten', []),
             ('dense', 720),
             ('dense', 128)
             ],
    'output': [('cross_attention', 1),
               ('dense', 64),
               ]
}
experiments['Time_CrossViVit_net'] = {
    'images': [('time_distr_vit_net', 1),
               ('Flatten', []),
               ('dense', 1024),
               ('dense', 128)
               ],

    'output': [
        ('concatenate', {1}),
        ('dense', 64),
    ]
}
experiments['trans_net'] = {
    'lstm': [('transformer', 128),
             ('Flatten', []),
             ('dense', 720),
             ('dense', 128)
             ],
    'output': [
        ('concatenate', {1}),
        ('dense', 64),
    ]
}

experiments['yolo'] = {
    'images': [('yolo', 1),
               ('Flatten', []),
               ('dense', 1024),
               ('dense', 256)
               ],
    'row_calendar': [('dense', 64),
                     ],
    # 'row_obs_nwp': [('dense', 4),
    #              ('dense', 720),
    #              ('dense', 128)
    #              ],
    'output': [('concatenate', {1}), ('dense', 64),
               ]
}
experiments['unet'] = {
    'images': [('unet', 1),
               ('Flatten', []),
               ('dense', 1024),
               ('dense', 64)
               ],
    'row_calendar': [('dense', 64),
                     ],
    # 'row_obs_nwp': [('dense', 4),
    #              ('dense', 720),
    #              ('dense', 128)
    #              ],
    'output': [('concatenate', {1}), ('dense', 256),
               ]
}

experiments['distributed_lstm1'] = {
    'lstm': [('lstm', 1),
             ('Flatten', []),
             ('dense', 0.25),
             ('dense', 0.5),
             ],
    'output': [('concatenate', {1}), ('dense', 0.25),
               ('dense', 0.5),
               ('dense', 32)
               ],
}
