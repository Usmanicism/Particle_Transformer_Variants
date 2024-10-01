import torch
from weaver.nn.model.SwinParticleTransformer import SwinParticleTransformer
from weaver.utils.logger import _logger

'''
Link to the full model implementation:
https://github.com/hqucms/weaver-core/blob/main/weaver/nn/model/ParticleMultiAxisTransformer.py
'''


class SwinParticleTransformerWrapper(torch.nn.Module):
    def __init__(self, **kwargs) -> None:
        super().__init__()
        self.mod = SwinParticleTransformer(**kwargs)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'mod.cls_token', }

    def forward(self, points, features, lorentz_vectors, mask):
        return self.mod(features, v=lorentz_vectors, mask=mask)


def get_model(data_config, **kwargs):

    cfg = dict(
        input_dim=len(data_config.input_dicts['pf_features']),
        num_classes=len(data_config.label_value),
        # network configurations
        pair_input_dim=4,
        use_pre_activation_pair=False,
        embed_dims=[128, 512, 128],
        pair_embed_dims=[64, 64, 64],
        num_heads=8,
        num_layers=8,
        num_cls_layers=2,
        block_params=None,
        cls_block_params={'dropout': 0, 'attn_dropout': 0, 'activation_dropout': 0},
        fc_params=[],
        activation='gelu',
        # misc
        trim=True,
        for_inference=False,
    )
    cfg.update(**kwargs)
    _logger.info('Model config: %s' % str(cfg))

    model = SwinParticleTransformerWrapper(**cfg)

    model_info = {
        'input_names': list(data_config.input_names),
        'input_shapes': {k: ((1,) + s[1:]) for k, s in data_config.input_shapes.items()},
        'output_names': ['softmax'],
        'dynamic_axes': {**{k: {0: 'N', 2: 'n_' + k.split('_')[0]} for k in data_config.input_names}, **{'softmax': {0: 'N'}}},
    }

    return model, model_info


def get_model_debug():

    cfg = dict(
        input_dim=128,
        num_classes=10,
        # network configurations
        pair_input_dim=4,
        use_pre_activation_pair=False,
        embed_dims=[128, 512, 128],
        pair_embed_dims=[64, 64, 64],
        num_heads=8,
        num_layers=8,
        num_cls_layers=2,
        block_params=None,
        cls_block_params={'dropout': 0, 'attn_dropout': 0, 'activation_dropout': 0},
        fc_params=[],
        activation='gelu',
        # misc
        trim=True,
        for_inference=False,
    )

    model = SwinParticleTransformerWrapper(**cfg)
    print('[model]')
    print(model)


def get_loss(data_config, **kwargs):
    return torch.nn.CrossEntropyLoss()


if __name__ == '__main__':
    get_model_debug()

"""
[2024-03-30 05:27:25,119] INFO: Model config: {'input_dim': 17, 'num_classes': 10, 'pair_input_dim': 4, 'use_pre_activation_pair': False, 'embed_dims': [128, 512, 128], 'pair_embed_dims': [64, 64, 64], 'num_heads': 8, 'num_layers': 8, 'num_cls_layers': 2, 'block_params': None, 'cls_block_params': {'dropout': 0, 'attn_dropout': 0, 'activation_dropout': 0}, 'fc_params': [], 'activation': 'gelu', 'trim': True, 'for_inference': False, 'use_amp': True}
[2024-03-30 05:27:25,119] INFO: cfg_block: {'embed_dim': 128, 'num_heads': 8, 'ffn_ratio': 4, 'dropout': 0.1, 'attn_dropout': 0.1, 'activation_dropout': 0.1, 'add_bias_kv': False, 'activation': 'gelu', 'scale_fc': True, 'scale_attn': True, 'scale_heads': True, 'scale_resids': True}
[2024-03-30 05:27:25,119] INFO: cfg_cls_block: {'embed_dim': 128, 'num_heads': 8, 'ffn_ratio': 4, 'dropout': 0, 'attn_dropout': 0, 'activation_dropout': 0, 'add_bias_kv': False, 'activation': 'gelu', 'scale_fc': True, 'scale_attn': True, 'scale_heads': True, 'scale_resids': True}
"""
