import os
import numpy as np
import torch
import torch.nn as nn
from fastcore.xtras import image_size
from openpyxl.styles.builtins import output
from transformers import AutoImageProcessor, AutoModel, AutoConfig
from eforecast.deep_models.pytorch_2x.ts_transformers.ts_transformer_models import get_transfromer_model
from torchvision.transforms import Pad
import segmentation_models_pytorch as smp
import torchvision.models as torchmodels
import torchvision.transforms as transforms
# from ultralytics import YOLO
# from eforecast.deep_models.pytorch_2x.image.yolo.model import DetectionModel
import timm
from segmentation_models_pytorch import create_model
from einops import rearrange
from einops import repeat
from einops import einsum


act_funcs = nn.ModuleDict({'elu': nn.ELU(),
                           'sigmoid': nn.Sigmoid(),
                           'relu': nn.ReLU(),
                           'gelu': nn.GELU(),
                           'tanh': nn.Tanh(),
                           'lrelu': nn.LeakyReLU(),
                           'prelu': nn.PReLU()
                           })


class conv_2d(nn.Module):
    def __init__(self, shape, params, size, name_scope, layer_id, train=True):
        super(conv_2d, self).__init__()
        if len(shape) == 3:
            shape.insert(1, 1)
        self.shape = shape
        self.size = size
        x_shape1 = shape
        x_shape1 = np.array(x_shape1[-3:-1])
        x_shape = x_shape1 // size
        x_shape[x_shape <= 1] = 2
        x_shape = np.minimum(x_shape, x_shape1)
        kernels = x_shape.tolist()
        self.act_func = act_funcs[params['act_func']]
        x_shape = x_shape // 2
        x_shape[x_shape == 0] = 1
        pool_size = x_shape.tolist()
        pool_size = [int(p) for p in pool_size]
        kernels = [int(k) for k in kernels]
        self.name = f'{name_scope}_conv_2d_{layer_id}'
        self.conv = nn.Conv2d(shape[1], int(params['filters']),
                              kernel_size=tuple(kernels[-2:]),
                              padding="valid")

        self.pool = nn.AvgPool2d(tuple(pool_size[-2:]), stride=1)

    def string(self):
        print(f"layer {self.name} has input shape {self.shape}")
        print(f"layer weights {self.name} has shape {list(self.conv.weight.size())}")

    def forward(self, x):
        x_shape1 = list(x.size())
        if len(x_shape1) == 3:
            x = torch.unsqueeze(x, dim=1)
        return self.pool(self.act_func(self.conv(x)))


class conv_3d(nn.Module):
    def __init__(self, shape, params, size, name_scope, layer_id, train=True):
        super(conv_3d, self).__init__()
        if len(shape) == 3:
            shape.insert(1, 1)
        if len(shape) == 4:
            shape.insert(1, 1)
        self.shape = shape
        self.size = size
        x_shape1 = shape
        x_shape1 = np.array(x_shape1[-3:-1])
        x_shape = x_shape1 // size
        x_shape[x_shape <= 2] = 3
        x_shape = np.minimum(x_shape, x_shape1)
        kernels = x_shape.tolist()
        if len(kernels) != 3:
            kernels = kernels + [1]
        self.act_func = act_funcs[params['act_func']]
        x_shape = x_shape // 2
        x_shape[x_shape == 0] = 1
        pool_size = x_shape.tolist()
        if len(pool_size) != 3:
            pool_size = pool_size + [1]
        pool_size = [int(p) for p in pool_size]
        kernels = [int(k) for k in kernels]
        self.name = f'{name_scope}_conv_3d_{layer_id}'
        self.conv = nn.Conv3d(shape[1], int((int(layer_id) + 1) * params['filters']),
                              kernel_size=tuple(kernels[-3:]),
                              stride=2,
                              padding="valid")

        self.pool = nn.AvgPool3d(tuple(pool_size[-3:]), stride=1)

    def string(self):
        print(f"layer {self.name} has input shape {self.shape}")
        print(f"layer weights {self.name} has shape {list(self.conv.weight.size())}")

    def forward(self, x):
        if len(x.size()) == 3:
            x = torch.unsqueeze(x, dim=1)
        if len(x.size()) == 4:
            x = torch.unsqueeze(x, dim=1)
        return self.pool(self.act_func(self.conv(x)))


class time_distr_conv_2d(nn.Module):
    def __init__(self, shape, params, size, name_scope, layer_id, train=True):
        super(time_distr_conv_2d, self).__init__()
        if len(shape) == 3:
            shape.insert(1, 1)
        if len(shape) == 4:
            shape.insert(1, 1)
        self.shape = shape
        self.size = size
        self.act_func = act_funcs[params['act_func']]
        self.name = f'{name_scope}_time_distr_{layer_id}'
        shape_conv = [s for ii, s in enumerate(shape) if ii != 1]
        self.conv_list = nn.ModuleList()
        for i in range(shape[1]):
            self.conv_list.append(conv_2d(shape_conv, params, size, self.name_conv, i))

    def string(self):
        print(f"layer {self.name} has input shape {self.shape}")
        print(
            f"layer weights {self.name} has {self.shape[1]} conv of shape{list(self.conv_list[0].weight.size())}")

    def forward(self, x):
        if list(x.size()) == 3:
            x = torch.unsqueeze(x, dim=1)
        if list(x.size()) == 4:
            x = torch.unsqueeze(x, dim=1)
        batch_size, time_steps, C, H, W = x.size()
        output = torch.tensor([])
        for i in range(time_steps):
            output_t = self.conv_list[i](x[:, i, ...])
            output_t = output_t.unsqueeze(1)
            output = torch.cat((output, output_t), 1)
        return output


class time_distr_conv_3d(nn.Module):
    def __init__(self, shape, params, size, name_scope, layer_id, train=True):
        super(time_distr_conv_3d, self).__init__()
        if len(shape) == 3:
            shape.insert(1, 1)
        if len(shape) == 4:
            shape.insert(1, 1)
        self.shape = shape
        self.size = size
        self.act_func = act_funcs[params['act_func']]
        self.name = f'{name_scope}_time_distr_{layer_id}'
        shape_conv = [s for ii, s in enumerate(shape) if ii != 1]
        self.conv_list = nn.ModuleList()
        for i in range(shape[1]):
            self.conv_list.append(conv_3d(shape_conv, params, size, self.name_conv, i))

    def string(self):
        print(f"layer {self.name} has input shape {self.shape}")
        print(
            f"layer weights {self.name} has {self.shape[1]} conv of shape{list(self.conv_list[0].weight.size())}")

    def forward(self, x):
        if list(x.size()) == 3:
            x = torch.unsqueeze(x, dim=1)
        if list(x.size()) == 4:
            x = torch.unsqueeze(x, dim=1)
        time_steps = x.size()[1]
        output = torch.tensor([])
        for i in range(time_steps):
            output_t = self.conv_list[i](x[:, i, ...])
            output_t = output_t.unsqueeze(1)
            output = torch.cat((output, output_t), 1)
        return output


class lstm(nn.Module):
    def __init__(self, shape, params, size, name_scope, layer_id, train=True):
        super(lstm, self).__init__()
        if len(shape) != 3:
            shape = [shape[0], shape[1], np.prod(shape[2:])]
        self.shape = shape
        self.size = size
        self.act_func = act_funcs[params['act_func']]
        self.name = f'{name_scope}_lstm_{layer_id}'
        self.lstm = nn.LSTM(shape[2], int(size * shape[2]), batch_first=True)

    def string(self):
        print(f"layer {self.name} has input shape {self.shape}")
        print(f"layer weights {self.name} has shape {list(self.lstm.weight.size())}")

    def forward(self, x):
        if list(x.size()) != 3:
            shape = x.size()
            x = torch.reshape(x, (shape[0], shape[1], torch.prod(shape[2:])))
        return self.act_func(self.lstm(x))



class RnnEmbedding(nn.Module):
    def __init__(self, shape, embed_size, layer_id):
        super().__init__()
        self.embed_size = embed_size
        self.name_embed = f'RnnEmbedding{layer_id}'
        self.rnn = nn.LSTM(shape[-1], embed_size, bias= False, batch_first=True)
        output, (hn, cn) = self.rnn(torch.rand(shape))
        self.output_shape = output[:, -1, :].unsqueeze(-1).shape


    def forward(self, x):
        output, (hn, cn) = self.rnn(x)
        return output[:, -1, :].unsqueeze(-1)

class transformer(nn.Module):
    def __init__(self, shape, params, size, name_scope, layer_id, train=True):
        super(transformer, self).__init__()
        B, Tpred, Ctime = shape['future_calendar']
        _, _, Cnwp = shape['future_nwp']
        _, Tpast, _ = shape['past_calendar']
        self.obs_embed = RnnEmbedding(shape['past_obs'], Tpred, layer_id)
        _, _, Cobs = shape['past_obs']
        self.shape = shape
        self.act_func = act_funcs[params['act_func']]
        self.name = f'{name_scope}_transformer_{layer_id}'
        self.transformer_name = params['transformer_name']
        enc_in = Cnwp + Cobs
        dec_in = Cnwp + Cobs
        self.size = size #if self.transformer_name not in {'Autoformer'} else enc_in
        self.transfromer_model = get_transfromer_model(self.transformer_name, Tpast, Tpred, enc_in,
                                                       dec_in, int(size), params)
        future_calendar = torch.cat([torch.rand(shape['past_calendar']),
                                     torch.rand(shape['future_calendar'])], 1)
        future_nwp = torch.cat([torch.rand(shape['past_nwp']),
                                torch.rand(shape['future_nwp'])], 1)

        past_calendar = torch.rand(shape['past_calendar'])
        past_nwp = torch.rand(shape['past_nwp'])
        past_obs = torch.rand(shape['past_obs'])
        future_obs = torch.cat([past_obs, self.obs_embed(past_obs)], 1)
        x_enc = torch.cat([past_obs, past_nwp], -1)
        x_mark_enc = past_calendar
        x_dec = torch.cat([future_obs, future_nwp], -1)
        x_mark_dec = future_calendar
        self.output_shape = self.transfromer_model(x_enc, x_mark_enc, x_dec, x_mark_dec).shape

    def string(self):
        print(f"layer {self.name} has input shape {self.shape}")
        print(f"layer weights {self.name} has shape {list(self.lstm.weight.size())}")

    def forward(self, x):
        future_calendar = torch.cat([x['past_calendar'], x['future_calendar']], 1)
        future_nwp = torch.cat([x['past_nwp'], x['future_nwp']], 1)
        past_calendar = x['past_calendar']
        past_nwp = x['past_nwp']
        past_obs = x['past_obs']
        future_obs = torch.cat([past_obs, self.obs_embed(past_obs)], 1)
        x_enc = torch.cat([past_obs, past_nwp], -1)
        x_mark_enc = past_calendar
        x_dec = torch.cat([future_obs, future_nwp], -1)
        x_mark_dec = future_calendar

        return self.act_func(self.transfromer_model(x_enc, x_mark_enc, x_dec, x_mark_dec))


class CrossAttention(nn.Module):
    def __init__(
        self, shape, params, size, name_scope, layer_id, train=True,
        heads=8,
        dim_head=64,
        dropout=0.0,
    ):
        super().__init__()
        if not isinstance(shape, dict):
            raise ValueError('shape must be dict of names of branches that should be cross-attention')

        if len(shape) != 2:
            raise ValueError('shape must include two branches that should be cross-attention')

        inner_dim = dim_head * heads
        self.heads = heads
        self.scale = dim_head**-0.5

        self.attend = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)
        names_branch = [k for k in shape.keys()]
        self.to_q = nn.Linear(shape[names_branch[0]][1], inner_dim, bias=False)

        self.to_kv = nn.Linear(shape[names_branch[1]][1], inner_dim * 2, bias=False)

        self.to_out = nn.Sequential(nn.Linear(inner_dim, shape[names_branch[0]][1]), nn.Dropout(dropout))
        self.output_shape = self.forward([torch.rand(shape[names_branch[0]]), torch.rand(shape[names_branch[1]])]).shape

    def forward(self, x):
        src, tgt = x
        q = self.to_q(tgt)

        qkv = (q, *self.to_kv(src).chunk(2, dim=-1))

        q, k, v = map(
            lambda t: rearrange(t, "b (h d) -> (b h) d", h=self.heads), qkv
        )

        dots = einsum(q, k, "b i, b i -> b i") * self.scale

        attn = self.attend(dots)
        attn = self.dropout(attn)

        out = einsum(attn, v, "b j, b j -> b j")
        out = rearrange(out, "(b h) d -> b (h d)", h=self.heads)
        return self.to_out(out)


# def transformer(input_name, shape, params, size, name_scope, layer_id, train=False):
#     x = dict()
#     x['observations'] = tf.keras.Input(shape=shape['observations'], dtype=tf.float32, name='observations')
#     x['future'] = tf.keras.Input(shape=shape['future'], dtype=tf.float32, name='future')
#     x['calendar'] = tf.keras.Input(shape=shape['calendar'], dtype=tf.float32, name='calendar')
#     name = f'{name_scope}_{params["transformer_type"]}_{layer_id}'
#     transformer = AutoModel(params["transformer_type"], shape, params, name=name)(x)
#     transformer_model = tf.keras.Model(x, transformer, name='model_' + name)
#     print(f"layer {name} has shape {transformer.get_shape().as_list()}")
#     print(f"layer weights {name} has shape {transformer_model.trainable_weights[0].get_shape().as_list()}")
#
#     return transformer_model
#
#

# # class yolo(nn.Module):
# #     def __init__(self, shape, params, size, name_scope, layer_id, train=True):
# #         super(yolo, self).__init__()
# #
# #         self.shape = shape
# #         self.size = size
# #         self.act_func = act_funcs[params['act_func']]
# #
# #         self.resize = transforms.Resize((224, 224), antialias=False)
# #         if params['pretrained'] in {'tune', 'finetune'}:
# #             is_train = True
# #         else:
# #             is_train = False
# #
# #         model_trained = YOLO(model=os.path.basename(params['model_yaml']).replace('yolov8', 'yolov8m').
# #                              replace('.yaml', '.pt'))
# #         self.yolo_model = DetectionModel(cfg=params['model_yaml'], ch=self.shape[-1])
# #         state_dict = dict()
# #         for k in self.yolo_model.state_dict().keys():
# #             for k2 in model_trained.state_dict().keys():
# #                 if k in k2:
# #                     state_dict[k] = model_trained.state_dict()[k2]
# #                     break
# #         if is_train:
# #             try:
# #                 self.yolo_model.load_state_dict(state_dict)
# #             except:
# #                 pass
# #         if params['pretrained'] == 'finetune':
# #             self.yolo_model.requires_grad = False
# #         self.flat = nn.Flatten()
#
#
#     def string(self):
#         print(f"layer {self.name} has input shape {self.shape}")
#
#     def forward(self, x):
#         x = rearrange(x, 'b h w c -> b c h w')
#         if self.resize is not None:
#             x = self.resize(x)
#         return self.act_func(self.flat(self.yolo_model(x)))


class unet(nn.Module):
    def __init__(self, shape, params, size, name_scope, layer_id, train=True):
        super(unet, self).__init__()

        self.shape = shape
        self.size = size
        self.act_func = act_funcs[params['act_func']]
        self.name = f'{name_scope}_{params["unet_type"]}_{layer_id}'
        arch = eval(f'smp.{params["unet_type"]}')

        self.resize = transforms.Resize((128, 128), antialias=False)
        if params['pretrained'] in {'tune', 'finetune'}:
            is_train = True
        else:
            is_train = False
        self.unet_model = arch(encoder_name=params["encoder_name"],
                               in_channels=shape[-1],
                               encoder_depth = 3,
                               decoder_channels= (64, 32, 16),
                               encoder_weights='imagenet' if is_train else None,
                               classes=1)
        if params['pretrained'] == 'finetune':
            self.unet_model.requires_grad = False
        self.flat = nn.Flatten()


    def string(self):
        print(f"layer {self.name} has input shape {self.shape}")

    def forward(self, x):
        x = rearrange(x, 'b h w c -> b c h w')
        if self.resize is not None:
            x = self.resize(x)
        return self.act_func(self.flat(self.unet_model(x)))

class timm_net(nn.Module):
    def __init__(self, shape, params, size, name_scope, layer_id, train=True):
        super(timm_net, self).__init__()

        self.shape = shape
        self.size = size
        self.act_func = act_funcs[params['act_func']]
        self.name = f'{name_scope}_{params["model_name"]}_{layer_id}'
        try:
            s = int(params["model_name"].split('_')[-1])
        except:
            s=  224
        if s > 600:
            raise ValueError('Too Large input size')
        self.resize = transforms.Resize((s, s), antialias=False)
        pre_trained = [m.split('.')[0] for m in timm.list_models(pretrained=True)]
        if params['pretrained'] in {'tune', 'finetune'}:
            is_train = params["model_name"] in pre_trained
        else:
            is_train = False
        if params["use_classes"]:
            self.timm_model = timm.create_model(params["model_name"], pretrained=is_train,
                                                num_classes=3192, in_chans=self.shape[-1])
        else:
            try:
                self.timm_model = timm.create_model(params["model_name"], pretrained=is_train,
                                                    num_classes=0, global_pool='', in_chans=self.shape[-1])
            except:
                self.timm_model = timm.create_model(params["model_name"], pretrained=is_train,
                                                    num_classes=3192, in_chans=self.shape[-1])
        if params['pretrained'] == 'finetune':
            self.timm_model.requires_grad = False
        self.flat = nn.Flatten()

    def string(self):
        print(f"layer {self.name} has input shape {self.shape}")

    def forward(self, x):
        x = rearrange(x, 'b h w c -> b c h w')
        x = self.resize(x)
        return self.act_func(self.flat(self.timm_model(x)))

class vit_net(nn.Module):
    def __init__(self, shape, params, size, name_scope, layer_id, train=True):
        super(vit_net, self).__init__()

        self.shape = shape['images']
        if len(self.shape) == 5:
            B, T, C, H, W = self.shape
            C= T * C
        else:
            B, C, H, W = self.shape
        x_sample = torch.rand([B, C, H, W])
        self.size = size
        self.act_func = act_funcs[params['act_func']]
        self.name = f'{name_scope}_{params["vit_name"]}_{layer_id}'
        self.vit_name = params["vit_name"]
        try:
            s = int(params["vit_name"].split('_')[-1])
        except:
            s =  H
        if s > 600:
            raise ValueError('Too Large input size')
        self.img_size = s
        if self.vit_name in params['timm_models'] or 'timm' in self.vit_name:
            try:
                self.vit_model = timm.create_model(self.vit_name, pretrained=True,
                                                    num_classes=0, global_pool='', in_chans=C)
            except:
                try:
                    self.vit_model = timm.create_model(self.vit_name, pretrained=True,
                                                        num_classes=self.img_size * C, in_chans=C)
                except:
                    self.vit_model = timm.create_model(self.vit_name, pretrained=False,
                                                       num_classes=self.img_size * C, in_chans=C)
            data_config = timm.data.resolve_model_data_config(self.vit_model)
            self.resize = transforms.Resize(data_config['input_size'][1:], antialias=False)

            img = self.resize(x_sample)
            outputs = self.vit_model(img)
            if len(outputs.shape) == 2:
                outputs = rearrange(outputs, 'b (c h) -> b c h',c=self.img_size)
            output_shape = outputs.shape
        else:
            try:
                self.processor = AutoImageProcessor.from_pretrained(self.vit_name)
                self.vit_model = AutoModel.from_pretrained(self.vit_name)
                inputs = self.processor(images=x_sample, return_tensors="pt", input_data_format='channels_first',
                                        do_rescale=False)
                outputs = self.vit_model(**inputs)
            except:
                self.config = AutoConfig.from_pretrained(self.vit_name)
                self.config.num_channels = C
                if hasattr(self.config, 'image_size'):
                    self.config.image_size = H if isinstance(self.config.image_size, int) else [H, W]
                self.vit_model = AutoModel.from_config(self.config)
                outputs = self.vit_model(x_sample)
            output_shape = outputs.last_hidden_state.shape
            outputs = outputs.last_hidden_state
        self.flat = nn.Flatten()
        self.pool = nn.AdaptiveAvgPool2d((1, output_shape[-1]))
        self.output_shape = self.flat(self.pool(outputs)).shape
        if params['pretrained'] == 'finetune':
            self.vit_model.requires_grad = False


    def string(self):
        print(f"layer {self.name} has input shape {self.shape}")
        print(f"layer {self.name} has output_shape {self.output_shape}")

    def forward(self, x):
        if isinstance(x, dict):
            x = x['images']
        if len(x.size()) == 5:
            x = rearrange(x, 'b t c h w -> b (t c) h w')
        if hasattr(self, 'resize'):
            x = self.resize(x)
            output = self.vit_model(x)
            if len(output.shape) == 2:
                output = rearrange(output, 'b (c h) -> b c h',c=self.img_size)
            output = self.act_func(self.flat(self.pool(output)))
        elif hasattr(self, 'config'):
            output = self.act_func(self.flat(self.pool(self.vit_model(x).last_hidden_state)))
        elif hasattr(self, 'processor') and not hasattr(self, 'config'):
            x = self.processor(images=x, return_tensors="pt")
            output = self.act_func(self.flat(self.pool(self.vit_model(**x).last_hidden_state)))

        return output

class time_distr_vit_net(nn.Module):
    def __init__(self, shape, params, size, name_scope, layer_id, train=True):
        super(time_distr_vit_net, self).__init__()
        self.shape = shape['images']
        B, T, C, H, W = self.shape
        self.size = size
        self.act_func = act_funcs['sigmoid']
        self.name = f'{name_scope}_time_distr_vit_net_{layer_id}'
        self.vit_list = nn.ModuleList()
        for i in range(T):
            self.vit_list.append(vit_net({'images': [B, C, H, W]}, params, size, self.name, i))
        self.lstm = nn.LSTM(int(self.vit_list[0].output_shape[-1]), int(self.vit_list[0].output_shape[-1]),
                            batch_first=True)
        self.flat = nn.Flatten()
        x = torch.rand([B, T, C, H, W])
        time_steps = x.size()[1]
        output = torch.tensor([])
        for i in range(time_steps):
            output_t = self.vit_list[i](x[:, i, ...])
            output_t = output_t.unsqueeze(1)
            output = torch.cat((output, output_t), 1)
        hidden_state = (torch.rand(1, B, output.shape[-1]), torch.rand(1, B, output.shape[-1]))
        lstm_out = self.lstm(output, hidden_state)[0]
        self.output_shape = self.act_func(self.flat(lstm_out)).shape

    def string(self):
        print(f"layer {self.name} has input shape {self.shape}")
        print(
            f"layer weights {self.name} has {self.shape[1]} conv of shape{list(self.conv_list[0].weight.size())}")
        print(f"layer {self.name} has output_shape {self.output_shape}")

    def forward(self, x):
        x = x['images']
        time_steps = x.size()[1]
        B = x.size()[0]
        output = torch.tensor([]).to(x.device)
        for i in range(time_steps):
            output_t = self.vit_list[i](x[:, i, ...])
            output_t = output_t.unsqueeze(1)
            output = torch.cat((output, output_t), 1)
        hidden_state = (torch.rand(1, B, output.shape[-1]).to(x.device),
                        torch.rand(1, B, output.shape[-1]).to(x.device))
        lstm_out = self.lstm(output, hidden_state)[0]
        return self.act_func(self.flat(lstm_out))

class dense(nn.Module):
    def __init__(self, shape, params, size, name_scope, layer_id, train=True):
        super(dense, self).__init__()
        self.shape = shape
        self.size = size
        if params['act_func'] is not None:
            self.act_func = act_funcs[params['act_func']]
        else:
            self.act_func = None
        self.name = f'{name_scope}_dense_{layer_id}'
        self.dense = nn.Linear(shape[1], int(size))
        self.output_shape = self.dense(torch.rand(self.shape)).shape

    def string(self):
        print(f"layer {self.name} has input shape {self.shape}")
        print(f"layer weights {self.name} has shape {list(self.dense.weight.size())}")

    def forward(self, x):
        if self.act_func is not None:
            return self.act_func(self.dense(x))
        else:
            return self.dense(x)


class Reshape(nn.Module):
    def __init__(self, shape, layer_id):
        super(Reshape, self).__init__()
        self.shape = shape
        self.name = f'Reshape_{layer_id}'

    def forward(self, x):
        return x.view(self.shape)


class Concat_(nn.Module):
    def __init__(self, shape, layer_id):
        super(Concat_, self).__init__()
        self.shape = shape
        shapes = [s for s in shape.values()]
        self.output_shape = [shapes[0][0], sum([s[-1] for s in shape.values()])]
        self.name = f'Concat_{layer_id}'

    def forward(self, x):
        return torch.cat(x, -1)

def layers_func():
    layers = {'conv_2d': conv_2d,
              'time_distr_conv_2d': time_distr_conv_2d,
              'conv_3d': conv_3d,
              'time_distr_conv_3d': time_distr_conv_3d,
              'lstm': lstm,
              'transformer': transformer,
              'Reshape': Reshape,
              'unet': unet,
              # 'yolo': yolo,
              'timm_net': timm_net,
              'cross_attention': CrossAttention,
              'concatenate': Concat_,
              'vit_net': vit_net,
              'time_distr_vit_net': time_distr_vit_net,
              'dense': dense
              }
    return layers
