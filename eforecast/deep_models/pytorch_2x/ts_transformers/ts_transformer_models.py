from eforecast.deep_models.pytorch_2x.ts_transformers.Autoformer import Autoformer
from eforecast.deep_models.pytorch_2x.ts_transformers.Crossformer import Crossformer
from eforecast.deep_models.pytorch_2x.ts_transformers.DLinear import DLinear
from eforecast.deep_models.pytorch_2x.ts_transformers.FEDformer import FEDformer
from eforecast.deep_models.pytorch_2x.ts_transformers.FiLM import FiLM
from eforecast.deep_models.pytorch_2x.ts_transformers.Informer import Informer
from eforecast.deep_models.pytorch_2x.ts_transformers.LightTS import LightTS
from eforecast.deep_models.pytorch_2x.ts_transformers.PatchTST import PatchTST
from eforecast.deep_models.pytorch_2x.ts_transformers.Reformer import Reformer



def get_transfromer_model(name, seq_len, pred_len, enc_in, dec_in, c_out, params):
    if name == 'Autoformer':
        return Autoformer(seq_len=seq_len,
                 label_len=seq_len,
                 pred_len=pred_len,
                 moving_avg=3,
                 output_attention=False,
                 enc_in=enc_in,
                 dec_in=dec_in,
                 d_model=64,
                 embed=64,
                  cal_vars=params['cal_vars'],
                 freq='t',
                 dropout=0.1,
                 e_layers=4,
                 d_layers=4,
                 d_ff=128,
                 n_heads=8,
                 factor=2,
                 activation='gelu',
                 c_out=c_out)
    elif name == 'Crossformer':
        return Crossformer(enc_in=enc_in,
                 seq_len=seq_len,
                 pred_len=pred_len,
                 d_model=64,
                 e_layers=5,
                 n_heads=8,
                 d_ff=256,
                 dropout=0.05,
                 factor=2,
                 output_attention=False,)
    elif name == 'DLinear':
        return DLinear(individual=False,
                 seq_len=seq_len,
                 pred_len=pred_len,
                 moving_avg=3,
                 enc_in=enc_in)
    elif name == 'FEDformer':
        return FEDformer(seq_len=seq_len,
                 label_len=seq_len,
                 pred_len=pred_len,
                 moving_avg=3,
                 enc_in=enc_in,
                 dec_in=dec_in,
                 d_model=64,
                 embed='fixed',
                 cal_vars=params['cal_vars'],
                 freq='t',
                 dropout=0.05,
                 e_layers=5,
                 d_layers=5,
                 d_ff=128,
                 n_heads=8,
                 factor=2,
                 activation='gelu',
                 c_out=c_out,
                 version='fourier',
                 mode_select='random',
                 modes=32)
    elif name == 'FiLM':
        return FiLM(seq_len=seq_len,
                 label_len=seq_len,
                 pred_len=pred_len,
                 output_attention=False,
                 e_layers=5,
                 enc_in=enc_in,
                 ratio=0.5,
                    device=params['device'])
    elif name == 'Informer':
        return Informer(enc_in=enc_in,
                 dec_in=dec_in,
                 c_out=c_out,
                 label_len=seq_len,
                 pred_len=pred_len,
                 d_model=64,
                 e_layers=5,
                 d_layers=5,
                 n_heads=8,
                 embed=64,
                 cal_vars=params['cal_vars'],
                 freq='t',
                 d_ff=256,
                 dropout=0.05,
                 factor=2,
                 output_attention=False,
                 activation='gelu',
                 distil=True)
    elif name == 'LightTS':
        return LightTS(seq_len,
                 pred_len,
                 0.05,
                 64,
                 enc_in,
                 chunk_size=1)
    elif name == 'PatchTST':
        return PatchTST(enc_in= enc_in,
                 c_out= c_out,
                 seq_len= seq_len,
                 pred_len= pred_len,
                 d_model= 128,
                 n_heads= 8,
                 e_layers= 5,
                 d_ff= 512,
                 factor= 2,
                 dropout= 0.05,
                 embed= 'timeF',
                 cal_vars=params['cal_vars'],
                 activation= 'gelu',
                 output_attention= False,
                 patch_len=2,
                 stride=2)
    elif name == 'Reformer':
        return Reformer(pred_len=pred_len,
                 seq_len=seq_len,
                 enc_in=enc_in,
                 d_model=128,
                 embed=64,
                 cal_vars=params['cal_vars'],
                 freq='t',
                 dropout=0.1,
                 e_layers=4,
                 d_ff=128,
                 n_heads=8,
                 factor=2,
                 activation='gelu',
                 c_out=c_out,
                 bucket_size=4,
                 n_hashes=4)
