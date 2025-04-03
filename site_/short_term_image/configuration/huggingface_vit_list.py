model_satellites = [
    # 'jucamohedano/dinov2-base-finetuned-eurosat-rgb',
    'mrm8488/convnext-tiny-finetuned-eurosat',
    'sawthiha/segformer-b0-finetuned-deprem-satellite'
]
model_ids = [
    'OpenGVLab/pvt_v2_b0',
    # 'Intel/dpt-hybrid-midas',
    'facebook/dinov2-base',
    # 'facebook/levit-128S',
    'facebook/dinov2-small-imagenet1k-1-layer',
    # 'facebook/convnextv2-tiny-22k-224',
    'facebook/deit-tiny-patch16-224',
    # 'facebook/vit-msn-small',
    # 'sail/poolformer_s12',
    # 'microsoft/swin-tiny-patch4-window7-224',
    # 'microsoft/swinv2-tiny-patch4-window8-256',
    # 'microsoft/rad-dino',
    # 'microsoft/focalnet-tiny',
    # 'google/vit-base-patch16-224',
    'Zetatech/pvt-tiny-224',
    'nvidia/mit-b0',
    # 'MBZUAI/swiftformer-xs',
    'Visual-Attention-Network/van-tiny',
    # 'facebook/vit-mae-base',
    'hustvl/yolos-tiny']

# timm_model_ids = ['timm/vit_base_patch16_clip_224.openai',
#                   'timm/vit_base_patch8_224.dino',
#                   'timm/vit_base_patch16_224.dino',
#                   'timm/vit_small_patch8_224.dino',
#                   'timm/vit_small_patch16_224.dino',
#                   'timm/vit_base_r50_s16_224.orig_in21k',
#                   'timm/vit_base_patch16_224.mae']
huggingface_models = set(model_satellites + model_ids)