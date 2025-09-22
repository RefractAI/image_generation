import torch
from models.image_model.image_model import ImageModel
from models.image_model.c2i_model import C2IModel
from models.image_model.ddt import DDT


def build_model(cfg, text_cfg):
    match cfg.type:
        case "ImageModel":
            if text_cfg.type == "c2i":
                num_classes = getattr(cfg, "num_classes", 1000)
                model = C2IModel(
                    in_channels=cfg.channels,
                    num_groups=cfg.n_heads,
                    hidden_size=cfg.dim,
                    decoder_hidden_size=cfg.decoder_hidden_size,
                    num_encoder_blocks=cfg.n_encoder_layers,
                    num_decoder_blocks=cfg.n_layers - cfg.n_encoder_layers,
                    patch_size=cfg.patch_size,
                    use_tread=cfg.use_tread,
                    num_classes=num_classes + 1,
                )
            else:
                model = ImageModel(
                    in_channels=cfg.channels,
                    num_groups=cfg.n_heads,
                    hidden_size=cfg.dim,
                    decoder_hidden_size=cfg.decoder_hidden_size,
                    num_encoder_blocks=cfg.n_encoder_layers,
                    num_decoder_blocks=cfg.n_layers - cfg.n_encoder_layers,
                    num_text_blocks=cfg.num_text_blocks,
                    patch_size=cfg.patch_size,
                    txt_embed_dim=text_cfg.txt_embed_dim,
                    txt_max_length=text_cfg.max_length,
                    use_tread=cfg.use_tread,
                )
            if cfg.compile and torch.cuda.is_available():
                model.compile()
            return model
        case "DDT":
            model = DDT(
                in_channels=cfg.channels,
                num_groups=cfg.n_heads,
                hidden_size=cfg.dim,
                num_blocks=cfg.n_layers,
                num_encoder_blocks=cfg.n_encoder_layers,
                patch_size=cfg.patch_size,
                num_classes=getattr(cfg, 'num_classes', 1000),
                learn_sigma=getattr(cfg, 'learn_sigma', True),
                deep_supervision=getattr(cfg, 'deep_supervision', 0),
                weight_path=None,
                load_ema=False,
                experiment=getattr(cfg, 'experiment', 'baseline'),
                txt_embed_dim=text_cfg.txt_embed_dim if getattr(cfg, 'use_cross_attention', False) else None,
                txt_max_length=text_cfg.max_length if getattr(cfg, 'use_cross_attention', False) else None,
                num_text_blocks=getattr(cfg, 'num_text_blocks', 4) if getattr(cfg, 'use_cross_attention', False) else None,
                use_cross_attention=getattr(cfg, 'use_cross_attention', False),
                use_tread=getattr(cfg, 'use_tread', False),
                tread_dropout_ratio=getattr(cfg, 'tread_dropout_ratio', 0.5),
                tread_prev_blocks=getattr(cfg, 'tread_prev_blocks', 3),
                tread_post_blocks=getattr(cfg, 'tread_post_blocks', 1),
            )
            if cfg.compile and torch.cuda.is_available():
                model.compile()
            return model
        case _:
            raise ValueError(f"Unsupported model type '{cfg.type}'.")
