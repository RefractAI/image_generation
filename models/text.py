import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def build_text_stack(cfg, device_type):
    match cfg.text_encoder.type:
        case "causal_lm":
            if cfg.text_encoder.path is None:
                raise ValueError("causal_lm text encoder requires a 'path'.")
            tokenizer = AutoTokenizer.from_pretrained(cfg.text_encoder.path, padding_side="right")
            dtype = torch.bfloat16 if device_type == "cuda" else torch.float32
            encoder = AutoModelForCausalLM.from_pretrained(cfg.text_encoder.path, torch_dtype=dtype)
            encoder.eval()
            if cfg.model.compile and torch.cuda.is_available():
                encoder.compile()
            return tokenizer, encoder
        case "c2i":
            return None, None
        case _:
            raise ValueError(f"Unsupported text encoder type '{cfg.text_encoder.type}'.")


def _parse_label(value, default_label):
    if isinstance(value, torch.Tensor):
        value = value.item()
    if isinstance(value, (int, float)):
        return int(value)
    if isinstance(value, str):
        stripped = value.strip()
        if stripped:
            try:
                return int(stripped)
            except ValueError as exc:
                raise ValueError(f"Could not parse label '{value}' for c2i text encoder.") from exc
        return default_label
    if value is None:
        return default_label
    raise ValueError(f"Unsupported label type '{type(value)}' for c2i text encoder.")


def encode_text(cfg, tokenizer, encoder, device, max_length, dtype, captions):
    match cfg.text_encoder.type:
        case "causal_lm":
            with torch.no_grad():
                tokens = tokenizer(
                    captions,
                    return_tensors="pt",
                    padding="max_length",
                    truncation=True,
                    max_length=max_length,
                )
                tokens = {k: v.to(device) for k, v in tokens.items()}
                outputs = encoder(**tokens, output_hidden_states=True)
                return outputs.hidden_states[-1].to(dtype)
        case "c2i":
            default_label = getattr(cfg.text_encoder, "default_label", 0)
            labels = [_parse_label(caption, default_label) for caption in captions]
            return torch.tensor(labels, device=device, dtype=torch.long)
        case _:
            raise ValueError(f"Unsupported text encoder type '{cfg.text_encoder.type}'.")
