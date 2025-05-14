from .ffn import MLP, SIREN
from .wire import GaborNet
from .sll import SLLNet
from .pe import GaussianFourierPE, GaussianGaborPE, FourierPE, GaborPE

from functools import partial
import inspect


model_zoo = {
    'siren': SIREN,
    'wire': GaborNet,
    'ffn': MLP,
    'sll': SLLNet,
}


def safe_call(func, args):
    sig = inspect.signature(func)
    valid_keys = sig.parameters.keys()
    return func(**{k: v for k, v in args.items() if k in valid_keys})


def get_model(out_dim, config):
    """Get model with positional encoding.

    Args:
        out_dim (int): Output dimension of the model.
        config (dict): Configuration dictionary containing model parameters.
    Returns:
        model: The model instance.
    Raises:
        ValueError: If the model type is not supported.
    """
    pos_enc = None
    if config.pos_enc_type == 'fourier':
        # pos_enc = partial(FourierPE, emb_size=config.pos_enc.emb_size)
        pos_enc = safe_call(GaussianFourierPE, config.pos_enc)
    elif config.pos_enc_type == 'gabor':
        # pos_enc = partial(GaborPE, emb_size=config.pos_enc.emb_size)
        pos_enc = safe_call(GaussianGaborPE, config.pos_enc)
    model = safe_call(
        partial(model_zoo.get(config.model_type), out_dim=out_dim, pos_enc=pos_enc), config.model
    )
    return model


__all__ = ['get_model']
