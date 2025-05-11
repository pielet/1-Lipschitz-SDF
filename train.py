import inspect
import os
from functools import partial

import jax
import optax
from flax import serialization
from flax.training import train_state
from jax import numpy as jnp
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from tqdm import tqdm

from data.dataset import SampleDataset
from gen_2d_dataset import parse_config, select_sdf
from models.ffn import MLP, SIREN
from models.sll import SLLNet
from utils.logger import WandbLogger
from utils.loss import eikonal, hKR, mse
from utils.metric import evaluate_sdf_2d
from utils.plot import render_sdf_2d


def save_pretrained(params, path):
    with open(path, 'wb') as f:
        f.write(serialization.to_bytes(params))


def load_pretrained(model, path, dummy_input):
    with open(path, 'rb') as f:
        bytes_data = f.read()
    template_params = model.init(jax.random.PRNGKey(0), dummy_input)['params']
    params = serialization.from_bytes(template_params, bytes_data)
    return params


def safe_call(func, args):
    sig = inspect.signature(func)
    valid_keys = sig.parameters.keys()
    return func(**{k: v for k, v in args.items() if k in valid_keys})


def train(config, logger, output_dir):
    # load dataset
    dataset = SampleDataset(f'input/{config.dataset}.npz')
    print(f'Loaded {len(dataset)} samples from {config.dataset}')
    dataloader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True)
    sdf = select_sdf(config)

    # initialize model
    models = {'siren': SIREN, 'ffn': MLP, 'sll': SLLNet}
    model = safe_call(partial(models.get(config.model_type), out_dim=dataset.out_dim), config.model)

    key = jax.random.PRNGKey(0)

    variables = model.init(key, jnp.zeros((1, dataset.in_dim)), mutable=['params', 'constants'])
    constants = variables['constants']
    tx = optax.adam(learning_rate=config.learning_rate)
    ts = train_state.TrainState.create(apply_fn=model.apply, params=variables['params'], tx=tx)

    # loss function
    # losses = {
    #     'mse': mse,
    #     'eikonal': eikonal,
    #     'hkr': hKR,
    # }
    # loss_fn = safe_call(
    #     partial(losses.get(config.loss_type), apply_fn=model.apply, constants=constants),
    #     config.loss,
    # )
    if config.loss_type == 'mse':
        loss_fn = mse(ts.apply_fn, constants)
    elif config.loss_type == 'eikonal':
        loss_fn = eikonal(ts.apply_fn, constants, lamb=config.loss.lamb)
    elif config.loss_type == 'hkr':
        loss_fn = hKR(
            ts.apply_fn,
            constants,
            margin=config.loss.margin,
            lamb=config.loss.lamb,
            rho=lambda x, y: 1.0,
        )

    @jax.jit
    def step(state, coords, field):
        value, grads = jax.value_and_grad(loss_fn)(state.params, coords, field)
        state = state.apply_gradients(grads=grads)
        return value, state

    # training loop
    loop = tqdm(range(config.epochs))
    for epoch in loop:
        epoch_loss = 0.0
        for batch in dataloader:
            coords, field = batch
            value, ts = step(ts, coords.numpy(), field.numpy())
            epoch_loss += value
        if epoch % config.log_interval == 0:
            logger.log('train/loss', epoch_loss / len(dataset))
            chamfer_dist, IoU, MSE, Eikonal = evaluate_sdf_2d(
                model,
                {'params': ts.params, 'constants': constants},
                sdf,
                config.domain_pivot,
                config.domain_size,
            )
            logger.log('train/chamfer_dist', chamfer_dist)
            logger.log('train/iou', IoU)
            logger.log('train/mse', MSE)
            logger.log('train/eikonal', Eikonal)
            contour_fig, grad_fig = render_sdf_2d(
                model,
                {'params': ts.params, 'constants': constants},
                config.domain_pivot,
                config.domain_size,
            )
            logger.log_image('contour', contour_fig, 'level-set contour')
            logger.log_image('grad', grad_fig, 'gradient magnitude')
        loop.set_description(f'Loss: {epoch_loss / len(dataset):.4f}')
        loop.update()

    # save model
    if config.save_pretrained:
        save_pretrained(ts.params, os.path.join(output_dir, 'model_final.msgpack'))
        print(f'Model saved to {os.path.join(output_dir, "model_final.msgpack")}')

    logger.finish()


if __name__ == '__main__':
    import datetime
    import sys

    config = parse_config(sys.argv[1])
    logger = WandbLogger(project='1-lip-sdf', config=OmegaConf.to_container(config, resolve=True))

    timestamp = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    output_dir = os.path.join(
        'output', f'{config.dataset}', f'{config.model_type}', f'{config.loss_type}', timestamp
    )
    os.makedirs(output_dir, exist_ok=True)

    train(config, logger, output_dir)
