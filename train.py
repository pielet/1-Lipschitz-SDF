import os

import jax
import optax
import yaml
from flax import serialization
from flax.training import train_state
from jax import numpy as jnp
from torch.utils.data import DataLoader
from tqdm import tqdm

from data.dataset import SampleDataset
from data.gen_2d_dataset import select_sdf
from models.siren import SIREN
from utils.logger import WandbLogger
from utils.loss import hKR, mse
from utils.metric import evaluate_sdf_2d
from utils.plot import render_sdf_2d


def parser_config(config_path):
    with open(config_path) as f:
        config_dict = yaml.safe_load(f)
    logger = WandbLogger(project='1-lip-sdf', config=config_dict)
    return logger


def save_pretrained(params, path):
    bytes_data = serialization.to_bytes(params)
    with open(path, 'wb') as f:
        f.write(bytes_data)


def load_pretrained(model, path, dummy_input):
    with open(path, 'rb') as f:
        bytes_data = f.read()
    template_params = model.init(jax.random.PRNGKey(0), dummy_input)['params']
    params = serialization.from_bytes(template_params, bytes_data)
    return params


def train(logger, output_dir):
    # load dataset
    dataset = SampleDataset(f'input/{logger.config.dataset}.npz')
    print(f'Loaded {len(dataset)} samples from {logger.config.dataset}')
    dataloader = DataLoader(dataset, batch_size=logger.config.batch_size, shuffle=True)
    sdf = select_sdf(logger.config)

    # initialize model
    if logger.config.model['type'] == 'siren':
        model = SIREN(
            out_dim=1,
            hidden_layers=logger.config.model['hidden_layers'],
            hidden_units=logger.config.model['hidden_units'],
        )
    elif logger.config.model['type'] == 'ffn':
        pass
    elif logger.config.model['type'] == 'ortho':
        pass
    elif logger.config.model['type'] == 'sll':
        pass
    else:
        raise ValueError(f'Unknown model type: {logger.config.model["type"]}')

    rng = jax.random.PRNGKey(0)

    params = model.init(rng, jnp.zeros((1, dataset.input_dim)))['params']
    tx = optax.adam(learning_rate=logger.config.learning_rate)
    ts = train_state.TrainState.create(apply_fn=model.apply, params=params, tx=tx)

    # loss function
    if logger.config.model['type'] == 'siren' or logger.config.model['type'] == 'ffn':
        loss_fn = mse(ts.apply_fn)
    elif logger.config.model['type'] == 'ortho' or logger.config.model['type'] == 'sll':
        loss_fn = hKR(ts.apply_fn, margin=0.1, lamb=1.0, rho=lambda x, y: 1.0)

    @jax.jit
    def step(state, coords, field):
        value, grads = jax.value_and_grad(loss_fn)(state.params, coords, field)
        state = state.apply_gradients(grads=grads)
        return value, state

    # training loop
    loop = tqdm(range(logger.config.epochs))
    for epoch in loop:
        epoch_loss = 0.0
        for batch in dataloader:
            coords, field = batch
            value, ts = step(ts, coords.numpy(), field.numpy())
            epoch_loss += value
        if epoch % logger.config.log_interval == 0:
            logger.log('train/loss', epoch_loss / len(dataset))
            chamfer_dist, IoU, MSE = evaluate_sdf_2d(
                model, ts.params, sdf, min=logger.config.sample_min, max=logger.config.sample_max
            )
            logger.log('train/chamfer_dist', chamfer_dist)
            logger.log('train/iou', IoU)
            logger.log('train/mse', MSE)
            contour_fig, grad_fig = render_sdf_2d(model, ts.params)
            logger.log_image('contour', contour_fig, 'level-set contour')
            logger.log_image('grad', grad_fig, 'gradient magnitude')
        loop.set_description(f'Loss: {epoch_loss / len(dataset):.4f}')
        loop.update()

    # save model
    if logger.config.save_pretrained:
        save_pretrained(ts.params, os.path.join(output_dir, 'model_final.msgpack'))
        print(f'Model saved to {os.path.join(output_dir, "model_final.msgpack")}')

    logger.finish()


if __name__ == '__main__':
    import datetime
    import sys

    logger = parser_config(sys.argv[1])
    timestamp = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    output_dir = os.path.join(
        'output', f'{logger.config.dataset}', f'{logger.config.model["type"]}', timestamp
    )
    os.makedirs(output_dir, exist_ok=True)

    train(logger, output_dir)
