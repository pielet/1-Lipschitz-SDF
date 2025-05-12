import inspect
import os

# os.environ['XLA_FLAGS'] = '--xla_gpu_autotune_level=0'  # disable autotune warnings

from functools import partial

import jax
import optax
import numpy as np
from flax import serialization
from flax.training import train_state
from jax import numpy as jnp
from omegaconf import OmegaConf
import torch
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm

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
    data = np.load(f'input/{config.dataset}.npz')
    coordiates, values = data['X'], data['Y']
    train_dataset = TensorDataset(torch.Tensor(coordiates), torch.Tensor(values))
    print(f'Loaded {len(train_dataset)} samples from {config.dataset}')
    train_dataloader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    sdf = select_sdf(config)

    # evaluation
    X, Y = np.mgrid[
        config.domain_pivot[0] : config.domain_pivot[0] + config.domain_size : config.domain_size
        / config.evaluation.resolution,
        config.domain_pivot[1] : config.domain_pivot[1] + config.domain_size : config.domain_size
        / config.evaluation.resolution,
    ]
    eval_coords = np.column_stack((X.ravel(), Y.ravel()))
    sdf_true = sdf(eval_coords).squeeze()
    eval_dataset = TensorDataset(torch.Tensor(eval_coords))
    eval_dataloader = DataLoader(eval_dataset, batch_size=100*config.batch_size, shuffle=False)

    # initialize model
    models = {'siren': SIREN, 'ffn': MLP, 'sll': SLLNet}
    model = safe_call(partial(models.get(config.model_type), out_dim=values.shape[1]), config.model)

    key = jax.random.PRNGKey(0)
    variables = model.init(key, jnp.zeros((1, coordiates.shape[1])), mutable=['params', 'constants'])
    constants = variables['constants']
    # print(jax.tree_map(lambda x: x.dtype, variables['params']))
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

    def evaluate(state, dataloader):
        """Evaluate the model and ground truth on a grid of points in the domain in minibatchs."""
        def forward(coords):
            return state.apply_fn({'params': state.params, 'constants': constants}, coords).squeeze()
        sdf_pred = []
        grad_pred = []
        for coords_batch in dataloader:
            sdf_batch, grad_batch = jax.vmap(jax.value_and_grad(forward))(coords_batch[0].numpy())
            sdf_pred.append(sdf_batch)
            grad_pred.append(grad_batch)
        sdf_pred = np.concatenate(sdf_pred)
        grad_pred = np.concatenate(grad_pred, axis=0)
        return sdf_pred, grad_pred

    @jax.jit
    def step(state, coords, field):
        value, grads = jax.value_and_grad(loss_fn)(state.params, coords, field)
        state = state.apply_gradients(grads=grads)
        return value, state

    # training loop
    loop = tqdm(range(config.epochs))
    for epoch in loop:
        epoch_loss = 0.0
        for batch in train_dataloader:
            coords, field = batch
            value, ts = step(ts, coords.numpy(), field.numpy())
            epoch_loss += value
        logger.log('train/loss', epoch_loss / len(train_dataset))
        if epoch % config.evaluation.interval == 0:
            logger.log('train/loss', epoch_loss / len(train_dataset))
            sdf_pred, grad_pred = evaluate(ts, eval_dataloader)
            chamfer_dist, IoU, MSE, Eikonal = evaluate_sdf_2d(
                eval_coords, sdf_pred, sdf_true, grad_pred
            )
            logger.log('train/chamfer_dist', chamfer_dist)
            logger.log('train/iou', IoU)
            logger.log('train/mse', MSE)
            logger.log('train/eikonal', Eikonal)
            contour_fig, grad_fig = render_sdf_2d(sdf_pred, grad_pred, config.evaluation.resolution)
            logger.log_image('contour', contour_fig, 'level-set contour')
            logger.log_image('grad', grad_fig, 'gradient magnitude')
        loop.set_description(f'Loss: {epoch_loss / len(train_dataset):.8f}')
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
