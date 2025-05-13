import inspect
import os

# os.environ['XLA_FLAGS'] = '--xla_gpu_autotune_level=0'  # disable autotune warnings

from functools import partial

import jax
import optax
import numpy as np
from flax.training import train_state, checkpoints
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


def save_pretrained(state, path):
    checkpoints.save_checkpoint(ckpt_dir=os.path.abspath(path), target=state.params, step=state.step, overwrite=True)


def load_pretrained(state, path):
    checkpoints.restore_checkpoint(ckpt_dir=os.path.abspath(path), target=state.params)


def safe_call(func, args):
    sig = inspect.signature(func)
    valid_keys = sig.parameters.keys()
    return func(**{k: v for k, v in args.items() if k in valid_keys})


def train(config, logger, ckpt_dir):
    # load training dataset
    data = np.load(f'input/{config.dataset}.npz')
    coordiates, values = data['X'], data['Y']
    train_dataset = TensorDataset(torch.Tensor(coordiates), torch.Tensor(values))
    print(f'Loaded {len(train_dataset)} samples from {config.dataset}')
    train_dataloader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    sdf = select_sdf(config)

    # evaluation dataset
    X, Y = np.mgrid[
        config.domain_pivot[0] : config.domain_pivot[0] + config.domain_size : config.domain_size
        / config.evaluation.resolution,
        config.domain_pivot[1] : config.domain_pivot[1] + config.domain_size : config.domain_size
        / config.evaluation.resolution,
    ]
    eval_coords = np.column_stack((X.ravel(), Y.ravel()))
    sdf_true = sdf(eval_coords).squeeze()
    eval_dataset = TensorDataset(torch.Tensor(eval_coords))
    eval_dataloader = DataLoader(eval_dataset, batch_size=config.batch_size, shuffle=False)

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

    @jax.jit
    def evaluate(state, coords):
        def forward(coords):
            coords = coords.reshape(-1, coords.shape[-1])  # [1, in_dim]
            return state.apply_fn({'params': state.params, 'constants': constants}, coords).squeeze()
        return jax.vmap(jax.value_and_grad(forward))(coords)

    @jax.jit
    def step(state, coords, field):
        value, grads = jax.value_and_grad(loss_fn)(state.params, coords, field)
        state = state.apply_gradients(grads=grads)
        return value, state
    
    # load pretrained model
    if config.load_pretrained:
        load_pretrained(ts, ckpt_dir)
        print('Model loaded')

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
            sdf_pred, grad_pred = [], []
            for batch in eval_dataloader:
                sdf_batch, grad_batch = evaluate(ts, batch[0].numpy())
                sdf_pred.append(sdf_batch)
                grad_pred.append(grad_batch)
            sdf_pred = np.concatenate(sdf_pred)
            grad_pred = np.concatenate(grad_pred, axis=0)
            
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
        save_pretrained(ts, ckpt_dir)
        print(f'Model saved to {ckpt_dir}')

    logger.finish()


if __name__ == '__main__':
    import sys

    config = parse_config(sys.argv[1])
    logger = WandbLogger(project='1-lip-sdf', config=OmegaConf.to_container(config, resolve=True))
    ckpt_dir = os.path.join(
                'output', f'{config.dataset}', f'{config.model_type}', f'{config.loss_type}'
            )
    os.makedirs(ckpt_dir, exist_ok=True)

    train(config, logger, ckpt_dir)
