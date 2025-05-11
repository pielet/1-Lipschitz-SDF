from io import BytesIO

import matplotlib.pyplot as plt
from PIL import Image


class WandbLogger:
    def __init__(self, project, config):
        import wandb

        self.wandb = wandb
        self.run = wandb.init(project=project, config=config)

    def log(self, name, data):
        self.wandb.log({name: data})

    def log_image(self, name, fig, caption):
        buf = BytesIO()
        fig.savefig(buf, format="png", bbox_inches="tight", pad_inches=0.05)
        image = Image.open(buf)
        self.wandb.log({name: self.wandb.Image(image, caption=caption)})
        plt.close(fig)

    @property
    def config(self):
        return self.wandb.config

    def finish(self):
        self.run.finish()
