import wandb


WANDB_API_KEY = "YOUR_API_KEY"


def setup_wandb():
    wandb.login(key=WANDB_API_KEY)
