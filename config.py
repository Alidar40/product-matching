import os
import yaml

config = yaml.safe_load(open("config.yaml", "rb"))

if config['wandb']['name'] == 'as_model':
    config['wandb']['name'] = config['model']

