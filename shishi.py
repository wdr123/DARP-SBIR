import torch
from config import get_config
import data_loader
import json


if __name__ == "__main__":
    config, unparsed = get_config()
    with open('./models/ram_6_8x8_2_params.json', 'rt') as f:
        config.__dict__.update(json.load(f))

    kwargs = {}
    if config.use_gpu and torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    if config.use_gpu:
        torch.cuda.manual_seed(config.random_seed)
        kwargs = {"num_workers": 1, "pin_memory": True}

    dloader = data_loader.get_test_loader(
                config.data_dir, config.batch_size, **kwargs,
            )

    for i, (x, y) in enumerate(dloader):
        x, y = x.to(device), y.to(device)

        if i == 0:
            print('x', x.shape)
            print('y', y.shape)
        break