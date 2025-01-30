import argparse
import torch
import datetime
import json
import yaml
import os

from model import diff_CSDI
from diffusion_process import CSDI_Generation
from dataset_generation import get_dataloader
from train import train, evaluate
from torch.optim import Adam
import torch.nn as nn

parser = argparse.ArgumentParser(description="CSDI")
parser.add_argument("--config", type=str, default="config.yaml")
parser.add_argument("--datatype", type=str, default="electricity")
parser.add_argument('--device', default='cuda:0', help='Device for Attack')
parser.add_argument("--seed", type=int, default=1)
parser.add_argument("--unconditional", action="store_true")
parser.add_argument("--modelfolder", type=str, default="")
parser.add_argument("--nsample", type=int, default=100)

args = parser.parse_args()
print(args)

path = "config/" + args.config
with open(path, "r") as f:
    config = yaml.safe_load(f)

if args.datatype == 'electricity':
    target_dim = 370

config["model"]["is_unconditional"] = args.unconditional

print(json.dumps(config, indent=4))

current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
foldername = "./save/generation_" + args.datatype + '_' + current_time + "/"
print('model folder:', foldername)
os.makedirs(foldername, exist_ok=True)
with open(foldername + "config.json", "w") as f:
    json.dump(config, f, indent=4)


# get dataloaders
train_loader, valid_loader, test_loader = get_dataloader(
    datatype=args.datatype,
    device=args.device,
    batch_size=config["train"]["batch_size"],
)

# get model
diff_model = diff_CSDI(config['diffusion'])
diff_model.to(args.device)
diffusion = CSDI_Generation(diff_model, config['diffusion'], device=args.device)
optimizer = Adam(diff_model.parameters(), lr=config["train"]["lr"], weight_decay=1e-6)
p1 = int(0.75 * config["train"]["epochs"])
p2 = int(0.9 * config["train"]["epochs"])
lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
    optimizer, milestones=[p1, p2], gamma=0.1
)
loss_fn = nn.MSELoss()

if args.modelfolder == "":
    train(
        diff_model,
        config["train"],
        train_loader,
        optimizer,
        lr_scheduler,
        loss_fn,
        diffusion,
        valid_loader=valid_loader,
        foldername=foldername,
    )
else:
    diff_model.load_state_dict(torch.load("./save/" + args.modelfolder + "/model.pth"))
    diff_model.target_dim = target_dim
    evaluate(
        diff_model,
        test_loader,
        nsample=args.nsample,
        foldername=foldername,
    )
