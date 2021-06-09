import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import utils
from data import dataset
from models import UNET

parser = argparse.ArgumentParser()
parser.add_argument("--config", type=str, required=True, help="Configuration file for training")

args = parser.parse_args()


config = utils.load_yaml(args.config)
print(config.DATA)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = UNET(bn=True).double().to(device)

train_dataset = dataset.DepthDataset(config.DATA.train_data, config.DATA.noise)
train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=1)

criterion = nn.SmoothL1Loss()
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

for i in range(0, config.TRAINING.epoch_size):
    for idx, batch in enumerate(train_dataloader):
        depth = batch.get("depth").unsqueeze(1).to(device)
        noisy_depth = batch.get("noisy_depth").unsqueeze(1).to(device)

        # zero the parameter gradients
        optimizer.zero_grad()

        output = model(noisy_depth)

        loss = criterion(output, depth)
        loss.backward()
        optimizer.step()



