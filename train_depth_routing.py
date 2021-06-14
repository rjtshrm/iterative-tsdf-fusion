import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import utils
from data import dataset
from models import UNET
from loss.routing_loss import RoutingLoss

parser = argparse.ArgumentParser()
parser.add_argument("--config", type=str, required=True, help="Configuration file for training")

args = parser.parse_args()

config = utils.load_yaml(args.config)
print(config.DATA)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = UNET(bn=False).double().to(device)

train_dataset = dataset.DepthDataset(config.DATA.train_data, config.DATA.noise)
train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=1)

criterion = RoutingLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

for epoch in range(0, config.TRAINING.epoch_size):
    for idx, batch in enumerate(train_dataloader):
        depth = batch.get("depth").unsqueeze(1).to(device)
        noisy_depth = batch.get("noisy_depth").unsqueeze(1).to(device)

        # zero the parameter gradients
        optimizer.zero_grad()

        output = model(noisy_depth)

        output_depth, output_confidence = output

        loss = criterion(output_depth, output_confidence, depth)
        loss.backward()
        optimizer.step()

        print(f"[Epoch: {epoch + 1}, {idx + 1}]"
              f"loss: {loss.item()}")
