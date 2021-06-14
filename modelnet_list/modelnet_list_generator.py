import os
import random
import argparse


parser = argparse.ArgumentParser()
parser.add_argument("--modelnet_data_path", type=str, required=True, help="Modelnet dataset path")
parser.add_argument("--train_num", type=int, default=120, help="Final number of train models")
parser.add_argument("--test_num", type=int, default=40, help="Final number of test models")
parser.add_argument("--val_num", type=int, default=40, help="Final number of val models")

args = parser.parse_args()


total = []
for subdir, dirs, files in os.walk(args.modelnet_data_path):
    for file in files:
        off_file = os.path.join(subdir, file)
        total.append(off_file)

train = sorted(random.sample(total, args.train_num))
test = sorted(random.sample(list(set(total) - set(train)), args.test_num))
val = sorted(random.sample(list(set(total) - set(train) - set(test)), args.val_num))

if os.path.exists("train.txt"):
    os.remove("train.txt")
if os.path.exists("test.txt"):
    os.remove("test.txt")
if os.path.exists("val.txt"):
    os.remove("val.txt")

with open("train.txt", "w") as f:
    for i in train:
        f.write(i + "\n")

with open("val.txt", "w") as f:
    for i in val:
        f.write(i + "\n")

with open("test.txt", "w") as f:
    for i in test:
        f.write(i + "\n")
