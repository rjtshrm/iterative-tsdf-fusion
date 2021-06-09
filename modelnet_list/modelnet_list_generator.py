import os
import random

modelnet_data_path = "/home/rash8327/Downloads/ModelNet40"

total = []
for subdir, dirs, files in os.walk(modelnet_data_path):
    for file in files:
        off_file = os.path.join(subdir, file)
        total.append(off_file)


train = sorted(random.sample(total, 10)) #60%
test = sorted(random.sample(list(set(total) - set(train)), 3)) #20%
val = sorted(random.sample(list(set(total) - set(train) - set(test)), 3)) #20%

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
print(train)
print(test)
print(val)