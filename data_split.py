import os
import shutil
import random
from tqdm import tqdm
from torchvision import datasets, transforms


random_seed = 42
random.seed(random_seed)


data_dir = "cutData"
new_data_dir = "WorkData"
train_dir = os.path.join(new_data_dir, "train")
valid_dir = os.path.join(new_data_dir, "val")
test_dir = os.path.join(new_data_dir, "test")


for directory in [train_dir, valid_dir, test_dir]:
    if not os.path.exists(directory):
        os.makedirs(directory)


full_dataset = datasets.ImageFolder(root=data_dir)


dataset_size = len(full_dataset)
train_size = int(0.8 * dataset_size)
valid_size = int(0.1 * dataset_size)
test_size = dataset_size - train_size - valid_size
print(test_size)

indices = list(range(dataset_size))
random.shuffle(indices)


for i, subset_dir in tqdm(enumerate([train_dir, valid_dir, test_dir])):
    subset_indices = []
    if i == 0:
        subset_indices = indices[0:train_size]
    elif i == 1:
        subset_indices = indices[train_size:train_size+valid_size]
    else:
        subset_indices = indices[train_size+valid_size:dataset_size]
    for idx in tqdm(subset_indices):
        img_path, _ = full_dataset.samples[idx]
        rel_path = os.path.relpath(img_path, data_dir)
        new_img_path = os.path.join(subset_dir, rel_path)
        os.makedirs(os.path.dirname(new_img_path), exist_ok=True)
        shutil.copy(img_path, new_img_path)

print("Finish")
