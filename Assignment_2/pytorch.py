import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets
import torchvision.transforms as transforms
from torch.utils.data.sampler import SubsetRandomSampler
import numpy as np
import matplotlib.pyplot as plt
import os


os.makedirs("output", exist_ok=True)

# -- DEVICE SETUP (Mac GPU MPS) ---
if torch.backends.mps.is_available():
    device = torch.device("mps")
    print("Using Apple GPU (MPS)")
else:
    device = torch.device("cpu")
    print("Using CPU")

# --- PARAMETERS ---
batch_size = 20
valid_size = 0.2
n_epochs = 20

# --- DATA ---
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
])

train_data = datasets.CIFAR10('data', train=True, download=True, transform=transform)
test_data  = datasets.CIFAR10('data', train=False, download=True, transform=transform)

num_train = len(train_data)
indices = list(range(num_train))
np.random.shuffle(indices)
split = int(np.floor(valid_size * num_train))
train_idx, valid_idx = indices[split:], indices[:split]

train_sampler = SubsetRandomSampler(train_idx)
valid_sampler = SubsetRandomSampler(valid_idx)

train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, sampler=train_sampler)
valid_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, sampler=valid_sampler)
test_loader  = torch.utils.data.DataLoader(test_data, batch_size=batch_size)

classes = ['airplane','automobile','bird','cat','deer','dog','frog','horse','ship','truck']

# -- MODEL ARCHITECTURE ---
class Net(nn.Module):
    def __init__(self):
        super(Net,self).__init__()
        self.conv1 = nn.Conv2d(3,32,3,padding=1)
        self.conv2 = nn.Conv2d(32,64,3,padding=1)
        self.conv3 = nn.Conv2d(64,128,3,padding=1)

        self.pool = nn.MaxPool2d(2,2)
        self.dropout = nn.Dropout(0.25)

        self.fc1 = nn.Linear(128*4*4,256)
        self.fc2 = nn.Linear(256,10)

    def forward(self,x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))

        x = x.view(x.size(0),-1)
        x = self.dropout(x)

        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

model = Net().to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# --- TRAINING ---
train_losses = []
valid_losses = []

for epoch in range(1,n_epochs+1):

    train_loss = 0.0
    valid_loss = 0.0

    model.train()
    for data,target in train_loader:
        data,target = data.to(device),target.to(device)

        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output,target)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()*data.size(0)

    model.eval()
    for data,target in valid_loader:
        data,target = data.to(device),target.to(device)

        output = model(data)
        loss = criterion(output,target)
        valid_loss += loss.item()*data.size(0)

    train_loss = train_loss/len(train_loader.sampler)
    valid_loss = valid_loss/len(valid_loader.sampler)

    train_losses.append(train_loss)
    valid_losses.append(valid_loss)

    print(f"Epoch {epoch}  Train Loss: {train_loss:.4f}  Val Loss: {valid_loss:.4f}")

print("Training Complete")

# -- TEST ---
test_loss = 0.0
correct = 0
total = 0

model.eval()
for data,target in test_loader:
    data,target = data.to(device),target.to(device)

    output = model(data)
    loss = criterion(output,target)
    test_loss += loss.item()*data.size(0)

    _,pred = torch.max(output,1)
    correct += pred.eq(target).sum().item()
    total += target.size(0)

test_loss = test_loss/len(test_loader.dataset)
accuracy = 100*correct/total

print("Test Loss:", test_loss)
print("Test Accuracy:", accuracy,"%")
plt.figure()
plt.plot(train_losses,label="train")
plt.plot(valid_losses,label="val")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.savefig("output/loss_curve.png")
plt.close()

# -- TASK 2A FEATURE MAPS ---
def show_feature_maps(model,image,label,index):
    model.eval()
    image = image.unsqueeze(0).to(device)

    with torch.no_grad():
        fmap = F.relu(model.conv1(image))

    fmap = fmap.cpu()

    fig = plt.figure(figsize=(14,3))

    ax = fig.add_subplot(1,9,1)
    img = image.cpu().squeeze()/2 + 0.5
    ax.imshow(np.transpose(img.numpy(),(1,2,0)))
    ax.set_title(classes[label])
    ax.axis('off')

    for i in range(8):
        ax = fig.add_subplot(1,9,i+2)
        ax.imshow(fmap[0][i],cmap='gray')
        ax.set_title(f"F{i}")
        ax.axis('off')

    plt.savefig(f"output/feature_maps_{index}.png")
    plt.close()

used = set()
count = 0

for idx in range(len(test_data)):
    img,label = test_data[idx]
    if label not in used:
        show_feature_maps(model,img,label,count)
        used.add(label)
        count += 1
    if count == 3:
        break

# -- TASK 2B TOP 5 FILTERS ---
def top5_for_filter(model,filter_index,activation_type="mean"):
    model.eval()
    scores = []

    with torch.no_grad():
        for idx in range(len(test_data)):
            img,label = test_data[idx]
            img = img.unsqueeze(0).to(device)

            act = F.relu(model.conv1(img))

            if activation_type == "max":
                value = act[0][filter_index].max().item()
            else:
                value = act[0][filter_index].mean().item()

            scores.append((value,idx))

    scores.sort(reverse=True)
    top5 = scores[:5]

    fig = plt.figure(figsize=(14,3))
    for i,(score,idx) in enumerate(top5):
        img,label = test_data[idx]
        ax = fig.add_subplot(1,5,i+1)

        img = img/2 + 0.5
        ax.imshow(np.transpose(img.numpy(),(1,2,0)))
        ax.set_title(f"{classes[label]}\n{activation_type}:{score:.2f}")
        ax.axis('off')

    plt.savefig(f"output/filter_{filter_index}.png")
    plt.close()

print("Activation definition: mean")

top5_for_filter(model,0,"mean")
top5_for_filter(model,5,"mean")
top5_for_filter(model,10,"mean")

print("All images saved inside output folder.")