#Imports
import torch
from torch import optim, nn
import torch.nn.functional as F
from torchvision import datasets, transforms, models
import numpy as np
import json
import argparse

# Load arguments
parser = argparse.ArgumentParser()
parser.add_argument('data_dir', help = 'Provide data directory.', type = str)
parser.add_argument ('--save_dir', help = 'Provide save directory (default = directory of script).', type = str, default = 'current directory')
parser.add_argument ('--arch', help = 'Provide the type of model (resnet18  or vgg16) (default = resnet18).', type = str, default = 'resnet18')
parser.add_argument ('--lr', help = 'Provide the learning rate (default = 0.001).', type = float, default = 0.001)
parser.add_argument ('--hidden_units', help = 'Provide the Classifier hidden units (default = 256).', type = int, default = 1024)
parser.add_argument ('--epochs', help = 'Provide the number of epochs (default = 3).', type = int, default = 3)
parser.add_argument ('--gpu', help = "Provide option to use gpu (default = false/cpu).", action='store_true')
args = parser.parse_args()


data_dir = args.data_dir
save_dir=args.save_dir
arch = args.arch
lr = args.lr
hidden_units = args.hidden_units
epochs_given = args.epochs

if args.gpu:

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

else:

    device = torch.device('cpu')
    

train_dir = data_dir + '/train'
valid_dir = data_dir + '/valid'
test_dir = data_dir + '/test'


train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                   transforms.RandomResizedCrop(224),
                                   transforms.RandomHorizontalFlip(),
                                   transforms.ToTensor(),
                                   transforms.Normalize([0.485, 0.456, 0.406],
                                                        [0.229, 0.224, 0.225])])

test_transforms = transforms.Compose([transforms.Resize(255),
                                  transforms.CenterCrop(224),
                                  transforms.ToTensor(),
                                  transforms.Normalize([0.485, 0.456, 0.406],
                                                       [0.229, 0.224, 0.225])])
train_data = datasets.ImageFolder(data_dir + '/train', transform=train_transforms)
test_data = datasets.ImageFolder(data_dir + '/test', transform=test_transforms)
valid_data = datasets.ImageFolder(data_dir + '/valid', transform=test_transforms)

trainloader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True)
validloader = torch.utils.data.DataLoader(test_data, batch_size=64)
testloader = torch.utils.data.DataLoader(test_data, batch_size=64)

with open('cat_to_name.json', 'r') as f:
    cat_to_name = json.load(f)

# Use GPU if it's available
#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#call on pretrained model
if arch == 'resnet18':
    model = models.resnet18(pretrained=True)


# Freeze parameters so we don't backprop through them
for param in model.parameters():
    param.requires_grad = False

#create model architecture
classifier = nn.Sequential(nn.Linear(512, hidden_units),
                                 nn.ReLU(),
                                 nn.Dropout(0.2),
                                 nn.Linear(hidden_units, 102),
                                 nn.LogSoftmax(dim=1))

model.fc = classifier

criterion = nn.NLLLoss()

# Only train the classifier parameters, feature parameters are frozen
optimizer = optim.Adam(model.fc.parameters(), lr=lr)

#send model to GPU if available
model.to(device)

epochs=epochs_given
steps=0
running_loss = 0
print_every = 5

for epoch in range(epochs):
    for images, labels in trainloader:
        steps += 1

        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()

        logps = model(images)
        loss = criterion(logps, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        # TODO: Do validation on the test set
        if steps % print_every == 0:
            model.eval()
            test_loss = 0
            accuracy = 0

            for images, labels in validloader:

                images, labels = images.to(device), labels.to(device)

                logps=model(images)
                loss = criterion(logps, labels)
                test_loss += loss.item()

                ps = torch.exp(logps)
                top_ps, top_class = ps.topk(1, dim=1)
                equality = top_class == labels.view(*top_class.shape)
                accuracy += torch.mean(equality.type(torch.FloatTensor)).item()

            print(f"Epoch {epoch+1}/{epochs}..."
              f"Train loss: {running_loss/print_every:.3f}.. "
              f"Validation loss: {test_loss/len(validloader):.3f}.. "
              f"Validation accuracy: {accuracy/len(validloader):.3f}")

            running_loss=0
            model.train()

checkpoint={'arch':arch,
           'classifier':'classifier',
           'input_size': 512,
           'output_size': 102,
           'hidden_layers': hidden_units,
           'dropout': 0.2,
           'state_dict': model.state_dict(),
           'epochs': epochs,
           'learning_rate': lr,
           'idx_to_class': {v: k for k, v in train_data.class_to_idx.items()}

           }

torch.save(save_dir, 'checkpoint1.pth')
