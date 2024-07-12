import os
import sys
import json
import torch
import torch.nn as nn
import numpy as np
import torch.optim as optim
import pandas as pd
from torchvision import transforms, datasets
from tqdm import tqdm
from tensorboardX import SummaryWriter
from model import resnext50_32x4d
import torchvision.utils as vutils
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="PIL")

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("using {} device.".format(device))
    data_transform = {
        "train": transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]),
        "val": transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    }


    data_root = os.path.abspath(os.path.join(os.getcwd(), "../.."))
    image_path = os.path.join(data_root, '')
    assert os.path.exists(image_path), "{} path does not exist.".format(image_path)
    train_dataset = datasets.ImageFolder(root=os.path.join(image_path, "train"),
                                         transform=data_transform["train"])
    train_num = len(train_dataset)


    flower_list = train_dataset.class_to_idx
    cla_dict = dict((val, key) for key, val in flower_list.items())
    json_str = json.dumps(cla_dict, indent=4)
    with open('', 'w') as json_file:
        json_file.write(json_str)

    batch_size =
    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])
    print('Using {} dataloader workers every process'.format(nw))

    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=batch_size, shuffle=True,
                                               num_workers=nw)

    validate_dataset = datasets.ImageFolder(root=os.path.join(image_path, "val"),
                                            transform=data_transform["val"])
    val_num = len(validate_dataset)
    validate_loader = torch.utils.data.DataLoader(validate_dataset,
                                                  batch_size=batch_size, shuffle=False,
                                                  num_workers=nw)

    print("using {} images for training, {} images for validation.".format(train_num, val_num))


    net = resnext50_32x4d()
    model_weight_path = ""
    assert os.path.exists(model_weight_path), "file {} does not exist.".format(model_weight_path)
    net.load_state_dict(torch.load(model_weight_path, map_location='cpu'))

    in_channel = net.fc.in_features
    net.fc = nn.Linear(in_channel, 4)
    net.to(device)


    loss_function = nn.CrossEntropyLoss()
    params = [p for p in net.parameters() if p.requires_grad]
    optimizer = optim.Adam(params, lr=0.0001)

    epochs =
    best_acc = 0.001
    save_path = ''
    weights_dir = ''
    os.makedirs(weights_dir, exist_ok=True)
    train_steps = len(train_loader)

    writer = SummaryWriter(log_dir='')

    train_loss_history = []
    val_loss_history = []
    train_acc_history = []
    val_acc_history = []

    for epoch in range(epochs):

        net.train()
        running_loss = 0.0
        train_bar = tqdm(train_loader, file=sys.stdout)
        correct = 0
        total = 0
        for step, data in enumerate(train_bar):
            images, labels = data
            optimizer.zero_grad()
            logits = net(images.to(device))
            loss = loss_function(logits, labels.to(device))
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(logits.data, 1)
            total += labels.size(0)
            correct += (predicted == labels.to(device)).sum().item()

            train_bar.desc = "train epoch[{}/{}] loss:{:.3f}".format(epoch + 1, epochs, loss)

        train_accuracy = correct / total
        train_loss_history.append(running_loss / train_steps)
        train_acc_history.append(train_accuracy)

        net.eval()
        val_loss = 0.0
        acc = 0.0
        with torch.no_grad():
            val_bar = tqdm(validate_loader, file=sys.stdout)
            for val_data in val_bar:
                val_images, val_labels = val_data
                outputs = net(val_images.to(device))
                loss = loss_function(outputs, val_labels.to(device))
                val_loss += loss.item()

                predict_y = torch.max(outputs, dim=1)[1]
                acc += torch.eq(predict_y, val_labels.to(device)).sum().item()

                val_bar.desc = "valid epoch[{}/{}]".format(epoch + 1, epochs)

        val_loss /= len(validate_loader)
        val_accuracy = acc / val_num
        val_loss_history.append(val_loss)
        val_acc_history.append(val_accuracy)

        writer.add_scalar('Train Loss', running_loss / train_steps, epoch)
        writer.add_scalar('Validation Loss', val_loss, epoch)
        writer.add_scalar('Train Accuracy', train_accuracy, epoch)
        writer.add_scalar('Validation Accuracy', val_accuracy, epoch)

        if val_accuracy > best_acc:
            best_acc = val_accuracy
            torch.save(net.state_dict(), save_path)
            torch.save(net.state_dict(), os.path.join(weights_dir, f'resNet50_epoch_{epoch+1}.pth'))

    losses = pd.DataFrame({
        'train_loss': train_loss_history,
        'val_loss': val_loss_history
    })
    losses.to_csv('runs/loss.csv', index=False)

    acces = pd.DataFrame({
        'train_acc': train_acc_history,
        'val_acc': val_acc_history
    })
    acces.to_csv('runs/acc.csv', index=False)

    writer.close()
    print('Finished Training')

if __name__ == '__main__':
    main()

