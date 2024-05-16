from datetime import datetime
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.tensorboard import SummaryWriter
from Model import Model
from torch.utils.data import DataLoader, Dataset
import pandas as pd
from tqdm import tqdm

class NamesDataset(Dataset):
    def __init__(self, csv_file, transform=None):
        self.data_frame = pd.read_csv(csv_file)
        self.transform = transform

    def __len__(self):
        return len(self.data_frame)

    def __getitem__(self, idx):
        name = self.data_frame.iloc[idx, 0]  # First column is the name
        gender = self.data_frame.iloc[idx, 1]  # Second column is the gender
        label = 0 if gender == 'MALE' else 1  # Convert gender to numerical labels
        sample = {'name': name, 'label': label}
        if self.transform:
            sample = self.transform(sample)
        return sample

class DataTransform:
    def __call__(self, sample):
        name = sample['name']
        # Convert name to ASCII values and pad/truncate to fixed length
        name_transformed = [ord(char) for char in name]
        max_length = 20  # Example fixed length
        if len(name_transformed) > max_length:
            name_transformed = name_transformed[:max_length]
        else:
            name_transformed += [0] * (max_length - len(name_transformed))  # Pad with zeros
        sample['name'] = torch.tensor(name_transformed, dtype=torch.float32)
        sample['label'] = torch.tensor(sample['label'], dtype=torch.long)
        return sample

def train():
    cuda = True if torch.cuda.is_available() else False
    device = torch.device('cpu')
    if cuda:
        device = torch.device('cuda')

    batch_size = 32
    load = False

    writer = SummaryWriter(log_dir='runs/' + "Name_Guesser " + datetime.now().strftime('%Y%m%d-%H%M'))

    # Define your data directory
    data_dir = 'Data'
    train_dir = os.path.join(data_dir, 'train')
    val_dir = os.path.join(data_dir, 'val')
    test_dir = os.path.join(data_dir, 'test')

    # Load the datasets
    trainset_female = NamesDataset(os.path.join(train_dir, 'female.csv'), transform=DataTransform())
    trainset_male = NamesDataset(os.path.join(train_dir, 'male.csv'), transform=DataTransform())

    valset_female = NamesDataset(os.path.join(val_dir, 'female.csv'), transform=DataTransform())
    valset_male = NamesDataset(os.path.join(val_dir, 'male.csv'), transform=DataTransform())

    testset_female = NamesDataset(os.path.join(test_dir, 'female.csv'), transform=DataTransform())
    testset_male = NamesDataset(os.path.join(test_dir, 'male.csv'), transform=DataTransform())

    # Concatenate the datasets for training, validation, and testing
    trainset = torch.utils.data.ConcatDataset([trainset_female, trainset_male])
    valset = torch.utils.data.ConcatDataset([valset_female, valset_male])
    testset = torch.utils.data.ConcatDataset([testset_female, testset_male])

    # Create data loaders
    train_loader = DataLoader(trainset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(valset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(testset, batch_size=batch_size, shuffle=False)

    input_size = 20  # Set input size according to your data preprocessing
    hidden_size = 64
    output_size = 2  # Binary classification

    model = Model.Model(input_size, hidden_size, output_size).to(device)

    loss_fn = nn.CrossEntropyLoss().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    epoch = 0

    if load:
        checkpoint = torch.load('checkpoints/chk1.pth')
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch = checkpoint['epoch'] + 1
        loss = checkpoint['loss']

    epochs = 200
    train_per_epoch = int(len(trainset) / batch_size)
    for e in range(epoch, epochs):
        loop = tqdm(enumerate(train_loader), total=len(train_loader), leave=True)
        for idx, batch in loop:
            inputs, labels = batch['name'], batch['label']
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = loss_fn(outputs, labels)
            loss.backward()
            optimizer.step()

            writer.add_scalar('loss', loss.item(), (e * train_per_epoch) + idx)
            predictions = outputs.argmax(dim=1, keepdim=True).squeeze()
            correct = (predictions == labels).sum().item()
            accuracy = correct / len(predictions)
            loop.set_description(f"Epoch [{e}/{epochs}")
            loop.set_postfix(loss=loss.item(), acc=accuracy)
            writer.add_scalar('acc', accuracy, (e * train_per_epoch) + idx)
        else:
            torch.save({
                'epoch': e,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss_fn
            }, 'checkpoints/chk' + str(e) + '.pth')

    num_correct = 0
    num_samples = 0
    model.eval()

    with torch.no_grad():
        for batch in test_loader:
            inputs, labels = batch['name'], batch['label']
            x, y = inputs.to(device), labels.to(device)


            scores = model(x)
            _, predictions = scores.max(1)
            num_correct += (predictions == y).sum()
            num_samples += predictions.size(0)
        acc = f'{float(num_correct) / float(num_samples) * 100:.2f}%'
        print(f'Dobio sam točnih {num_correct} od ukupno {num_samples} što čini točnost od ' + acc)

if __name__ == '__main__':
    train()




