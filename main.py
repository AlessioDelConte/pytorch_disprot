import os
import shutil
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
from matplotlib import pyplot as plt
from ray import air, tune
from ray.tune.schedulers import ASHAScheduler
from sklearn import metrics
from torch import optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset.disprot_dataset import DisprotDataset, Sequence, collate_fn
from dataset.utils import PadRightTo


class Net(nn.Module):
    def __init__(self, in_features, channels=None, kernels=None):
        super().__init__()

        def conv_out_len(layer, length_in):
            return (length_in + 2 * layer.padding[0] - layer.dilation[0] * (layer.kernel_size[0] - 1) - 1) // \
                   layer.stride[0] + 1

        if not channels:
            channels = [in_features, 70, 60, 50, 30, 20, 10, 5, 3, 1]
        else:
            channels = [in_features] + channels + [1]

        if not kernels:
            kernel_sizes = [81, 51, 41, 31, 21, 11, 7, 5, 1]
        else:
            kernel_sizes = kernels + [1]

        paddings = [int((k_size - 1) / 2) for k_size in kernel_sizes]

        assert len(channels) - 1 == len(kernel_sizes) == len(paddings)

        self.conv_layers = nn.ModuleList([
                nn.Conv1d(channels[i], c_size, kernel_size=k_size, padding=p_size) for i, (k_size, c_size, p_size) in
                enumerate(zip(kernel_sizes, channels[1:], paddings))
        ])

        # Every 2 convolution layers, we add a batch normalization layer
        self.bn_layers = dict()
        # self.bn_layers = {l: nn.BatchNorm1d(l.out_channels) for l in self.conv_layers[1::2] if l.out_channels != 1}

        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU()

    def forward(self, x):
        # Cycle through convolution layers except last one, that has to be used with a different activation function
        for i, layer in enumerate(self.conv_layers[:-1]):
            # Add batch normalization layer every 2 convolution layers, except last one
            if layer in self.bn_layers:
                x = self.relu(self.bn_layers[layer](layer(x)))
            else:
                x = self.relu(layer(x))
        # Last convolution layer with the sigmoid activation function
        x = self.sigmoid(self.conv_layers[-1](x))
        x = x.flatten(start_dim=1)
        return x


def trim_padding_and_flat(sequences: List[Sequence], pred):
    all_target = np.array([])
    all_trimmed_pred = np.array([])
    for i, seq in enumerate(sequences):
        tmp_pred = pred[i][:len(seq)].cpu().detach().numpy()
        if seq.clean_target is not None:
            all_target = np.concatenate([all_target, seq.clean_target])
        all_trimmed_pred = np.concatenate([all_trimmed_pred, tmp_pred])
    return all_target, all_trimmed_pred


def batch_auc(sequences: List[Sequence], pred):
    target, pred = trim_padding_and_flat(sequences, pred)
    fpr, tpr, thresholds = metrics.roc_curve(target, pred, pos_label=1)
    auc = metrics.auc(fpr, tpr)
    return auc


def plot_auc_and_loss(train_losses, test_losses, test_aucs, epoch, title="AUC and Loss"):
    plt.close('all')
    hundred_epochs = int((epoch + 2) / 100)
    x_size = 8.5 + hundred_epochs * 2.5
    fig, ax1 = plt.subplots(figsize=(x_size, 7.5))

    x_test = np.arange(1, epoch + 2)
    # train_losses = train_losses.reshape(-1, 4).mean(axis=1)
    x_train = np.linspace(0, epoch + 1, len(train_losses))

    ax1.plot(x_train, train_losses, color='slategrey', linewidth=1, marker='o', markersize=2, label='Train Loss')
    ax1.plot(x_test, test_losses, color='dodgerblue', linewidth=1, marker='o', markersize=2, label='Test Loss')
    max_ticks = 22 * hundred_epochs if hundred_epochs > 0 else 22
    ax1.set_xticks(np.linspace(0, epoch + 1, max_ticks, dtype=int))
    plt.xticks(rotation=90)
    ax1.tick_params(axis='y', color='slategrey', labelcolor='slategrey')
    ax1.set_ylabel('Loss (MSE)', color='slategrey')
    ax1.set_xlabel('Time (epochs)')
    ax1.set_yscale('log')

    ax2 = ax1.twinx()
    ax2.plot(x_test, test_aucs, color='orange', linewidth=1, marker='o', markersize=2, label='Test AUC')
    ax2.tick_params(axis='y', color='orange', labelcolor='orange')
    ax2.set_yticks(np.linspace(0, 1, 11))
    ax2.set_ylabel('AUC', color='orange')
    # Set the minimum y-axis value to 0.0 and maximum y-axis value to 1.0 (AUC is between 0.0 and 1.0)
    ax2.set_ylim(0.0, 1.0)
    ax2.grid(True, which='major', axis='y', linestyle='dotted')

    plt.title(title)
    fig.legend(ncol=1, bbox_to_anchor=(0, 0, 1, 1), bbox_transform=ax1.transAxes)

    plt.tight_layout()
    plt.show()


# Function that get the results from the model on the test set and plot the ROC curve
def plot_roc_curve(model, data_loader, device, set='Test'):
    model.eval()
    all_output, all_target = np.array([]), np.array([])
    with torch.no_grad():
        for sequences, data, target in data_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)

            target, output = trim_padding_and_flat(sequences, output)
            all_target = np.concatenate([all_target, target])
            all_output = np.concatenate([all_output, output])

    fpr, tpr, thresholds = metrics.roc_curve(all_target, all_output, pos_label=1)
    auc = metrics.auc(fpr, tpr)
    fig, ax = plt.subplots(figsize=(7.5, 7.5))
    r = np.linspace(0, 1, 1000)
    fs = np.mean(np.array(np.meshgrid(r, r)).T.reshape(-1, 2), axis=1).reshape(1000, 1000)
    cs = ax.contour(r[::-1], r, fs, levels=np.linspace(0.1, 1, 10), colors='silver', alpha=0.7, linewidths=1,
                    linestyles='--')
    ax.clabel(cs, inline=True, fmt='%.1f', fontsize=10, manual=[(l, 1 - l) for l in cs.levels[:-1]])
    ax.plot(fpr, tpr, color='orange', linewidth=1, label=f'{set} AUC = %0.3f' % auc)
    ax.plot([0, 1], [0, 1], color='k', linestyle='--')
    ax.set_xlabel("FPR")
    ax.set_ylabel("TPR")
    plt.legend(loc='lower right')
    plt.title(f'ROC Curve for {set} Set')
    plt.show()
    return fpr, tpr, thresholds


# To get the loss we cut the output and target to the length of the sequence, removing the padding.
# This helps the network to focus on the actual sequence and not the padding.
def get_loss(sequences, output, criterion, device) -> torch.Tensor:
    loss = 0.0
    # Cycle through the sequences and accumulate the loss, removing the padding
    for i, seq in enumerate(sequences):
        seq_loss = criterion(output[i][:len(seq)], torch.tensor(seq.clean_target, device=device, dtype=torch.float))
        loss += seq_loss
    # Return the average loss over the sequences of the batch
    return loss / len(sequences)


def train(model, train_loader, optimizer, criterion, device, epoch):
    model.train()
    optimizer.zero_grad(set_to_none=True)
    running_loss = 0.0
    losses = np.array([])
    for batch_idx, (sequences, data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        output = model(data)
        loss = get_loss(sequences, output, criterion, device)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        running_loss += loss.item() * data.size(0)
        losses = np.append(losses, [loss.item()])
        # if batch_idx == len(train_loader) - 1:
        #     print('\nTrain Epoch: {} [{:4d}/{} ({:2.0f}%)] Loss: {:.3f}'.format(
        #             epoch + 1, batch_idx * len(data), len(train_loader.dataset),
        #             100. * batch_idx / len(train_loader), loss.item()))

    epoch_loss = running_loss / len(train_loader.dataset)
    return epoch_loss, losses


def test(model, test_loader, criterion, device):
    model.eval()
    test_loss = 0
    test_auc = 0
    with torch.no_grad():
        for sequences, data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += get_loss(sequences, output, criterion, device).cpu()
            test_auc += batch_auc(sequences, output) * data.size(0)
    test_loss /= len(test_loader)
    test_auc /= len(test_loader.dataset)
    return test_loss, test_auc


def predict_one_sequence(model, sequence: Sequence, n_features, device):
    model.eval()
    data = sequence.data.reshape(1, n_features, -1).to(device)
    output = model(data)
    _, output = trim_padding_and_flat([sequence], output)
    return output


def caid_format(net, device, dataset, n_features, idx=0):
    sequence: Sequence = dataset[idx]
    prediction = predict_one_sequence(net, sequence, device, n_features)
    for idx, (aa, pred) in enumerate(zip(sequence.sequence, prediction)):
        print(f'{idx + 1:3d}\t{aa}\t{pred:.3f}')


def pred_heatmap(net, device, dataset, idx=0):
    sequence: Sequence = dataset[idx]
    prediction = predict_one_sequence(net, sequence, device)
    x = np.vstack((sequence.clean_target, prediction))
    fig, ax = plt.subplots(figsize=(12, 4))
    cmap = sns.light_palette("seagreen", as_cmap=True)
    sns.heatmap(x, vmin=0, vmax=1, cmap=cmap)
    ax.set_yticklabels(["Truth", "Prediction"])
    plt.show()


def load_data(data_dir, use_pssm, use_spd3, use_pca):
    print("Loading data...")
    # Load the data
    train_data = pd.read_json(data_dir / "dataset/disorder_train.json", orient='records', dtype=False)
    test_data = pd.read_json(data_dir / "dataset/disorder_test.json", orient='records', dtype=False)

    pad = PadRightTo(4000)

    # Defining the dataset
    train_disorder = DisprotDataset(data=train_data, feature_root=data_dir / 'features',
                                    pssm=use_pssm, spd3=use_spd3, transform=pad, target_transform=pad, pca=use_pca)
    test_disorder = DisprotDataset(data=test_data, spd3=use_spd3, feature_root=data_dir / 'features',
                                   pssm=use_pssm, transform=pad, target_transform=pad, pca=use_pca)

    print("Data loaded.")

    return train_disorder, test_disorder


def load_caid_data(caid_dir, use_pssm, use_spd3, use_pca):
    print("Loading CAID data...")
    data = []
    pad = PadRightTo(4000)
    for fasta in tqdm(caid_dir.glob("fastas/*.fasta")):
        with open(fasta, 'r') as f:
            for line in f:
                if line.startswith(">"):
                    seq_id = line[1:].split()[0].strip()
                    sequence = f.readline().strip()
                    data.append(Sequence(seq_id=seq_id, sequence=sequence, features_path=caid_dir, load_pssm=use_pssm,
                                         load_spd3=use_spd3, pca=use_pca, data_transform=pad, target_transform=pad))
    return data


def get_feature_num(config):
    return 1 + (4 if config["pca"] else 20) * config["use_pssm"] + (5 if config["pca"] else 12) * config["use_spd3"]


def train_disprot(config):
    n_features = get_feature_num(config)
    train_epochs = 1000

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Using device:', device)

    train_disorder, test_disorder = load_data(Path("/home/alessio/pytorch_disprot/data"), config["use_pssm"],
                                              config["use_spd3"], config["pca"])

    # Defining the dataloader for the training set and the test set
    train_loader = DataLoader(train_disorder, batch_size=config["batch_size"], shuffle=True, num_workers=8,
                              collate_fn=collate_fn, pin_memory=True)
    test_loader = DataLoader(test_disorder, batch_size=340, shuffle=True, num_workers=8, collate_fn=collate_fn,
                             pin_memory=True)

    channels, kernels = None, None
    if "channels_kernels" in config:
        channels, kernels = config["channels_kernels"]

    net = Net(in_features=n_features, channels=channels, kernels=kernels).to(device)
    # Instantiate the model
    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        net = nn.DataParallel(net).to(device)

    # Define the loss function and the optimizer
    criterion = nn.MSELoss(reduction='mean')

    # optimizer = optim.SGD(net.parameters(), lr=1e-6, momentum=0.9)
    optimizer = optim.Adam(net.parameters(), lr=config["lr"])

    all_train_loss, all_test_loss, all_test_aucs = np.array([]), np.array([]), np.array([])
    for epoch in range(train_epochs):
        epoch_mean_loss, losses = train(net, train_loader, optimizer, criterion, device, epoch)
        all_train_loss = np.append(all_train_loss, epoch_mean_loss)

        test_loss, test_auc = test(net, test_loader, criterion, device)
        all_test_loss = np.append(all_test_loss, [test_loss])
        all_test_aucs = np.append(all_test_aucs, [test_auc])

        with tune.checkpoint_dir(epoch) as checkpoint_dir:
            path = os.path.join(checkpoint_dir, 'checkpoint.pt')
            torch.save((net.state_dict(), optimizer.state_dict()), path)

        tune.report(train_loss=epoch_mean_loss, test_loss=test_loss.item(), test_auc=test_auc)

    print("Training finished")


def create_channels(i):
    return [int(np.random.uniform(10, 50)) for _ in range(i)]


def create_kernels(i):
    k = []
    for _ in range(i):
        x = int(np.random.uniform(30, 100))
        if x % 2 == 0:
            x += 1
        k.append(x)
    return k


def main(num_samples, max_num_epochs=300, gpus_per_trial=1):

    # Performance tuning
    # torch.multiprocessing.set_sharing_strategy('file_system')
    torch.backends.cudnn.benchmark = True
    ######

    rng = np.random.default_rng()
    search_space = {
            "lr"              : tune.loguniform(5e-6, 1e-5),
            "batch_size"      : tune.randint(70, 100),
            "pca"             : False,
            "use_pssm"        : True,
            "use_spd3"        : True,
            "channels_kernels": tune.choice([(create_channels(i), create_kernels(i)) for i in range(5, 10)]),
    }

    scheduler = ASHAScheduler(
            metric="test_auc",
            mode="max",
            max_t=max_num_epochs,
            grace_period=5,
    )

    reporter = tune.CLIReporter(
            metric_columns=["train_loss", "test_loss", "test_auc", "training_iteration"],
    )

    resources_per_trial = {"cpu": 8, "gpu": gpus_per_trial}

    tuner = tune.Tuner(
            tune.with_resources(
                    train_disprot,
                    resources=resources_per_trial
            ),
            tune_config=tune.TuneConfig(
                    num_samples=num_samples,
                    scheduler=scheduler,
            ),
            run_config=air.RunConfig(
                    name="disorder_in_disprot",
                    local_dir="./runs",
                    checkpoint_config=air.CheckpointConfig(
                            checkpoint_score_attribute="test_auc",
                            num_to_keep=3,
                    ),
                    progress_reporter=reporter,
            ),
            param_space=search_space,

    )

    results = tuner.fit()

    best_trial = results.get_best_result(metric="test_auc", mode="max")
    print("Best trial config:", best_trial.config)
    print("Best trial final test loss:", best_trial.metrics["test_loss"])
    print("Best trial final test auc:", best_trial.metrics["test_auc"])

    # best_n_features = get_feature_num(best_trial.config)

    # best_trained_model = Net(best_n_features)

    best_checkpoint_pth = os.path.join(best_trial.checkpoint.to_directory(), "checkpoint.pt")
    # model_state, optimizer_state = torch.load(best_checkpoint_pth)
    # best_trained_model.load_state_dict(model_state)

    shutil.copyfile(best_checkpoint_pth, "./models/best_model_2.pt")
    best_trial.metrics_dataframe.to_csv("./models/best_model_metrics_2.csv")

    # device = "cpu"
    # if torch.cuda.is_available():
    #     device = "cuda"
    #     if gpus_per_trial > 1:
    #         best_trained_model = nn.DataParallel(best_trained_model)
    # best_trained_model.to(device)

    # pred_heatmap(best_trained_model, device, )


if __name__ == '__main__':
    main(num_samples=50, max_num_epochs=300, gpus_per_trial=1)

# if __name__ == '__main__':
#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#     print("Device:", device)
#
#     config = {
#             "lr"              : 9e-6,
#             "batch_size"      : 90,
#             "use_pssm"        : True,
#             "use_spd3"        : True,
#             "pca"             : False,
#             "channels_kernels": ([13, 39, 21, 14, 40, 22], [31, 91, 33, 59, 87, 37]),
#     }
#
#     best_n_features = get_feature_num(config)
#     channels, kernels = config["channels_kernels"]
#     best_trained_model = Net(best_n_features, channels, kernels).to(device)
#     model_state_dict, optimizer = torch.load("./models/best_model.pt")
#     best_trained_model.load_state_dict(model_state_dict)
#
#     _, test_disorder = load_data(Path("./data"), config["use_pssm"], config["use_spd3"], config["pca"])
#     test_loader = DataLoader(test_disorder, batch_size=config["batch_size"], shuffle=False, num_workers=0, collate_fn=collate_fn)
#     tpr, fpr, thr = plot_roc_curve(best_trained_model, test_loader, device)
#
#     gmeans = np.sqrt(tpr * (1 - fpr))
#     # locate the index of the largest g-mean
#     ix = np.argmax(gmeans)
#     best_threshold = thr[ix]
#
#     data = load_caid_data(Path("./caid"), use_pssm=config["use_pssm"], use_spd3=config["use_spd3"], use_pca=config["pca"])
#
#     with open(f"caid/ConvSPRITZ.caid", 'w') as f:
#         for seq in data:
#             prediction = predict_one_sequence(best_trained_model, seq, best_n_features, device)
#             f.write(f">{seq.seq_id}\n")
#             for idx, (aa, pred) in enumerate(zip(seq.sequence, prediction)):
#                 f.write(f'{idx + 1}\t{aa}\t{pred:.3f}\t{int(pred > best_threshold)}\n')
