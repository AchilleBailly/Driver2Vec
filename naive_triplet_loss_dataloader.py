# %%
from cmath import nan
import matplotlib.pyplot as plt
from tsne_torch import TorchTSNE as TSNE
import lightgbm as lgbm
import os
import pandas as pd
import pywt
import numpy as np
import torch
import torch.nn as nn
from torch.nn.utils import weight_norm
from torch.utils.data import DataLoader
from tqdm.notebook import tqdm
from math import nan

from haar_part_other_group import HaarWavelet


use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")
#torch.backends.cudnn.benchmark = True


# %% [markdown]
# Temporal Convolution Network

# The following classes implement the TCN as explained in the paper ["Temporal Convolutional Networks: A Unified Approach to Action Segmentation"](https://link.springer.com/chapter/10.1007/978-3-319-49409-8_7).

# %%


class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()


class TemporalBlock(nn.Module):
    def __init__(
            self,
            n_inputs,
            n_outputs,
            kernel_size,
            stride,
            dilation,
            padding,
            dropout=0.2):
        super(TemporalBlock, self).__init__()
        self.conv1 = weight_norm(
            nn.Conv1d(
                n_inputs,
                n_outputs,
                kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation))
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = weight_norm(
            nn.Conv1d(
                n_outputs,
                n_outputs,
                kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation))
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(
            self.conv1,
            self.chomp1,
            self.relu1,
            self.dropout1,
            self.conv2,
            self.chomp2,
            self.relu2,
            self.dropout2)

        self.downsample = nn.Conv1d(
            n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()
        self.init_weights()

    def init_weights(self):
        nn.init.kaiming_uniform_(self.conv1.weight)
        nn.init.kaiming_uniform_(self.conv2.weight)
        if self.downsample is not None:
            nn.init.kaiming_uniform_(self.downsample.weight)

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return out + res


class TemporalConvNet(nn.Module):
    def __init__(self, num_inputs, num_channels, kernel_size=2, dropout=0.2):
        super(TemporalConvNet, self).__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = num_inputs if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            layers += [TemporalBlock(in_channels,
                                     out_channels,
                                     kernel_size,
                                     stride=1,
                                     dilation=dilation_size,
                                     padding=(kernel_size-1) * dilation_size,
                                     dropout=dropout)]

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        # the data should have  dimension (N,C,L) where N is the batch size,
        # C is the number of channels and L the in input_length
        # this is the same as input dims for the nn.Conv1d
        return self.network(x)

# %% [markdown]
# Haar Wavelet Transorm

# The following two blocks implement the second part of the Driver2Vec architecture, related to the Haar wavelet transorm. Its aim is to capture spectral components of the inputs.

# %%


def reference_transform(tensor):
    """
    Apply the Haar wavelet transform to a tensor

    input:
        tensor: a tensor with dimension (N, C, L), with N batch size, C number of channels and L the input length

    output:
        a tensor with dimensions (N, C, L) where the the two output channels of the transform are concatenated along the L dimension
    """
    array = tensor.numpy()
    out1, out2 = pywt.dwt(array, "haar")
    out1 = torch.from_numpy(out1)
    out2 = torch.from_numpy(out2)

    # concatenate each channel to be able to concatenate it to the untransformed data
    # everything will then be split when fed to the network
    return torch.cat((out1, out2), -1)


# %%
class WaveletPart(nn.Module):
    """
    Module to map the (N, C, L) output of the Haar transform to a (N, 2*O) tensor
    """

    def __init__(self, input_length, input_size, output_size):
        """
        inputs:
            input_length: length of the initial sequence fed to the network
            input_size: size of the inputs of the FC layer 
            output_size: output size of the FC layer
        """
        super(WaveletPart, self).__init__()

        # used two different layers here as in the paper but in the github code, they are the same
        self.fc1 = nn.Linear(input_size, output_size)
        self.fc2 = nn.Linear(input_size, output_size)

        self.input_size = input_size
        self.input_length = input_length

        self.haar = reference_transform

    def init_weight(self):
        nn.init.kaiming_uniform_(self.fc1)
        nn.init.kaiming_uniform_(self.fc2)

    def forward(self, x):
        # split the wavelet transformed data along third dim
        # the data should have  dimension (N,C,L*2) where N is the batch size,
        # C is the number of channels and L the in input_length (*2 because of wavelet transforma concatenation)
        x1, x2 = torch.split(x, self.input_length//2, 2)

        # reshape everything to feed to the linear layer
        bsize = x.size()[0]
        x1 = self.fc1(x1.reshape((bsize, -1, 1)).squeeze())
        x2 = self.fc2(x2.reshape((bsize, -1, 1)).squeeze())
        x1 = x1.reshape(bsize, -1)
        x2 = x2.reshape(bsize, -1)
        return torch.cat((x1, x2), -1)

# %% [markdown]
# Full architecture

# The following class implements the full architecture of Driver2Vec
# %%


class Driver2Vec(nn.Module):
    def __init__(
            self,
            input_size,
            input_length,
            num_channels,
            output_size,
            kernel_size,
            dropout,
            do_wavelet=True,
            fc_output_size=15):
        super(Driver2Vec, self).__init__()

        self.tcn = TemporalConvNet(input_size,
                                   num_channels,
                                   kernel_size=kernel_size,
                                   dropout=dropout)
        self.wavelet = do_wavelet
        if self.wavelet:
            self.haar = WaveletPart(
                input_length, input_size*input_length//2, fc_output_size)

            linear_size = num_channels[-1] + fc_output_size*2
        else:
            linear_size = num_channels[-1]
        self.input_length = input_length

        self.input_bn = nn.BatchNorm1d(linear_size)
        self.linear = nn.Linear(linear_size, output_size)
        self.activation = nn.Sigmoid()

    def forward(self, inputs, print_temp=False):
        """Inputs have to have dimension (N, C_in, L_in*2)
        the base time series, and the two wavelet transform channel are concatenated along the third dim"""

        # split the inputs, in the last dim, first is the unchanged data, then
        # the wavelet transformed data
        input_tcn, input_haar = torch.split(inputs, self.input_length, 2)

        # feed each one to their corresponding network
        y1 = self.tcn(input_tcn)
        # for the TCN, only the last output element interests us
        y1 = y1[:, :, -1]

        if self.wavelet:
            y2 = self.haar(input_haar)

            out = torch.cat((y1, y2), 1)
        else:
            out = y1

        bsize = out.shape[0]
        if bsize > 1:  # issue when the batch size is 1, can't batch normalize it
            out = self.input_bn(out)
        else:
            out = out
        #out = self.linear(out)
        out = self.activation(out)

        if print_temp:
            print(out)

        return out

# %% [markdown]

# Hard Triplet Loss

# The following class implements the hard triplet loss. Hard means that the closest negative and furthest positive are choosen instead of random.


# %% [markdown]
# Datasets

# The following 3 classes are used to handle the dataset that we have.
# %%


def preprocess(df):
    return (
        df.drop(
            ["FOG",
             "FOG_LIGHTS",
             "FRONT_WIPERS",
             "HEAD_LIGHTS",
             "RAIN",
             "REAR_WIPERS",
             "SNOW",
             ], axis=1
        )
    )


class Dataset(torch.utils.data.Dataset):
    """
    This class is used to handle the train dataset to work witht the Triplet Loss.
    The __getitem__ method (to be used with a dataloader) returns the anchor, a random postive and a random negative for that anchor
    as well as the anchor's label.
    """

    def __init__(self, data, labels, input_length):
        self.data, self.labels = data, labels
        self.index = [i for i in range(len(self.data))]
        self.length = input_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        X_anchor = self.data[index]
        anchor_wvlt = reference_transform(X_anchor)
        y_anchor = self.labels[index]

        # create list of possible positive index samples
        positive_list = [
            i for i in self.index if self.labels[i] == y_anchor and i != index]
        positive_index = np.random.choice(positive_list)

        positive = self.data[positive_index]
        # get the wavelet transform of that sample (the two channels are concatenated along the last dim)
        positive_wvlt = reference_transform(positive)

        # same here
        negative_list = [
            i for i in self.index if self.labels[i] != y_anchor]
        negative_index = np.random.choice(negative_list)

        negative = self.data[negative_index]
        negative_wvlt = reference_transform(negative)

        # concatenate the data for the TCN and the haar wavelet transform
        # they will be split in the forward pass
        return torch.cat((X_anchor, anchor_wvlt), 1), \
            torch.cat((positive, positive_wvlt), 1), \
            torch.cat((negative, negative_wvlt), 1), \
            y_anchor

    @torch.no_grad()
    def get_classifier_data(self, model: nn.Module):
        data = []
        device = torch.device("cuda:0")
        for i in self.index:
            anchor = self[i][0].unsqueeze(0)
            anchor = anchor.to(device)
            embed = model(anchor)

            data.append(embed.cpu().detach().numpy().squeeze())

        data = np.array(data)
        return data, self.labels


class FromFiles:
    """
    Used to load the data from the files and split between test and train sets
    Also segments the 1000-length inputs as suggested in the original paper
    """

    def __init__(self, input_dir, input_length, seg_offset=40):
        self.input_length = input_length
        self.seg_offset = seg_offset

        self.raw_data, self.raw_labels = self._load_dataset(input_dir)
        self.seg_data, self.seg_labels = self._segment(
            input_length, seg_offset)
        self.index = [i for i in range(len(self.seg_data))]

    def _load_dataset(self, input_dir):
        x = []
        y = []
        for dir, _, files in os.walk(input_dir):
            for file in files:
                label = int(file.split("_")[1])-1
                df = pd.read_csv(dir + "/" + file, index_col=0)
                df = preprocess(df).to_numpy().transpose()
                x.append(torch.from_numpy(df).float()+1e-8)
                y.append(label)
        return x, y

    def _segment(self, input_length, offset):
        new_data, new_labels = [], []
        for i in range(len(self.raw_data)):
            cur_offset = 0
            data = self.raw_data[i].clone()
            label = self.raw_labels[i]
            while cur_offset + input_length < 1000:
                new_data.append(data[:, cur_offset:cur_offset+input_length])
                new_labels.append(label)
                cur_offset += offset

        return new_data, new_labels

    def split_train_test(self, ratio=0.8):
        x_test, y_test = [], []
        x_train, y_train = [], []

        for label in range(5):
            possible_list = [
                i for i in self.index if self.seg_labels[i] == label]
            number_to_select = int(len(possible_list)*ratio)
            train_list = set(np.random.choice(
                possible_list, number_to_select, replace=False))
            test_list = set(possible_list)-train_list

            for i in train_list:
                x_train.append(self.seg_data[i])
            for i in test_list:
                x_test.append(self.seg_data[i])

            y_train += [label] * len(train_list)
            y_test += [label] * len(test_list)

        return x_train, y_train, x_test, y_test

# %% [markdown]

# Custom Dataloader

# This custom dataset is useful to load bigger batches than the 19 sample sequences that we have.
# %%


class Dataloader():
    """Homemade dataloader for our needs in training
    This is different from the other one as it allows for "infinite" batches, even when the data only has 
    19 points. When setting bacht_size*number_batches bigger that the total number of points in the dataset,
    this dataloader will just loop again from the beginning."""

    def __init__(self, dataset: Dataset, batch_size: int, shuffle=True, number_batch=None):
        self.dataset = dataset
        self.b_size = batch_size
        self.n_batches = number_batch
        self.current_batch = 0
        self.shuffle = shuffle
        if number_batch == None:
            self.n_batches = len(self.dataset)//batch_size

    def __iter__(self):
        self.current_batch = 0
        self.index_list = [i % len(self.dataset) for i in range(
            max(self.b_size, len(self.dataset)))]
        if self.shuffle:
            np.random.shuffle(self.index_list)
        self.i = -1
        return self

    def __next__(self):
        dataset_length = len(self.dataset)
        if self.current_batch < dataset_length:
            a_out, p_out, n_out, l_out = [], [], [], []
            for _ in range(self.b_size):
                self.i = (self.i+1) % dataset_length
                a, p, n, l = self.dataset[self.index_list[self.i]]
                a_out.append(a)
                p_out.append(p)
                n_out.append(n)
                l_out.append(torch.Tensor([l]))
            self.current_batch += 1
            a_out = torch.stack(a_out)
            p_out = torch.stack(p_out)
            n_out = torch.stack(n_out)
            l_out = torch.stack(l_out)
            return a_out, p_out, n_out, l_out

        else:
            self.current_batch = 0
            raise StopIteration

# %% [markdown]
# The model

# The following code is the Driver2Vec model setup.


# %%
input_channels = 31
input_length = 300
channel_sizes = [25, 25]
output_size = 62
kernel_size = 16
dropout = 0.1
model = Driver2Vec(input_channels, input_length, channel_sizes, output_size,
                   kernel_size=kernel_size, dropout=dropout, do_wavelet=False)
model.to(device)

# %% [markdown]

# Next are the dataloader, loss and optimizer.

# %%

# datasets parameters
params = {'batch_size': 4,
          'shuffle': True,
          'num_workers': 1}


fromfiles = FromFiles("./dataset", input_length)
x_train, y_train, x_test, y_text = fromfiles.split_train_test()
training_set = Dataset(x_train, y_train, input_length)
training_generator = DataLoader(training_set, **params)

loss = nn.TripletMarginLoss(margin=1)
optimizer = torch.optim.Adam(
    model.parameters(), lr=0.00004, weight_decay=0.001)

# %% [markdown]

# Finally, here is the training loop.
# %%
epochs = 200


model.train()
for epoch in (pbar := tqdm(range(epochs))):
    loss_list = []
    dap_list = []
    dan_list = []
    for anchor, positive, negative, label in training_generator:
        anchor = anchor.to(device)
        positive = positive.to(device)
        negative = negative.to(device)

        optimizer.zero_grad()

        y_positive = model(positive)
        y_negative = model(negative)

        y_anchor = model(anchor)

        loss_value = loss(y_anchor, y_positive, y_negative)
        loss_int = float(loss_value) == nan
        dap_list.append((y_anchor-y_positive).norm(2).cpu().detach().numpy())
        dan_list.append((y_anchor-y_negative).norm(2).cpu().detach().numpy())
        loss_value.backward()

        optimizer.step()

        loss_list.append(loss_value.cpu().detach().numpy())
    #print("Epoch: {}/{} - Loss: {:.4f}".format(epoch+1, epochs, np.mean(loss_list)))
    pbar.set_description("Loss: %0.5g/%0.3g, Dap: %0.3g, Dan: %0.3g, Epochs" % (
        np.mean(loss_list), loss_int, np.mean(dap_list), np.mean(dan_list)))
# %% [markdown]

# LightGBM classifier

# The following is the setup and the training of the LightGBM classifier.
# %%

params1 = {'batch_size': 19,
           'shuffle': False,
           'num_workers': 2}

params2 = {'batch_size': 38,
           'shuffle': False,
           'number_batch': 1}

classifier_train_set = Dataset(x_train, y_train, input_length)

x_train_classifier, y_train_classifier = classifier_train_set.get_classifier_data(
    model)

lgb_train = lgbm.Dataset(x_train_classifier, y_train_classifier)

params = {
    'boosting_type': 'gbdt',
    'objective': 'multiclass',
    'num_class': 5,
    'metric': 'multi_logloss',
    'num_leaves': 32,
    'feature_fraction': 0.8,
    'bagging_fraction': 0.9,
    'max_depth': 8,
    'num_trees': 30,
    'verbose': 0,
    'min_data_in_leaf': 2  # May need to change that with a real test set
}

clf = lgbm.train(params, lgb_train)

# %% [markdown]

# Testing the classifier on the test set we made and also a part of the training set to evaluate it.
# %%

classifier_test_set = Dataset(x_test, y_text, input_length)

x_test_classifier, y_test_classifier = classifier_test_set.get_classifier_data(
    model)


y_pred = clf.predict(x_test_classifier)
print(y_pred, y_test_classifier)

# %%


def distance_matrix(tensor: torch.Tensor):
    """
    computes the distance matrix between each point of the input Tensor
    tensor should have dimension (L,D) where D is the dimension of the vectors"""

    res = torch.cdist(tensor, tensor)
    return res


# %% [markdown]

# T-SNE visualisation

# As in the original paper, we us t-SNE to visualise the embeddings. It can help to spot the issues in the model.
# %%
params = {'batch_size': 60,
          'shuffle': True,
          'number_batch': 1}

generator = Dataloader(training_set, **params)
x_tsne = []
y_tsne = []
for data, _, _, label in generator:
    data = data.to(device)
    embed = model(data)

print(embed.shape)

for i in range(embed.shape[0]):
    x_tsne.append(embed[i, :].cpu().detach().numpy().squeeze())
    y_tsne.append(int(label[i]))


# %%

X_emb = TSNE(n_components=2, perplexity=60, n_iter=10000,
             verbose=True).fit_transform(embed)


# %%

plt.scatter(X_emb[:, 0], X_emb[:, 1], marker="+", c=y_tsne)
plt.show()

# %%
