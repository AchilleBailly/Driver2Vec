# %%
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

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
#torch.backends.cudnn.benchmark = True

# %% [markdown]
# ## Wavelet transform

# %%


def van_haar(data):
    ncol = data.shape[1]
    nrow = data.shape[0]
    for i in range(ncol):
        cur_col = data[:, i].copy()
        (cA, cD) = pywt.dwt(cur_col, 'haar')
        new_col = np.reshape(np.concatenate((cA, cD), 0), (nrow, 1))
        data = np.hstack((data, new_col))
    data = data.reshape(nrow, -1)
    return data

# van_haar(test.to_numpy()).shape

# %% [markdown]
# ## PyTorch

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
        nn.init.xavier_uniform(self.conv1.weight)
        nn.init.xavier_uniform(self.conv2.weight)
        if self.downsample is not None:
            nn.init.xavier_uniform(self.downsample.weight)

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


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

# %%


def reference_transform(tensor):
    array = tensor.numpy()
    out1, out2 = pywt.dwt(array, "haar")
    out1 = torch.from_numpy(out1)
    out2 = torch.from_numpy(out2)

    # concatenate each channel to be able to concatenate it to the untransformed data
    # everything will then be split when fed to the network
    return torch.cat((out1, out2), -1)


# %%
class WaveletPart(nn.Module):

    def __init__(self, input_length, input_size, output_size):
        super(WaveletPart, self).__init__()

        # used two different layers here as in the paper but in the github code, they are the same
        self.fc1 = nn.Linear(input_size, output_size)
        self.fc2 = nn.Linear(input_size, output_size)

        self.input_size = input_size
        self.input_length = input_length

        self.haar = reference_transform

    def init_weight(self):
        self.fc1.weight.data.normal_(0, 0.01)
        self.fc1.bias.data.normal_(0, 0.01)
        self.fc2.weight.data.normal_(0, 0.01)
        self.fc2.bias.data.normal_(0, 0.01)

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
        the base time series, and the two wavelet transform channel are concatenated along dim2"""

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
        out = self.linear(out)
        out = self.activation(out)

        if print_temp:
            print(out)

        return out

# %%


class HardTripletLoss():

    def __init__(self, device, margin=1.0):
        self.margin = margin
        self.device = device

    def _get_anchor_positive_triplet_mask(self, labels):
        labels_mat = labels.unsqueeze(0).repeat(labels.shape[0], 1)
        res = (labels_mat == labels_mat.T).int().to(self.device)
        return res

    def _get_anchor_negative_triplet_mask(self, labels):
        labels_mat = labels.unsqueeze(0).repeat(labels.shape[0], 1)
        res = (labels_mat != labels_mat.T).int().to(self.device)
        return res

    def _get_dist_matrix(self, embeddings):
        return torch.cdist(embeddings, embeddings).to(self.device)

    def __call__(self, embeddings, labels):
        """Build the triplet loss over a batch of embeddings.

        For each anchor, we get the hardest positive and hardest negative to form a triplet.

        Args:
            labels: labels of the batch, of size (batch_size,)
            embeddings: tensor of shape (batch_size, embed_dim)

        Returns:
            triplet_loss: scalar tensor containing the triplet loss
        """
        # Get the pairwise distance matrix
        pairwise_dist = self._get_dist_matrix(embeddings)

        # For each anchor, get the hardest positive
        # First, we need to get a mask for every valid positive (they should have same label)
        mask_anchor_positive = self._get_anchor_positive_triplet_mask(labels)

        # We put to 0 any element where (a, p) is not valid (valid if a != p and label(a) == label(p))
        anchor_positive_dist = mask_anchor_positive * pairwise_dist

        # shape (batch_size, 1)
        hardest_positive_dist, _ = torch.max(
            anchor_positive_dist, axis=1, keepdims=True)

        # For each anchor, get the hardest negative
        # First, we need to get a mask for every valid negative (they should have different labels)
        mask_anchor_negative = self._get_anchor_negative_triplet_mask(labels)

        # We add the maximum value in each row to the invalid negatives (label(a) == label(n))
        max_anchor_negative_dist, _ = torch.max(
            pairwise_dist, axis=1, keepdims=True)
        anchor_negative_dist = pairwise_dist + \
            max_anchor_negative_dist * (1.0 - mask_anchor_negative)

        # shape (batch_size,)
        hardest_negative_dist, _ = torch.min(
            anchor_negative_dist, axis=1, keepdims=True)

        # Combine biggest d(a, p) and smallest d(a, n) into final triplet loss
        triplet_loss = torch.max(hardest_positive_dist -
                                 hardest_negative_dist + self.margin, torch.zeros_like(hardest_negative_dist))

        # Get final mean triplet loss
        triplet_loss = torch.mean(triplet_loss)

        return triplet_loss


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


class TrainDataset(torch.utils.data.Dataset):
    'Characterizes a dataset for PyTorch'

    def __init__(self, data, labels, input_length=500):
        'Initialization'
        self.data, self.labels = data, labels
        self.index = [i for i in range(len(self.data))]
        self.length = input_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        max_length = self.data[index].shape[1]
        # int(np.random.uniform(0, max_length - self.length+1))
        start_pos_anchor = int(np.random.uniform(0, max_length - self.length+1))
        X_anchor = self.data[index][:,
                                    start_pos_anchor:start_pos_anchor+self.length]
        anchor_wvlt = reference_transform(X_anchor)
        y_anchor = self.labels[index]

        # concatenate the data for the TCN and the haar wavelet transform
        # they will be split in the forward pass
        return torch.cat((X_anchor, anchor_wvlt), 1), \
            y_anchor


class TestDataset(torch.utils.data.Dataset):
    """
    Is to be used with Dataloader for testing only
    """

    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        x = self.data[index]
        wvlt = reference_transform(x)
        out = torch.cat((x, wvlt), 1)
        return out, self.labels[index]


class FromFiles:
    """
    Used to load the data from the files and split between test and train sets
    """

    def __init__(self, input_dir, input_length):
        self.input_dir = input_dir
        self.input_length = input_length

        self.data, self.labels = self.__load_dataset()
        self.index = [i for i in range(len(self.data))]

    def __load_dataset(self):
        x = []
        y = []
        for dir, _, files in os.walk(self.input_dir):
            for file in files:
                label = int(file.split("_")[1])-1
                df = pd.read_csv(dir + "/" + file, index_col=0)
                df = preprocess(df).to_numpy().transpose()
                x.append(torch.from_numpy(df).float())
                y.append(label)
        return x, y

    def split_train_test(self):
        x_test, y_test = [], []
        for label in range(5):
            possible_list = [i for i in self.index if self.labels[i] == label]
            choosen_index = np.random.choice(possible_list)

            positive = self.data[choosen_index]
            max_length = positive.shape[1]
            train, test = torch.split(
                positive, [max_length-self.input_length, self.input_length], dim=1)

            x_test.append(test)
            y_test.append(label)
            self.data[choosen_index] = train

        return self.data, self.labels, x_test, y_test


# %%
class Dataloader():
    """Homemade dataloader for our needs in training
    This is different from the other one as it allows for "infinite" batches, even when the data only has 
    19 points. When setting bacht_size*number_batches bigger that the total number of points in the dataset,
    this dataloader will just loop again from the beginning."""

    def __init__(self, dataset: TrainDataset, batch_size: int, number_batch: int, shuffle=True):
        self.dataset = dataset
        self.b_size = batch_size
        self.n_batches = number_batch
        self.current_batch = 0
        self.shuffle = shuffle

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
            a_out, l_out = [], []
            for _ in range(self.b_size):
                self.i = (self.i+1) % dataset_length
                a, l = self.dataset[self.index_list[self.i]]
                a_out.append(a)
                l_out.append(l)
            self.current_batch += 1
            a_out = torch.stack(a_out)
            l_out = torch.Tensor(l_out)
            return a_out, l_out

        else:
            self.current_batch = 0
            raise StopIteration


# %%
input_channels = 31
input_length = 300
channel_sizes = [25,25]
output_size = 62
kernel_size = 16
dropout = 0.1
model = Driver2Vec(input_channels, input_length, channel_sizes, output_size,
                   kernel_size=kernel_size, dropout=dropout, do_wavelet=False)
model.to(device)

# %%

# # datasets parameters
# params = {'batch_size': 4,
#           'shuffle': True,
#           'num_workers': 1}


fromfiles = FromFiles("./dataset", input_length)
x_train, y_train, x_test, y_text = fromfiles.split_train_test()
training_set = TrainDataset(x_train, y_train, input_length)
training_generator = Dataloader(training_set, 20, 4)

loss = HardTripletLoss(device, margin=0.5)
optimizer = torch.optim.Adam(model.parameters(), lr=0.004, weight_decay=0.0001)

# %%
epochs = 50


model.train()
for epoch in (pbar := tqdm(range(epochs))):
    loss_list = []
    for anchor, label in training_generator:
        anchor = anchor.to(device)

        optimizer.zero_grad()

        y_anchor = model(anchor)

        loss_value = loss(y_anchor, label)
        loss_value.backward()

        optimizer.step()

        loss_list.append(loss_value.cpu().detach().numpy())
    #print("Epoch: {}/{} - Loss: {:.4f}".format(epoch+1, epochs, np.mean(loss_list)))
    pbar.set_description("Loss: %0.5g, Epochs" % (np.mean(loss_list)))

# %%

params1 = {'batch_size': 19,
           'shuffle': False,
           'num_workers': 2}

params2 = {'batch_size': 38,
           'shuffle': False,
           'number_batch': 1}

classifier_train_set = TrainDataset(x_test, y_text, input_length)
classifier_train_generator = DataLoader(training_set, **params1)

x_train_classifier = []
y_train_classifier = []
model.train(False)
for data, label in classifier_train_generator:
    data = data.to(device)
    embed = model(data)


for i in range(5):
    x_train_classifier.append(embed[i, :].cpu().detach().numpy().squeeze())
    y_train_classifier.append(int(label[i]))

x_train_classifier = np.array(x_train_classifier)
y_train_classifier = np.array(y_train_classifier)

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

# %%
params = {'batch_size': 5,
          'shuffle': False,
          'num_workers': 1}

classifier_test_set = TestDataset(x_test, y_text)
classifier_test_generator = DataLoader(classifier_test_set, **params)

x_test_classifier = []
y_test_classifier = []
for data, label in classifier_test_generator:
    data = data.to(device)
    embed = model(data)


for i in range(5):
    x_test_classifier.append(embed[i, :].cpu().detach().numpy().squeeze())
    y_test_classifier.append(int(label[i]))


y_pred = clf.predict(x_test_classifier)
print(y_pred, y_test_classifier)

# %%


def distance_matrix(tensor: torch.Tensor):
    """
    computes the distance matrix between each point of the input Tensor
    tensor should have dimension (L,D) where D is the dimension of the vectors"""

    res = torch.cdist(tensor, tensor)
    return res


# %%
params = {"batch_size": 19,
          "shuffle": False,
          "num_workers": 1}

generator = DataLoader(training_set, **params)
x_test_classifier = []
y_test_classifier = []
for data, label in generator:
    data = data.to(device)
    embed = model(data)


for i in range(19):
    x_test_classifier.append(embed[i, :].cpu().detach().numpy().squeeze())
    y_test_classifier.append(int(label[i]))


y_pred = clf.predict(x_test_classifier)
print(y_pred, y_test_classifier)

# %%

params = {'batch_size': 60,
          'shuffle': False,
          'number_batch': 1}

generator = Dataloader(training_set, **params)
x_tsne = []
y_tsne = []
for data, label in generator:
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
