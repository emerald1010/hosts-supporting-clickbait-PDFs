from itertools import combinations
import numpy as np
import timm
import torch
import torch.nn as nn
import pytorch_lightning as pl


from Utilities.Logger.Logger import Logger
import torch.nn.functional as F


class EmbeddingResnet(pl.LightningModule, Logger):
    """
    Attributes:

        DownBlock1,DownBlock2,DownBlock3,DownBlock4: nn.Conv2d
            Convolutional LAYERS

    """

    def __init__(self, img_channels: int = 3, n_features=128, dropout_p=.0, margin=1.):
        super().__init__()
        self.save_hyperparameters()

        self.resnet = timm.create_model('resnet18d', in_chans=img_channels, pretrained=True, num_classes=n_features)

        self.negative_samples = NegativeSamples(margin)
        self.loss = OnlineTripletLoss(margin, AllNegativeTripletSelector(margin))

    def forward(self, x):
        return self.resnet(x)

    def training_step(self, batch, batch_idx):
        samples, labels = batch
        features = self.forward(samples)
        loss_value = self.loss(features, np.array(labels))
        n_negative_samples = self.negative_samples(features, np.array(labels))
        self.log('training_loss', loss_value, on_step=True, on_epoch=True, batch_size=128, prog_bar=True)
        self.log('negative_samples', n_negative_samples, on_step=True, on_epoch=True, batch_size=128, prog_bar=True)
        return {"loss": loss_value}

    def validation_step(self, batch, batch_idx):
        samples, labels = batch
        features = self.forward(samples)
        loss_value = self.loss(features, np.array(labels))
        n_negative_samples = self.negative_samples(features, np.array(labels))
        self.log('validation_loss', loss_value, on_step=False, on_epoch=True, batch_size=128, prog_bar=True)
        self.log('negative_samples', n_negative_samples, on_step=True, on_epoch=True, batch_size=128, prog_bar=True)
        return {"val_loss": loss_value}

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.0001)


class DownBlock(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels, pooling_kernel=(2, 2), kernel_size=(3, 3), dropout_p=None):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(pooling_kernel),
            ConvBlock(in_channels, out_channels, kernel_size, dropout_p)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class ConvBlock(nn.Module):
    """(convolution => [BN] => ReLU)"""

    def __init__(self, in_channels, out_channels, kernel_size=(3, 3), dropout_p=None, padding=1):
        super().__init__()
        self.conv_block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.PReLU(),
        )

        if dropout_p:
            self.conv_block.add_module("dropout", nn.Dropout(dropout_p))

    def forward(self, x):
        return self.conv_block(x)


class OnlineTripletLoss(nn.Module):
    """
    Online Triplets' loss
    Takes a batch of embeddings and corresponding labels.
    Triplets are generated using triplet_selector object that take embeddings and targets and return indices of
    triplets
    """

    def __init__(self, margin, triplet_selector):
        super(OnlineTripletLoss, self).__init__()
        self.margin = margin
        self.triplet_selector = triplet_selector

    def forward(self, embeddings, target):
        triplets = self.triplet_selector.get_triplets(embeddings, target)

        if len(triplets) == 0:
            triplets.append([0, 0, 0])

        triplets = np.array(triplets)

        triplets = torch.LongTensor(triplets)

        if embeddings.is_cuda:
            triplets = triplets.cuda()

        ap_distances = (embeddings[triplets[:, 0]] - embeddings[triplets[:, 1]]).pow(2).sum(1)  # .pow(.5)
        an_distances = (embeddings[triplets[:, 0]] - embeddings[triplets[:, 2]]).pow(2).sum(1)  # .pow(.5)
        losses = F.relu(ap_distances - an_distances + self.margin)
        return losses.mean()


class NegativeSamples(nn.Module):
    """
    Computes the number of hard examples per minibatch
    """

    def __init__(self, margin):
        super(NegativeSamples, self).__init__()
        self.margin = margin
        self.triplet_selector = AllNegativeTripletSelector(margin)

    def forward(self, embeddings, target):
        triplets = self.triplet_selector.get_triplets(embeddings, target)
        return len(triplets)


class TripletSelector:
    """

    Implementation of the triplet selector class from here:
    https://github.com/adambielski/siamese-triplet/blob/master/utils.py

    Implementation should return indices of anchors, positive and negative samples
    return np array of shape [N_triplets x 3]
    """

    def __init__(self):
        pass

    def get_triplets(self, embeddings, labels):
        raise NotImplementedError


class FunctionNegativeTripletSelector(TripletSelector):
    """
    For each positive pair, takes the hardest negative sample (with the greatest triplet loss value) to create a triplet
    Margin should match the margin used in triplet loss.
    negative_selection_fn should take array of loss_values for a given anchor-positive pair and all negative samples
    and return a negative index for that pair
    """

    def __init__(self, margin, negative_selection_fn, cpu=True):
        super(FunctionNegativeTripletSelector, self).__init__()
        self.cpu = cpu
        self.margin = margin
        self.negative_selection_fn = negative_selection_fn

    def get_triplets(self, embeddings, labels):
        if self.cpu:
            embeddings = embeddings.cpu()
        distance_matrix = pdist(embeddings)
        distance_matrix = distance_matrix.cpu()

        if type(labels) == list:
            labels = np.array(labels)

        triplets = []

        for label in set(labels):
            label_mask = (labels == label)
            label_indices = np.where(label_mask)[0]

            if len(label_indices) < 2:
                continue

            negative_indices = np.where(np.logical_not(label_mask))[0]
            anchor_positives = list(combinations(label_indices, 2))  # All anchor-positive pairs
            anchor_positives = np.array(anchor_positives)

            ap_distances = distance_matrix[anchor_positives[:, 0], anchor_positives[:, 1]]
            for anchor_positive, ap_distance in zip(anchor_positives, ap_distances):
                loss_values = ap_distance - distance_matrix[
                    torch.LongTensor(np.array([anchor_positive[0]])), torch.LongTensor(negative_indices)] + self.margin
                loss_values = loss_values.data.cpu().numpy()
                hard_negatives = self.negative_selection_fn(loss_values)
                if hard_negatives is not None:
                    for hard_negative_index in hard_negatives:
                        hard_negative_sample = negative_indices[hard_negative_index]
                        triplets.append([anchor_positive[0], anchor_positive[1], hard_negative_sample])

        return triplets


def random_hard_negative(loss_values):
    hard_negatives = np.where(loss_values > 0)[0]
    negative_loss_values = loss_values[hard_negatives]

    # p = np.asarray(negative_loss_values).astype('float64')
    # p /= p.sum().astype('float64')

    return [np.random.choice(hard_negatives)] if len(hard_negatives) > 0 else None


def all_negatives(loss_values):
    all_negatives = np.where(loss_values > 0)[0]

    return all_negatives.tolist() if len(all_negatives) > 0 else None


def hardest_negative(loss_values):
    hard_negative = np.argmax(loss_values)
    return hard_negative if loss_values[hard_negative] > 0 else None


def RandomNegativeTripletSelector(margin, cpu=False): return FunctionNegativeTripletSelector(margin=margin,
                                                                                             negative_selection_fn=random_hard_negative,
                                                                                             cpu=cpu)


def AllNegativeTripletSelector(margin, cpu=False): return FunctionNegativeTripletSelector(margin=margin,
                                                                                          negative_selection_fn=all_negatives,
                                                                                          cpu=cpu)


def pdist(vectors):
    distance_matrix = -2 * vectors.mm(torch.t(vectors)) + vectors.pow(2).sum(dim=1).view(1, -1) + vectors.pow(2).sum(
        dim=1).view(-1, 1)
    return distance_matrix
