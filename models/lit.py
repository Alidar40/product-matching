import numpy as np
import torch
import torch.nn.functional as F
from torch import optim
from torch.nn import functional as F
import pytorch_lightning as pl
from torchmetrics.functional import f1_score, precision, recall
from pytorch_metric_learning import losses, miners, distances, reducers, regularizers
import wandb


class LitBase(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.save_hyperparameters()

    def log_metrics(self, loss, probs=None, target=None, mode="train"):
        if mode == "train":
            on_step = True
            on_epoch = False
        else:
            on_step = False
            on_epoch = True

        self.log(f"{mode}_loss", loss)
        if probs is not None and target is not None:
            self.log(f"{mode}/f1", f1_score(probs, target, threshold=0.5), on_step=on_step, on_epoch=on_epoch)
            self.log(f"{mode}/precision", precision(probs, target, threshold=0.5), on_step=on_step, on_epoch=on_epoch)
            self.log(f"{mode}/recall", recall(probs, target, threshold=0.5), on_step=on_step, on_epoch=on_epoch)

    def configure_optimizers(self):
        # optimizer = optim.Adam(self.parameters(), lr=1e-3)
        optimizer = optim.SGD(self.parameters(), lr=0.1, momentum=0.9)
        return optimizer
        # scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, "min")
        # return [optimizer], [{"scheduler": scheduler, "interval": "epoch", "monitor": "train_loss"}]


class LitSiamese(LitBase):
    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder

        # self.distance = distances.CosineSimilarity
        # self.distance = distances.DotProductSimilarity
        self.distance = distances.LpDistance

        # self.criterion = losses.TripletMarginLoss()
        # self.criterion = losses.ContrastiveLoss(distance=self.distance(), pos_margin=1, neg_margin=0)
        self.criterion = losses.ContrastiveLoss()

        self.miner = None
        # self.miner = miners.MultiSimilarityMiner()
        # self.miner = miners.UniformHistogramMiner(distance=self.distance())

    def forward(self, x):
        return self.encoder(x)

    def training_step(self, batch, batch_idx):
        sentence_embeddings, labels = batch
        encodings = self(sentence_embeddings)
        if self.miner is not None:
            hard_pairs = self.miner(encodings, labels)
            loss = self.criterion(encodings, labels, hard_pairs)
        else:
            loss = self.criterion(encodings, labels)
        self.log_metrics(loss, mode="train")
        return loss

    def validation_step(self, batch, batch_idx):
        sentence_embeddings, labels = batch
        encoding = self(sentence_embeddings)
        loss = self.criterion(encoding, labels)
        self.log_metrics(loss, mode="val")

    def test_step(self, batch, batch_idx):
        sentence_embeddings, labels = batch
        # sentence_embeddings = batch
        encoding = self(sentence_embeddings)
        return encoding
        # return encoding, labels

    def test_epoch_end(self, outputs):
        self.encodings = torch.cat(outputs)
