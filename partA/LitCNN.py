from cnn import CNN

import torch
from torch import nn
from torch.nn import CrossEntropyLoss

from torchmetrics.functional import accuracy
import lightning as L


class LitCNN(L.LightningModule):
    def __init__(self,
                 input_dim = (3,256,256),
                 output_num_classes = 10,
                 activation_fn = 'ReLU',
                 num_layers = 5,
                 num_filters = 64,
                 filter_sizes = 3,
                 conv_padding = [0,0,0,0,0],
                 conv_strides = 1,
                 pooling_filter_sizes = 3,
                 pooling_strides = 1,
                 pooling_padding = [0,0,0,0,0],
                 num_dense_neurons = 128,
                 add_batchNorm = True,
                 add_dropout = True,
                 dl_dropout_prob = 0.5,
                 ap_dropout_prob=0.2,
                 lr=1e-4):
        super().__init__()
        self.cnn = CNN(
            input_dim,
            output_num_classes,
            activation_fn,
            num_layers,
            num_filters,
            filter_sizes,
            conv_padding,
            conv_strides,
            pooling_filter_sizes,
            pooling_strides,
            pooling_padding,
            num_dense_neurons,
            add_batchNorm,
            add_dropout,
            dl_dropout_prob,
            ap_dropout_prob,
        )
        self.loss = CrossEntropyLoss()

        self.lr = lr
        
        self.save_hyperparameters()
        
    def training_step(self,batch,batch_idx):
        _,loss,acc = self._get_preds_loss_accuracy(batch)

        #Log loss and metric
        self.log('train_loss',loss,sync_dist=True)
        self.log('train_accuracy',acc,sync_dist=True)

        # print("train_loss", loss)
        # print("train_accuracy",acc)

        return loss

    def test_step(self,batch,batch_idx):
        _, loss,acc = self._get_preds_loss_accuracy(batch)

        #Log loss and Metric
        self.log('test_loss',loss,sync_dist=True)
        self.log('test_accuracy',acc,sync_dist=True)
        
    def validation_step(self,batch,batch_idx):
        preds,loss,acc = self._get_preds_loss_accuracy(batch)

        # Log loss and metric
        self.log('val_loss', loss,sync_dist=True)
        self.log('val_accuracy',acc,sync_dist=True)

        return preds
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.cnn.parameters(),lr = self.lr)
        return optimizer

    def _get_preds_loss_accuracy(self,batch):
        images,labels = batch
        logits = self.cnn(images)
        preds = torch.argmax(logits,dim=1)
        loss = self.loss(logits, labels)
        acc = accuracy(preds,labels,'multiclass', num_classes = 10)
        return preds, loss, acc