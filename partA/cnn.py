import torch
from torch import nn
from torch.nn import CrossEntropyLoss
import torch.nn.functional as F
from torchvision import transforms, datasets
from torch.utils.data import DataLoader, random_split, Subset
from torchmetrics.functional import accuracy

class CNN(nn.Module):
    def __init__(self,
                 input_dim,
                 output_num_classes,
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
                 ap_dropout_prob=0.2):
        '''
        Params:
            input_shape: the shape of the input image
            output_num_classes: Number of classes in multiclass classification
            num_layers: Total number of "convolution - activation - pooling" layers
            num_filters: (int/list): Total number of filters in each conv layers
            filter_sizes: (int/list): Size of filters in each conv layer
            conv_padding: (int/list): Padding in each conv layer
            conv_strides: (int/list)
            pooling_filter_sizes: (int/list)
            pooling_strides: (int/list)
            pooling_padding: (int/list)
            dl_dropout_prob: Dropout probability in the dense layers in CNN architecture
            ap_dropout_prob: Dropout probability after the pooling layer in CNN architecture 
            add_batchNorm: Add Batch Normalisation in the architecture
        '''
        super().__init__()

        if isinstance(num_filters,int):
            self.num_filters = [num_filters] * num_layers
        elif isinstance(num_filters,list):
            self.num_filters = num_filters
        else:
            raise ValueError("num_filters should be either of type int or list")

        if isinstance(filter_sizes,int):
            self.filter_sizes = [filter_sizes] * num_layers
        elif isinstance(filter_sizes,list):
            self.filter_sizes = filter_sizes
        else:
            raise ValueError("filter_sizes should be either of type int or list")

        if isinstance(conv_padding,int):
            self.conv_padding = [conv_padding] * num_layers
        elif isinstance(conv_padding,list):
            self.conv_padding = conv_padding
        else:
            raise ValueError("conv_padding should be either of type int or list")

        if isinstance(conv_strides,int):
            self.conv_strides = [conv_strides] * num_layers
        elif isinstance(conv_strides,list):
            self.conv_strides = conv_strides
        else:
            raise ValueError("conv_strides should be either of type int or list")

        if isinstance(pooling_filter_sizes,int):
            self.pooling_filter_sizes = [pooling_filter_sizes] * num_layers
        elif isinstance(pooling_filter_sizes,list):
            self.conv_strides = pooling_filter_sizes
        else:
            raise ValueError("pooling_filter_sizes should be either of type int or list")

        if isinstance(pooling_strides,int):
            self.pooling_strides = [pooling_strides] * num_layers
        elif isinstance(pooling_strides,list):
            self.pooling_strides = pooling_strides
        else:
            raise ValueError("pooling_strides should be either of type int or list")

        if isinstance(pooling_padding,int):
            self.pooling_padding = [pooling_padding] * num_layers
        elif isinstance(pooling_padding,list):
            self.pooling_padding = pooling_padding
        else:
            raise ValueError("pooling_padding should be either of type int or list")

        self.activation_fn = None

        if activation_fn == 'ReLU':
            self.activation_fn = nn.ReLU
        elif activation_fn=='GELU':
            self.activation_fn = nn.GELU
        elif activation_fn == 'SiLU':
            self.activation_fn = nn.SiLU
        elif activation_fn == 'Mish':
            self.activation_fn = nn.Mish
        else: 
            raise ValueError(f"{activation_fn} is not supported")
        
        layers = []
        dimensions = input_dim
        for i in range(num_layers):
            if i!=0 and add_dropout:
                layers.append(nn.Dropout(p=ap_dropout_prob))
            
            layers.append(nn.Conv2d(dimensions[0],self.num_filters[i],self.filter_sizes[i],self.conv_strides[i],self.conv_padding[i]))
            
            if add_batchNorm:
                layers.append(nn.BatchNorm2d(self.num_filters[i]))
            
            height = (dimensions[1] + 2 * self.conv_padding[i] - self.filter_sizes[i])//self.conv_strides[i] + 1
            width = (dimensions[2] + 2 * self.conv_padding[i] - self.filter_sizes[i])//self.conv_strides[i] + 1
            dimensions = (self.num_filters[i],height,width)

            layers.append(self.activation_fn())
            layers.append(nn.MaxPool2d(self.pooling_filter_sizes[i],self.pooling_strides[i], self.pooling_padding[i]))

            height = (dimensions[1] + 2 * self.pooling_padding[i] - self.pooling_filter_sizes[i])//self.pooling_strides[i] + 1
            width = (dimensions[2] + 2 * self.pooling_padding[i] - self.pooling_filter_sizes[i])//self.pooling_strides[i] + 1
            dimensions = (self.num_filters[i],height,width)


        self.features = nn.Sequential(
            *layers,
            nn.Flatten()
        )

        classifier_layers = []
        if add_dropout:
            classifier_layers.append(nn.Dropout(p = dl_dropout_prob))
        classifier_layers.append(nn.Linear(dimensions[0]*dimensions[1]*dimensions[2],num_dense_neurons))
        if add_batchNorm:
            classifier_layers.append(nn.BatchNorm1d(num_dense_neurons))
        classifier_layers.append(self.activation_fn())
        classifier_layers.append(nn.Linear(num_dense_neurons,output_num_classes))
        self.classifier = nn.Sequential(
            *classifier_layers             
        )
    def forward(self,X):
        return self.classifier(self.features(X))