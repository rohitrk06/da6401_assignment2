# Assignment 2: Part A - Training from scratch, hyperparameter tuning and visualizing the filters
submitted by: Rohit Kumar (DA24S003)

In order to find the best hyperparameters for the model, `cnn_sweep.ipynb` was used to perform a bayesian optimization sweep over the hyperparameters. In order to utilise the GPU, the code was run on kaggle notebooks which utilises 2 T4 GPUs. In order to rerun the code using kaggle, please ensure that `wandb_api` secret is set and dataset is uploaded to the kaggle notebook. 

`cnn_sweep.ipynb` is the main file that was used to perform all the experiments. Use this ipynb file to reproduce the results.

In order to allow users to run the code from command line, `train.py` is provided. This file can be used to run the code from command line. The command to run the code is as follows:

```bash
python train.py --num_layers 5 --activation_fn SiLU --num_filters 512 256 128 64 32 --filter_sizes 5 --conv_padding 1 1 1 1 1 --conv_strides 1 1 1 1 1 --pooling_filter_sizes 5 --pooling_strides 2 --pooling_padding 1 --num_dense_neurons 512 --dl_dropout_prob 0.01 --ap_dropout_prob 0.55 --add_batchNorm --max_epochs 10 
```

`cnn.py` contains the architecture of the CNN model
`LitCNN.py` contains the architecture of the CNN model using PyTorch Lightning
`train.py` contains the code to train the model using PyTorch Lightning

NOTE: The code is not tested on local machine because of hardware constraints. It is recommended to run the code on kaggle notebooks. The code is tested on kaggle notebooks with 2 T4 GPUs. 