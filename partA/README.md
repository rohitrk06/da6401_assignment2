# Assignment 2: Part A - Training from scratch, hyperparameter tuning and visualizing the filters
submitted by: Rohit Kumar (DA24S003)

In order to find the best hyperparameters for the model, `cnn_sweep.ipynb` was used to perform a bayes sian optimization sweep over the hyperparameters. In order to utilise the GPU, the code was run on kaggle notebooks which utilises 2 T4 GPUs. In order to rerun the code using kaggle, please ensure that `wandb_api` secret is set and dataset is uploaded to the kaggle notebook. 
