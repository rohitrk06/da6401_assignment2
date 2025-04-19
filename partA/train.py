from LitCNN import LitCNN
import torch
import torch.nn as nn
import wandb
import uuid
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch  import Trainer
from lightning.pytorch.callbacks import Callback
from lightning.pytorch.callbacks.early_stopping import EarlyStopping

wandb_logger = WandbLogger(project = "da6401_assignment2",)
def create_cnn_sweep_config_name(config):
    return (
        f"final_run_"
        f"nl{config.num_layers}_"
        f"nf{config.num_filters}_"
        f"act{config.activation_fn}_"
        f"fs{config.filter_sizes}_"
        f"cp{config.conv_padding}_"
        f"cs{config.conv_strides}_"
        f"pfs{config.pooling_filter_sizes}_"
        f"ps{config.pooling_strides}_"
        f"pp{config.pooling_padding}_"
        f"dense{config.num_dense_neurons}_"
        f"do{int(config.add_dropout)}_"
        f"dl_do{config.dl_dropout_prob:.3f}_"
        f"ap_do{config.ap_dropout_prob:.2f}_"
        f"bn{int(config.add_batchNorm)}_"
        f"dag{int(config.add_data_augmentation)}"
        f"ep{config.max_epochs}_"
        f"lr{config.lr:.0e}"
    )

class LogPredictionsCallback(Callback):
    def on_validation_batch_end(
        self, trainer,pl_module,outputs,batch,batch_idx
    ):
        if batch_idx == 0:
            no_samples = 20
            images,labels = batch

            columns = ['Image', 'Ground Truth', 'prediction']
            data = [[wandb.Image(x_i), y_i, y_pred] for x_i,y_i,y_pred in list(zip(images[:no_samples], labels[:no_samples],outputs[:no_samples]))]
            wandb_logger.log_table(key = 'Prediction on Validation Set', columns = columns, data = data)

class TransformDataset(Dataset):
    def __init__(self,dataset, transform=None):
        self.dataset = dataset
        self.transform = transform

    def __getitem__(self,idx):
        image, label = self.dataset[idx]
        if self.transform:
            image = self.transform(image)
        return image, label

    def __len__(self):
        return len(self.dataset)

model = None
train_dataset_dir = "..\\inaturalist_12K\\train"
test_dataset_dir = "..\\inaturalist_12K\\val"
def main(config = None, test = False):
    wandb.init(project = "da6401_assignment2",
               config = config)
    config = wandb.config
    config_group = create_cnn_sweep_config_name(config)
    wandb.config.update({"config_group": config_group}, allow_val_change=True)
    wandb.run.name = name=f"{config_group}_run_{uuid.uuid4().hex[:4]}"   


    train_transform = transforms.Compose([
                                    transforms.Resize((256, 256)),
                                    transforms.RandomHorizontalFlip(),
                                    transforms.RandomRotation(20),
                                    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2), 
                                    transforms.ToTensor(),
                                ])
    transform = transforms.Compose([
                                    transforms.Resize((256,256)),
                                    transforms.ToTensor(),
                                ])

    train_dataset = datasets.ImageFolder(root = train_dataset_dir)
    test_dataset = datasets.ImageFolder(root = test_dataset_dir,
                                transform=transforms.Compose([
                                    transforms.Resize((256,256)),
                                    transforms.ToTensor(),
                                ])
                                )

    train_set_size = int(len(train_dataset)*0.8)
    valid_set_size = len(train_dataset) - train_set_size

    training_set, validation_set = random_split(train_dataset,[train_set_size, valid_set_size])

    val = TransformDataset(validation_set, transform)

    validation_loader = DataLoader(val,batch_size = 32,num_workers = 4)
    test_loader = DataLoader(test_dataset,batch_size = 32, num_workers = 4)

    if config.add_data_augmentation:
        train = TransformDataset(training_set,train_transform)
    else: 
        train = TransformDataset(training_set, transform)
            
    training_loader = DataLoader(train,batch_size = 32, shuffle = True,num_workers = 4)
    
    
    model = LitCNN(
        input_dim = (3,256,256),
        output_num_classes = 10,
        activation_fn = config.activation_fn,
        num_layers = config.num_layers,
        num_filters = config.num_filters,
        filter_sizes = config.filter_sizes,
        conv_padding = config.conv_padding,
        conv_strides = config.conv_strides,
        pooling_filter_sizes = config.pooling_filter_sizes,
        pooling_strides = config.pooling_strides,
        pooling_padding = config.pooling_padding,
        num_dense_neurons = config.num_dense_neurons,
        add_batchNorm = config.add_batchNorm,
        add_dropout = config.add_dropout,
        dl_dropout_prob = config.dl_dropout_prob,
        ap_dropout_prob=config.ap_dropout_prob,
        lr = config.lr
    )
    model = torch.compile(model)

    log_predictions_callback = LogPredictionsCallback()
    # checkpoint_callback = ModelCheckpoint(monitor='val_accuracy', mode='max')

    trainer = Trainer(
        logger = wandb_logger,
        callbacks = [EarlyStopping(monitor="val_accuracy", mode = "max"),
                    log_predictions_callback],
        max_epochs = config.max_epochs,
        precision="16-mixed",
    )
    
    trainer.fit(model,training_loader,validation_loader)
    if test:
        trainer.test(model,test_loader)
    wandb.finish()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", action="store_true", help="Run the test set")
    parser.add_argument("--activation_fn", type=str, default="ReLU", choices=["ReLU", "GELU", "SiLU", "Mish"], help="Activation function for convolution layers")
    parser.add_argument("--num_layers", type=int, default=5, help="Number of convolutional layers")
    parser.add_argument("--num_filters", type=int, nargs='+', default=[64, 64, 64, 64, 64], help="List of filters per layer")
    parser.add_argument("--filter_sizes", type=int, default=3, help="Kernel size of convolution layers")
    parser.add_argument("--conv_padding", type=int, nargs='+', default=[1, 1, 1, 1, 1], help="List of padding values per conv layer")
    parser.add_argument("--conv_strides", type=int, nargs='+', default=[1, 1, 1, 1, 1], help="List of stride values per conv layer")
    parser.add_argument("--pooling_filter_sizes", type=int, default=2, help="Kernel size for pooling layers")
    parser.add_argument("--pooling_strides", type=int, default=2, help="Stride for pooling layers")
    parser.add_argument("--pooling_padding", type=int, nargs='+', default=[0, 0, 0, 0, 0], help="List of padding values per pooling layer")
    parser.add_argument("--num_dense_neurons", type=int, default=128, help="Number of neurons in the dense layer")
    parser.add_argument("--add_batchNorm", action="store_true", help="Whether to use BatchNorm")
    parser.add_argument("--add_dropout", action="store_true", help="Whether to use dropout layers")
    parser.add_argument("--dl_dropout_prob", type=float, default=0.5, help="Dropout probability after dense layers")
    parser.add_argument("--ap_dropout_prob", type=float, default=0.1, help="Dropout probability after adaptive pooling")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--max_epochs", type=int, default=10, help="Maximum number of epochs")
    parser.add_argument("--add_data_augmentation", action="store_true", help="Whether to use data augmentation")
    parser.add_argument("--wandb_project", type=str, default="da6401_assignment2", help="Wandb project name")
    parser.add_argument("--wandb_entity", type=str, default="your_entity", help="Wandb entity name")

    args = parser.parse_args()

    config = {
        "activation_fn": args.activation_fn,
        "num_layers": args.num_layers,
        "num_filters": args.num_filters,
        "filter_sizes": args.filter_sizes,
        "conv_padding": args.conv_padding,
        "conv_strides": args.conv_strides,
        "pooling_filter_sizes": args.pooling_filter_sizes,
        "pooling_strides": args.pooling_strides,
        "pooling_padding": args.pooling_padding,
        "num_dense_neurons": args.num_dense_neurons,
        "add_batchNorm": args.add_batchNorm,
        "add_dropout": args.add_dropout,
        "dl_dropout_prob": args.dl_dropout_prob,
        "ap_dropout_prob": args.ap_dropout_prob,
        "lr": args.lr,
        "max_epochs": args.max_epochs,
        "add_data_augmentation": args.add_data_augmentation
    }

    main(config,test=args.test)