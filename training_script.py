import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.nn.functional import interpolate
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import transforms, models
from torchvision.datasets import Food101
from torchvision.models import ResNet50_Weights, ViT_B_16_Weights
from tqdm import tqdm
import os
import argparse

# Set environment variable to ensure CUDA operations are synchronous (for debugging)
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

# Define the classes to be used in the dataset
classes = ['churros', 'carrot_cake', 'pork_chop', 'panna_cotta', 'greek_salad']


# Data transformations for image preprocessing
def get_transforms(image_size):
    """
    Create a composition of image transformations for preprocessing.

    Args:
        image_size (int): The size to which the images will be resized (height and width).

    Returns:
        torchvision.transforms.Compose: A composition of image transformations including resizing,
                                        conversion to tensor, and normalization.
    """
    return transforms.Compose([
        # Resize the images to the specified size (height and width)
        transforms.Resize((image_size, image_size)),

        # Convert the images from PIL format or NumPy arrays to PyTorch tensors
        transforms.ToTensor(),

        # Normalize the images using the mean and standard deviation of the ImageNet dataset
        # This is a common practice when using models pre-trained on ImageNet
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])


# Custom dataset class to filter and remap labels
class FilteredDataset(Dataset):
    """
    A custom dataset class that filters a given dataset to include only samples from specified classes.

    Args:
        dataset (torch.utils.data.Dataset): The original dataset to filter.
        selected_classes (list): A list of class names to include in the filtered dataset.

    Attributes:
        dataset (torch.utils.data.Dataset): The original dataset.
        selected_classes (list): The list of selected class names.
        class_to_idx (dict): A mapping from class names to their original indices in the dataset.
        selected_class_indices (list): The indices of the selected classes in the original dataset.
        label_mapping (dict): A mapping from original class indices to new indices in the filtered dataset.
        indices (list): The indices of samples in the original dataset that belong to the selected classes.
        labels (list): The remapped labels for the filtered dataset.
    """

    def __init__(self, dataset, selected_classes):
        """
        Initialize the FilteredDataset.

        Args:
            dataset (torch.utils.data.Dataset): The original dataset to filter.
            selected_classes (list): A list of class names to include in the filtered dataset.
        """
        self.dataset = dataset  # The original dataset
        self.selected_classes = selected_classes  # The list of selected class names

        # Create a mapping from class names to their original indices in the dataset
        self.class_to_idx = {cls: idx for idx, cls in enumerate(dataset.classes)}

        # Get the indices of the selected classes in the original dataset
        self.selected_class_indices = [self.class_to_idx[cls] for cls in selected_classes]

        # Create a mapping from original class indices to new indices in the filtered dataset
        self.label_mapping = {old_idx: new_idx for new_idx, old_idx in enumerate(self.selected_class_indices)}

        # Get the indices of samples in the original dataset that belong to the selected classes
        self.indices = [i for i, label in enumerate(dataset._labels) if label in self.selected_class_indices]

        # Remap the labels to new indices for the filtered dataset
        self.labels = [self.label_mapping[dataset._labels[i]] for i in self.indices]

    def __len__(self):
        """
        Get the number of samples in the filtered dataset.

        Returns:
            int: The number of samples in the filtered dataset.
        """
        return len(self.indices)

    def __getitem__(self, idx):
        """
        Get a sample from the filtered dataset.

        Args:
            idx (int): The index of the sample to retrieve.

        Returns:
            tuple: A tuple containing:
                - image: The image corresponding to the sample.
                - label: The remapped label for the sample.
        """
        # Get the image and ignore the original label from the original dataset
        image, _ = self.dataset[self.indices[idx]]

        # Get the remapped label for the sample
        label = self.labels[idx]

        # Return the image and the remapped label
        return image, label


# Function to load and filter the dataset
def load_and_filter_dataset(root, split, selected_classes, transform):
    """
    Load the Food101 dataset and filter it to include only the specified classes.

    Args:
        root (str): The root directory where the dataset is stored or will be downloaded.
        split (str): The dataset split to load. Must be either "train" or "test".
        selected_classes (list): A list of class names to include in the filtered dataset.
        transform (callable): A function/transform to apply to the images in the dataset.

    Returns:
        FilteredDataset: A dataset containing only the images and labels from the specified classes.

    Raises:
        ValueError: If `split` is not "train" or "test".
    """
    # Validate the dataset split
    if split not in ["train", "test"]:
        raise ValueError("split must be either 'train' or 'test'.")

    # Load the Food101 dataset
    dataset = Food101(root=root, split=split, download=True, transform=transform)

    # Filter the dataset to include only the specified classes
    filtered_dataset = FilteredDataset(dataset, selected_classes)

    # Return the filtered dataset
    return filtered_dataset


# Function to split dataset into training and validation sets
def split_dataset(dataset, train_ratio=0.8):
    """
    Split a dataset into training and validation subsets based on a specified ratio.

    Args:
        dataset (torch.utils.data.Dataset): The dataset to split.
        train_ratio (float, optional): The proportion of the dataset to include in the training subset.
                                      Must be between 0 and 1. Defaults to 0.8.

    Returns:
        tuple: A tuple containing two subsets:
            - train_subset: The training subset of the dataset.
            - val_subset: The validation subset of the dataset.

    Raises:
        ValueError: If `train_ratio` is not between 0 and 1.
    """
    # Validate the training ratio
    if not 0 <= train_ratio <= 1:
        raise ValueError("train_ratio must be between 0 and 1.")

    # Calculate the size of the training subset
    train_size = int(train_ratio * len(dataset))

    # Calculate the size of the validation subset
    val_size = len(dataset) - train_size

    # Split the dataset into training and validation subsets using random_split
    return random_split(dataset, [train_size, val_size])


# Function to create data loaders with reproducible shuffling if a seed is provided
def create_data_loaders(train_dataset, val_dataset, test_dataset, batch_size, seed=None):
    """
    Create DataLoader instances for training, validation, and testing datasets.

    Args:
        train_dataset (torch.utils.data.Dataset): The dataset for training.
        val_dataset (torch.utils.data.Dataset): The dataset for validation.
        test_dataset (torch.utils.data.Dataset): The dataset for testing.
        batch_size (int): The batch size to use for all DataLoader instances.
        seed (int, optional): The random seed for reproducibility. If provided, ensures that shuffling
                             and other random operations in the DataLoader are deterministic. Defaults to None.

    Returns:
        tuple: A tuple containing three DataLoader instances:
            - train_loader: DataLoader for the training dataset.
            - val_loader: DataLoader for the validation dataset.
            - test_loader: DataLoader for the testing dataset.
    """
    # Define a worker initialization function to set the random seed for each worker process.
    # This ensures reproducibility when using multi-threaded data loading.
    def seed_worker():
        worker_seed = torch.initial_seed() % 2**32  # Get the seed for the current worker
        np.random.seed(worker_seed)  # Set the seed for NumPy
        random.seed(worker_seed)  # Set the seed for Python's random module

    # Create a random number generator for the DataLoader.
    # If a seed is provided, the generator is initialized with the seed to ensure reproducibility.
    generator = torch.Generator()
    if seed is not None:
        generator.manual_seed(seed)

    # Create the DataLoader for the training dataset.
    train_loader = DataLoader(
        dataset=train_dataset,  # Training dataset
        batch_size=batch_size,  # Batch size
        shuffle=True,  # Shuffle the training data for each epoch
        num_workers=os.cpu_count(),  # Use all available CPU cores for data loading
        worker_init_fn=seed_worker if seed is not None else None,  # Worker initialization function for reproducibility
        generator=generator if seed is not None else None  # Random number generator for reproducibility
    )

    # Create the DataLoader for the validation dataset.
    val_loader = DataLoader(
        dataset=val_dataset,  # Validation dataset
        batch_size=batch_size,  # Batch size
        shuffle=False,  # Do not shuffle the validation data
        num_workers=os.cpu_count()  # Use all available CPU cores for data loading
    )

    # Create the DataLoader for the testing dataset.
    test_loader = DataLoader(
        dataset=test_dataset,  # Testing dataset
        batch_size=batch_size,  # Batch size
        shuffle=False,  # Do not shuffle the testing data
        num_workers=os.cpu_count()  # Use all available CPU cores for data loading
    )

    # Return the DataLoader instances for training, validation, and testing
    return train_loader, val_loader, test_loader


# Function to load and modify the pre-trained model
def load_model(model_type, num_classes, image_size=224, fine_tune_layers=None, final_layer_dropout=0.0, mlp_dropout=0.0,
               attention_dropout=0.0):
    """
    Load and configure a neural network model based on the specified type and parameters.

    Args:
        model_type (str): The type of model to load. Supported values are "resnet" and "vit".
        num_classes (int): The number of output classes for the final classification layer.
        image_size (int, optional): The size of the input image. Defaults to 224.
        fine_tune_layers (list, optional): A list of layer names to fine-tune. If "all" is included, all layers are fine-tuned. Defaults to None.
        final_layer_dropout (float, optional): Dropout probability for the final classification layer. Defaults to 0.0.
        mlp_dropout (float, optional): Dropout probability for the MLP blocks in the Vision Transformer (ViT). Defaults to 0.0.
        attention_dropout (float, optional): Dropout probability for the attention mechanism in the Vision Transformer (ViT). Defaults to 0.0.

    Returns:
        torch.nn.Module: A configured neural network model ready for training or inference.

    Raises:
        ValueError: If an unsupported model type is provided, or if invalid or ambiguous layer names are specified for fine-tuning.
    """
    # Initialize a set to store unique layer names for validation and logging purposes
    unique_names = set()

    # Load and configure a ResNet model if specified
    if model_type == "resnet":
        # Load a pre-trained ResNet50 model with default weights
        model = models.resnet50(weights=ResNet50_Weights.DEFAULT)

        # Fine-tune all layers if 'all' is specified in fine_tune_layers
        if fine_tune_layers and fine_tune_layers[0] == "all":
            for param in model.parameters():
                param.requires_grad = True
        else:
            # Freeze all layers by default
            for param in model.parameters():
                param.requires_grad = False

        # Replace the final fully connected layer with a new Sequential block
        # that includes dropout and a linear layer with the specified number of classes
        model.fc = nn.Sequential(
            nn.Dropout(final_layer_dropout),
            nn.Linear(model.fc.in_features, num_classes)
        )
        model.fc.requires_grad = True

        # Track the status of each layer (fine-tuned or frozen)
        params_status = {}
        for name, param in model.named_parameters():
            # Determine the layer name based on the parameter name
            if "conv1" in name:
                layer_name = "conv1"
            elif "bn1" in name:
                layer_name = "bn1"
            elif "layer1" in name:
                layer_name = "layer1"
            elif "layer2" in name:
                layer_name = "layer2"
            elif "layer3" in name:
                layer_name = "layer3"
            elif "layer4" in name:
                layer_name = "layer4"
            elif "fc" in name:
                layer_name = "fc"
            else:
                layer_name = "other"

            # Add the layer name to the set of unique names
            unique_names.add(layer_name)

            # Fine-tune specific layers if they are listed in fine_tune_layers
            if fine_tune_layers and fine_tune_layers[0] != "all" and any(layer in layer_name for layer in fine_tune_layers):
                param.requires_grad = True
                params_status[layer_name] = "fine-tuned"
            else:
                if "fc" in layer_name:
                    params_status[layer_name] = "fine-tuned"
                else:
                    params_status[layer_name] = "frozen"

        # Print the status of each layer (fine-tuned or frozen)
        print("NN Layers:")
        for layer in sorted(unique_names, key=lambda x: int(x.split("layer")[-1]) if "layer" in x else -1):
            print(f"{layer} - {params_status[layer]}")

    # Load and configure a Vision Transformer (ViT) model if specified
    elif model_type == "vit":
        # Load a pre-trained Vision Transformer (ViT-B/16) model with default weights
        model = models.vit_b_16(weights=ViT_B_16_Weights.DEFAULT)

        # Fine-tune all layers if 'all' is specified in fine_tune_layers
        if fine_tune_layers and fine_tune_layers[0] == "all":
            for param in model.parameters():
                param.requires_grad = True
        else:
            # Freeze all layers by default
            for param in model.parameters():
                param.requires_grad = False

        # Track the status of each layer (fine-tuned or frozen)
        params_status = {}
        for name, param in model.named_parameters():
            # Determine the layer name based on the parameter name
            if "encoder.layers" in name:
                layer_name = ".".join(name.split(".")[:3])
            elif "conv_proj" in name:
                layer_name = "conv_proj"
            else:
                layer_name = ".".join(name.split(".")[:2])

            # Add the layer name to the set of unique names
            unique_names.add(layer_name)

            # Fine-tune specific layers if they are listed in fine_tune_layers
            if fine_tune_layers and fine_tune_layers[0] != "all" and any(layer in layer_name for layer in fine_tune_layers):
                param.requires_grad = True
                params_status[layer_name] = "fine-tuned"
            else:
                if "head" in layer_name:
                    params_status[layer_name] = "fine-tuned"
                else:
                    params_status[layer_name] = "frozen"

        # Print the status of each layer (fine-tuned or frozen)
        print("NN Layers:")
        for layer in sorted(unique_names, key=lambda x: int(x.split("layer_")[-1]) if "layer_" in x else -1):
            print(f"{layer} - {params_status[layer]}")

        # Add dropout to the MLP blocks if specified
        if mlp_dropout > 0.0:
            for i, block in enumerate(model.encoder.layers):
                if fine_tune_layers[0] == "all" or any(
                        layer in f"encoder.layers.encoder_layer_{i}" for layer in fine_tune_layers):
                    print(f"Adding dropout to MLP block in layer: encoder.layers.encoder_layer_{i}")
                    block.mlp = nn.Sequential(
                        nn.Linear(block.mlp[0].in_features, block.mlp[0].out_features),
                        nn.GELU(),
                        nn.Dropout(mlp_dropout),
                        nn.Linear(block.mlp[3].in_features, block.mlp[3].out_features)
                    )

        # Add dropout to the attention mechanism if specified
        if attention_dropout > 0.0:
            for i, block in enumerate(model.encoder.layers):
                if fine_tune_layers[0] == "all" or any(
                        layer in f"encoder.layers.encoder_layer_{i}" for layer in fine_tune_layers):
                    if hasattr(block, "self_attention"):
                        block.self_attention.dropout = attention_dropout
                    elif hasattr(block, "attention"):
                        block.attention.dropout = attention_dropout
                    else:
                        raise AttributeError(
                            f"Could not find attention mechanism in block encoder.layers.encoder_layer_{i}")
                    print(f"Added dropout to attention mechanism in layer: encoder.layers.encoder_layer_{i}")

        # Replace the final classification head with a new Sequential block
        # that includes dropout and a linear layer with the specified number of classes
        model.heads.head = nn.Sequential(
            nn.Dropout(final_layer_dropout),
            nn.Linear(model.heads.head.in_features, num_classes)
        )
        model.heads.head.requires_grad = True

        # Define the patch size used by the Vision Transformer (ViT).
        # ViT-B/16 uses patches of size 16x16 pixels.
        patch_size = 16

        # Check if the provided image size is divisible by the patch size.
        # This is necessary because the image is divided into non-overlapping patches,
        # and the patch size must evenly divide the image dimensions.
        if image_size % patch_size != 0:
            raise ValueError(f"Image size {image_size} must be divisible by patch size {patch_size}.")

        # Calculate the number of patches along one dimension of the image.
        # For example, if the image size is 224x224 and the patch size is 16x16,
        # the number of patches per side is 224 / 16 = 14.
        # The total number of patches is 14 * 14 = 196.
        num_patches = (image_size // patch_size) ** 2

        # Retrieve the original positional embeddings from the model.
        # ViT's positional embeddings include:
        # - A class token embedding (used for classification).
        # - Patch embeddings (one for each patch in the image).
        old_pos_embed = model.encoder.pos_embedding

        # Separate the class token embedding from the patch embeddings.
        # The class token embedding is the first embedding vector (index 0).
        class_pos_embed = old_pos_embed[:, 0:1, :]

        # Extract the patch embeddings, which are all embeddings except the class token.
        patch_pos_embed = old_pos_embed[:, 1:, :]

        # Permute the patch embeddings to prepare for interpolation.
        # The original shape is (batch_size, num_patches, embedding_dim).
        # We permute it to (batch_size, embedding_dim, num_patches) to perform interpolation
        # along the patch dimension.
        patch_pos_embed = patch_pos_embed.permute(0, 2, 1)

        # Interpolate the patch embeddings to match the new number of patches.
        # This is necessary because the number of patches changes when the image size changes.
        # For example, if the original image size was 224x224 (196 patches) and the new size is
        # 384x384 (576 patches), we need to interpolate the embeddings to match the new patch count.
        patch_pos_embed = interpolate(
            patch_pos_embed,  # Input tensor to interpolate
            size=num_patches,  # Target size (number of patches)
            mode='linear',  # Interpolation mode (linear for 1D interpolation)
            align_corners=False  # Whether to align corners during interpolation
        )

        # Permute the interpolated patch embeddings back to the original shape.
        # The shape is restored to (batch_size, num_patches, embedding_dim).
        patch_pos_embed = patch_pos_embed.permute(0, 2, 1)

        # Concatenate the class token embedding with the interpolated patch embeddings.
        # The class token embedding remains unchanged, while the patch embeddings are updated.
        new_pos_embed = torch.cat([class_pos_embed, patch_pos_embed], dim=1)

        # Replace the model's positional embeddings with the new interpolated embeddings.
        # This ensures the model can handle the new image size.
        model.encoder.pos_embedding = nn.Parameter(new_pos_embed)

        # Store the new image size in the model for reference.
        model.image_size = image_size

    # Raise an error if an unsupported model type is specified
    else:
        raise ValueError(f"Unsupported model type: {model_type}. Choose 'resnet' or 'vit'.")

    # Validate the fine_tune_layers argument
    if fine_tune_layers:
        if fine_tune_layers[0] != "all":
            # Check for invalid layer names
            invalid_layers = []
            for layer in fine_tune_layers:
                if not any(layer in unique_layer for unique_layer in unique_names):
                    invalid_layers.append(layer)
            if invalid_layers:
                raise ValueError(f"Invalid layer name(s): {invalid_layers}")

        # Check for ambiguous layer names
        ambiguous_layers = [layer for layer in fine_tune_layers if
                            sum(layer in unique_layer for unique_layer in unique_names) > 1]
        if ambiguous_layers:
            raise ValueError(f"Ambiguous layer name(s): {ambiguous_layers}")

    # Return the configured model
    return model


# Function to get the learning rate scheduler
def get_scheduler(optimizer, lr_scheduler, num_epochs, initial_lr, min_lr=None):
    """
    Create a learning rate scheduler for the given optimizer.

    Args:
        optimizer (torch.optim.Optimizer): The optimizer whose learning rate will be scheduled.
        lr_scheduler (str): The type of learning rate scheduler to use. Currently supports "linear".
        num_epochs (int): The total number of training epochs.
        initial_lr (float): The initial learning rate.
        min_lr (float, optional): The minimum learning rate for linear decay. Defaults to None.

    Returns:
        torch.optim.lr_scheduler._LRScheduler: A learning rate scheduler, or None if no scheduler is specified.
    """
    # Check if the learning rate scheduler is set to "linear"
    if lr_scheduler == "linear":
        # Ensure that the number of epochs is greater than 1 for linear decay.
        # Linear decay requires at least 2 epochs to transition from the initial learning rate to the final learning rate.
        if num_epochs <= 1:
            raise ValueError("num_epochs must be greater than 1 for linear decay.")

        # If a minimum learning rate (min_lr) is provided, validate it.
        if min_lr:
            # Ensure that min_lr is greater than 0 and less than the initial learning rate.
            # This ensures the learning rate decays properly.
            if min_lr <= 0 or min_lr >= initial_lr:
                raise ValueError("min_lr must be greater than 0 and less than initial_lr for linear decay.")

            # Calculate the decrement in learning rate per epoch.
            # The learning rate decreases linearly from initial_lr to min_lr over (num_epochs - 1) steps.
            decrement = (initial_lr - min_lr) / (num_epochs - 1)

            # Create a LambdaLR scheduler that applies linear decay.
            # The lr_lambda function computes the multiplicative factor for the learning rate at each epoch.
            scheduler = torch.optim.lr_scheduler.LambdaLR(
                optimizer,
                lr_lambda=lambda epoch: 1.0 - epoch * decrement / initial_lr
            )
        else:
            # If no min_lr is provided, the learning rate decays linearly from initial_lr to 0 over num_epochs.
            scheduler = torch.optim.lr_scheduler.LambdaLR(
                optimizer,
                lr_lambda=lambda epoch: 1.0 - (epoch / num_epochs)
            )
    else:
        # If the scheduler type is not supported, return None.
        scheduler = None

    # Return the configured scheduler.
    return scheduler


# Function to plot training loss, validation accuracy, and learning rate
def plot_graphs(train_loss, val_accuracies, learning_rates):
    """
    Plot training loss, validation accuracy, and learning rate over epochs.

    Args:
        train_loss (list): A list of training loss values for each epoch.
        val_accuracies (list): A list of validation accuracy values for each epoch.
        learning_rates (list): A list of learning rate values for each epoch.

    Returns:
        None: Displays a matplotlib figure with three subplots.
    """
    # Generate a list of epoch numbers starting from 1 to the length of the training loss list.
    epochs = range(1, len(train_loss) + 1)

    # Create a matplotlib figure with a size of 15x5 inches.
    plt.figure(figsize=(15, 5))

    # Subplot 1: Training Loss
    plt.subplot(1, 3, 1)  # 1 row, 3 columns, 1st subplot
    plt.plot(epochs, train_loss, label='Training Loss', color='blue', marker='o', linestyle='-', markersize=8)
    plt.title('Training Loss Over Epochs')  # Set the title of the subplot
    plt.xlabel('Epochs')  # Label for the x-axis
    plt.ylabel('Loss')  # Label for the y-axis
    plt.xticks(epochs)  # Set x-axis ticks to match the epoch numbers
    plt.grid(True)  # Enable grid lines for better readability

    # Subplot 2: Validation Accuracy
    plt.subplot(1, 3, 2)  # 1 row, 3 columns, 2nd subplot
    plt.plot(epochs, val_accuracies, label='Validation Accuracy', color='green', marker='o', linestyle='-', markersize=8)
    plt.title('Validation Accuracy Over Epochs')  # Set the title of the subplot
    plt.xlabel('Epochs')  # Label for the x-axis
    plt.ylabel('Accuracy (%)')  # Label for the y-axis
    plt.xticks(epochs)  # Set x-axis ticks to match the epoch numbers
    plt.grid(True)  # Enable grid lines for better readability

    # Subplot 3: Learning Rate
    plt.subplot(1, 3, 3)  # 1 row, 3 columns, 3rd subplot
    plt.plot(epochs, learning_rates, label='Learning Rate', color='red', marker='o', linestyle='-', markersize=8)
    plt.title('Learning Rate Over Epochs')  # Set the title of the subplot
    plt.xlabel('Epochs')  # Label for the x-axis
    plt.ylabel('Learning Rate')  # Label for the y-axis
    plt.yscale('linear')  # Use a linear scale for the y-axis
    plt.ylim(bottom=0)  # Set the minimum y-axis value to 0
    plt.xticks(epochs)  # Set x-axis ticks to match the epoch numbers
    # Format y-axis ticks to display learning rates in scientific notation
    plt.yticks(learning_rates, labels=[f'{lr:.1e}' for lr in learning_rates])
    plt.grid(True)  # Enable grid lines for better readability

    # Adjust the layout to prevent overlapping of subplots
    plt.tight_layout()

    # Display the figure
    plt.show()


# Function to train the model
def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, num_epochs, device, args):
    """
    Train a neural network model and evaluate its performance on a validation set.

    Args:
        model (torch.nn.Module): The neural network model to train.
        train_loader (torch.utils.data.DataLoader): DataLoader for the training dataset.
        val_loader (torch.utils.data.DataLoader): DataLoader for the validation dataset.
        criterion (torch.nn.Module): The loss function (e.g., CrossEntropyLoss).
        optimizer (torch.optim.Optimizer): The optimizer (e.g., Adam, SGD).
        scheduler (torch.optim.lr_scheduler._LRScheduler): The learning rate scheduler.
        num_epochs (int): The number of epochs to train the model.
        device (torch.device): The device to use for training (e.g., "cuda" or "cpu").
        args (argparse.Namespace): Command-line arguments or configuration parameters.

    Returns:
        tuple: A tuple containing three lists:
            - train_losses: Training loss values for each epoch.
            - val_accuracies: Validation accuracy values for each epoch.
            - learning_rates: Learning rate values for each epoch.
    """
    # Initialize variables to track the best validation accuracy and early stopping
    best_val_accuracy = 0.0
    epochs_without_improvement = 0

    # Lists to store training metrics
    train_losses = []  # Training loss for each epoch
    val_accuracies = []  # Validation accuracy for each epoch
    learning_rates = []  # Learning rate for each epoch

    # Training loop over epochs
    for epoch in range(num_epochs):
        model.train()  # Set the model to training mode
        accumulated_loss = 0.0  # Accumulated loss for the epoch
        analyzed_batches = 0  # Number of batches processed

        # Progress bar for the training loop
        train_loop = tqdm(train_loader, desc=f"Epoch [{epoch + 1}/{num_epochs}] Training")
        for images, labels in train_loop:
            # Move data to the specified device (e.g., GPU or CPU)
            images, labels = images.to(device), labels.to(device)
            labels = labels.long()  # Ensure labels are of type long for loss computation

            # Forward pass: compute model predictions
            outputs = model(images)
            loss = criterion(outputs, labels)  # Compute the loss

            # Backward pass: compute gradients and update model parameters
            optimizer.zero_grad()  # Clear previous gradients
            loss.backward()  # Compute gradients
            optimizer.step()  # Update model parameters

            # Update metrics
            analyzed_batches += 1
            accumulated_loss += loss.item()
            current_lr = optimizer.param_groups[0]['lr']  # Get the current learning rate

            # Update the progress bar with the current batch loss and learning rate
            train_loop.set_postfix(batch_loss=loss.item(), lr=current_lr, avg_loss=accumulated_loss / analyzed_batches)

        # Compute the average training loss for the epoch
        epoch_train_loss = accumulated_loss / analyzed_batches
        train_losses.append(epoch_train_loss)  # Store the training loss
        learning_rates.append(current_lr)  # Store the learning rate

        # Evaluate the model on the validation set
        validation_accuracy, macro_accuracy = evaluate_model(
            model, val_loader, device, desc=f"Epoch [{epoch + 1}/{num_epochs}] Validation"
        )
        val_accuracies.append(validation_accuracy)  # Store the validation accuracy

        # Save the model checkpoint if it achieves the best validation accuracy
        if args.save_best_model and validation_accuracy > best_val_accuracy:
            if args.early_stop is None:
                best_val_accuracy = validation_accuracy
            save_model(
                model, args.model_name, args.model, args.image_size, len(classes), classes,
                args.learning_rate, args.batch_size, args.lr_scheduler, len(train_losses),
                args.weight_decay, args.fine_tune_params, args.final_layer_dropout,
                args.mlp_dropout, args.attention_dropout
            )
            print(f"\nNew best checkpoint saved at epoch {epoch + 1} with validation samples accuracy {validation_accuracy:.2f}%, macro accuracy {macro_accuracy:.2f}% and training loss {epoch_train_loss:.4f}")
        else:
            print(f"\nDone epoch {epoch + 1} with validation samples accuracy {validation_accuracy:.2f}%, macro accuracy {macro_accuracy:.2f}% and training loss {epoch_train_loss:.4f}")

        # Early stopping logic
        if args.early_stop is not None:
            if validation_accuracy > best_val_accuracy:
                best_val_accuracy = validation_accuracy
                epochs_without_improvement = 0  # Reset the counter
            else:
                epochs_without_improvement += 1
                if epochs_without_improvement >= args.early_stop:
                    print(f"Early stopping triggered after {epoch + 1} epochs. No improvement for the last {epochs_without_improvement} epochs.")
                    break  # Stop training if no improvement for the specified number of epochs

        # Update the learning rate scheduler
        if scheduler:
            scheduler.step()

    # Plot the training loss, validation accuracy, and learning rate over epochs
    plot_graphs(train_losses, val_accuracies, learning_rates)

    # Save the final model checkpoint if not saving the best model only
    if not args.save_best_model:
        save_model(
            model, args.model_name, args.model, args.image_size, len(classes), classes,
            args.learning_rate, args.batch_size, args.lr_scheduler, len(train_losses),
            args.weight_decay, args.fine_tune_params, args.final_layer_dropout,
            args.mlp_dropout, args.attention_dropout
        )
        print("Model checkpoint saved after training completion")

    # Return the training metrics
    return train_losses, val_accuracies, learning_rates


# Function to evaluate the model with macro accuracy
def evaluate_model(model, data_loader, device, desc="Evaluating"):
    """
    Evaluate the performance of a neural network model on a given dataset.

    Args:
        model (torch.nn.Module): The neural network model to evaluate.
        data_loader (torch.utils.data.DataLoader): DataLoader for the evaluation dataset.
        device (torch.device): The device to use for evaluation (e.g., "cuda" or "cpu").
        desc (str, optional): Description for the progress bar. Defaults to "Evaluating".

    Returns:
        tuple: A tuple containing two values:
            - Overall accuracy (float): The percentage of correctly classified samples.
            - Macro accuracy (float): The average accuracy across all classes.
    """
    # Set the model to evaluation mode (disables dropout and batch normalization)
    model.eval()

    # Initialize counters for overall accuracy
    correct = 0  # Number of correctly classified samples
    total = 0  # Total number of samples

    # Initialize dictionaries to track class-wise accuracy
    class_correct = {}  # Number of correctly classified samples per class
    class_total = {}  # Total number of samples per class

    # Initialize class-wise counters
    for class_idx in range(len(classes)):
        class_correct[class_idx] = 0
        class_total[class_idx] = 0

    # Progress bar for the evaluation loop
    eval_loop = tqdm(data_loader, desc=desc)

    # Disable gradient computation for evaluation
    with torch.no_grad():
        for images, labels in eval_loop:
            # Move data to the specified device (e.g., GPU or CPU)
            images, labels = images.to(device), labels.to(device)

            # Forward pass: compute model predictions
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)  # Get the predicted class labels

            # Update overall accuracy counters
            total += labels.size(0)  # Increment total number of samples
            correct += (predicted == labels).sum().item()  # Increment correct predictions

            # Update class-wise accuracy counters
            for label, pred in zip(labels, predicted):
                if label == pred:
                    class_correct[label.item()] += 1  # Increment correct predictions for the class
                class_total[label.item()] += 1  # Increment total samples for the class

            # Update the progress bar with the current accuracy
            eval_loop.set_postfix(accuracy=100 * correct / total)

    # Calculate macro accuracy (average accuracy across all classes)
    macro_accuracy = 0.0
    for class_idx in range(len(classes)):
        if class_total[class_idx] > 0:
            class_accuracy = 100 * class_correct[class_idx] / class_total[class_idx]
            macro_accuracy += class_accuracy
    macro_accuracy /= len(classes)  # Average the class accuracies

    # Print class-wise accuracy
    print("\nClass-wise Accuracy:")
    for class_idx in range(len(classes)):
        if class_total[class_idx] > 0:
            print(f"{classes[class_idx]}: {100 * class_correct[class_idx] / class_total[class_idx]:.2f}%")
        else:
            print(f"{classes[class_idx]}: No samples")  # Handle cases where a class has no samples

    # Return overall accuracy and macro accuracy
    return 100 * correct / total, macro_accuracy


# Function to save the model checkpoint with configurations
def save_model(model, model_name, model_type, image_size, num_classes, class_names, learning_rate, batch_size, lr_scheduler, num_epochs, weight_decay, fine_tune_layers=None, final_layer_dropout=0.0,
               mlp_dropout=0.0, attention_dropout=0.0):
    """
    Save a trained model checkpoint along with its configuration and hyperparameters.

    Args:
        model (torch.nn.Module): The trained neural network model to save.
        model_name (str): The name of the model file (e.g., "my_model.pth").
        model_type (str): The type of the model (e.g., "resnet" or "vit").
        image_size (int): The size of the input image used during training.
        num_classes (int): The number of output classes in the model.
        class_names (list): A list of class names corresponding to the output classes.
        learning_rate (float): The learning rate used during training.
        batch_size (int): The batch size used during training.
        lr_scheduler (str): The learning rate scheduler used during training.
        num_epochs (int): The number of epochs the model was trained for.
        weight_decay (float): The weight decay (L2 regularization) used during training.
        fine_tune_layers (list, optional): A list of layers that were fine-tuned. Defaults to None.
        final_layer_dropout (float, optional): Dropout probability for the final layer. Defaults to 0.0.
        mlp_dropout (float, optional): Dropout probability for the MLP blocks in ViT. Defaults to 0.0.
        attention_dropout (float, optional): Dropout probability for the attention mechanism in ViT. Defaults to 0.0.

    Returns:
        None: The model checkpoint is saved to disk.
    """
    # Directory to save the model checkpoints
    models_dir = "models"

    # Ensure the model file has a ".pth" extension
    model_name = model_name if model_name.endswith(".pth") else f"{model_name}.pth"

    # Create the models directory if it does not exist
    if not os.path.exists(models_dir):
        os.makedirs(models_dir)

    # Handle fine-tuning layers for ResNet and ViT models
    if model_type == "resnet":
        # If no fine-tuning layers are specified, default to the final fully connected layer ("fc")
        if fine_tune_layers is None:
            fine_tune_layers = "fc"
        else:
            # Append "fc" to the list of fine-tuned layers
            fine_tune_layers += ",fc"
    elif model_type == "vit":
        # If no fine-tuning layers are specified, default to the classification head ("head")
        if fine_tune_layers is None:
            fine_tune_layers = "head"
        else:
            # Append "head" to the list of fine-tuned layers
            fine_tune_layers += ",head"

    # Create a dictionary to store the model checkpoint
    checkpoint = {
        'model_state_dict': model.state_dict(),  # Model parameters
        'model_type': model_type,  # Type of the model (e.g., "resnet" or "vit")
        'image_size': image_size,  # Input image size
        'num_classes': num_classes,  # Number of output classes
        'class_names': class_names,  # List of class names
        'fine_tune_layers': fine_tune_layers,  # Layers that were fine-tuned
        'final_layer_dropout': final_layer_dropout,  # Dropout for the final layer
        'learning_rate': learning_rate,  # Learning rate used during training
        'batch_size': batch_size,  # Batch size used during training
        'lr_scheduler': lr_scheduler,  # Learning rate scheduler used during training
        'num_epochs': num_epochs,  # Number of epochs the model was trained for
        'weight_decay': weight_decay,  # Weight decay (L2 regularization) used during training
    }

    # Add ViT-specific parameters to the checkpoint
    if model_type == "vit":
        checkpoint['mlp_dropout'] = mlp_dropout  # Dropout for MLP blocks in ViT
        checkpoint['attention_dropout'] = attention_dropout  # Dropout for attention mechanism in ViT
        checkpoint['pos_embedding'] = model.encoder.pos_embedding  # Positional embeddings in ViT

    # Save the checkpoint to disk
    model_path = os.path.join(models_dir, model_name)
    torch.save(checkpoint, model_path)


# Function to set random seed for reproducibility
def set_seed(seed):
    """
    Set the random seed for reproducibility across multiple libraries.

    Args:
        seed (int): The seed value to use for random number generation.

    Returns:
        None: The random seed is set for all relevant libraries.
    """
    # Set the random seed for PyTorch's CPU operations
    torch.manual_seed(seed)

    # Set the random seed for PyTorch's CUDA operations (GPU)
    torch.cuda.manual_seed(seed)

    # Set the random seed for all GPUs (if multiple GPUs are available)
    torch.cuda.manual_seed_all(seed)

    # Set the random seed for NumPy
    np.random.seed(seed)

    # Set the random seed for Python's built-in random module
    random.seed(seed)

    # Ensure deterministic behavior in CuDNN (CUDA Deep Neural Network library)
    torch.backends.cudnn.deterministic = True

    # Disable CuDNN benchmarking (which can introduce non-determinism)
    torch.backends.cudnn.benchmark = False


# Main function
def main():
    parser = argparse.ArgumentParser(description="Fine-tune ResNet or ViT on Food-101 dataset.")
    parser.add_argument("--model", type=str, required=True, choices=["resnet", "vit"],
                        help="Model type: 'resnet' (default) or 'vit'.")
    parser.add_argument("--image_size", type=int, default=224,
                        help="Input image size (default: 224). Must be divisible by 16 for ViT.")
    parser.add_argument("--batch_size", type=int, default=40,
                        help="Batch size for training and evaluation (default: 50).")
    parser.add_argument("--learning_rate", type=float, default=0.001,
                        help="Learning rate for the optimizer (default: 0.001).")
    parser.add_argument("--lr_scheduler", type=str, default="static", choices=["static", "linear"],
                        help="Learning rate scheduler: 'static' (default) or 'linear'.")
    parser.add_argument("--num_epochs", type=int, default=10,
                        help="Number of epochs for training (default: 10).")
    parser.add_argument("--model_name", type=str, required=True,
                        help="Name of the saved model checkpoint (e.g., 'model.pth').")
    parser.add_argument("--fine_tune_params", type=str, default=None,
                        help="Comma-separated list of layers to fine-tune (e.g., 'layer1,layer2').")
    parser.add_argument("--weight_decay", type=float, default=0.0,
                        help="Weight decay for regularization (default: 0.0).")
    parser.add_argument("--final_layer_dropout", type=float, default=0.0,
                        help="Dropout rate for the final layer (default: 0.0).")
    parser.add_argument("--mlp_dropout", type=float, default=0.0,
                        help="Dropout rate for MLP blocks in ViT (default: 0.0).")
    parser.add_argument("--attention_dropout", type=float, default=0.0,
                        help="Dropout rate for attention mechanisms in ViT (default: 0.0).")
    parser.add_argument("--save_best_model", action="store_true",
                        help="Save the model with the best validation accuracy.")
    parser.add_argument("--early_stop", type=int, default=None,
                        help="Number of epochs to wait before early stopping if validation accuracy doesn't improve.")
    parser.add_argument("--seed", type=int, default=None,
                        help="Random seed for reproducibility. If not provided, randomness will not be controlled.")
    parser.add_argument("--min_lr", type=float, default=None,
                        help="Minimum learning rate for linear decay scheduler (default: None).")
    args = parser.parse_args()

    if args.num_epochs <= 0:
        raise ValueError("Number of epochs must be greater than 0.")
    if args.early_stop is not None and (args.early_stop <= 0 or args.early_stop >= args.num_epochs):
        raise ValueError("Early stopping must be greater than 0 and less than the number of epochs.")
    if args.image_size < 0:
        raise ValueError("Image size must be a positive integer.")
    if args.batch_size <= 0:
        raise ValueError("Batch size must be a positive integer.")
    if args.learning_rate <= 0:
        raise ValueError("Learning rate must be a positive float.")
    if args.min_lr is not None and args.min_lr <= 0:
        raise ValueError("Minimum learning rate must be a positive float.")
    if args.weight_decay < 0:
        raise ValueError("Weight decay must be a non-negative float.")
    if args.final_layer_dropout < 0 or args.final_layer_dropout >= 1:
        raise ValueError("Final layer dropout must be in the range [0, 1).")
    if args.mlp_dropout < 0 or args.mlp_dropout >= 1:
        raise ValueError("MLP dropout must be in the range [0, 1).")
    if args.attention_dropout < 0 or args.attention_dropout >= 1:
        raise ValueError("Attention dropout must be in the range [0, 1).")
    if args.seed is not None and args.seed > 0:
        print(f"Setting random seed for reproducibility: {args.seed}")
        set_seed(args.seed)
    elif args.seed is not None and args.seed <= 0:
        raise ValueError("Random seed must be a positive integer.")
    if args.save_best_model:
        print("Saving the best model based on validation accuracy")
    if args.early_stop is not None:
        print(f"Early stopping is enabled with a patience of {args.early_stop} epochs")
        if not args.save_best_model:
            print("[WARNING]: Early stopping enabled without enabling the save_best_model option")
            print("The model saved will be the last model checkpoint after the specified number of epochs")
            print("If this is not the desired behavior, consider enabling the save_best_model option")
    fine_tune_params = args.fine_tune_params.split(",") if args.fine_tune_params else None
    if fine_tune_params and len(fine_tune_params) > 1 and "all" in fine_tune_params:
        raise ValueError("Cannot fine-tune all layers and specific layers at the same time.")
    print(f"Model: {args.model}")
    print(f"Model name: {args.model_name}")
    print("\nTraining configurations:")
    print(f"Image size: {args.image_size}")
    print(f"Batch size: {args.batch_size}")
    print(f"Learning rate scheduler: {args.lr_scheduler}")
    if args.lr_scheduler == "linear":
        print(f"Minimum learning rate: {args.min_lr}")
    print(f"Learning rate: {args.learning_rate}")
    print(f"Number of epochs: {args.num_epochs}")
    print(f"Fine-tune params: {args.fine_tune_params}")
    print(f"Weight decay: {args.weight_decay}")
    print(f"Final layer dropout: {args.final_layer_dropout}")
    print(f"Save best model: {args.save_best_model}")
    print(f"Early stop: {args.early_stop}")
    if args.model == "vit":
        print(f"MLP dropout: {args.mlp_dropout}")
        print(f"Attention dropout: {args.attention_dropout}")
    fine_tune_layers = args.fine_tune_params.split(",") if args.fine_tune_params else []
    if fine_tune_layers:
        invalid_layers = []
        for layer in fine_tune_layers:
            if layer.lower() in ["fc", "heads.head", "head"]:
                invalid_layers.append(layer)
        if invalid_layers:
            raise ValueError(
                f"Invalid layer(s) in fine_tune_params: {invalid_layers}. "
                f"These layers are automatically fine-tuned and should not be specified."
            )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size = args.batch_size
    num_epochs = args.num_epochs
    learning_rate = args.learning_rate
    transform = get_transforms(args.image_size)
    train_dataset = load_and_filter_dataset(root='./data', split='train',
                                            selected_classes=classes,
                                            transform=transform)
    test_dataset = load_and_filter_dataset(root='./data', split='test',
                                           selected_classes=classes,
                                           transform=transform)
    train_dataset, val_dataset = split_dataset(train_dataset)
    train_loader, val_loader, test_loader = create_data_loaders(train_dataset, val_dataset, test_dataset, batch_size)
    print("\nDataset Information:")
    print(f"Filtered classes: {classes}")
    print(f"Total filtered dataset size: {len(train_dataset) + len(val_dataset)}")
    print(f"Training set size: {len(train_dataset)}")
    print(f"Validation set size: {len(val_dataset)}")
    print(f"Steps per epoch (training): {len(train_loader)}")
    print(f"Steps per epoch (validation): {len(val_loader)}")
    print("\nLoading model...")
    model = load_model(args.model, len(classes), args.image_size, fine_tune_layers, args.final_layer_dropout,
                       args.mlp_dropout, args.attention_dropout)
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=args.weight_decay)
    scheduler = get_scheduler(optimizer, args.lr_scheduler, num_epochs, learning_rate, args.min_lr)
    print("\nTraining the model...")
    t_loss, v_acc, lr_values = train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, num_epochs,
                                           device, args)
    print("\nTraining completed.")
    t_loss = [round(elem, 4) for elem in t_loss]
    v_acc = [round(elem, 2) for elem in v_acc]
    lr_values = [f'{elem:.1e}' for elem in lr_values]
    print(f"Training losses: {t_loss}")
    print(f"Validation accuracies: {v_acc}")
    print(f"Learning rates: {lr_values}")
    if args.save_best_model:
        print("\nLoading the best model checkpoint for testing...")
        model_path = os.path.join("models",
                                  args.model_name if args.model_name.endswith(".pth") else f"{args.model_name}.pth")
        checkpoint = torch.load(model_path, weights_only=True)
        model.load_state_dict(checkpoint['model_state_dict'])
        print("Best model checkpoint loaded.")
    print("\nTesting the model...")
    test_accuracy, test_macro_acc = evaluate_model(model, test_loader, device, desc="Testing")
    print("\nTesting completed.")
    print("\nResults:")
    print(f'Test Samples Accuracy: {test_accuracy:.2f}%')
    print(f'Test Macro Accuracy: {test_macro_acc:.2f}%')
    print("\nModel saved as:", args.model_name if ".pth" in args.model_name else f"{args.model_name}.pth")


if __name__ == '__main__':
    main()