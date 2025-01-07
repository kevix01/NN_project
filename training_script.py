import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import transforms, models
from torchvision.datasets import Food101
from torchvision.models import ResNet50_Weights, ViT_B_16_Weights
from tqdm import tqdm
import os
import argparse

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'


# Data transformations
def get_transforms(image_size):
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),  # Resize to the desired input size
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])


# Custom dataset class to filter and remap labels
class FilteredDataset(Dataset):
    def __init__(self, dataset, selected_classes):
        self.dataset = dataset
        self.selected_classes = selected_classes
        self.class_to_idx = {cls: idx for idx, cls in enumerate(dataset.classes)}
        self.selected_class_indices = [self.class_to_idx[cls] for cls in selected_classes]
        self.label_mapping = {old_idx: new_idx for new_idx, old_idx in enumerate(self.selected_class_indices)}
        self.indices = [i for i, label in enumerate(dataset._labels) if label in self.selected_class_indices]
        self.labels = [self.label_mapping[dataset._labels[i]] for i in self.indices]

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        image, _ = self.dataset[self.indices[idx]]  # Get the image and ignore the original label
        label = self.labels[idx]  # Use the remapped label
        return image, label


# Function to load and filter the dataset
def load_and_filter_dataset(root, split, selected_classes, transform):
    dataset = Food101(root=root, split=split, download=True, transform=transform)
    filtered_dataset = FilteredDataset(dataset, selected_classes)
    return filtered_dataset


# Function to split dataset into training and validation sets
def split_dataset(dataset, train_ratio=0.8):
    train_size = int(train_ratio * len(dataset))
    val_size = len(dataset) - train_size
    return random_split(dataset, [train_size, val_size])


# Function to create data loaders
def create_data_loaders(train_dataset, val_dataset, test_dataset, batch_size):
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader, test_loader


# Function to load and modify the pre-trained model
def load_model(model_type, num_classes, image_size=224, fine_tune_layers=None, final_layer_dropout=0.0, mlp_dropout=0.0, attention_dropout=0.0):
    if model_type == "resnet":
        # Load the pre-trained ResNet50 model
        model = models.resnet50(weights=ResNet50_Weights.DEFAULT)
        # Freeze all layers by default
        for param in model.parameters():
            param.requires_grad = False

        # Modify the final fully connected layer to match the number of classes
        model.fc = nn.Sequential(
            nn.Dropout(final_layer_dropout),  # Dropout in the final layer
            nn.Linear(model.fc.in_features, num_classes)
        )
        model.fc.requires_grad = True  # Always fine-tune the final layer

        # Fine-tune specified layers
        unique_names = set()  # To store unique layer names
        params_status = {}  # To store the status of each layer (frozen or fine-tuned)

        # Group layers into broader categories
        for name, param in model.named_parameters():
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

            unique_names.add(layer_name)  # Add the layer name to the set

            # Check if the layer should be fine-tuned
            if fine_tune_layers and any(layer in layer_name for layer in fine_tune_layers):
                param.requires_grad = True
                params_status[layer_name] = "fine-tuned"
            else:
                if param.requires_grad:  # if was already set to True (fc)
                    params_status[layer_name] = "fine-tuned"
                else:
                    params_status[layer_name] = "frozen"

        # Print the status of all parameters
        print("NN status:")
        for layer in sorted(unique_names):
            print(f"{layer} - {params_status[layer]}")

    elif model_type == "vit":
        # Load the pre-trained Vision Transformer (ViT) model
        model = models.vit_b_16(weights=ViT_B_16_Weights.DEFAULT)
        # Freeze all layers by default
        for param in model.parameters():
            param.requires_grad = False

        # Add dropout to MLP blocks if specified and not frozen
        if mlp_dropout > 0.0 and fine_tune_layers:
            for i, block in enumerate(model.encoder.layers):
                layer_name = f"encoder.layers.encoder_layer_{i}"
                if any(layer in layer_name for layer in fine_tune_layers):
                    print(f"Adding dropout to fine-tuned MLP block in layer: {layer_name}")
                    block.mlp = nn.Sequential(
                        nn.Linear(block.mlp[0].in_features, block.mlp[0].out_features),
                        nn.GELU(),
                        nn.Dropout(mlp_dropout),
                        nn.Linear(block.mlp[3].in_features, block.mlp[3].out_features)
                    )

        # Add dropout to attention mechanisms if specified and not frozen
        if attention_dropout > 0.0 and fine_tune_layers:
            for i, block in enumerate(model.encoder.layers):
                # Check if the current encoder layer is being fine-tuned
                layer_name = f"encoder.layers.encoder_layer_{i}"
                if any(layer in layer_name for layer in fine_tune_layers):
                    # Access the MultiheadAttention module within the block
                    if hasattr(block, "self_attention"):
                        # For some versions of ViT, the attention mechanism is named "self_attention"
                        block.self_attention.dropout = attention_dropout  # Set dropout probability directly
                    elif hasattr(block, "attention"):
                        # For other versions, it might be named "attention"
                        block.attention.dropout = attention_dropout  # Set dropout probability directly
                    else:
                        raise AttributeError(f"Could not find attention mechanism in block {layer_name}")
                    print(f"Added dropout to attention mechanism in layer: {layer_name}")

        # Modify the final classification head to match the number of classes
        model.heads.head = nn.Sequential(
            nn.Dropout(final_layer_dropout),  # Dropout in the final layer
            nn.Linear(model.heads.head.in_features, num_classes)
        )
        model.heads.head.requires_grad = True  # Always fine-tune the final layer

        # Fine-tune specified layers
        unique_names = set()  # To store unique layer names
        params_status = {}  # To store the status of each layer (frozen or fine-tuned)

        for name, param in model.named_parameters():
            # Extract the layer name at the desired level of granularity
            if "encoder.layers" in name:
                # Group by encoder layer (e.g., "encoder.layers.encoder_layer_0")
                layer_name = ".".join(name.split(".")[:3])  # Keep only up to "encoder.layers.encoder_layer_X"
            elif "conv_proj" in name:
                # Group conv_proj.weight and conv_proj.bias under "conv_proj"
                layer_name = "conv_proj"
            else:
                # Handle other params (e.g., "class_token", "encoder.pos_embedding", "encoder.ln", "heads.head")
                layer_name = ".".join(name.split(".")[:2])  # Keep only the first two parts (e.g., "encoder.pos_embedding")

            unique_names.add(layer_name)  # Add the layer name to the set

            # Check if the layer should be fine-tuned
            if fine_tune_layers and any(layer in layer_name for layer in fine_tune_layers):
                param.requires_grad = True
                params_status[layer_name] = "fine-tuned"
            else:
                if param.requires_grad:  # if was already set to True (head)
                    params_status[layer_name] = "fine-tuned"
                else:
                    params_status[layer_name] = "frozen"

        # Sort the encoder layers numerically
        def sort_encoder_layers(layer_name):
            if "encoder.layers.encoder_layer_" in layer_name:
                # Extract the numerical part of the layer name
                return int(layer_name.split("_")[-1])
            else:
                # Return a high number to ensure non-encoder layers are sorted last
                return float('inf')

        # Print the status of all parameters in sorted order
        print("NN status:")
        for layer in sorted(unique_names, key=sort_encoder_layers):
            print(f"{layer} - {params_status[layer]}")

        # Check if the image size is compatible with the patch size (16x16 for vit_b_16)
        patch_size = 16
        if image_size % patch_size != 0:
            raise ValueError(f"Image size {image_size} must be divisible by patch size {patch_size}.")

        # Calculate the new number of patches
        num_patches = (image_size // patch_size) ** 2

        # Get the original positional embeddings
        old_pos_embed = model.encoder.pos_embedding

        # Separate the class token positional embedding from the patch positional embeddings
        class_pos_embed = old_pos_embed[:, 0:1, :]  # Shape: (1, 1, embed_dim)
        patch_pos_embed = old_pos_embed[:, 1:, :]  # Shape: (1, 196, embed_dim)

        # Resize the patch positional embeddings to match the new number of patches
        from torch.nn.functional import interpolate
        patch_pos_embed = patch_pos_embed.permute(0, 2, 1)  # Shape: (1, embed_dim, 196)
        patch_pos_embed = interpolate(
            patch_pos_embed,
            size=num_patches,
            mode='linear',
            align_corners=False
        )  # Shape: (1, embed_dim, num_patches)
        patch_pos_embed = patch_pos_embed.permute(0, 2, 1)  # Shape: (1, num_patches, embed_dim)

        # Combine the class token positional embedding with the resized patch positional embeddings
        new_pos_embed = torch.cat([class_pos_embed, patch_pos_embed], dim=1)  # Shape: (1, num_patches + 1, embed_dim)

        # Update the positional embeddings in the model
        model.encoder.pos_embedding = nn.Parameter(new_pos_embed)

        # Update the model's image_size attribute
        model.image_size = image_size  # Explicitly set the image_size attribute
    else:
        raise ValueError(f"Unsupported model type: {model_type}. Choose 'resnet' or 'vit'.")

    return model


# Function to get the learning rate scheduler
def get_scheduler(optimizer, lr_scheduler, num_epochs):
    if lr_scheduler == "linear":
        # Linear decay: lr decreases linearly from initial value to 0 over epochs
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer,
            lr_lambda=lambda epoch: 1.0 - (epoch / num_epochs)
        )
    else:
        # Static learning rate (no scheduler)
        scheduler = None
    return scheduler


# Function to train the model
def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, num_epochs, device):
    final_train_loss = 0.0
    validation_accuracy = 0.0
    for epoch in range(num_epochs):
        model.train()
        # reset loss at each new epoch
        accumulated_loss = 0.0
        analyzed_batches = 0
        # Wrap train_loader with tqdm for a progress bar
        train_loop = tqdm(train_loader, desc=f"Epoch [{epoch + 1}/{num_epochs}] Training")
        for images, labels in train_loop:
            images, labels = images.to(device), labels.to(device)
            labels = labels.long()

            outputs = model(images)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            analyzed_batches += 1
            accumulated_loss += loss.item()
            # get the current learning rate
            current_lr = optimizer.param_groups[0]['lr']
            train_loop.set_postfix(batch_loss=loss.item(), lr=current_lr, avg_loss=accumulated_loss / analyzed_batches)

        final_train_loss = accumulated_loss / analyzed_batches
        if scheduler:
            scheduler.step()

        # Validation loop with progress bar
        validation_accuracy = evaluate_model(model, val_loader, device,
                                             desc=f"Epoch [{epoch + 1}/{num_epochs}] Validation")
    return final_train_loss, validation_accuracy


# Function to evaluate the model
def evaluate_model(model, data_loader, device, desc="Evaluating"):
    model.eval()
    correct = 0
    total = 0
    # Wrap data_loader with tqdm for a progress bar
    eval_loop = tqdm(data_loader, desc=desc)
    with torch.no_grad():
        for images, labels in eval_loop:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            # Update the progress bar description with the current accuracy
            eval_loop.set_postfix(accuracy=100 * correct / total)
    return 100 * correct / total


# Main function
def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Fine-tune ResNet or ViT on Food-101 dataset.")
    parser.add_argument("--model", type=str, default="resnet", choices=["resnet", "vit"],
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
    parser.add_argument("--model_name", type=str, default="model",
                        help="Name of the saved model checkpoint (default: 'model').")
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
    args = parser.parse_args()

    # Validate arguments based on the selected model
    if args.model == "vit":
        if args.mlp_dropout > 0.0 or args.attention_dropout > 0.0:
            print(f"Using MLP dropout: {args.mlp_dropout} and attention dropout: {args.attention_dropout} for ViT.")
    elif args.model == "resnet":
        if args.mlp_dropout > 0.0 or args.attention_dropout > 0.0:
            raise ValueError("mlp_dropout and attention_dropout are not applicable for ResNet.")

    # Print the selected hyperparameters (only relevant ones)
    print(f"Model: {args.model}")
    print(f"Image size: {args.image_size}")
    print(f"Learning rate scheduler: {args.lr_scheduler}")
    print(f"Batch size: {args.batch_size}")
    print(f"Learning rate: {args.learning_rate}")
    print(f"Number of epochs: {args.num_epochs}")
    print(f"Model name: {args.model_name}")
    print(f"Fine-tune params: {args.fine_tune_params}")
    print(f"Weight decay: {args.weight_decay}")
    print(f"Final layer dropout: {args.final_layer_dropout}")

    if args.model == "vit":
        print(f"MLP dropout: {args.mlp_dropout}")
        print(f"Attention dropout: {args.attention_dropout}")

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Hyperparameters
    batch_size = args.batch_size
    num_epochs = args.num_epochs
    learning_rate = args.learning_rate
    num_classes = 5  # Fine-tune on 5 classes

    # Parse fine-tune params
    fine_tune_layers = args.fine_tune_params.split(",") if args.fine_tune_params else None

    # Data transformations
    transform = get_transforms(args.image_size)

    # Load and filter datasets
    train_dataset = load_and_filter_dataset(root='./data', split='train',
                                            selected_classes=['apple_pie', 'pizza', 'sushi', 'ice_cream', 'hamburger'],
                                            transform=transform)
    test_dataset = load_and_filter_dataset(root='./data', split='test',
                                           selected_classes=['apple_pie', 'pizza', 'sushi', 'ice_cream', 'hamburger'],
                                           transform=transform)

    # Split dataset into training and validation sets
    train_dataset, val_dataset = split_dataset(train_dataset)

    # Create data loaders
    train_loader, val_loader, test_loader = create_data_loaders(train_dataset, val_dataset, test_dataset, batch_size)

    # Debug: Print dataset sizes and steps
    print(f"Total filtered dataset size: {len(train_dataset) + len(val_dataset)}")
    print(f"Training set size: {len(train_dataset)}")
    print(f"Validation set size: {len(val_dataset)}")
    print(f"Batch size: {batch_size}")
    print(f"Steps per epoch (training): {len(train_loader)}")
    print(f"Steps per epoch (validation): {len(val_loader)}")

    # Load and modify the pre-trained model
    model = load_model(args.model, num_classes, args.image_size, fine_tune_layers, args.final_layer_dropout, args.mlp_dropout, args.attention_dropout)
    model = model.to(device)

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=args.weight_decay)

    # Get the learning rate scheduler
    scheduler = get_scheduler(optimizer, args.lr_scheduler, num_epochs)

    # Train the model
    t_loss, v_accuracy = train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, num_epochs,
                                     device)

    # Test the model with progress bar
    test_accuracy = evaluate_model(model, test_loader, device, desc="Testing")

    # Print final train loss, validation accuracy, and test accuracy
    print(f'Final Train Loss: {t_loss:.4f}')
    print(f'Final Validation Accuracy: {v_accuracy:.2f}%')
    tqdm.write(f'Test Accuracy: {test_accuracy:.2f}%')

    # Save the model checkpoint
    model_name = args.model_name if args.model_name.endswith(".pth") else f"{args.model_name}.pth"
    torch.save(model.state_dict(), model_name)
    print(f"Model saved as {model_name}")


if __name__ == '__main__':
    main()