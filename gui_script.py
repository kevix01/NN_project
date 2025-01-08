import sys
import torch
import torch.nn as nn
from tkinter import Tk, filedialog, Button, Label, StringVar, Listbox, Scrollbar, messagebox, Frame, OptionMenu
from PIL import Image, ImageTk
import os
from torchvision import models, transforms
from torchvision.models import ResNet50_Weights, ViT_B_16_Weights
import numpy as np

# Set initial device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Function to load the model and extract configurations
def load_model(model_path, device):
    # Load the checkpoint
    checkpoint = torch.load(model_path, weights_only=True)  # Set weights_only=True for security

    # Extract configurations
    model_type = checkpoint.get('model_type', 'resnet')  # Default to 'resnet' if not found
    saved_image_size = checkpoint.get('image_size', 256)  # Default to 256 if not found
    num_classes = checkpoint.get('num_classes', 5)  # Default to 5 if not found

    # Load class names from the checkpoint or generate default class names
    class_names = checkpoint.get('class_names', None)
    if class_names is None:
        # Generate default class names if not found in the checkpoint
        class_names = [f"class_{i}" for i in range(num_classes)]

    # Load the model based on the model type
    if model_type == "resnet":
        model = models.resnet50(weights=ResNet50_Weights.DEFAULT)
        # Modify the final fully connected layer to match the number of classes
        model.fc = nn.Sequential(
            nn.Dropout(checkpoint.get('final_layer_dropout', 0.0)),  # Dropout in the final layer
            nn.Linear(model.fc.in_features, num_classes)
        )
        # Get fine_tune_layers
        fine_tune_layers = checkpoint.get('fine_tune_layers', None)
    elif model_type == "vit":
        model = models.vit_b_16(weights=ViT_B_16_Weights.DEFAULT)
        # Modify the final classification head to match the number of classes
        model.heads.head = nn.Sequential(
            nn.Dropout(checkpoint.get('final_layer_dropout', 0.0)),  # Dropout in the final layer
            nn.Linear(model.heads.head.in_features, num_classes)
        )

        # Load the saved positional embeddings
        if 'pos_embedding' in checkpoint:
            model.encoder.pos_embedding = nn.Parameter(checkpoint['pos_embedding'])
        else:
            raise ValueError("Positional embeddings not found in the checkpoint.")
        model.image_size = saved_image_size  # Set the image size for the model

        # Get fine_tune_layers and dropout values
        fine_tune_layers = checkpoint.get('fine_tune_layers', None)
        mlp_dropout = checkpoint.get('mlp_dropout', 0.0)
        attention_dropout = checkpoint.get('attention_dropout', 0.0)

        # Apply MLP and attention dropout based on fine_tune_layers
        if fine_tune_layers:
            if fine_tune_layers == ["all"]:
                # Apply dropout to all layers
                if mlp_dropout > 0.0:
                    for block in model.encoder.layers:
                        block.mlp = nn.Sequential(
                            nn.Linear(block.mlp[0].in_features, block.mlp[0].out_features),
                            nn.GELU(),
                            nn.Dropout(mlp_dropout),
                            nn.Linear(block.mlp[3].in_features, block.mlp[3].out_features)
                        )
                    print("Applied MLP dropout to all MLP blocks.")

                if attention_dropout > 0.0:
                    for block in model.encoder.layers:
                        if hasattr(block, "self_attention"):
                            block.self_attention.dropout = attention_dropout
                        elif hasattr(block, "attention"):
                            block.attention.dropout = attention_dropout
                        else:
                            raise AttributeError("Could not find attention mechanism in block.")
                    print("Applied attention dropout to all attention mechanisms.")
            else:
                # Apply dropout only to specified layers
                for i, block in enumerate(model.encoder.layers):
                    layer_name = f"encoder.layers.encoder_layer_{i}"
                    if any(layer in layer_name for layer in fine_tune_layers):
                        if mlp_dropout > 0.0:
                            block.mlp = nn.Sequential(
                                nn.Linear(block.mlp[0].in_features, block.mlp[0].out_features),
                                nn.GELU(),
                                nn.Dropout(mlp_dropout),
                                nn.Linear(block.mlp[3].in_features, block.mlp[3].out_features)
                            )
                            print(f"Applied MLP dropout to layer: {layer_name}")

                        if attention_dropout > 0.0:
                            if hasattr(block, "self_attention"):
                                block.self_attention.dropout = attention_dropout
                            elif hasattr(block, "attention"):
                                block.attention.dropout = attention_dropout
                            else:
                                raise AttributeError(f"Could not find attention mechanism in block {layer_name}.")
                            print(f"Applied attention dropout to layer: {layer_name}")

    else:
        raise ValueError("Unsupported model type: choose 'resnet' or 'vit'.")

    # Load the model weights
    model.load_state_dict(checkpoint['model_state_dict'], strict=True)  # Enforce strict loading
    model.eval()

    # Move the model to the specified device
    model = model.to(device)

    # Return the model and configurations for display
    configs = {
        'model_type': model_type,
        'image_size': saved_image_size,  # Use the saved image size
        'num_classes': num_classes,
        'class_names': class_names,  # Include class names
        'fine_tune_layers': fine_tune_layers,
        'final_layer_dropout': checkpoint.get('final_layer_dropout', 0.0),
        'learning_rate': checkpoint.get('learning_rate', 0.001),
        'batch_size': checkpoint.get('batch_size', 40),
        'lr_scheduler': checkpoint.get('lr_scheduler', 'static'),
        'num_epochs': checkpoint.get('num_epochs', 10),
        'weight_decay': checkpoint.get('weight_decay', 0.0),
        'mlp_dropout': checkpoint.get('mlp_dropout', 0.0),
        'attention_dropout': checkpoint.get('attention_dropout', 0.0),
    }
    return model, configs


# Function to display configurations on the GUI
def display_configs(configs):
    # Clear previous content
    for widget in model_info_frame.winfo_children():
        widget.destroy()
    for widget in training_info_frame.winfo_children():
        widget.destroy()

    # Model Info
    Label(model_info_frame, text="Model Info", font=("Arial", 12, "bold")).grid(row=0, column=0, sticky="w", pady=5)
    Label(model_info_frame, text=f"Model Type: {configs['model_type']}").grid(row=1, column=0, sticky="w")
    Label(model_info_frame, text=f"Image Size: {configs['image_size']}").grid(row=2, column=0, sticky="w")
    Label(model_info_frame, text=f"Num Classes: {configs['num_classes']}").grid(row=3, column=0, sticky="w")
    Label(model_info_frame, text=f"Fine-Tuned Params: {configs['fine_tune_layers']}").grid(row=4, column=0, sticky="w")
    Label(model_info_frame, text=f"Final Layer Dropout: {configs['final_layer_dropout']}").grid(row=5, column=0, sticky="w")
    if configs['model_type'] == "vit":
        Label(model_info_frame, text=f"MLP Dropout: {configs['mlp_dropout']}").grid(row=6, column=0, sticky="w")
        Label(model_info_frame, text=f"Attention Dropout: {configs['attention_dropout']}").grid(row=7, column=0, sticky="w")

    # Training Info
    Label(training_info_frame, text="Training Info", font=("Arial", 12, "bold")).grid(row=0, column=0, sticky="w", pady=5)
    Label(training_info_frame, text=f"Learning Rate: {configs['learning_rate']}").grid(row=1, column=0, sticky="w")
    Label(training_info_frame, text=f"Batch Size: {configs['batch_size']}").grid(row=2, column=0, sticky="w")
    Label(training_info_frame, text=f"LR Scheduler: {configs['lr_scheduler']}").grid(row=3, column=0, sticky="w")
    Label(training_info_frame, text=f"Num Epochs: {configs['num_epochs']}").grid(row=4, column=0, sticky="w")
    Label(training_info_frame, text=f"Weight Decay: {configs['weight_decay']}").grid(row=5, column=0, sticky="w")


# Function to classify an image
def classify_image(model, image_path, image_size=256):
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),  # Resize to 256x256
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    image = Image.open(image_path).convert("RGB")
    image = transform(image).unsqueeze(0)  # Add batch dimension

    # Move the input tensor to the same device as the model
    device = next(model.parameters()).device  # Get the device of the model
    image = image.to(device)

    output = model(image)
    _, predicted = torch.max(output, 1)
    return predicted.item()


# Function to browse and select an image
def browse_files():
    # Specify the file types to include in the file dialog
    filetypes = [("Image files", "*.jpg *.jpeg *.png"), ("All files", "*.*")]

    # Open the file dialog to select an image
    filename = filedialog.askopenfilename(
        initialdir="/",  # Start from the root directory
        title="Select an Image",
        filetypes=filetypes  # Include .jpg, .jpeg, and .png files
    )

    # If a file is selected, update the image path and display the image
    if filename:
        image_path.set(filename)  # Update the image path variable
        show_image(filename)  # Display the selected image


# Function to display the selected image
def show_image(image_path):
    img = Image.open(image_path)
    img.thumbnail((250, 250))  # Resize for display
    img = ImageTk.PhotoImage(img)
    panel.config(image=img)
    panel.image = img


# Function to handle listbox selection
def on_model_select(event):
    # Get the selected model from the listbox
    selected_model = model_listbox.get(model_listbox.curselection())
    model_path = os.path.join("models", selected_model)
    try:
        # Load the model and configurations
        global loaded_model, loaded_configs
        loaded_model, loaded_configs = load_model(model_path, device)
        display_configs(loaded_configs)  # Display configurations on the GUI
        result_label.config(text="Model loaded successfully!")
    except Exception as e:
        messagebox.showerror("Error", f"Failed to load model: {e}")


# Function to handle device changes
def change_device(new_device):
    global device, loaded_model, loaded_configs
    device = torch.device(new_device)  # Update the device
    if loaded_model is not None:
        # Reload the model on the new device
        selected_model = model_listbox.get(model_listbox.curselection())
        model_path = os.path.join("models", selected_model)
        loaded_model, loaded_configs = load_model(model_path, device)
        display_configs(loaded_configs)  # Update the displayed configurations
        result_label.config(text=f"Model reloaded on {new_device}!")


# Function to perform inference using the loaded model
def perform_inference():
    if not image_path.get():
        messagebox.showerror("Error", "Please select an image first!")
        return
    if not loaded_model:
        messagebox.showerror("Error", "Please select a model first!")
        return

    try:
        # Get the image size from the model configurations
        image_size = loaded_configs['image_size']

        # Perform inference
        predicted_class_index = classify_image(loaded_model, image_path.get(), image_size)

        # Get the class name from the loaded configurations
        class_names = loaded_configs['class_names']
        predicted_class_name = class_names[predicted_class_index]

        # Display the predicted class name
        result_label.config(text=f"Predicted Class: {predicted_class_name}")
    except Exception as e:
        messagebox.showerror("Error", f"Failed to perform inference: {e}")


# Function to get available devices
def get_available_devices():
    devices = ["cpu"]  # CPU is always available
    if torch.cuda.is_available():
        devices.append("cuda")  # Add CUDA if available
    return devices

# Create the Tkinter window
window = Tk()
window.title("Image Classifier")

# Variables
image_path = StringVar()
loaded_model = None
loaded_configs = None

# Get available devices
available_devices = get_available_devices()

# Set the default device
device_var = StringVar(value="cuda" if "cuda" in available_devices else "cpu")

# Layout
Label(window, text="Select Image:").grid(row=0, column=0, padx=10, pady=10)
browse_button = Button(window, text="Browse Files", command=browse_files)
browse_button.grid(row=0, column=1, padx=10, pady=10)

# Image display panel
panel = Label(window)
panel.grid(row=1, column=0, columnspan=2, pady=10)

# Listbox to display available models
Label(window, text="Available Models:").grid(row=2, column=0, padx=10, pady=10)
model_listbox = Listbox(window, width=50, height=10)
model_listbox.grid(row=3, column=0, columnspan=2, padx=10, pady=10)

# Bind the listbox selection event to the on_model_select function
model_listbox.bind('<<ListboxSelect>>', on_model_select)

# Scrollbar for the listbox
scrollbar = Scrollbar(window, orient="vertical")
scrollbar.config(command=model_listbox.yview)
scrollbar.grid(row=3, column=2, sticky="ns")
model_listbox.config(yscrollcommand=scrollbar.set)

# Populate the listbox with models from the models folder
models_dir = "models"
if os.path.exists(models_dir):
    for model_file in os.listdir(models_dir):
        if model_file.endswith(".pth"):
            model_listbox.insert("end", model_file)
else:
    messagebox.showwarning("Warning", "No models folder found!")

# Device selection dropdown
Label(window, text="Select Device:").grid(row=7, column=0, padx=10, pady=10)
device_dropdown = OptionMenu(window, device_var, *available_devices, command=change_device)
device_dropdown.grid(row=7, column=1, padx=10, pady=10)

# Frames for Model Info and Training Info
model_info_frame = Frame(window, borderwidth=2, relief="groove")
model_info_frame.grid(row=4, column=0, padx=10, pady=10, sticky="nsew")

training_info_frame = Frame(window, borderwidth=2, relief="groove")
training_info_frame.grid(row=4, column=1, padx=10, pady=10, sticky="nsew")

# Button to perform inference
predict_button = Button(window, text="Classify", command=perform_inference)
predict_button.grid(row=5, column=0, columnspan=2, pady=10)

# Label for displaying predictions
result_label = Label(window, text="Predicted Class: ", font=("Arial", 12))
result_label.grid(row=6, column=0, columnspan=2, pady=10)

# Run the application
window.mainloop()