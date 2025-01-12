import torch
import torch.nn as nn
from tkinter import Tk, filedialog, Button, Label, StringVar, Listbox, Scrollbar, messagebox, Frame, OptionMenu
from PIL import Image, ImageTk
import os
from torchvision import models, transforms
from torchvision.models import ResNet50_Weights, ViT_B_16_Weights

# Set initial device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Function to load the model and extract configurations
def load_model(model_path, device):
    """
    Load a pre-trained model from a checkpoint file and apply configurations.

    Args:
        model_path (str): The path to the model checkpoint file.
        device (torch.device): The device to load the model onto (e.g., "cuda" or "cpu").

    Returns:
        tuple: A tuple containing:
            - model (torch.nn.Module): The loaded model with weights and configurations applied.
            - configs (dict): A dictionary of configurations extracted from the checkpoint.

    Raises:
        ValueError: If the model type is unsupported or if positional embeddings are missing for ViT.
    """
    # Load the checkpoint file
    checkpoint = torch.load(model_path, weights_only=True, map_location=device)  # Set weights_only=True for security

    # Extract configurations from the checkpoint
    model_type = checkpoint.get('model_type', None)  # Default to 'resnet' if not found
    saved_image_size = checkpoint.get('image_size', None)  # Default to 256 if not found
    num_classes = checkpoint.get('num_classes', None)  # Default to 5 if not found

    # Load class names from the checkpoint or generate default class names
    class_names = checkpoint.get('class_names', None)
    if class_names is None:
        # Generate default class names if not found in the checkpoint
        class_names = [f"class_{i}" for i in range(num_classes)]

    # Load the model based on the model type
    if model_type == "resnet":
        # Load a pre-trained ResNet50 model
        model = models.resnet50(weights=ResNet50_Weights.DEFAULT)

        # Modify the final fully connected layer to match the number of classes
        model.fc = nn.Sequential(
            nn.Dropout(checkpoint.get('final_layer_dropout', None)),  # Dropout in the final layer
            nn.Linear(model.fc.in_features, num_classes)
        )

        # Get fine_tune_layers from the checkpoint
        fine_tune_layers = checkpoint.get('fine_tune_layers', None)

    elif model_type == "vit":
        # Load a pre-trained Vision Transformer (ViT-B/16) model
        model = models.vit_b_16(weights=ViT_B_16_Weights.DEFAULT)

        # Modify the final classification head to match the number of classes
        model.heads.head = nn.Sequential(
            nn.Dropout(checkpoint.get('final_layer_dropout', None)),  # Dropout in the final layer
            nn.Linear(model.heads.head.in_features, num_classes)
        )

        # Load the saved positional embeddings
        if 'pos_embedding' in checkpoint:
            model.encoder.pos_embedding = nn.Parameter(checkpoint['pos_embedding'])
        else:
            raise ValueError("Positional embeddings not found in the checkpoint.")
        model.image_size = saved_image_size  # Set the image size for the model

        # Get fine_tune_layers and dropout values from the checkpoint
        fine_tune_layers = checkpoint.get('fine_tune_layers', None)
        mlp_dropout = checkpoint.get('mlp_dropout', None)
        attention_dropout = checkpoint.get('attention_dropout', None)

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

    # Load the model weights from the checkpoint
    model.load_state_dict(checkpoint['model_state_dict'], strict=True)  # Enforce strict loading
    model.eval()  # Set the model to evaluation mode

    # Move the model to the specified device (e.g., GPU or CPU)
    model = model.to(device)

    # Return the model and configurations for display
    configs = {
        'model_type': model_type,
        'image_size': saved_image_size,  # Use the saved image size
        'num_classes': num_classes,
        'class_names': class_names,  # Include class names
        'fine_tune_layers': fine_tune_layers,
        'final_layer_dropout': checkpoint.get('final_layer_dropout', None),
        'learning_rate': checkpoint.get('learning_rate', None),
        'batch_size': checkpoint.get('batch_size', None),
        'lr_scheduler': checkpoint.get('lr_scheduler', None),
        'num_epochs': checkpoint.get('num_epochs', None),
        'weight_decay': checkpoint.get('weight_decay', None),
        'mlp_dropout': checkpoint.get('mlp_dropout', None),
        'attention_dropout': checkpoint.get('attention_dropout', None),
    }
    return model, configs


# Function to display configurations on the GUI
def display_configs(configs):
    """
    Display the model and training configurations in a graphical user interface (GUI).

    Args:
        configs (dict): A dictionary containing model and training configurations.

    Behavior:
        - Clears any previous content in the `model_info_frame` and `training_info_frame` widgets.
        - Displays the model configurations (e.g., model type, image size, number of classes) in the `model_info_frame`.
        - Displays the training configurations (e.g., learning rate, batch size, number of epochs) in the `training_info_frame`.
        - Handles ViT-specific configurations (e.g., MLP dropout, attention dropout) if the model type is ViT.
    """
    # Clear previous content in the model info frame
    for widget in model_info_frame.winfo_children():
        widget.destroy()

    # Clear previous content in the training info frame
    for widget in training_info_frame.winfo_children():
        widget.destroy()

    # Display Model Info
    Label(model_info_frame, text="Model Info", font=("Arial", 12, "bold")).grid(row=0, column=0, sticky="w", pady=5)
    Label(model_info_frame, text=f"Model Type: {configs['model_type']}").grid(row=1, column=0, sticky="w")
    Label(model_info_frame, text=f"Image Size: {configs['image_size']}").grid(row=2, column=0, sticky="w")
    Label(model_info_frame, text=f"Num Classes: {configs['num_classes']}").grid(row=3, column=0, sticky="w")
    Label(model_info_frame, text=f"Fine-Tuned Params: {configs['fine_tune_layers']}").grid(row=4, column=0, sticky="w")
    Label(model_info_frame, text=f"Final Layer Dropout: {configs['final_layer_dropout']}").grid(row=5, column=0, sticky="w")

    # Display ViT-specific configurations if the model type is ViT
    if configs['model_type'] == "vit":
        Label(model_info_frame, text=f"MLP Dropout: {configs['mlp_dropout']}").grid(row=6, column=0, sticky="w")
        Label(model_info_frame, text=f"Attention Dropout: {configs['attention_dropout']}").grid(row=7, column=0, sticky="w")

    # Display Training Info
    Label(training_info_frame, text="Training Info", font=("Arial", 12, "bold")).grid(row=0, column=0, sticky="w", pady=5)
    Label(training_info_frame, text=f"Learning Rate: {configs['learning_rate']}").grid(row=1, column=0, sticky="w")
    Label(training_info_frame, text=f"Batch Size: {configs['batch_size']}").grid(row=2, column=0, sticky="w")
    Label(training_info_frame, text=f"LR Scheduler: {configs['lr_scheduler']}").grid(row=3, column=0, sticky="w")
    Label(training_info_frame, text=f"Num Epochs: {configs['num_epochs']}").grid(row=4, column=0, sticky="w")
    Label(training_info_frame, text=f"Weight Decay: {configs['weight_decay']}").grid(row=5, column=0, sticky="w")

# Function to classify an image
def classify_image(model, image_path, image_size=256):
    """
    Classify an image using a pre-trained model.

    Args:
        model (torch.nn.Module): The pre-trained model used for classification.
        image_path (str): The path to the image file to classify.
        image_size (int, optional): The size to which the image will be resized. Defaults to 256.

    Returns:
        int: The predicted class index for the image.

    Behavior:
        - Loads the image from the specified path and converts it to RGB format.
        - Applies transformations to the image, including resizing, conversion to a tensor, and normalization.
        - Adds a batch dimension to the image tensor.
        - Moves the image tensor to the same device as the model (e.g., GPU or CPU).
        - Passes the image through the model to obtain predictions.
        - Returns the predicted class index.
    """
    # Define the image transformations
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),  # Resize the image to the specified size
        transforms.ToTensor(),  # Convert the image to a PyTorch tensor
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize using ImageNet stats
    ])

    # Load the image and convert it to RGB format
    image = Image.open(image_path).convert("RGB")

    # Apply the transformations and add a batch dimension
    image = transform(image).unsqueeze(0)  # Add batch dimension (shape: [1, C, H, W])

    # Move the input tensor to the same device as the model
    device = next(model.parameters()).device  # Get the device of the model (e.g., "cuda" or "cpu")
    image = image.to(device)

    # Pass the image through the model to obtain predictions
    output = model(image)

    # Get the predicted class index (the class with the highest score)
    _, predicted = torch.max(output, 1)  # torch.max returns (max_value, max_index)

    # Return the predicted class index
    return predicted.item()


# Function to browse and select an image
def browse_files():
    """
    Open a file dialog to allow the user to select an image file.

    Behavior:
        - Opens a file dialog that allows the user to browse and select an image file.
        - Filters the file types to include only image files (e.g., .jpg, .jpeg, .png) and all files.
        - If a file is selected, updates the global `image_path` variable with the selected file's path.
        - Displays the selected image in the GUI using the `show_image` function.
    """
    # Specify the file types to include in the file dialog
    filetypes = [("Image files", "*.jpg *.jpeg *.png"), ("All files", "*.*")]

    # Open the file dialog to select an image
    filename = filedialog.askopenfilename(
        initialdir="/",  # Start browsing from the root directory
        title="Select an Image",  # Set the title of the file dialog
        filetypes=filetypes  # Filter files to include only image files (jpg, jpeg, png) and all files
    )

    # If a file is selected, update the image path and display the image
    if filename:
        image_path.set(filename)  # Update the global `image_path` variable with the selected file's path
        show_image(filename)  # Call the `show_image` function to display the selected image in the GUI

# Function to display the selected image
def show_image(image_path):
    """
    Display an image in the GUI.

    Args:
        image_path (str): The path to the image file to display.

    Behavior:
        - Opens the image file using the `PIL.Image.open` method.
        - Resizes the image to a thumbnail size of 250x250 pixels for display.
        - Converts the image to a `PhotoImage` object compatible with `tkinter`.
        - Updates the `panel` widget to display the image.
        - Stores a reference to the image to prevent it from being garbage collected.
    """
    # Open the image file using PIL (Python Imaging Library)
    img = Image.open(image_path)

    # Resize the image to a thumbnail size of 250x250 pixels for display
    img.thumbnail((250, 250))  # Maintains aspect ratio while resizing

    # Convert the PIL image to a PhotoImage object compatible with tkinter
    img = ImageTk.PhotoImage(img)

    # Update the `panel` widget to display the image
    panel.config(image=img)

    # Store a reference to the image to prevent it from being garbage collected
    # (tkinter requires a reference to the image to keep it displayed)
    panel.image = img


# Function to handle listbox selection
def on_model_select(event):
    """
    Handle the event when a model is selected from the listbox.

    Behavior:
        - Displays a loading message in the GUI to indicate that the model is being loaded.
        - Forces an immediate update of the GUI to ensure the loading message is displayed.
        - Retrieves the selected model from the listbox and constructs the full path to the model.
        - Attempts to load the model and its configurations using the `load_model` function.
        - If successful, displays the model configurations in the GUI and updates the status message.
        - If an error occurs during loading, displays an error message using a messagebox.
        - Clears the loading message regardless of whether the model was loaded successfully or not.
    """
    # Show loading message
    loading_label.config(text="Loading model...")
    window.update_idletasks()  # Force GUI update

    # Get the selected model from the listbox
    selected_model = model_listbox.get(model_listbox.curselection())
    model_path = os.path.join(models_dir, selected_model)
    try:
        # Load the model and configurations
        global loaded_model, loaded_configs
        loaded_model, loaded_configs = load_model(model_path, device)
        display_configs(loaded_configs)  # Display configurations on the GUI
        show_info_frames()  # Show the info frames
        result_label.config(text="Model loaded successfully!")
    except Exception as e:
        messagebox.showerror("Error", f"Failed to load model: {e}")
    finally:
        # Clear loading message
        loading_label.config(text="")


# Function to handle device changes
def change_device(new_device):
    """
    Handle the event when the device (e.g., CPU or GPU) is changed.

    Behavior:
        - Displays a loading message in the GUI to indicate that the device is being switched.
        - Forces an immediate update of the GUI to ensure the loading message is displayed.
        - Updates the global `device` variable to the new device (e.g., "cuda" or "cpu").
        - If a model is already loaded, attempts to reload the model on the new device:
            - Retrieves the selected model from the listbox and constructs the full path to the model.
            - Reloads the model and its configurations using the `load_model` function.
            - Updates the displayed configurations in the GUI and sets a success message.
        - If an error occurs during reloading, displays an error message using a messagebox.
        - If no model is loaded, displays a message prompting the user to select a model first.
        - Clears the loading message regardless of the outcome.
    """
    # Show loading message
    loading_label.config(text=f"Switching to {new_device}...")
    window.update_idletasks()  # Force GUI update

    global device, loaded_model, loaded_configs
    device = torch.device(new_device)  # Update the device
    if loaded_model is not None:
        try:
            # Reload the model on the new device
            selected_model = model_listbox.get(model_listbox.curselection())
            model_path = os.path.join("models", selected_model)
            loaded_model, loaded_configs = load_model(model_path, device)
            display_configs(loaded_configs)  # Update the displayed configurations
            result_label.config(text=f"Model reloaded on {new_device}!")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to reload model: {e}")
        finally:
            # Clear loading message
            loading_label.config(text="")
    else:
        result_label.config(text="No model loaded. Please select a model first.")
        # Clear loading message
        loading_label.config(text="")


# Function to perform inference using the loaded model
def perform_inference():
    """
    Perform inference on the selected image using the loaded model.

    Behavior:
        - Checks if an image has been selected. If not, displays an error message and exits.
        - Checks if a model has been loaded. If not, displays an error message and exits.
        - Displays a loading message in the GUI to indicate that inference is in progress.
        - Forces an immediate update of the GUI to ensure the loading message is displayed.
        - Retrieves the required image size from the loaded model configurations.
        - Calls the `classify_image` function to perform inference on the selected image.
        - Maps the predicted class index to the corresponding class name using the loaded configurations.
        - Displays the predicted class name in the GUI.
        - If an error occurs during inference, displays an error message using a messagebox.
        - Clears the loading message regardless of the outcome.
    """
    if not image_path.get():
        messagebox.showerror("Error", "Please select an image first!")
        return
    if not loaded_model:
        messagebox.showerror("Error", "Please select a model first!")
        return

    # Show loading message
    loading_label.config(text="Classifying image...")
    window.update_idletasks()  # Force GUI update

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
    finally:
        # Clear loading message
        loading_label.config(text="")


# Function to get available devices
def get_available_devices():
    """
    Retrieve a list of available devices for running the model.

    Behavior:
        - Always includes "cpu" as a default device since it is universally available.
        - Checks if CUDA (GPU support) is available using `torch.cuda.is_available()`.
        - If CUDA is available, adds "cuda" to the list of devices.
        - Returns the list of available devices.

    Returns:
        list: A list of strings representing the available devices (e.g., ["cpu", "cuda"]).
    """
    devices = ["cpu"]  # CPU is always available
    if torch.cuda.is_available():
        devices.append("cuda")  # Add CUDA if available
    return devices


# Function to show the model and training info frames
def show_info_frames():
    """
    Make the model and training info frames visible.
    """
    model_info_frame.grid()  # Show the model info frame
    training_info_frame.grid()  # Show the training info frame


# Create the Tkinter window
window = Tk()
window.title("Image Classifier")

# Disable resizing of the window to maintain a fixed layout
window.resizable(False, False)  # Disable resizing in both width and height

# Variables
image_path = StringVar()  # Stores the path of the selected image
loaded_model = None  # Stores the currently loaded model
loaded_configs = None  # Stores the configurations of the loaded model

# Get available devices (e.g., CPU, CUDA)
available_devices = get_available_devices()

# Set the default device to CUDA if available, otherwise fallback to CPU
device_var = StringVar(value="cuda" if "cuda" in available_devices else "cpu")

# Layout

# Label and button for selecting an image
Label(window, text="Select Image:").grid(row=0, column=0, padx=10, pady=10)
browse_button = Button(window, text="Browse Files", command=browse_files)
browse_button.grid(row=0, column=1, padx=10, pady=10)

# Panel to display the selected image
panel = Label(window)
panel.grid(row=1, column=0, columnspan=2, pady=10)

# Listbox to display available models
Label(window, text="Available Models:").grid(row=2, column=0, padx=10, pady=10)
model_listbox = Listbox(window, width=50, height=10)
model_listbox.grid(row=3, column=0, columnspan=2, padx=10, pady=10)

# Bind the listbox selection event to the `on_model_select` function
model_listbox.bind('<<ListboxSelect>>', on_model_select)

# Scrollbar for the listbox to handle overflow
scrollbar = Scrollbar(window, orient="vertical")
scrollbar.config(command=model_listbox.yview)
scrollbar.grid(row=3, column=2, sticky="ns")
model_listbox.config(yscrollcommand=scrollbar.set)

# Populate the listbox with models from the "models" folder
models_dir = "models"
if os.path.exists(models_dir):
    for model_file in os.listdir(models_dir):
        if model_file.endswith(".pth"):  # Only include files with the .pth extension
            model_listbox.insert("end", model_file)
else:
    messagebox.showwarning("Warning", "No models folder found!")

# Device selection dropdown menu
Label(window, text="Select Device:").grid(row=0, column=3, padx=10, pady=10)
device_dropdown = OptionMenu(window, device_var, *available_devices, command=change_device)
device_dropdown.grid(row=0, column=4, padx=10, pady=10)

# Frames for displaying model and training information
model_info_frame = Frame(window, borderwidth=2, relief="groove")
model_info_frame.grid(row=1, column=3, padx=10, pady=10, sticky="nsew")
model_info_frame.grid_remove()  # Hide the frame initially

training_info_frame = Frame(window, borderwidth=2, relief="groove")
training_info_frame.grid(row=1, column=4, padx=10, pady=10, sticky="nsew")
training_info_frame.grid_remove()  # Hide the frame initially

# Button to perform inference on the selected image
predict_button = Button(window, text="Classify", command=perform_inference)
predict_button.grid(row=2, column=3, columnspan=2, pady=10)

# Label for displaying the predicted class or status messages
result_label = Label(window, text="Ready", font=("Arial", 12), wraplength=400)
result_label.grid(row=3, column=3, columnspan=2, pady=10, padx=20)

# Label for displaying loading or status messages
loading_label = Label(window, text="", font=("Arial", 12), fg="blue")
loading_label.grid(row=4, column=3, columnspan=2, pady=10)

# Run the Tkinter event loop to display the window and handle user interactions
window.mainloop()