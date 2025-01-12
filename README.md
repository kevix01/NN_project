# Food Image Classification with ResNet and Vision Transformer (ViT)

This project provides a Python-based solution for fine-tuning and using pre-trained ResNet and Vision Transformer (ViT) models for food image classification. The project includes a training script (`training_script.py`) for fine-tuning the models on a subset of the Food-101 dataset and a GUI script (`gui_script.py`) for loading trained models and performing inference on new images.

## Project Description

The goal of this project is to classify food images into one of five categories: `churros`, `carrot_cake`, `pork_chop`, `panna_cotta`, and `greek_salad`. The project leverages transfer learning by fine-tuning pre-trained ResNet and ViT models on the Food-101 dataset. The trained models can be saved and later used for inference via a user-friendly GUI.

### Key Features:
- Fine-tuning of ResNet and Vision Transformer (ViT) models.
- Support for custom image sizes, learning rate scheduling, and dropout configurations.
- Early stopping and saving the best model based on validation accuracy.
- A GUI for loading trained models, selecting images, and performing inference.
- Display of model and training configurations in the GUI.
  
## Project Structure

The project is structured as follows:

```bash
food-classification/
│
├── data/                   # Directory for storing the Food-101 dataset
├── models/                 # Directory for saving trained model checkpoints
├── examples/               # Directory containing sample images for testing models with the GUI
├── training_script.py      # Script for training and fine-tuning models
├── gui_script.py           # Script for loading models and performing inference via GUI
├── README.md               # Project documentation
└── LICENSE                 # License file for the project
```

## Description

This project is designed for fine-tuning and using pre-trained ResNet and Vision Transformer (ViT) models for food image classification. Below is a detailed description of the project structure and its components:

- **data/**:  
  This directory contains the Food-101 dataset, which is automatically downloaded when running the training script. The dataset includes images of 101 food categories, but this project focuses on a subset of 5 classes: `churros`, `carrot_cake`, `pork_chop`, `panna_cotta`, and `greek_salad`.

- **models/**:  
  This directory is used to save trained model checkpoints. Each checkpoint includes the model's state, training configurations, and hyperparameters. Models are saved with a `.pth` extension and can be loaded later for inference or further training.

- **examples/**:  
  This directory contains sample images that can be used to test the trained models with the GUI. Users can select these images from the GUI to see how the model performs on unseen data. The images should represent the five classes (`churros`, `carrot_cake`, `pork_chop`, `panna_cotta`, and `greek_salad`) to ensure the model's predictions can be verified.

- **training_script.py**:  
  This script is used for training and fine-tuning ResNet or Vision Transformer (ViT) models on the Food-101 dataset. It supports custom configurations such as image size, batch size, learning rate, and fine-tuning specific layers. The script also includes features like early stopping, learning rate scheduling, and saving the best model based on validation accuracy.

- **gui_script.py**:  
  This script provides a user-friendly graphical interface (GUI) for loading trained models and performing inference on new images. Users can select an image, choose a model, and view the predicted class along with model and training configurations. The GUI also supports switching between CPU and GPU for inference.

- **README.md**:  
  This file contains the project documentation, including instructions for setting up the environment, training models, and using the GUI for inference. It also provides details about the project structure, requirements, and acknowledgments.

- **LICENSE**:  
  This file contains the license under which the project is distributed. For this project, the MIT License is used, allowing users to freely use, modify, and distribute the code with proper attribution.


## How to Use

### 1. Training the Model

To train a model, use the `training_script.py` script. This script allows you to fine-tune either a ResNet or Vision Transformer (ViT) model on the Food-101 dataset.

#### Example Command:
```bash
python training_script.py --model resnet --image_size 224 --batch_size 40 --learning_rate 0.001 --num_epochs 10 --model_name resnet_model.pth --fine_tune_params layer1,layer2 --save_best_model
```

#### Arguments:
- `--model`: Model type (`resnet` or `vit`).
- `--image_size`: Input image size (default: 224). Must be divisible by 16 for ViT.
- `--batch_size`: Batch size for training (default: 40).
- `--learning_rate`: Initial learning rate (default: 0.001).
- `--num_epochs`: Number of training epochs (default: 10).
- `--model_name`: Name of the saved model checkpoint (e.g., `resnet_model.pth`).
- `--fine_tune_params`: Comma-separated list of layers to fine-tune (e.g., `layer1,layer2`).
- `--save_best_model`: Save the model with the best validation accuracy.
- `--early_stop`: Number of epochs to wait before early stopping if validation accuracy doesn't improve.
- `--seed`: Random seed for reproducibility.
- `--weight_decay`: Weight decay (L2 regularization) for the optimizer (default: 0.0).
- `--final_layer_dropout`: Dropout probability for the final classification layer (default: 0.0).
- `--mlp_dropout`: Dropout probability for the MLP blocks in Vision Transformer (ViT) (default: 0.0).
- `--attention_dropout`: Dropout probability for the attention mechanism in Vision Transformer (ViT) (default: 0.0).
- `--lr_scheduler`: Learning rate scheduler type (`static` or `linear`) (default: `static`).
- `--min_lr`: Minimum learning rate for the linear decay scheduler (default: None).

#### Output:
- The trained model is saved in the `models/` directory with the specified name.
- Training loss, validation accuracy, and learning rate are plotted at the end of training.

### 2. Using the GUI for Inference

To use the GUI for loading trained models and performing inference, run the `gui_script.py` script.

#### Example Command:
```bash
python gui_script.py
```

#### Steps:
1. **Select an Image**: Click the "Browse Files" button to select an image for classification.
2. **Load a Model**: Select a trained model from the list of available models in the `models/` directory.
3. **Select Device**: Choose the device (CPU or GPU) for inference.
4. **Classify**: Click the "Classify" button to perform inference on the selected image.
5. **View Results**: The predicted class and model/training configurations will be displayed in the GUI.

#### Features:
- Displays model and training configurations (e.g., model type, image size, learning rate).
- Supports switching between CPU and GPU for inference.
- Shows the selected image and predicted class in the GUI.

## Requirements

To run this project, you need the following Python packages:

- **`torch`**:  
  The core library for PyTorch, used for building, training, and evaluating neural networks. It provides tensor operations and automatic differentiation.

- **`torchvision`**:  
  A library that provides datasets, model architectures, and image transformations for computer vision tasks. It is used for loading the Food-101 dataset and applying image preprocessing.

- **`numpy`**:  
  A fundamental library for numerical computing in Python. It is used for array manipulations and mathematical operations during data processing.

- **`matplotlib`**:  
  A plotting library used for visualizing training metrics such as loss, accuracy, and learning rate over epochs.

- **`tqdm`**:  
  A progress bar library used to display progress during training, validation, and inference loops for better user feedback.

- **`PIL` (Pillow)**:  
  The Python Imaging Library (Pillow) is used for loading, resizing, and displaying images in the GUI. It is essential for handling image input and output operations.

- **`tkinter`**:  
  The standard Python interface to the Tk GUI toolkit. It is used to create the graphical user interface (GUI) for loading models, selecting images, and displaying results.

You can install the required packages using `pip`:

```bash
pip install torch torchvision numpy matplotlib tqdm pillow
```

## Notes

- The Food-101 dataset will be automatically downloaded to the `data/` directory when you run the training script.
- Ensure that the `models/` directory exists to save trained models.
- The GUI script requires trained models to be saved in the `models/` directory with a `.pth` extension.

## License

This project is licensed under the MIT License. See the LICENSE file for details.

## Acknowledgments

- The Food-101 dataset is provided by the [Food-101 project](https://data.vision.ee.ethz.ch/cvl/datasets_extra/food-101/).
- Pre-trained ResNet and Vision Transformer models are provided by [PyTorch](https://pytorch.org/vision/stable/models.html).
