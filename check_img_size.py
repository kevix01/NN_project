import os
from torchvision.datasets import Food101
from torchvision import transforms
from PIL import Image
from collections import defaultdict
from tqdm import tqdm

# Function to get the size of an image
def get_image_size(image_path):
    with Image.open(image_path) as img:
        return img.size  # Returns (width, height)

# Function to check min and max image sizes for each class in both splits
def check_image_sizes(root='./data'):
    # Load the Food-101 dataset for both splits
    train_dataset = Food101(root=root, split='train', download=True)
    test_dataset = Food101(root=root, split='test', download=True)

    # Dictionary to store min and max sizes for each class in both splits
    class_sizes = defaultdict(lambda: {
        'train': {'min': (float('inf'), float('inf')), 'max': (0, 0)},
        'test': {'min': (float('inf'), float('inf')), 'max': (0, 0)}
    })

    # Process the training split
    for idx in tqdm(range(len(train_dataset)), desc="Processing train images"):
        image_path, label = train_dataset._image_files[idx], train_dataset._labels[idx]
        class_name = train_dataset.classes[label]

        # Get the image size
        width, height = get_image_size(image_path)

        # Update min and max sizes for the class in the training split
        if width < class_sizes[class_name]['train']['min'][0] and height < class_sizes[class_name]['train']['min'][1]:
            class_sizes[class_name]['train']['min'] = (width, height)
        if width > class_sizes[class_name]['train']['max'][0] and height > class_sizes[class_name]['train']['max'][1]:
            class_sizes[class_name]['train']['max'] = (width, height)

    # Process the test split
    for idx in tqdm(range(len(test_dataset)), desc="Processing test images"):
        image_path, label = test_dataset._image_files[idx], test_dataset._labels[idx]
        class_name = test_dataset.classes[label]

        # Get the image size
        width, height = get_image_size(image_path)

        # Update min and max sizes for the class in the test split
        if width < class_sizes[class_name]['test']['min'][0] and height < class_sizes[class_name]['test']['min'][1]:
            class_sizes[class_name]['test']['min'] = (width, height)
        if width > class_sizes[class_name]['test']['max'][0] and height > class_sizes[class_name]['test']['max'][1]:
            class_sizes[class_name]['test']['max'] = (width, height)

    # Print the results in a compact format
    print("\nImage size statistics for each class:")
    for class_name, sizes in class_sizes.items():
        print(f"Class: {class_name}")
        print(f"  Train split: Min size = {sizes['train']['min']}, Max size = {sizes['train']['max']}")
        print(f"  Test split:  Min size = {sizes['test']['min']}, Max size = {sizes['test']['max']}")
        print()

if __name__ == '__main__':
    check_image_sizes()