<table>
  <tr>
    <td><img src="https://github.com/harshjuly12/Hand-Gesture-Recognition-Using-CNN/assets/112745312/b7445b6c-7e0e-415b-af64-bca47a29079c" width="120" style="margin-right: 10;"></td>
    <td><h1 style="margin: 0;">Hand Gesture Recognition Using Convolutional Neural Networks (CNN)</h1></td>
  </tr>
</table>

## Table of Contents
1. [Project Overview](#project-overview)
2. [Dataset](#dataset)
3. [Project Structure](#project-structure)
4. [Requirements](#requirements)
5. [Installation](#installation)
6. [Usage](#usage)
7. [Example](#example)
8. [Analysis and Results](#analysis-and-results)
9. [Contributing](#contributing)
10. [License](#license)
11. [Author](#author)
    
## Project Overview
This project implements a hand gesture recognition system using Convolutional Neural Networks (CNN). The dataset comprises near-infrared images acquired by the Leap Motion sensor, featuring 10 different hand gestures performed by 10 subjects (5 men and 5 women).

## Dataset
The dataset includes 10 categories of hand gestures:
- 01_palm
- 02_l
- 03_fist
- 04_fist_moved
- 05_thumb
- 06_index
- 07_ok
- 08_palm_moved
- 09_c
- 10_down

Images are resized to 50x50 pixels and converted to grayscale for processing.

## Project Structure
The project is structured as follows:
- **Data Loading and Preprocessing**: Images are loaded from the dataset, resized, and converted to grayscale.
- **Model Building**: A Convolutional Neural Network (CNN) architecture is defined using Keras.
- **Model Training**: The CNN model is trained on the processed dataset.
- **Evaluation**: Model performance is evaluated using accuracy and other relevant metrics.

## Requirements
To run the project, you need the following libraries:
- NumPy
- Pandas
- Matplotlib
- Seaborn
- Plotly
- Scikit-learn

## Installation
1. Clone the repository:
   ```sh
   git clone <repository-url>
   cd HandGestureRecognitionUsingCNN
   ```
2. Install the required Python libraries:
   ```sh
   pip install -r requirements.txt
   ```
## Usage
1. Navigate to the project directory:
   ```sh
   cd HandGestureRecognitionUsingCNN
   ```
2. Run the Jupyter notebook for detailed steps and execution:
   ```sh
    jupyter notebook HandGestureRecognitionUsingCNN.ipynb
   ```

## Example
# Import necessary libraries
```sh
import warnings
warnings.filterwarnings('ignore')
import keras
import matplotlib.pyplot as plt
import os
import cv2
import numpy as np
from keras.layers import Conv2D, Activation, MaxPool2D, Dense, Flatten, Dropout

# Define categories and image size
CATEGORIES = ["01_palm", "02_l", "03_fist", "04_fist_moved", "05_thumb", "06_index", "07_ok", "08_palm_moved", "09_c", "10_down"]
IMG_SIZE = 50

# Path to the dataset
data_path = "../input/leapgestrecog/leapGestRecog"

# Load and preprocess images
image_data = []
for dr in os.listdir(data_path):
    for category in CATEGORIES:
        class_index = CATEGORIES.index(category)
        path = os.path.join(data_path, dr, category)
        for img in os.listdir(path):
            try:
                img_arr = cv2.imread(os.path.join(path, img), cv2.IMREAD_GRAYSCALE)
                image_data.append([cv2.resize(img_arr, (IMG_SIZE, IMG_SIZE)), class_index])
            except Exception as e:
                pass
```

## Analysis and Results
The notebook contains the following steps:
1. Importing Libraries: Importing necessary libraries for analysis and visualization.
2. Data Exploration: Exploring the dataset to understand the distribution and relationships between different variables.
3. Data Preprocessing: Preparing the data for clustering by scaling the features.
4. K-means Clustering: Implementing K-means clustering to group customers into segments.
5. Visualization: Visualizing the clusters to interpret the results.

## Contributing
Contributions are welcome! Please open an issue or submit a pull request for any improvements or bug fixes.

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Author
For any questions or suggestions, please contact:
- Harsh Singh: [harshjuly12@gmail.com](harshjuly12@gmail.com)
- GitHub: [harshjuly12](https://github.com/harshjuly12)
