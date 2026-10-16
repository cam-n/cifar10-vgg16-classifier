# CIFAR-10 Image Classification with VGG16 (CNN, Raw Dataset)

A convolutional neural network (VGG16-style, with BatchNorm and Dropout) trained from scratch on the raw CIFAR-10 data batches. The project covers two problems:

- **Problem 0:** convert the raw stored image arrays into a standard, library-friendly image format.
- **Problem 1:** build an image classifier that predicts the class label of each image.

## Results

| Setup | Validation accuracy |
|---|---|
| VGG16 + data augmentation | **91.58%** |
| VGG16 without data augmentation | 82.13% |

- Trained for 111 epochs with early stopping (patience 15); the best accuracy was reached at epoch 96.
- Training and validation curves stay close and there is no large variation in the validation line, so the model is stable and does not show clear overfitting.

## Repository Structure

```
cnn-raw-dataset/
├── vgg16.ipynb        # Data loading, augmentation, VGG16 model, training, evaluation
├── classes.names      # Class labels, one per line, in label-id order
├── data_batch_1 ...   # Raw CIFAR-10 training batches (data_batch_1 to data_batch_5)
└── README.md
```

## Dataset

The data is [CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar.html): 32×32 RGB images in 10 classes.

| ID | Class | ID | Class |
|---|---|---|---|
| 0 | airplane | 5 | dog |
| 1 | automobile | 6 | frog |
| 2 | bird | 7 | horse |
| 3 | cat | 8 | ship |
| 4 | deer | 9 | truck |

The labels are in `classes.names`, in the exact order of the label ids stored in the data batches.

**Splits used in the notebook:** the 5 batches (50,000 images) are combined and split into 45,000 images for training and 5,000 for validation.

### Loading the batches

There are 5 batches of 10,000 images each. Load them with Python's `pickle` module:

```python
import pickle

with open(file, 'rb') as batch_file:
    data = pickle.load(batch_file, encoding='latin1')  # or encoding='bytes'

images = data['data']      # shape (10000, 3072)
labels = data['labels']    # list of 10000 label ids
```

With `encoding='bytes'` the dictionary keys come back as byte strings (`b'data'`, `b'labels'`). Using `encoding='latin1'`, as the notebook does, returns normal string keys.

### Image layout (important)

`data` is a 10000 × 3072 NumPy array. Each row is one 32×32 RGB image stored in row-major order: the first 1024 values are the red channel, the next 1024 the green, and the last 1024 the blue.

## Problem 0: Converting the raw arrays

Reshape each row into channels × height × width, then move the channel axis to the end so the images work with matplotlib and OpenCV:

```python
# (N, 3072) -> (N, 3, 32, 32) -> (N, 32, 32, 3)
images = images.reshape(N, 3, 32, 32).transpose(0, 2, 3, 1)
```

## Problem 1: Image classifier

**Model.** A VGG16-style network ([Simonyan & Zisserman](https://arxiv.org/pdf/1409.1556.pdf)):

- 5 convolutional blocks (64, 128, 256, 512, 512 filters) with 3×3 kernels and max pooling
- Batch normalization after every convolution ([Ioffe & Szegedy](https://arxiv.org/pdf/1502.03167.pdf))
- Dropout (0.25) after each block and in the dense layers to reduce overfitting
- Dense layers of 512 and 256 units, then a 10-way softmax

**Preprocessing.** Images are standardized using the training-set mean and standard deviation, and labels are one-hot encoded.

**Data augmentation** (training set only), chosen by trying several parameter subsets:

| Parameter | Value |
|---|---|
| `rotation_range` | 15 |
| `horizontal_flip` | True |
| `width_shift_range` | 0.1 |
| `height_shift_range` | 0.1 |
| `zoom_range` | 0.3 |

**Training.**

| Setting | Value |
|---|---|
| Optimizer | SGD, initial learning rate 0.1 with exponential decay (rate 0.9 every 10,000 steps) |
| Loss | Categorical cross-entropy |
| Batch size | 128 |
| Max epochs | 200 |
| Early stopping | Monitors validation accuracy, patience 15, restores the best weights |

The best model is saved after training and reloaded for evaluation. The validation set can be replaced by a test set at that step.

## Getting Started

**Environment** (versions used for this project):

- Jupyter Notebook 6.1.4
- Python 3.8.5
- TensorFlow 2.3.1
- Keras 2.4.3
- NumPy 1.18.5
- Matplotlib 3.3.2

**Run:**

```bash
git clone https://github.com/cam-n/cifar10-vgg16-classifier.git
cd cifar10-vgg16-classifier
pip install tensorflow==2.3.1 numpy==1.18.5 matplotlib==3.3.2 notebook
jupyter notebook vgg16.ipynb
```

Run the notebook from the repo folder, since it reads the data batches and `classes.names` from the current directory.
