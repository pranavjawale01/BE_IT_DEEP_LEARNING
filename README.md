# BE_IT_DEEP_LEARNING

## Required Libraries Installation

```bash
pip install tensorflow          # Used for building and training neural networks.
pip install keras               # High-level neural networks API, running on top of TensorFlow.
pip install torch               # An alternative deep learning framework.
pip install numpy               # Fundamental package for numerical computations in Python.
pip install pandas              # Data analysis and manipulation library.
pip install matplotlib          # Plotting library for creating visualizations.
pip install seaborn             # Statistical data visualization library based on Matplotlib.
pip install scikit-learn        # Machine learning library for data preprocessing and evaluation.
pip install scipy               # Library for scientific and technical computing.
pip install nltk spacy          # For natural language processing tasks.
pip install theano
pip install scikit-learn
```
# NEW Problem Statement

| No | Project Description                                                             |
|----|-------------------------------------------------------------------------------|
| 1  | Demonstrate use of TensorFlow and PyTorch by implementing simple code in Python |
| 2  | Demonstrate use of TensorFlow and Keras (Diabetes Dataset) by implementing simple code in Python |
| 3  | Implement Feedforward neural networks with Keras and TensorFlow MNIST Digit dataset |
| 4  | Implement Feedforward neural networks with Keras and TensorFlow CIFAR image dataset |
| 5  | Build image classification model using CNN on Fashion MNIST dataset           |
| 6  | Build image classification model using CNN on pneumonia X-ray image dataset   |
| 7  | Build Brain Tumor classification model with CNN                               |
| 8  | Build Recurrent Neural Network by using the NumPy library                      |
| 9  | Use Autoencoder to implement anomaly detection on credit card dataset          |
| 10 | Implement the concept of image denoising using autoencoders on MNIST dataset  |
| 11 | Implement object detection using Transfer Learning on flower dataset           |
| 12 | Implement image classification using Transfer Learning on animal dataset       |
| 13 | Implement the Continuous Bag of Words (CBOW) Model                           |


# Syllabus Problem Statement

| Assignment | Description | Steps |
|------------|-------------|-------|
| **1. Study of Deep Learning Packages** | Compare TensorFlow, Keras, Theano, and PyTorch on features and functionality. | 1. **Overview**: Brief overview of each framework. <br> 2. **Comparison**: Evaluate usability, community, deployment, and performance. <br> 3. **Demo Implementation**: Run a simple example on MNIST or CIFAR10. |
| **2. Feedforward Neural Networks with Keras & TensorFlow** | Build a basic feedforward network for image classification on MNIST or CIFAR10. | 1. **Import Packages**: TensorFlow, Keras, numpy, etc. <br> 2. **Load Data**: Training/testing data. <br> 3. **Define Network Architecture**: Input, hidden, and output layers with ReLU and Softmax activations. <br> 4. **Train**: Use SGD optimizer. <br> 5. **Evaluate**: Test set accuracy. <br> 6. **Plot Metrics**: Loss and accuracy across epochs. |
| **3. Image Classification Model** | Develop an image classification pipeline with preprocessing, architecture, training, and evaluation. | 1. **Load & Preprocess Data**: Resize, normalize, and split. <br> 2. **Define Architecture**: Convolutional and dense layers with ReLU/Softmax. <br> 3. **Train**: Choose SGD or Adam optimizer. <br> 4. **Evaluate**: Check accuracy, precision, recall. |
| **4. Anomaly Detection with Autoencoder** | Use autoencoder to detect anomalies, compressing and reconstructing inputs. | 1. **Import Libraries**: TensorFlow, Keras, numpy, etc. <br> 2. **Load Data**: Anomaly detection dataset. <br> 3. **Build Encoder**: Map inputs to latent space. <br> 4. **Build Decoder**: Reconstruct input. <br> 5. **Compile Model**: Set optimizer, loss (e.g., MSE), and evaluation metrics. |
| **5. CBOW Model** | Create a CBOW model for word embeddings, focusing on data prep, training, and output. | 1. **Data Preparation**: Clean, tokenize, and prepare text. <br> 2. **Generate Training Data**: Context-target pairs. <br> 3. **Train Model**: Predict target based on context. <br> 4. **Output Embeddings**: Extract learned embeddings. |
| **6. Object Detection with Transfer Learning** | Build an object detection model with a pre-trained CNN, adding custom layers. | 1. **Load Pre-trained CNN**: E.g., ResNet or VGG on ImageNet. <br> 2. **Freeze Layers**: Lock lower conv layers. <br> 3. **Add Classifier**: Custom trainable layers for object detection. <br> 4. **Train Classifier**: Task-specific data. <br> 5. **Tune**: Adjust hyperparameters, unfreeze layers as needed. |



# College Problem Statement

| Sr.No | Title of Assignment | Description |
|-------|----------------------|-------------|
| 1 | **Demonstrate use of TensorFlow and PyTorch by implementing simple code in Python** | Implement basic operations and models in both TensorFlow and PyTorch to demonstrate their functionality and differences. |
| 2 | **Implement Feedforward Neural Networks with Keras and TensorFlow on MNIST Digit Dataset** | Use Keras and TensorFlow to create, train, and evaluate a feedforward neural network for digit classification with the MNIST dataset. |
| 3 | **Implement Feedforward Neural Networks with Keras and TensorFlow on CIFAR Dataset** | Create a feedforward neural network model using Keras and TensorFlow, focusing on image classification tasks with the CIFAR dataset. |
| 4 | **Build Image Classification Model Using CNN on Fashion MNIST Dataset** | Build and train a convolutional neural network (CNN) for image classification using the Fashion MNIST dataset. |
| 5 | **Build Image Classification Model Using CNN on Pneumonia X-Ray Image Dataset** | Design and implement a CNN to classify pneumonia from chest X-ray images. Utilize transfer learning if applicable. |
| 6 | **Build Image Classification Model Using CNN on Food Dataset** | Construct and train a CNN model to classify various types of food images, focusing on accuracy and generalization. |
| 7 | **Build Brain Tumor Classification Model with CNN** | Create a CNN model to classify brain tumor images, evaluating its accuracy in distinguishing between tumor and non-tumor images. |
| 8 | **Build Recurrent Neural Network by Using the Numpy Library** | Implement an RNN from scratch using only Numpy for sequence processing tasks, demonstrating fundamental RNN operations. |
| 9 | **Implement Simple Autoencoder to Reconstruct MNIST Digits with Sparsity Constraint** | Build an autoencoder model with a sparsity constraint on encoded representations, focusing on reconstructing MNIST digit images. |
| 10 | **Use Autoencoder to Implement Anomaly Detection on Credit Card Dataset** | Develop an autoencoder model to detect anomalies (fraud) in a credit card transactions dataset by reconstructing normal patterns. |
| 11 | **Implement Image Denoising Using Autoencoders on MNIST Dataset** | Use an autoencoder to remove noise from images in the MNIST dataset, focusing on achieving clear reconstructions. |
| 12 | **Implement Object Detection Using Transfer Learning on Food Dataset** | Implement object detection using a pre-trained CNN model fine-tuned on a food dataset to identify different food items. |
| 13 | **Implement Image Classification Using Transfer Learning on Animal Dataset** | Utilize transfer learning to classify images in an animal dataset, adjusting and fine-tuning layers as needed for performance. |
| 14 | **Implement the Continuous Bag of Words (CBOW) Model** | Create a CBOW model for word embeddings, preparing data, training, and generating vector representations for words. |
