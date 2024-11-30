
Sketch Classifier Web App
=========================

Sketch Classification with Deep Learning

\=====================

Overview
========

* * *

This project focuses on creating an interactive web application for sketch classification. Users can draw sketches on a canvas, and the system predicts the sketch's class using a **deep learning model**. The application provides real-time predictions, displaying the top-5 probabilities.

Features
========

* * *

*   **Interactive Drawing Canvas**: Enables users to create sketches directly within the app.
*   **Real-time Predictions**: Utilizes a deep learning model to deliver real-time class predictions with confidence scores for the top 5 classes.
*   **Streamlit Deployment**: The app is deployed using **Streamlit**, ensuring a simple and interactive user experience.

Model Architecture
==================

* * *

The model is built using TensorFlow and consists of the following layers:

1.  **Convolutional Layers**:
    *   3 Conv1D layers with ReLU activation and Batch Normalization.
    *   Kernel sizes: 5, 5, and 3; strides: 2 for each layer.
2.  **Recurrent Layers**:
    *   2 LSTM layers: the first returns sequences, and the second outputs the final state.
3.  **Dense Layer**:
    *   Fully connected layer with a softmax activation function for classification across all sketch classes.

**Optimizer**: Stochastic Gradient Descent (SGD) with a learning rate of 0.01 and gradient clipping (`clipnorm=1.0`).  
**Loss Function**: Sparse categorical cross-entropy.  
**Metrics**: Accuracy and top-k categorical accuracy (k=5).

Model Training
==============

* * *

The model was trained on the cropped TFRecord dataset using the following setup:

Parameter

Value

**Dataset Format**

TFRecords

**Training Epochs**

2

**Validation Data**

Cropped Valid Set

**Batch Size**

Default

Installation
============

* * *

1.  Clone the repository:
    
    ```bash
    git clone https://github.com/your_username/sketch-classifier.git
    cd sketch-classifier
    ```
    
2.  Install the required dependencies:
    
    ```bash
    pip install -r requirements.txt
    ```
    
3.  Download the dataset and place it in the appropriate directory.
    

Usage
=====

* * *

1.  Run the Streamlit app:
    
    ```bash
    streamlit run app.py
    ```
    
2.  Use the canvas to draw sketches and view real-time predictions.
    

Dataset
=======

* * *

The dataset consists of sketches stored in **TFRecords** format. While it is not yet available in TensorFlow Datasets (TFDS), a [pull request](https://github.com/tensorflow/datasets/pull/361) is in progress.

The dataset includes:

*   **3,450,000 training samples**
*   **345,000 test samples**

Results
=======

* * *

The model achieved robust performance after two training epochs, with promising results for real-time sketch classification. Performance metrics will be updated after extended training and fine-tuning.

Future Enhancements
===================

* * *

*   Increase training epochs for improved accuracy and performance.
*   Optimize the model for real-time use on mobile and edge devices.
*   Extend the dataset with more diverse examples to improve robustness.

Contributing
------------

* * *

Contributions are welcome! Feel free to fork the repository and submit a pull request.

Contact
-------

* * *

For any inquiries, reach out to ebrahemelsherif666i@gmail.com
