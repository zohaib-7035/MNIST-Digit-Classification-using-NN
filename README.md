
# 🧠 MNIST Handwritten Digit Classification using Deep Learning (Neural Network)

This project is a deep learning-based solution for classifying handwritten digits (0–9) using the MNIST dataset. The model is trained using TensorFlow and Keras, achieving high accuracy on both training and test data.

---

## 📌 Project Overview

- 📊 Dataset: MNIST (60,000 training images + 10,000 test images)
- 🖼 Image Size: 28x28 grayscale images
- 🔍 Goal: Predict the digit in a given image
- 🤖 Model: Feedforward Neural Network (2 hidden layers)
- 🧪 Accuracy:
  - Training Accuracy: ~98.9%
  - Test Accuracy: ~97.1%

---

## 🔧 Technologies Used

- Python
- TensorFlow / Keras
- NumPy
- OpenCV
- Matplotlib
- Seaborn
- Google Colab

---

## 🚀 How the Model Works

1. **Data Loading & Preprocessing**:
   - MNIST dataset is loaded using `keras.datasets`.
   - Input data is normalized by dividing by 255.
   - Images are visualized using `matplotlib`.

2. **Model Architecture**:
   ```python
   keras.Sequential([
       keras.layers.Flatten(input_shape=(28, 28)),
       keras.layers.Dense(50, activation='relu'),
       keras.layers.Dense(50, activation='relu'),
       keras.layers.Dense(10, activation='sigmoid')
   ])
````

* Optimizer: Adam
* Loss: Sparse Categorical Crossentropy

3. **Training**:

   * Trained for 10 epochs.
   * Evaluation done using `model.evaluate()` on test data.

4. **Prediction**:

   * Accepts custom handwritten digit images using OpenCV.
   * Preprocesses the input (grayscale, resize, normalize).
   * Predicts digit using trained model.

5. **Evaluation**:

   * Displays confusion matrix with `seaborn.heatmap`.

---

## 📷 Predicting Custom Images

To predict your own handwritten digit image:

* Upload an image to Colab.
* Provide the file path when prompted.
* Image should be a clear 28x28 or larger digit on a white background.

```python
Path of the image to be predicted: /content/MNIST_digit.png
```

---

## 📁 Project Files

* `DL_Project_MNIST.ipynb` – Main notebook
* `MNIST_digit.png` – Sample input image for prediction
* `README.md` – This file

---

## 📊 Sample Output

```
The Handwritten Digit is recognised as: 7
```

---

## 🙋‍♂️ Author

**Zohaib Shahid**
Contributions welcome! Fork it, improve it, and don’t forget to ⭐ the repo if you find it helpful.

---

## 📎 References

* [MNIST Dataset](http://yann.lecun.com/exdb/mnist/)
* [TensorFlow](https://www.tensorflow.org/)
* [Keras Documentation](https://keras.io/)

```

---

Let me know if you'd like the markdown file in `.md` format or if you're adding this to a GitHub repository so I can tailor it better (e.g., adding badges, project links, etc.).
```
