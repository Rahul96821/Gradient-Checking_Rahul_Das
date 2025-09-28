# Fraud Detection Using Deep Learning

This project implements a **deep learning model** for detecting fraudulent transactions in mobile payments. It demonstrates **forward propagation, backward propagation, and gradient checking** to ensure the model's reliability, which is critical for mission-critical applications.

---

## 📝 Project Overview

Fraud detection is a high-stakes problem in mobile payments. This project focuses on:

* Building a multi-layer neural network for fraud detection.
* Implementing forward and backward propagation.
* Using **gradient checking** to verify the correctness of backpropagation.
* Ensuring high confidence in model performance before deploying in production.

---

## ⚙️ Features

* **1D Gradient Checking:** Simple demonstration to validate gradient computation.
* **Multi-layer Neural Network:** ReLU activations in hidden layers, Sigmoid in output.
* **Gradient Checking for Multi-layer Networks:** Confirms that backpropagation is correctly implemented.
* **Mission-critical Validation:** Ideal for high-stakes applications like fraud detection.

---

## 📂 File Structure

```
.
├── gc_utils.py           # Utility functions (sigmoid, relu, dictionary/vector conversions)
├── testCases.py          # Test cases for 1D and multi-layer gradient checks
├── public_tests.py       # Public test functions
├── gradient_checking.ipynb  # Jupyter Notebook with implementation
└── README.md             # Project documentation
```

---

## 🧮 Implementation

### Forward Propagation (1D Example)

```python
def forward_propagation(x, theta):
    return theta * x
```

### Backward Propagation (1D Example)

```python
def backward_propagation(x, theta):
    return x
```

### Gradient Checking (1D)

```python
def gradient_check(x, theta, epsilon=1e-7):
    theta_plus = theta + epsilon
    theta_minus = theta - epsilon
    gradapprox = (forward_propagation(x, theta_plus) - forward_propagation(x, theta_minus)) / (2 * epsilon)
    grad = backward_propagation(x, theta)
    difference = np.linalg.norm(grad - gradapprox) / (np.linalg.norm(grad) + np.linalg.norm(gradapprox))
    return difference
```

### Multi-layer Network

* **Forward propagation:** ReLU → ReLU → Sigmoid
* **Backward propagation:** Compute gradients for all weights and biases
* **Gradient checking:** Verify multi-layer backpropagation using vectorized approach

---

## ✅ Usage

1. Clone the repository:

```bash
git clone https://github.com/yourusername/fraud-detection.git
cd fraud-detection
```

2. Install dependencies:

```bash
pip install numpy
```

3. Run Jupyter Notebook:

```bash
jupyter notebook gradient_checking.ipynb
```

4. Follow notebook to see:

   * Forward and backward propagation
   * Gradient checking results
   * Model validation

---

## 📊 Results

* **1D Gradient Checking:** Difference ~ 0 → backpropagation is correct
* **Multi-layer Gradient Checking:** Confirms correctness of all parameter gradients
* Ensures confidence in model implementation for production use

---

## ⚠️ Notes

* Gradient checking is **computationally expensive**, so only use for testing.
* Do not run gradient checking with **dropout enabled**.
* Once gradients are verified, the model is ready for **training and deployment**.

---

## 🏆 Conclusion

With gradient checking, you can be confident that your deep learning model computes correct gradients. This provides a **strong foundation for accurate fraud detection**, ensuring high reliability in real-world applications.

If you want, I can also **add badges, Python version info, and a “Getting Started” section with example outputs** to make it more professional for GitHub.
