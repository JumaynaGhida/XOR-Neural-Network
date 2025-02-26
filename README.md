# XOR Neural Network

This project is a simple neural network implemented in Python using **NumPy** to solve the **XOR problem**, a fundamental challenge in early artificial neural networks.  

## 📌 Why is this Code Important?

The XOR problem is a classic example in machine learning that demonstrates the need for **non-linear models**. A simple perceptron cannot solve XOR because it is not linearly separable. This project implements a **basic feedforward neural network with one hidden layer**, allowing it to learn the XOR function through **backpropagation** and **gradient descent**.  

### 🔹 **Key Features:**
- **Custom-built neural network** (without frameworks like TensorFlow/PyTorch).  
- **Backpropagation and gradient descent** for weight updates.  
- **Sigmoid activation function** for non-linearity.  
- **Mean Squared Error (MSE) loss function** for training.  

## 🚀 Running the Code on GitHub Codespaces

You can run this code directly on GitHub without installing anything on your local machine using **GitHub Codespaces**.

### 🔹 **Steps:**
1. **Open the repository** on GitHub.
2. Click the **"Code"** button (top right of the page).
3. Go to the **"Codespaces"** tab.
4. Click **"Create codespace on main"**.
5. Wait for the development environment to load in your browser.
6. Once **Codespaces** is open, open the **Terminal** from the top menu or press `Ctrl + Shift + P` and type **"New Terminal"**.
7. Make sure the required dependencies are installed:
   ```bash
   pip install numpy
8. Run the code with:
python problem-solution.py
