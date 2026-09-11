# Neural Network from Scratch 🧠

A beginner-friendly, educational mini-project implementing a Neural Network in Python. 

We recently reformatted the project into a new notebook `copy.ipynb` divided into **three progressive categories** to help you build intuition from the absolute basics up to GPU-accelerated training using PyTorch.

---

## Category 1: Single Node Neural Network (The Building Block)

At the core of a neural network is a **Neuron**. It takes numerical inputs, applies some math, and produces an output.

<div align="center">
  <img src="https://victorzhou.com/a74a19dc0599aae11df7493c718abaf9/perceptron.svg" alt="A Single Neuron" />
</div>

Here is the step-by-step math for a single neuron with two inputs ($x_1$ and $x_2$):
1. **Weights and Bias**: Each input is multiplied by a **weight** ($w_1$, $w_2$), and a **bias** ($b$) is added to the sum.
   $$ z = (x_1 \times w_1) + (x_2 \times w_2) + b $$
2. **Activation Function**: The result $z$ is then passed through an **activation function** to map the output to a predictable range (usually between 0 and 1). We use the **Sigmoid** function:
   $$ f(z) = \frac{1}{1 + e^{-z}} $$

<div align="center">
  <img src="https://victorzhou.com/static/dd5a39500acbef371d8d791d2cd381e0/8c557/sigmoid.png" alt="Sigmoid Function" />
</div>

---

## Category 2: Multi-Layer Neural Network & Training (From Scratch)

In this category, we connect multiple neurons to form a network and train it on a small dataset purely using `numpy`.

<div align="center">
  <img src="https://victorzhou.com/27cf280166d7159c0465a58c68f99b39/network3.svg" alt="A Simple Neural Network" />
</div>

### Forward Propagation
The process of passing data from the **Input Layer**, through the **Hidden Layer(s)**, and to the **Output Layer** to get a prediction is called Forward Propagation.

### Measuring Success: The Loss Function
<div align="center">
  <img src="https://victorzhou.com/static/99e7886af56d6f41b484d17a52f9241b/3ebb1/loss.png" alt="Loss Function" />
</div>

We calculate the **Loss** to see how accurate our predictions are. We initially use **Mean Squared Error (MSE)**, which takes the difference between the true answer ($y_{true}$) and our prediction ($y_{pred}$), squares it, and averages it.
$$ MSE = \frac{1}{n} \sum_{i=1}^n (y_{true} - y_{pred})^2 $$

### Learning: Backpropagation and Gradient Descent
To minimize the loss, we calculate the **derivative** of the loss with respect to each weight and bias using the Chain Rule (Backpropagation). We then update our weights using **Gradient Descent**:
$$ w = w - (\text{Learning Rate} \times \text{Derivative}) $$
By repeating this over many epochs, the network learns!

---

## Category 3: Training on GPU using PyTorch

While the pure `numpy` implementation is great for learning, it processes data slowly on the CPU. To train on our larger 3,000-sample dataset, we transition to **PyTorch**.

### Why PyTorch & GPU?
Instead of using Python `for` loops which process one sample at a time, PyTorch leverages your GPU (e.g., RTX 3050). The GPU has thousands of CUDA cores that can execute matrix multiplications in parallel.

### Major Upgrades from Category 2:
1. **Network Capacity**: We expanded the hidden layer from 2 neurons to **16 neurons** (`nn.Linear(2, 16)`). A larger dataset requires more parameters to capture complex patterns.
2. **Binary Cross-Entropy (BCE) Loss**: We swapped out Mean Squared Error (MSE) for BCE. MSE suffers from the *vanishing gradient problem* when used with Sigmoid, which causes the network's loss to plateau (get stuck at a high number like 0.96). BCE provides strong gradients and is the mathematically correct loss function for binary classification (0 or 1 labels).
   $$ \text{BCE} = -[y_{true} \times \log(y_{pred}) + (1 - y_{true}) \times \log(1 - y_{pred})] $$
3. **Accuracy Tracking**: Loss tells us how confident the network is, but **Accuracy** tells us how often it guesses correctly (e.g. 95% accuracy). We track this during inference.
4. **Z-Score Normalization**: We scaled the input data by subtracting the mean and dividing by the standard deviation. Feeding massive, unscaled numbers directly into a Sigmoid activation causes the gradient to become mathematically zero, permanently freezing the network's learning (Vanishing Gradient).
5. **Data Cleaning (Outlier Removal)**: We filtered out dirty, physically impossible data points (e.g. individuals listed as 3050 cm tall) before training. If left in, these outliers completely warp the Z-score normalization math and cripple the network's performance.

### Implementation Details:
- **Tensors & VRAM**: We convert our `numpy` arrays into `torch.tensor` and move them to the GPU using `.to(device)`.
- **nn.Module**: We replace manual weights and biases with `nn.Linear()`, which handles initialization and matrix operations automatically.
- **Autograd**: We completely remove our manual calculus code. PyTorch tracks operations and calculates gradients automatically when we call `loss.backward()`.
- **Batches**: We use a `DataLoader` to pass data to the GPU in mini-batches (e.g., 64 samples at a time) for more efficient and stable training.

---

## Prerequisites
To run the notebooks, install the required packages:

```bash
pip install numpy pandas jupyter torch
```

## Usage
Open the notebook in your preferred environment:
```bash
jupyter notebook copy.ipynb
```
Follow along with the markdown cells and run the code sequentially!
