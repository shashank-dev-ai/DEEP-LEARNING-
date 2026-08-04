# Vanishing Gradient Problem

## Overview

The vanishing gradient problem is a common issue encountered while training deep neural networks using the backpropagation algorithm.

During backpropagation, gradients of the loss function with respect to the weights are computed using the chain rule. When activation functions such as sigmoid or tanh are used, their derivatives are very small (between 0 and 1). Multiplying these small values repeatedly across many layers causes the gradients to become extremely small.

As a result, the weights in the earlier layers receive negligible updates, making the network difficult to train.

---

## Why Does It Happen?

Suppose the derivative of an activation function is 0.5.

After passing through multiple layers:

```text
0.5 × 0.5 × 0.5 × 0.5 × 0.5 = 0.03125
```

For a very deep network:

```text
0.5^20 ≈ 0.000001
```

The gradient becomes almost zero.

---

## Consequences

- Very slow learning in the initial layers.
- Negligible weight updates.
- Slow convergence.
- Poor performance on deep neural networks.

---

## Experiment

I trained two neural networks:

1. Neural network using the Sigmoid activation function.
2. Neural network using the ReLU activation function.

### Observations

- Sigmoid resulted in very small weight updates.
- ReLU resulted in larger weight updates.
- ReLU converged much faster.

---

## Solutions

- Use ReLU activation instead of Sigmoid or Tanh.
- Use Batch Normalization.
- Use He initialization.
- Use Residual Networks (ResNet).
- Use LSTM or GRU for recurrent networks.

---

## Key Takeaway

The choice of activation function plays a crucial role in training deep neural networks. ReLU helps mitigate the vanishing gradient problem by maintaining larger gradients during backpropagation.

---

## Technologies Used

- Python
- TensorFlow
- Keras
- NumPy
- Matplotlib

