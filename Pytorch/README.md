# PyTorch Complete Syllabus

## Phase 1: PyTorch Fundamentals
1. Introduction to PyTorch
2. PyTorch vs NumPy
3. Tensors
4. Tensor Dimensions (0D, 1D, 2D, 3D)
5. Tensor Creation
6. Tensor Attributes (shape, dtype, device)
7. Indexing and Slicing
8. Reshaping (view, reshape, flatten)
9. Tensor Operations (Addition, Subtraction, Multiplication)
10. Matrix Multiplication (matmul, mm)
11. Broadcasting
12. CPU vs GPU (CUDA)

## Phase 2: Autograd & Mathematical Foundations
13. Computational Graph
14. Automatic Differentiation
15. requires_grad
16. Gradients (.grad)
17. backward()
18. Gradient Accumulation
19. torch.no_grad()
20. detach()
21. Partial Derivatives
22. Chain Rule
23. Gradient Descent
24. Backpropagation

## Phase 3: Neural Networks
25. Introduction to Neural Networks
26. Artificial Neuron
27. Weights and Biases
28. Activation Functions
29. Sigmoid, Tanh, ReLU, GELU, Softmax
30. torch.nn
31. nn.Module
32. nn.Linear
33. Forward Propagation
34. Loss Functions
35. MSE Loss
36. Cross Entropy Loss
37. Optimizers
38. SGD
39. Adam
40. AdamW

## Phase 4: Model Training
41. Training Loop
42. Forward Pass
43. Loss Calculation
44. Backward Pass
45. Optimizer Step
46. Zero Gradients
47. Training vs Evaluation Mode
48. model.train()
49. model.eval()
50. torch.no_grad()
51. Epochs and Batches
52. Overfitting and Underfitting
53. Regularization
54. Dropout
55. Weight Initialization
56. Learning Rate Scheduling

## Phase 5: Dataset & DataLoader
57. Custom Dataset
58. torch.utils.data.Dataset
59. DataLoader
60. Batch Processing
61. Shuffling
62. Transformations
63. Train/Test/Validation Split
64. Custom Collate Function
65. Data Preprocessing

## Phase 6: Advanced PyTorch
66. GPU Training with CUDA
67. Mixed Precision Training
68. torch.autocast
69. Gradient Scaling
70. Gradient Clipping
71. Distributed Training Basics
72. Multi-GPU Training
73. Model Checkpointing
74. Saving and Loading Models
75. TorchScript / torch.compile
76. Performance Profiling
77. Memory Optimization

## Phase 7: Deep Learning Architectures
78. Multi-Layer Perceptron (MLP)
79. Convolutional Neural Networks (CNN)
80. Recurrent Neural Networks (RNN)
81. LSTM
82. GRU
83. Attention Mechanism
84. Self-Attention
85. Transformer Architecture
86. Encoder and Decoder
87. Positional Encoding

## Phase 8: LLM Development with PyTorch
88. Tokenization
89. Embeddings
90. Embedding Layers
91. Causal Language Modeling
92. Transformer Blocks
93. Multi-Head Attention
94. Layer Normalization
95. Residual Connections
96. Feed Forward Networks
97. Causal Masking
98. Next Token Prediction
99. Pretraining
100. Fine-tuning
101. LoRA and QLoRA
102. Quantization
103. Model Evaluation
104. Inference Optimization

## Practical Projects

| Level | Project |
|---|---|
| Beginner | Linear Regression |
| Beginner | House Price Prediction |
| Beginner | MNIST Digit Classification |
| Intermediate | Image Classifier |
| Intermediate | Sentiment Analysis |
| Intermediate | Custom Neural Network |
| Advanced | Character-Level Text Generator |
| Advanced | Mini GPT from Scratch |
| Advanced | Fine-tune a Small Language Model |
