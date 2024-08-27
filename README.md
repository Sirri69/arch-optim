## Arch optimizations:
  
There are 3 major optimizations that can be made to the architecture:

1. Add loss aware output nodes at various layers.
2. Compress the model reducing the hidden size and adding skip connections to compenstate for the loss of information.
3. DWA combined with 2nd point - paper: https://arxiv.org/abs/2402.02622


Goal is to make the model training more efficient.

Another thing to try is to use a custom differentiable scheduler for the learning rate which takes into account the training speed for each iteration and adjusts the learning rate accordingly.

Also, during inference, train a tree based model to predict the next tokens and assist the model during high logprob regions. The classifier will be able to switch to the main model by haveing a class that switches to __MODEL__ class.

