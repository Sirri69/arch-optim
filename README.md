## Arch optimizations:
  
There are 3 major optimizations that can be made to the architecture:

1. Add loss aware output nodes at various layers.
2. Compress the model reducing the hidden size and adding skip connections to compenstate for the loss of information.
3. DWA combined with 2nd point - paper: https://arxiv.org/abs/2402.02622


Goal is to make the model training more efficient.

Another thing to try is to use a custom differentiable scheduler for the learning rate which takes into account the training speed for each iteration and adjusts the learning rate accordingly.

Also, during inference, train a tree based model to predict the next tokens and assist the model during high logprob regions. The classifier will be able to switch to the main model by haveing a class that switches to __MODEL__ class.




------------------------------------------------------------------
## Scratch Pad

GPT arch:

GPT(
  (transformer): ModuleDict(
    (wte): Embedding(50304, 768)
    (wpe): Embedding(1024, 768)
    (drop): Dropout(p=0.0, inplace=False)
    (h): ModuleList(
      (0-11): 12 x Block(
        (ln_1): LayerNorm()
        (attn): CausalSelfAttention(
          (c_attn): Linear(in_features=768, out_features=2304, bias=False)
          (c_proj): Linear(in_features=768, out_features=768, bias=False)
          (attn_dropout): Dropout(p=0.0, inplace=False)
          (resid_dropout): Dropout(p=0.0, inplace=False)
        )
        (ln_2): LayerNorm()
        (mlp): MLP(
          (c_fc): Linear(in_features=768, out_features=3072, bias=False)
          (gelu): GELU(approximate='none')
          (c_proj): Linear(in_features=3072, out_features=768, bias=False)
          (dropout): Dropout(p=0.0, inplace=False)
        )
      )
    )
    (ln_f): LayerNorm()
  )
  (lm_head): Linear(in_features=768, out_features=50304, bias=False)
)

So, to add multiple output nodes, let's add them at the blocks 8, 10, 12, the loss ratios will be 0.05, 0.15, 0.8 respectively. These intermidiate nodes will get their own lm_heads.

The last output is the final output, others are intermediate outputs which will be used for helping the model during training.
