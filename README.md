Related repositories:
- https://github.com/tancik/fourier-feature-networks/
- https://github.com/jmclong/random-fourier-features-pytorch/
- https://github.com/matajoh/fourier_feature_nets/


https://www.robots.ox.ac.uk/~az/lectures/ia/lect2.pdf
https://pytorch.org/docs/stable/generated/torch.fft.rfft.html

# Questions

1. Why fourier features are made with sin and cos? Why not just sin or just cos or e to the power of i?

2. Should we mutliply with 2*pi? (I think yes, but not sure)

3. Can we use torch.fft instead of custom implementation?

4. Why the visualization of the features is so identical to each other?

5. Is positional encoding helpful for this exact task?

6. Is there a risk of information leakage if 2d fourier features are used?

7. What does euler's formula have to do with fourier transform?

8. Kernel methods? RBF sampler?

# what i plan

- I am planning to see if a speacial input encofing can be used to boost the performance of the model. I will keep these things same for every model:
    - same model architecture
    - same optimizer
    - same configs (learning rate, number of epochs, batch size, etc)
    - same data (no randomization nor test/validation set)
-I will only add a layer before the model that will encode the input. Model should still have inputs and outputs are mapped to the range of 0 to 1. (will try -1 to 1 for inputs as well) **r, g, b = model(x, y)**
- I must be able to visualize the encoded features.

- After i get the results i am satisfied with:
- i make a video about it, and write a blog post in my site (i really hate medium but we'll see).
- i might try to add some experiments with NeRF model (i am not sure if i can do it, but i will try, i really like the idea of NeRF)
- post links in twitter, linkedin

# what i did

