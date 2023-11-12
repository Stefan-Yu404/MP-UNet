# MPU-Net
The code is for the article 3D Medical Image Segmentation based on multi-scale MP-UNet


# Structure
- Contraction Phase (Encoder):
  - The input image is passed through a Convolutional Neural Network (CNN) layer to extract high-level feature representations.
  - The CNN's output is then serialized, meaning it is converted into a one-dimensional sequence of data.
  - The Position Attention Module (PAM) is applied, which allows the model to focus on important features in the sequence.
- Bottleneck:

  - The bottleneck processes the serialized feature maps to capture global dependencies.
- Expansion Phase (Decoder):

  - The decoder uses skip connections to integrate low-level feature information from the encoder with the high-level representations from the bottleneck.
![image](https://github.com/Stefan-Yu404/MP-UNet/assets/80494218/0226ec52-41e4-43a2-aafc-61b2eff59b18)


# Citing MPU-Net
If you use the code, please cite the article:
```
@article{zeqiu2023MP-UNet,
  author    = {Zeqiu Yu and Shuo Han},
  title     = {3D Medical Image Segmentation based on multi-scale MPU-Net},
  journal   = {arXiv preprint arXiv:2307.05799},
  year      = {2023},
  url       = {https://arxiv.org/abs/2307.05799}
}
```
