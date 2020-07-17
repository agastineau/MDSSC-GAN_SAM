# MDSSC-GAN_SAM
Multi-Discriminator with Spectral and Spatial Constraints Adversarial Network for Pansharpening

Abstract:
The pansharpening problem amounts to fusing a high resolution panchromatic image with a low resolution multispectral image so as to obtain a high resolution multispectral image. So the preservation of the spatial resolution of the panchromatic image and the spectral resolution of the multipsectral image are of key importance for the pansharpening problem. To cope with it, we propose a new method based on multi-discriminator in a Generative Adversarial Network (GAN) framework. Two discriminators are considered. The first one is optimized to preserve textures of images by taking as input the luminance and the near infrared band of images, and the second one preserves the color by comparing the chroma components Cb and Cr. Thus, this method allows to train two discriminators, each one with a different and complementary task. Moreover, to enhance these aspects, the proposed method based on multi-discriminator, and called MDSSC-GAN SAM, considers a spatial and a spectral constraints in the loss function of the generator. We show on numerous examples the advantages of this new method.

# Generator Architecture


# Discriminator Architecture
