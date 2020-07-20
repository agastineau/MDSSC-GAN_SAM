# MDSSC-GAN SAM

Implementation of Multi-Discriminator with Spectral and Spatial Constraints Adversarial Network for Pansharpening

Anaïs GASTINEAU (1,2), Jean-François AUJOL (1), Yannick BERTHOUMIEU (2) and Christian GERMAIN (2)

(1) Univ. Bordeaux, Bordeaux INP, CNRS, IMB, UMR 5251, F-33400 Talence, France 
(2) Univ. Bordeaux, Bordeaux INP, CNRS, IMS, UMR 5218, F-33400 Talence, France 

contact : anais.gastineau@u-bordeaux.fr

To run the code with GPU:
tensorflow-gpu=1.2.0, 

cuda=8 

cuDNN=5.1 

python 3.6

First step:
Use the file tfrecord.py to create a file with the .tfrecords extension

Second step:
Use file MDSSC-GAN_SAM.py to train and test the network

Usage for training:
pyhton MDSSC-GAN_SAM.py --mode=train --output_dir=path_output_train_folder

--mode and --output_dir options are required. Other options are optionals. If not indicate, default values will be used. Other options are:
    
    -batch_size : nunber of images in the batch
    
    -beta1 : weight for ADAM
    
    -checkpoint : path to the checkpoint 
    
    -display_freq : frequency for saving images during training
    
    -gan_weight : weight for cross entropy term in the loss function of the generator
    
    -l1_weight : weight for l1 term in the loss function of the generator
    
    -lr : learing rate initial pour ADAM
    
    -max_epochs : maximal number of epochs
    
    -max_steps : maximal number of itérations
    
    -mode : train or test
    
    -ndf : number of filters in the first layer of the generator
    
    -output_dir : path for the output directory. The directory will be created if it doesn't exist.
    
    -progress_freq : frequency to display the progression in the terminal 
    
    -save_freq : frequency of saving the model
    
    -test_count : number of test images
    
    -test_tfrecord : path for the .tfrecords file obtained with test images
    
    -train_count : number of train images
    
    -train_tfrecord : path for the .tfrecords file obtained with train images

Usage for testing:
pyhton MDSSC-GAN_SAM.py --mode=test --checkpoint=path_folder_with_saved_model --output_dir=path_output_test_folder

