# Morphological classification of galaxies out of DECaLS in Legacy Survey

## Directory Arrange

### /

- decalsDataset.py: rewrite the torch.Dataset
- main.py: training the network
- train.py: train code
- validate.py: validate the trained model and save the confusion metric.
- predict.py: apply the model to images in the directory

#### log

here is nothing important. just some running log about my code

#### models

Here is all the neural network architecture and loss function in this project.

- denseNet.py
- focal_loss.py
- simpleCNN.py
- swinTransformer.py
- xception.py

#### preprocess

- catalog_handle.ipynb: first code about this project to handle the catalog and raw data. I also get the labels and plot the picture here.
- data_augmentation.py: do data augmentation before training
- data_handle.py: consist of all the function handling data I need
- download_sdss.py: download sdss image in decals
- generate_txt.py: generate txt file of training dataset
- get_jpg.py: download jpg in decals
- out_decals.ipynb: analize the data out of decals
- out_decals_normalize.py: normalize the data to be applied

#### seperate

here is the code about all the process about seperate background and sources

- separate.ipynb

