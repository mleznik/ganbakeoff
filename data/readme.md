# Data

This data folder provides the time series, which are used to train our GANs.

Please enter your own time series data to this folder to use our code.

In general, our code expects numpy files within the folders.

If you want to achieve a conditional setup with multiple classes please provide the training data for each class in a separate numpy file.

The saved numpy arrays must have the following shapes:

[#training_examples, #window_size, #channels]

As an examples we provide our artifical multi-class data set in the *preprocessed-data/periodic-waves* folder.
Within this folder there are 2 numpy files which contain sine and cosine waves.

You can use this data set to explore our code.
