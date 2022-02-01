# GAN Evaluation

This Code folder contains the python scripts, which include the GAN Models.

## Run via Docker

After creating the docker image (evalgan) and having a wandb account you can start training a GAN model with a defined configuration as follows:

```
docker run -d -it\
  --name EvalGAN \
  --mount type=bind,source="$(pwd)"/code/,target=/opt1/program \
  --mount type=bind,source="$(pwd)"/results/,target=/opt1/out \
  --mount type=bind,source="$(pwd)"/data/,target=/opt1/data/ \
  evalgan --epochs 5 --config_path "tcn_tcn_sinus_256.json" --wandb_projekt "DockerDemo" --wandb_run_name "DockerDemoRun" --datapath "../data/preprocessed-data/periodic-waves/" --num_frequencies 1
```

Within this command the following parameters have to be passed to the docker container:

- epochs: Number of epochs 
- config_path: Path to json file, which determines the configuration of the GAN model, which should be trained
- wandb_projekt: Name of Wandb project, where the metrics are logged to
- wandb_run_name: Name of wandb run, where the metrics are logged to
- datapath: Path to data folder, which contains the training data
- num_frequencies: Number of frequency components that should be considered for the temporal correlation analysis. For further detail please refer to our paper.

## Explore the Code

Please refer to our jupyter notebooks in order to explore our code.

## Configuration

As mentioned above a json file is required to run our GAN model via docker image.

An example json file is provided in *tcn_tcn_sinus_256.json* looks like this:

```yaml
{
 "lr": 5e-05,
 "n_gan": false, 
 "optim": "ADAM",
 "d_steps": 3, 
 "g_steps": 3, 
 "Generator": {
     "dropout": 0.2, 
     "channels": 10, 
     "num_layers": 8, 
     "kernel_size": 7, 
     "architecture": "TCN"},
 "batch_size": 512,
 "architecture": ["TCN", "TCN"],
 "z_latent_dim": 2,
 "Discriminator": {
     "dropout": 0.2, 
     "channels": 20,
     "num_layers": 8,
     "kernel_size": 7,
     "architecture": "TCN"}
  }
```

In general the following parameters must be defined:

- lr: Learning Rate of the optimizers for the discriminator as well as the generator
- n_gan: We provide the option to use a semi-supervised training approach in a conditional setup, only set this parameter to true in a conditional setup.
- optim: Chose between "ADAM" and "RMSProp" 
- d_steps: Number of training iterations of the discriminator before training the generator
- g_steps: Number of training iterations of the generator before training the discriminator
- Generator: Json Dict, which includes the architecture specific configuration, please refer to the next sections for more information.
- Discriminator: Json Dict, which includes the architecture specific configuration, please refer to the next sections for more information.
- batch_size: Batch Size while training the GAN
- architecture: First entry defines the generator architecture, second entry the discriminator part.
- z_latent_dim: Number of channels of the noise vector, which is sampled from the latent space. Generally, you can set this parameter to the number of channels in your training data.


## Generator

We provide the opportunity to train mixed GAN architectures.
The generator can be TCN-based, LSTM-based or TFT-based.

### TCN

The json dict must define the following parameters:

- dropout: Dropout value in range of [0,1].
- channels: Number of channels of the convolutional layers.
- num_layers: Number of convolutional layers.
- kernel_size: Size of the convoluted kernel.

Note: The receptive field size of the TCN must be larger than the number of timestamps of the training time windows.

### LSTM

The json dict must define the following parameters:

- hidden_layer_size: Number of hidden neurons
- num_layers: Number of LSTM layers

### TFT

The json dict must define the following parameters:

- lstm_hidden_dimension: Number of hidden neurons of the LSTM block
- lstm_layers: Number of LSTM layers
- dropout: Dropout value in range of [0,1]
- embedding_dim: Dimensionality of the embedding layers
- attn_heads: Number of attention heads

## Discriminator

### TCN

The json dict must define the same parameters as the generator part.

### LSTM

The json dict must define the following parameters:

- hidden_layer_size: Number of hidden neurons
- num_layers: Number of LSTM layers
- dropout: Dropout value in range of [0,1]

### TFT

The json dict must define the same parameters as the generator part.

## Evaluation Metrics

The Python Code logs the evaluation metrics every epoch (default).
The evaluation metrics include:

- Temporal Correlation
- Spatial Correlation
- ICD
- INND
- ONND
- Appx. Entropy

The numbers reported in the paper are averaged over all possible channels and classes.
The logged numbers correspond to the values computed on each channel and class.




