# Evaluating GANs

This repository provides the python code used for our research for our evaluation of different GAN architectures.

For demo purposes, we provide the code and configurations used to synthesise time series in our multi-class setup.

## Dependencies
In general, we utilize [Weights and Biases](https://wandb.ai/site) to monitor and orchestrate our experiments. 
Therefore, it is necessary to install WandB locally and create and account to fully use our repository.

Once you have create an account and have the corresponding API-Key this API-Key has to be entered in Docker/wandbkey.json


## Pipeline

This repository contains these 3 folders:

1. [Docker](https://omi-gitlab.e-technik.uni-ulm.de/aml/ganevaluation/-/tree/dev/Docker) 
    - Files to create docker image to run our code

2. [code](https://omi-gitlab.e-technik.uni-ulm.de/aml/ganevaluation/-/tree/dev/code)
    - Code to train and evaluate the GAN models

3. [data](https://omi-gitlab.e-technik.uni-ulm.de/aml/ganevaluation/-/tree/dev/data)
    - Please enter the time series with anomalies to this data folder in order to utilize our approach.

Lastly, we also provide Jupyter-Notebooks to explore our code / approach.

4. [Notebooks](https://omi-gitlab.e-technik.uni-ulm.de/aml/ganevaluation/-/tree/dev/notebooks)







