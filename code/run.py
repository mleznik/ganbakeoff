from gan_lib import GANModel
import argparse
import os
import json
import pandas as pd
import numpy as np
import torch
from datetime import datetime
import sklearn
import sys
import uuid


def main():
    
    now = datetime.now() # current date and time
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--save_path", type=str, default="/opt/out/")
    parser.add_argument("--config_path", type=str)
    parser.add_argument("--wandb_projekt", type=str, default="default_project")
    parser.add_argument("--wandb_run_name", type=str, default="default_run")
    parser.add_argument("--num_frequencies", type=int, default=1)
    parser.add_argument("--datapath", default="CDN")
    
    
    args = parser.parse_args()

    if args.config_path == "../":
        sys.exit()
    
    
    
    os.environ["WANDB_API_KEY"] = json.load(open("../config/wandbkey.json"))["APIKey"]

        
        
    ## Load Configs / Data
    model_config = json.load(open(args.config_path))
    data_list, training_data_config = load_data(args)
    
    print(model_config)
    print(training_data_config)
    
    print(args.save_path)
    
    date_time = now.strftime("%Y_%d_%m/")
    
    save_path = "/opt/out/" + date_time + args.wandb_projekt + "/" + args.wandb_run_name

    if not os.path.exists(save_path):
        os.makedirs(save_path)
        print("Directory " , save_path ,  " Created ")
    else:    
        print("Directory " , save_path ,  " already exists")
            
    
    ganModel = GANModel(training_data_config, model_config, data_list, notebook=False, save_path = save_path)
    ganModel.train(num_epochs=args.epochs, use_wandb=True, wandb_run_name=args.wandb_run_name, wandb_projekt=args.wandb_projekt, docker_run=True)
    
    save_model(ganModel, save_path)
    print("Finished")


def save_model(ganModel,save_path):
    
    generated_examples = ganModel.img_list
    
    with open(f"{save_path}/generated_examples.npy", "wb") as file:
        np.save(file,generated_examples)
        
    torch.save(ganModel.netD.state_dict(), f"{save_path}/discriminator.pt")
    
    torch.save(ganModel.netG.state_dict(), f"{save_path}/generator.pt")
    
    with open(f"{save_path}/model_config.json", 'w') as file:
     file.write(json.dumps(ganModel.model_config)) 
    
    with open(f"{save_path}/training_data_config.json", 'w') as file:
     file.write(json.dumps(ganModel.training_data_config)) 
    
def load_data(args):
    
    files = os.listdir(args.datapath)
    
    data = []
    
    for file in files:
        data.append(np.load(args.datapath + "/" + file))
    
    training_data_config = {
        "num_data_points": data[0].shape[1],
        "z_seq_length": data[0].shape[1],
        "multi_variates": data[0].shape[2],
        "num_frequencies": args.num_frequencies
    }
    
    return data, training_data_config    


if __name__ == "__main__":
    main()
