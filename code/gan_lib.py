from torch.utils.data import TensorDataset, DataLoader
import torch
import torch.nn as nn
from torch.distributions import normal
from scipy.spatial.distance import cdist
import torch.optim as optim
import numpy as np
import json
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib.animation as animation
from IPython.display import HTML
import wandb
import itertools
from tcn_lib import *
from lstm_lib import *
from tft_lib import *
from evaluation import *
from tst.transformer import *
from tcn_classifier import *
import torch.nn.functional as F




class GANModel:
            
        
    def __init__(self, training_data_config, model_config, data_list=[], notebook=True, save_path="./save_plot", save=False, classifier_path=None):
        
        
        self.num_data_points = training_data_config["num_data_points"]
        self.z_seq_length = training_data_config["z_seq_length"]
        
        self.multi_variates = training_data_config["multi_variates"]
        self.fft_num_frequencies = training_data_config["num_frequencies"]

        self.generator_output_function = torch.nn.Sigmoid()
        
        self.n_gan = model_config["n_gan"]
        self.batch_size = model_config["batch_size"]
        self.d_config = model_config["Discriminator"]
        self.g_config = model_config["Generator"]
        self.lr = model_config["lr"]
        self.d_steps = model_config["d_steps"]
        self.z_latent_dim = model_config["z_latent_dim"]
        self.g_steps = model_config["g_steps"]
        self.optim = model_config["optim"]

        self.classifier_path = classifier_path
        
        
        self.model_config = model_config
        self.training_data_config = training_data_config
        
        self.notebook = notebook
        self.save_path = save_path
        self.save = save
                           
        architecture = model_config["architecture"]
        
        if (architecture[0] not in ["LSTM", "TCN", "Transformer", "TFT"]) or (architecture[1] not in ["LSTM", "TCN", "Transformer", "TFT"]):
            print("Invalid architecture was chosen. Default architecture LSTM is used.")
            architecture = ["LSTM","LSTM"]
        
            
        self.architecture = architecture
        self.num_classes = len(data_list)                
        self.conditional = self.num_classes > 1        
        self.discr_num_classes = (self.num_classes + 1 if (self.n_gan and self.conditional) else 1)
        
        
        data = []
        cond_data = []
        
        for i,d in enumerate(data_list):
            
            c = np.zeros(shape=[d.shape[0], self.num_classes + (1 if self.n_gan else 0)])
            
            c[:,i] = 1
            
            data.extend(d)
            
            cond_data.extend(c)
            
        self.data = np.array(data)
        self.cond_data = np.array(cond_data)


    def create_models(self,device):
        
        if self.n_gan:
            
            self.gen_cond_input_size = self.num_classes + 1
            
        else:
            
            self.gen_cond_input_size = self.num_classes
        

        if self.architecture[0] == "LSTM":
            
            self.netG = LSTMGenerator(batch_size=self.batch_size,hidden_layer_size=self.g_config["hidden_layer_size"],output_size=self.num_data_points, input_size=self.z_latent_dim + (self.gen_cond_input_size if self.conditional else 0), seq_length=self.z_seq_length, multi_variate=self.multi_variates, output_function=self.generator_output_function,num_layers=self.g_config["num_layers"], flatten=True, bidirectional=True, device=device).to(device)
            
            
        elif self.architecture[0] == "TCN":
            
            self.netG = TCNGenerator(input_size=self.z_latent_dim + (self.gen_cond_input_size if self.conditional else 0), channels=self.g_config["channels"], num_layers=self.g_config["num_layers"],output_size=self.num_data_points, kernel_size=self.g_config["kernel_size"], dropout=self.g_config["dropout"], input_length=self.num_data_points, multi_variate=self.multi_variates, output_function=self.generator_output_function).to(device)
            
            
        elif self.architecture[0] == "TFT":
                           
            
            self.netG = TFTGenerator(self.g_config, z_latent_dim=self.z_latent_dim +  (self.gen_cond_input_size if self.conditional else 0), multi_variates = self.multi_variates, batch_size = self.batch_size, seq_length=self.num_data_points,output_function=self.generator_output_function,device=device).to(device)
            
            
        if self.n_gan or (not self.conditional):
            
            self.discr_cond_input_size = 0
            
        else:
            
            self.discr_cond_input_size = self.num_classes
            
                   
            
        if self.architecture[1] == "LSTM":
                      
            self.netD = LSTMDiscriminator(self.d_config, multi_variates=self.multi_variates + self.discr_cond_input_size, batch_size = self.batch_size, seq_length=self.num_data_points, device=device, output_size=self.discr_num_classes).to(device)
            
        elif self.architecture[1] == "TCN":
            
            self.netD = TCNDiscriminator(input_size=self.multi_variates + self.discr_cond_input_size, input_length=self.num_data_points,channels=self.d_config["channels"], num_layers=self.d_config["num_layers"], kernel_size=self.d_config["kernel_size"], dropout=self.d_config["dropout"], num_classes=self.discr_num_classes).to(device)
            
        
        elif self.architecture[1] == "TFT":

            
            self.netD = TFTDiscriminator(self.d_config, multi_variates=self.multi_variates + self.discr_cond_input_size, batch_size = self.batch_size, seq_length=self.num_data_points, device=device, output_size=self.discr_num_classes).to(device)
            
    
    
    def train(self, num_epochs=10, use_wandb=False, seed=None, wandb_path="/opt/out", wandb_run_name=None, wandb_projekt=None, print_epochs=1, docker_run=False,num_fixed_noises=100, num_reference_samples=500):
        
        if seed:
            torch.manual_seed(seed)
        
        self.use_wandb = use_wandb
        if use_wandb:
            if not docker_run:
                os.environ["WANDB_API_KEY"] = json.load(open("../Docker/wandbkey.json"))['APIKey']
            wandb.init(config={"model_conf": self.model_config, "training_conf": self.training_data_config}, dir=wandb_path, project=wandb_projekt)
            if wandb_run_name:
                wandb.run.name = wandb_run_name
                wandb.run.save()
            print("wandb init")

        
        device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")
        
        self.device = device

        if self.classifier_path is not None:
            self.classifier = TCNClassifier(seq_length=self.num_data_points).to(self.device)
            self.classifier.load_state_dict(torch.load(self.classifier_path))
            self.classifier.tcnNet.eval()
        
        print(self.device)
        
        data_tensor = torch.Tensor(self.data).to(device)
        
        if self.conditional:
        
            cond_data_tensor = torch.Tensor(self.cond_data).to(device)

            my_dataset = TensorDataset(data_tensor,cond_data_tensor)
        
        else:
            
            my_dataset = TensorDataset(data_tensor)

        self.dataloader = torch.utils.data.DataLoader(my_dataset, batch_size=self.batch_size,shuffle=True)
        self.create_models(device)

                    
        return self.train_loop(num_epochs,device,self.lr,self.d_steps,print_epochs=print_epochs)
    
    def train_discriminator(self, data, optimizerD, device, real_label,criterion, fake_label, noise, C):
        
        self.netD.zero_grad()
        real_cpu = data[0].to(device)
                
        b_size = real_cpu.size(0)
        label = torch.full((b_size,), real_label, dtype=torch.float, device=self.device)
        



        self.netD.zero_grad()
        if self.conditional:
            if self.n_gan:
                output = self.netD((real_cpu, None))
                errD_real = criterion(output, data[1])
            else:
                output = self.netD((real_cpu, data[1] if self.conditional else None)).view(-1)
                errD_real = criterion(output, label)
        else:
            output = self.netD((real_cpu, data[1] if self.conditional else None)).view(-1)
            errD_real = criterion(output, label)

        # Calculate loss on all-real batch


        
        # Calculate gradients for D in backward pass
        errD_real.backward()
        D_x = output.mean().item()


        # Generate fake image batch with G


        fake = self.netG((noise,C))
        label.fill_(fake_label)
        #label = torch.cat((torch.full((b_size,1), fake_label, dtype=torch.float, device=device),torch.full((b_size,1), real_label, dtype=torch.float, device=device)), dim=1)    
        # Classify all fake batch with D

        
        
        

        # Calculate D's loss on the all-fake batch
        if self.conditional:
            if self.n_gan:

                output = self.netD((fake.detach(), None))
                crit_input = torch.zeros_like(data[1])

                crit_input[:,-1] = 1
                errD_fake = criterion(output, crit_input)
            else:
                output = self.netD((fake.detach(), C)).view(-1)
                errD_fake = criterion(output, label)
        
        else:
            output = self.netD((fake.detach(), C)).view(-1)
            errD_fake = criterion(output, label)
        # Calculate the gradients for this batch, accumulated (summed) with previous gradients
        errD_fake.backward()
        D_G_z1 = output.mean().item()
        # Compute error of D as sum over the fake and the real batches
        errD = errD_real + errD_fake
        # Update D


                        #Clipping weights

        optimizerD.step()
        
        return errD, D_G_z1, D_x
        
        
    def train_generator(self, real_label, noise, C, criterion, optimizerG):
        
        self.netG.zero_grad()
        label = torch.full((self.batch_size,), real_label, dtype=torch.float, device=self.device)
        label.fill_(real_label)  # fake labels are real for generator cost
        
        fake = self.netG((noise,C))

        if self.n_gan:
            output = self.netD((fake,None))
            errG = criterion(output, C)
        else:
            output = self.netD((fake,C)).view(-1)
            errG = criterion(output, label)
            
        # Calculate G's loss based on this output
        
        # Calculate gradients for G
        errG.backward()
        D_G_z2 = output.mean().item()

        # Update G
        optimizerG.step()
        return errG, D_G_z2
        

    def train_loop(self,num_epochs,device,lr,d_steps, num_fixed_noises=100, num_reference_samples=500, print_epochs=1):
                
        
        lambda_evaluation_keys = {}
        
        if self.multi_variates > 1:
            
            corr_combinations = list(itertools.combinations(range(self.multi_variates), 2))
            
            lambda_evaluation_keys = {
                "correlation": {
                    "cross-corr": (lambda x: np.transpose(np.array([x[:,a,b] for (a,b) in corr_combinations]),(1,0)), lambda x: np.mean(x,axis=0),None)
                },
            }
        
        lambda_evaluation_keys["fft"] = {
                        "eval": (lambda x: x, lambda x: [np.mean(y, axis=0) for y in x], lambda x: np.sum(x, axis=1))
                    }
        lambda_evaluation_keys["entropy"] = {"app_entropy": (lambda x: x, lambda x: np.mean(x,axis=0),None)}

            
        
        
        img_list = []
        evaluation_list = []
        fft_values = []
        eval_values = []
        G_losses = []
        D_losses = []
        iters = 0
        
        
        if self.conditional:
            c_list = []
            for i in range(self.num_classes):
                
                if self.n_gan:
                    C1 = torch.zeros(size=(num_fixed_noises,self.num_classes+1)).to(device)
                else:
                    C1 = torch.zeros(size=(num_fixed_noises,self.num_classes)).to(device)
                C1[:,i] = 1
                
                c_list.append(C1)
                
            
            fixed_noise = torch.randn(self.num_classes,num_fixed_noises, self.z_seq_length, self.z_latent_dim, device=device)
            
            fixed_noise_label = torch.cat(c_list,axis=0)
            
            
        
        else:
            
            fixed_noise = torch.randn(num_fixed_noises, self.z_seq_length, self.z_latent_dim, device=device)
            fixed_noise_label = None

        # Initialize BCELoss function
        criterion = nn.BCELoss()
        #criterion = F.nll_loss

        # Establish convention for real and fake labels during training
        real_label = 1
        fake_label = 0

        #if True:
        if self.optim == "RMSprop":
            optimizerD = torch.optim.RMSprop(self.netD.parameters(), lr=lr)
            optimizerG = torch.optim.RMSprop(self.netG.parameters(), lr=lr)
            
        else: 
            # Setup Adam optimizers for both G and D
            optimizerD = optim.Adam(self.netD.parameters(), lr=lr, betas=(0.5, 0.999))
            #optimizerD = optim.RMSprop(self.netD.parameters(), lr=d_lr)
            optimizerG = optim.Adam(self.netG.parameters(), lr=lr, betas=(0.5, 0.999))
            
    
        training_data_per_class = self.data.shape[0] // self.num_classes
        
        self.reference_metric = [
            evaluation_pipeline(self.data[np.random.randint(training_data_per_class*i,training_data_per_class*(i+1),size=num_reference_samples)], num_frequencies=self.fft_num_frequencies) for i in range(self.num_classes)]


        print("Starting Training Loop...")
        # For each epoch
        for epoch in range(num_epochs):
            # For each batch in the dataloader
            
            print(f"Epoch {epoch}")
            for i in range((len(self.dataloader) // self.d_steps)+1):
                
                b_size = self.batch_size
                
                noise = torch.randn(b_size, self.z_seq_length, self.z_latent_dim, device=device)

                C = None

                if self.conditional:
                        if self.n_gan:
                            C = torch.zeros(size=(b_size,self.num_classes+1)).to(device)
                        else:
                            C = torch.zeros(size=(b_size,self.num_classes)).to(device)
                        # locations
                        labels = torch.randint(0,self.num_classes,(b_size,))
                        C[torch.arange(b_size),labels] = 1
                
                for k in range(self.d_steps):
                    
                    data = next(iter(self.dataloader))
                    
                    errD, D_G_z1, D_x = self.train_discriminator(data, optimizerD, device, real_label, criterion, fake_label, noise, C)
                    
                    iters += 1
                    

                    
                for k in range(self.g_steps):
                    
                    errG, D_G_z2 = self.train_generator(real_label, noise, C, criterion, optimizerG)
    
            
            if (epoch % print_epochs) == 0:
                self.log_metrics(fixed_noise, fixed_noise_label, evaluation_list, fft_values, eval_values, epoch, num_epochs, errD, errG, D_x, D_G_z1, D_G_z2, img_list, lambda_evaluation_keys)

                
                
        
        #evaluation_dict = evaluation_pipeline(img_list[-1])       
                
        self.img_list = np.transpose(img_list,(1,0,2,3))
                
    
    def log_metrics(self, fixed_noise, fixed_noise_label, evaluation_list, fft_values, eval_values, epoch, num_epochs, errD, errG, D_x, D_G_z1, D_G_z2, img_list, lambda_evaluation_keys):
        
        print('[%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f'
                          % (epoch, num_epochs,
                             errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))
                  
        with torch.no_grad():
            
            if self.n_gan:
                fake = self.netG((fixed_noise.view(-1,self.z_seq_length,self.z_latent_dim),fixed_noise_label)).detach().cpu()
            else:
                fake = self.netG((fixed_noise.view(-1,self.z_seq_length,self.z_latent_dim),fixed_noise_label)).detach().cpu()
                
                
            img_list.append(fake.numpy())
            
            fixed_noise_per_class = fake.shape[0] // self.num_classes
            
            evaluation_list.append([
                    evaluation_pipeline(fake.numpy()[fixed_noise_per_class*i:fixed_noise_per_class*(i+1)], num_frequencies=self.fft_num_frequencies)
                for i in range(self.num_classes)
                ])

            self.evaluation_metric_calculation(self.reference_metric, evaluation_list[-1],  lambda_evaluation_keys, fake, fixed_noise_label)


        
    def calculate_distance(self,data, num_classes=2, dist_type="DTW"):
       
    
        num_data_points_per_class = data.shape[0]//num_classes
    
        inner_dist_list = []

        innd_list = []

        onnd_list = []

        for class_idx in range(num_classes):
            
            training_data = self.data[class_idx * (self.data.shape[0] // num_classes):(class_idx + 1) * (self.data.shape[0] // num_classes)]
            
            class_data = data[class_idx * num_data_points_per_class:(class_idx+1)*num_data_points_per_class].numpy()
            
            if dist_type=="DTW":
                inter_class_dist = dtw_ndim.distance_matrix_fast(class_data.astype(np.double),ndim=2)
                nnd = dtw_ndim.distance_matrix_fast(np.concatenate([class_data,training_data]).astype(np.double),ndim=2, block=((0,class_data.shape[0]),(class_data.shape[0],class_data.shape[0]+training_data.shape[0])))

                nnd = nnd[:class_data.shape[0],class_data.shape[0]:]

            elif dist_type=="ED":
                inter_class_dist = cdist(class_data.reshape(num_data_points_per_class,-1), class_data.reshape(num_data_points_per_class,-1), 'sqeuclidean')

                nnd = cdist(class_data.reshape(num_data_points_per_class,-1), training_data.reshape(self.data.shape[0] // num_classes,-1), 'sqeuclidean')

            innd = np.mean(np.min(nnd,axis=1))

            onnd = np.mean(np.min(nnd,axis=0))
            

            mean_dist = np.mean(inter_class_dist)

            inner_dist_list.append(mean_dist)
            innd_list.append(innd)
            onnd_list.append(onnd)

        return inner_dist_list,innd_list,onnd_list  
    
    def evaluation_metric_calculation(self, reference_dict, evaluation_dict, lambda_evaluation_keys, fake, fixed_noise_label):
        
        
   
        logging_dict = {
            
        }
           
    
        for stat in lambda_evaluation_keys.keys():
            for key in lambda_evaluation_keys[stat]:
                difference_list = []
                for i in range(self.num_classes):
                    
                    
                
                
                    ref_value = reference_dict[i][stat][key]
                    eval_value = evaluation_dict[i][stat][key]
                    

                    ref_value = lambda_evaluation_keys[stat][key][0](ref_value)

                    
                    ref_value = lambda_evaluation_keys[stat][key][1](ref_value)

                    eval_value = lambda_evaluation_keys[stat][key][0](eval_value)


                    eval_value = lambda_evaluation_keys[stat][key][1](eval_value)

                
                    difference = (np.array(ref_value)-np.array(eval_value))**2
                    
                    
                    difference_list.append(difference)
                    

                difference = np.mean(difference_list,axis=0)
                
                if lambda_evaluation_keys[stat][key][2] is not None:
                    difference = lambda_evaluation_keys[stat][key][2](difference)
                
                for i,val in enumerate(difference):
                    logging_dict[key + "_" + str(i)] = val
            
        if self.classifier_path:
            logging_dict["classifier_accuracy"] = self.calculate_classifier_accuracy(fake,fixed_noise_label)

        inter_distance, innd, onnd = self.calculate_distance(fake, self.num_classes, "ED")
        
        for i in range(self.num_classes):
            logging_dict[f"inter_class_{i}_distance"] = inter_distance[i]
            logging_dict[f"innd_{i}"] = innd[i]
            logging_dict[f"onnd_{i}"] = onnd[i]
        

                           
        if self.use_wandb:

            fig = self.visualize_training_wandb(fake)

            logging_dict["chart"] = wandb.Image(fig)
            
            plt.close(fig)

            wandb.log(logging_dict)
        else:
            print(logging_dict)

    def calculate_classifier_accuracy(self, fake, fixed_noise_label):
        
        class_output = self.classifier.tcnNet((fake.to(self.device),None))
        
        labels = class_output.argmax(dim=-1)
        
        correct = (labels == fixed_noise_label.argmax(dim=-1)).float().sum()
        
        return correct/fake.shape[0]  
       
    def visualize_training_wandb(self, fake):

        img_list = fake
        
        if self.conditional:
            k = len(img_list)//2
            img_list = img_list[[i for i in range(10)] + [i+k for i in range(10)]]
        else:
            img_list = img_list[:10]

        if self.conditional:
            num_samples = img_list.shape[0] // 2
            fig, axs = plt.subplots(10,2,figsize=(20,2*num_samples))
            
            cols = ['Class {}'.format(col) for col in range(1,3)]
            
            for ax, col in zip(axs[0], cols):
                ax.set_title(col)
                
            ln_list = [
                axs[i%num_samples][int(i/10)].plot(img_list[i,:,j])[0]
                for i in range(img_list.shape[0]) for j in range(self.multi_variates)
            ]

        else:
            num_samples = img_list.shape[0]
            fig, axs = plt.subplots(10,1,figsize=(10,2*num_samples),squeeze=False)
            
            ln_list = [
                axs[i%num_samples][int(i/10)].plot(img_list[i,:,j])[0]
                for i in range(img_list.shape[0]) for j in range(self.multi_variates)
            ]

        return fig
    
    
    def visualize_training(self, mnist=False):
        
        if mnist:
            return self.visualize_mnist()
            
        
        img_list = self.img_list
        
        if self.conditional:
            k = len(img_list)//2
            img_list = img_list[[i for i in range(10)] + [i+k for i in range(10)]]
        else:
            img_list = img_list[:10]
            
        num_data_points = self.num_data_points
        
        
        if self.conditional:
            num_samples = int(len(img_list)/2)
            fig, axs = plt.subplots(int(len(img_list)/2),2,figsize=(20,2*num_samples))
            
            cols = ['Class {}'.format(col) for col in range(1,3)]
            
            for ax, col in zip(axs[0], cols):
                ax.set_title(col)
        else:
            num_samples = len(img_list)
            fig, axs = plt.subplots(len(img_list),1,figsize=(20,2*num_samples),squeeze=False)
            

        xdata, ydata = [], []
        ln_list = [None for i in range(img_list.shape[0]*self.multi_variates)]
        
        
        ln_list = [
            axs[i%num_samples][int(i/num_samples)].plot(img_list[i,0,:,j])[0]
            for i in range(img_list.shape[0]) for j in range(self.multi_variates)
        ]

        def init():
            for i in range(len(img_list)):    
                    axs[i%num_samples][int(i/num_samples)].set_xlim(0, num_data_points)
                    axs[i%num_samples][int(i/num_samples)].set_ylim(np.min(img_list[i]), np.max(img_list[i]))
            return ln_list[0],

        def update(frame):
            #xdata.append(frame)
            #ydata.append(np.sin(frame))
            for index,ln in enumerate(ln_list):  
                
                i = index // self.multi_variates
                j = index % self.multi_variates
                #for j in range(self.multi_variates):
                ln.set_data(np.arange(0,num_data_points), img_list[i,frame,:,j])
                #axs[i%num_samples][int(i/num_samples)].plot(img_list[i,frame])
                axs[i%num_samples][int(i/num_samples)].set_ylim(np.min(img_list[i][frame]), np.max(img_list[i][frame]))
            return ln_list[0],

        ani = FuncAnimation(fig, update, frames=np.arange(0,len(img_list[0])),
                            init_func=init, blit=True)
                 
        if self.save:
        
            writer = animation.writers['ffmpeg']

            writer = writer(fps=3)
            ani.save(f'{self.save_path + "/"}visualize_training.mp4', writer=writer)
        
        else:
        
            plt.tight_layout()
            plt.close(fig)
            return HTML(ani.to_jshtml())
            

        



