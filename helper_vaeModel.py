import torch
import torch.nn as nn

class Reshape(nn.Module):
    def __init__(self, *args):
        super().__init__()
        self.shape = args

    def forward(self, x):
        return x.view(self.shape)


class Trim(nn.Module):
    def __init__(self, *args):
        super().__init__()

    def forward(self, x):
        return x[:, :, :1024, :1024]
    
class VAE(nn.Module):
    def __init__(self, latent_dim: int, act_fn: object = nn.LeakyReLU):
        
        super().__init__()
        
        self.encoder=nn.Sequential(
            # (N , 1, 1024, 1024)
            ## downsammling blocK 1
            nn.Conv2d(in_channels = 1, out_channels = 8, kernel_size = 1),
            nn.Conv2d(in_channels = 8, out_channels = 16, kernel_size = 3, stride = 2, padding = 1),
            nn.BatchNorm2d(num_features=16),# same shape as input 
            act_fn(),
            #nn.MaxPool2d(kernel_size=2, stride=2),
            
            ## downsammling blocK 2
            nn.Conv2d(in_channels = 16, out_channels = 8, kernel_size = 1),
            nn.Conv2d(in_channels = 8, out_channels = 32, kernel_size = 3, stride = 2, padding = 1),
            nn.BatchNorm2d(num_features=32),# same shape as input 
            act_fn(),
           # nn.MaxPool2d(kernel_size=2, stride=2),
            
            ## downsammling blocK 3
            nn.Conv2d(in_channels = 32, out_channels = 8, kernel_size = 1),
            nn.Conv2d(in_channels = 8, out_channels = 64, kernel_size = 3, stride = 2, padding = 1),
            nn.BatchNorm2d(num_features=64),# same shape as input 
            act_fn(),
            ##nn.MaxPool2d(kernel_size=2, stride=2),
            
            ## downsammling blocK 4
            nn.Conv2d(in_channels = 64, out_channels = 8, kernel_size = 1),
            nn.Conv2d(in_channels = 8, out_channels = 128, kernel_size = 3, stride = 2, padding = 1),
            nn.BatchNorm2d(num_features=128),# same shape as input 
            act_fn(),
            #nn.MaxPool2d(kernel_size=2, stride=2),
            
            ## downsammling blocK 5
            nn.Conv2d(in_channels = 128, out_channels = 8, kernel_size = 1),
            nn.Conv2d(in_channels = 8, out_channels = 256, kernel_size = 3, stride = 2, padding = 1),
            nn.BatchNorm2d(num_features=256),# same shape as input 
            act_fn(),
            #nn.MaxPool2d(kernel_size=2, stride=2),
            
            ## downsammling blocK 6
            nn.Conv2d(in_channels = 256, out_channels = 8, kernel_size = 1),
            nn.Conv2d(in_channels = 8, out_channels = 512, kernel_size = 3, stride = 2, padding = 1),
            nn.BatchNorm2d(num_features=512),# same shape as input 
            act_fn(),
           # nn.MaxPool2d(kernel_size=2, stride=2),
            
            ## downsammling blocK 7
            nn.Conv2d(in_channels = 512, out_channels = 8, kernel_size = 1),
            nn.Conv2d(in_channels = 8, out_channels = 1024, kernel_size = 3, stride = 2, padding = 1),
            nn.BatchNorm2d(num_features = 1024),# same shape as input 
            act_fn(),
            #nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Flatten(),
    
        
        )  
        self.z_mean = torch.nn.Linear(65536, latent_dim)#32768#65536
        self.z_log_var = torch.nn.Linear(65536, latent_dim)
        
        # decoder 
        self.decoder=nn.Sequential(
            
            torch.nn.Linear(latent_dim, 65536),
            Reshape(-1, 1024, 8, 8),
            
            # Block 1
            nn.Conv2d(in_channels = 1024, out_channels = 8, kernel_size = 1),
            nn.ConvTranspose2d(in_channels = 8, out_channels = 512, kernel_size = 3,padding = 1,stride = 2,output_padding = 1),# 
            nn.BatchNorm2d(num_features = 512),
            act_fn(),
            
            # Block 2
            nn.Conv2d(in_channels = 512, out_channels = 8, kernel_size = 1),
            nn.ConvTranspose2d(in_channels = 8, out_channels = 256, kernel_size = 3,padding = 1,stride = 2,output_padding = 1),# 
            nn.BatchNorm2d(num_features = 256),
            act_fn(),
            
            # Block 3
            nn.Conv2d(in_channels = 256, out_channels = 8, kernel_size = 1),
            nn.ConvTranspose2d(in_channels = 8, out_channels = 128, kernel_size = 3,padding = 1,stride = 2,output_padding = 1),# 
            nn.BatchNorm2d(num_features=128),
            act_fn(),
            
            # Block 4
            nn.Conv2d(in_channels = 128, out_channels = 8, kernel_size = 1),
            nn.ConvTranspose2d(in_channels = 8, out_channels = 64, kernel_size = 3,padding = 1,stride = 2,output_padding = 1),# 
            nn.BatchNorm2d(num_features=64),
            act_fn(),
            
            # Block 5
            nn.Conv2d(in_channels = 64, out_channels = 8, kernel_size = 1),
            nn.ConvTranspose2d(in_channels = 8, out_channels = 32, kernel_size = 3,padding = 1,stride = 2,output_padding = 1),# 
            nn.BatchNorm2d(num_features=32),
            #nn.Dropout2d(p=0.5),
            act_fn(),
            
            # Block 6
            nn.Conv2d(in_channels = 32, out_channels = 8, kernel_size = 1),
            nn.ConvTranspose2d(in_channels = 8, out_channels = 16, kernel_size = 3,padding = 1,stride = 2,output_padding = 1),# 
            nn.BatchNorm2d(num_features=16),
            act_fn(),
            
            # Block 7
            nn.ConvTranspose2d(in_channels = 16, out_channels = 1, kernel_size = 3,padding = 1,stride = 2,output_padding = 1),# 
            nn.BatchNorm2d(num_features=1),
           # nn.Dropout2d(p=0.5),
            #Trim(),
            nn.Sigmoid(),
        )
    def decode(self,x):
        x = self.decoder(x)
        return x
    
    def encode(self, x):
        x = self.encoder(x)
        z_mean, z_log_var = self.z_mean(x), self.z_log_var(x)
        encoded = self.reparameterize(z_mean, z_log_var)
        return encoded
    
    def reparameterize(self, z_mu, z_log_var):
        eps = torch.randn(z_mu.size(0), z_mu.size(1)).to(z_mu.get_device())
        z = z_mu + eps * torch.exp(z_log_var/2.) 
        return z
        
    def forward(self, x):
        x = self.encoder(x)
        z_mean, z_log_var = self.z_mean(x), self.z_log_var(x)
        encoded = self.reparameterize(z_mean, z_log_var)
        decoded = self.decoder(encoded)
        return encoded, z_mean, z_log_var, decoded
    
class NVAE(nn.Module):
    
    def __init__(self, latent_dim: int, act_fn: object = nn.LeakyReLU):
        
        super().__init__()
        
        self.encoder=nn.Sequential(
            # (N , 1, 1024, 1024)
            # Block 1
            nn.Conv2d(in_channels = 1, out_channels = 16, kernel_size = 3, stride = 1, padding = 1),
            nn.BatchNorm2d(num_features=16),# same shape as input 
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Block 2
            #nn.Conv2d(in_channels = 16, out_channels = 8, kernel_size = 1),
            nn.Conv2d(in_channels = 16, out_channels = 32, kernel_size = 3, stride = 2, padding = 1),
            nn.BatchNorm2d(num_features=32),# same shape as input 
            nn.ReLU(),
            #nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Block 3
           #nn.Conv2d(in_channels = 32, out_channels = 8, kernel_size = 1),
            nn.Conv2d(in_channels = 32, out_channels = 64, kernel_size = 3, stride = 2, padding = 1),
            nn.BatchNorm2d(num_features=64),# same shape as input 
            nn.ReLU(),
            #nn.Dropout2d(p=0.5),
            #nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Block 4
            #nn.Conv2d(in_channels = 64, out_channels = 8, kernel_size = 1),
            nn.Conv2d(in_channels = 64, out_channels = 128, kernel_size = 3, stride = 2, padding = 1),
            nn.BatchNorm2d(num_features=128),# same shape as input 
            nn.ReLU(),
            #nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Block 5
            #nn.Conv2d(in_channels = 128, out_channels = 8, kernel_size = 1),
            nn.Conv2d(in_channels = 128, out_channels = 256, kernel_size = 3, stride = 2, padding = 1),
            nn.BatchNorm2d(num_features=256),# same shape as input 
            nn.ReLU(),
            #nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Block 6
            #nn.Conv2d(in_channels = 256, out_channels = 8, kernel_size = 1),
            nn.Conv2d(in_channels = 256, out_channels = 512, kernel_size = 3, stride = 2, padding = 1),
            nn.BatchNorm2d(num_features=512),# same shape as input 
            nn.ReLU(),
            nn.Dropout2d(p=0.5),
           # nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Block 7
            #nn.Conv2d(in_channels = 512, out_channels = 8, kernel_size = 1),
            nn.Conv2d(in_channels = 512, out_channels = 1024, kernel_size = 3, stride = 2, padding = 1),
            nn.BatchNorm2d(num_features = 1024),# same shape as input 
            nn.ReLU(),
            #nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Flatten(),
    
        
        )  
        self.z_mean = torch.nn.Linear(32768, latent_dim)#65536
        self.z_log_var = torch.nn.Linear(32768, latent_dim)
        
        # decoder 
        self.decoder=nn.Sequential(
            
            torch.nn.Linear(latent_dim, 32768),
            Reshape(-1, 1024, 8, 4),
            
            # Block 1
            #nn.Conv2d(in_channels = 1024, out_channels = 8, kernel_size = 1),
            nn.ConvTranspose2d(in_channels = 1024, out_channels = 512, kernel_size = 3,padding = 1,stride = 2,output_padding = 1),# 
            nn.BatchNorm2d(num_features = 512),
            act_fn(),
            
            # Block 2
            #nn.Conv2d(in_channels = 512, out_channels = 8, kernel_size = 1),
            nn.ConvTranspose2d(in_channels = 512, out_channels = 256, kernel_size = 3,padding = 1,stride = 2,output_padding = 1),# 
            nn.BatchNorm2d(num_features = 256),
            act_fn(),
            
            # Block 3
            #nn.Conv2d(in_channels = 256, out_channels = 8, kernel_size = 1),
            nn.ConvTranspose2d(in_channels = 256, out_channels = 128, kernel_size = 3,padding = 1,stride = 2,output_padding = 1),# 
            nn.BatchNorm2d(num_features=128),
            act_fn(),
            
            # Block 4
           # nn.Conv2d(in_channels = 128, out_channels = 8, kernel_size = 1),
            nn.ConvTranspose2d(in_channels = 128, out_channels = 64, kernel_size = 3,padding = 1,stride = 2,output_padding = 1),# 
            nn.BatchNorm2d(num_features=64),
            act_fn(),
            
            # Block 5
            #nn.Conv2d(in_channels = 64, out_channels = 8, kernel_size = 1),
            nn.ConvTranspose2d(in_channels = 64, out_channels = 32, kernel_size = 3,padding = 1,stride = 2,output_padding = 1),# 
            nn.BatchNorm2d(num_features=32),
            nn.Dropout2d(p=0.5),
            act_fn(),
            
            # Block 6
            #nn.Conv2d(in_channels = 32, out_channels = 8, kernel_size = 1),
            nn.ConvTranspose2d(in_channels = 32, out_channels = 16, kernel_size = 3,padding = 1,stride = 2,output_padding = 1),# 
            nn.BatchNorm2d(num_features=16),
            act_fn(),
            
            # Block 7
            nn.ConvTranspose2d(in_channels = 16, out_channels = 1, kernel_size = 3,padding = 1,stride = 2,output_padding = 1),# 
            nn.BatchNorm2d(num_features=1),
            nn.Dropout2d(p=0.5),
            #Trim(),
            nn.Sigmoid(),
        )
    def decode(self,x):
        x = self.decoder(x)
        return x
    
    def encode(self, x):
        encoded = self.encoder(x)
        z_mean, z_log_var = self.z_mean(x), self.z_log_var(x)
        encoded = self.reparameterize(z_mean, z_log_var)
        return encoded
    
    def reparameterize(self, z_mu, z_log_var):
        eps = torch.randn(z_mu.size(0), z_mu.size(1)).to(z_mu.get_device())
        z = z_mu + eps * torch.exp(z_log_var/2.) 
        return z
        
    def forward(self, x):
        x = self.encoder(x)
        z_mean, z_log_var = self.z_mean(x), self.z_log_var(x)
        encoded = self.reparameterize(z_mean, z_log_var)
        decoded = self.decoder(encoded)
        return encoded, z_mean, z_log_var, decoded
    