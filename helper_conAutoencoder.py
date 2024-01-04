import torch
import torch.nn as nn


class CAE(nn.Module):
    def __init__(self, latent_dim: int, act_fn: object = nn.LeakyReLU):
        
        super().__init__()
        
        self.encoder=nn.Sequential(
            # (N , 1, 1024, 1024)
            
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
        
        )  
        self.linear1 = nn.Sequential(
            nn.Flatten(),
            nn.Linear(65536, latent_dim)#32768 #65536
        )
        self.linear2 = nn.Sequential(
            nn.Linear(latent_dim, 65536))#32768
        
        # decoder 
        self.decoder=nn.Sequential(
            ## (N , 1024, 11, 8)
            
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
            #nn.Sigmoid(),
        )
    def encoding_fn(self, x):
        encoded = self.encoder(x)
        return encoded
                
    def forward(self, x):
        encoded = self.encoder(x)
        first_linearLy = self.linear1(encoded)
        second_linearLy = self.linear2(first_linearLy)
        reshaped_output = second_linearLy.reshape(-1, 1024, 8, 8)#(-1, 1024, 8, 4)
        decoded = self.decoder(reshaped_output)
        return decoded
    def encode(self, x):
        x = self.encoder(x)
        x = self.linear1(x)
        return x

    
    
class AE(nn.Module):
    def __init__(self, act_fn: object = nn.LeakyReLU):
        super().__init__()
        self.encoder=nn.Sequential(
            
            # (N , 1, 1408, 1024)
            
            nn.Conv2d(in_channels = 1, out_channels = 16, kernel_size = 3, stride = 1, padding = 1),
            nn.BatchNorm2d(num_features=16),# same shape as input 
            act_fn(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(in_channels = 16, out_channels = 8, kernel_size = 1),
            nn.Conv2d(in_channels = 8, out_channels = 32, kernel_size = 3, stride = 1, padding = 1),
            nn.BatchNorm2d(num_features=32),# same shape as input 
            act_fn(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(in_channels = 32, out_channels = 8, kernel_size = 1),
            nn.Conv2d(in_channels = 8, out_channels = 64, kernel_size = 3, stride = 1, padding = 1),
            nn.BatchNorm2d(num_features=64),# same shape as input 
            act_fn(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(in_channels = 64, out_channels = 8, kernel_size = 1),
            nn.Conv2d(in_channels = 8, out_channels = 128, kernel_size = 3, stride = 1, padding = 1),
            nn.BatchNorm2d(num_features=128),# same shape as input 
            act_fn(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(in_channels = 128, out_channels = 8, kernel_size = 1),
            nn.Conv2d(in_channels = 8, out_channels = 256, kernel_size = 3, stride = 1, padding = 1),
            nn.BatchNorm2d(num_features=256),# same shape as input 
            act_fn(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(in_channels = 256, out_channels = 8, kernel_size = 1),
            nn.Conv2d(in_channels = 8, out_channels = 512, kernel_size = 3, stride = 1, padding = 1),
            nn.BatchNorm2d(num_features=512),# same shape as input 
            act_fn(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(in_channels = 512, out_channels = 8, kernel_size = 1),
            nn.Conv2d(in_channels = 8, out_channels = 1024, kernel_size = 3, stride = 1, padding = 1),
            nn.BatchNorm2d(num_features = 1024),# same shape as input 
            act_fn(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
        
        )  
        # decoder 
        self.decoder=nn.Sequential(
            ## (N , 1024, 11, 8)
            
            nn.Conv2d(in_channels = 1024, out_channels = 8, kernel_size = 1),
            nn.ConvTranspose2d(in_channels = 8, out_channels = 512, kernel_size = 3,padding = 1,stride = 2,output_padding = 1),# 
            nn.BatchNorm2d(num_features = 512),
            act_fn(),
            
            nn.Conv2d(in_channels = 512, out_channels = 8, kernel_size = 1),
            nn.ConvTranspose2d(in_channels = 8, out_channels = 256, kernel_size = 3,padding = 1,stride = 2,output_padding = 1),# 
            nn.BatchNorm2d(num_features = 256),
            act_fn(),
            
            nn.Conv2d(in_channels = 256, out_channels = 8, kernel_size = 1),
            nn.ConvTranspose2d(in_channels = 8, out_channels = 128, kernel_size = 3,padding = 1,stride = 2,output_padding = 1),# 
            nn.BatchNorm2d(num_features=128),
            act_fn(),
            
            nn.Conv2d(in_channels = 128, out_channels = 8, kernel_size = 1),
            nn.ConvTranspose2d(in_channels = 8, out_channels = 64, kernel_size = 3,padding = 1,stride = 2,output_padding = 1),# 
            nn.BatchNorm2d(num_features=64),
            act_fn(),
            
            nn.Conv2d(in_channels = 64, out_channels = 8, kernel_size = 1),
            nn.ConvTranspose2d(in_channels = 8, out_channels = 32, kernel_size = 3,padding = 1,stride = 2,output_padding = 1),# 
            nn.BatchNorm2d(num_features=32),
            act_fn(),
            
            nn.Conv2d(in_channels = 32, out_channels = 8, kernel_size = 1),
            nn.ConvTranspose2d(in_channels = 8, out_channels = 16, kernel_size = 3,padding = 1,stride = 2,output_padding = 1),# 
            nn.BatchNorm2d(num_features=16),
            act_fn(),
            
            nn.ConvTranspose2d(in_channels = 16, out_channels = 1, kernel_size = 3,padding = 1,stride = 2,output_padding = 1),# 
            nn.BatchNorm2d(num_features=1),
            #nn.Dropout2d(p=0.5),
           # nn.Sigmoid(),
        )
    def encoding_fn(self, x):
        encoded = self.encoder(x)
        return encoded
                
    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded
            
class AE1(nn.Module):
    def __init__(self, act_fn: object = nn.LeakyReLU):
        super().__init__()
     
        
        self.encoder=nn.Sequential(
            # (N , 1, 1024, 1024)
            # Block 1
            nn.Conv2d(in_channels = 1, out_channels = 16, kernel_size = 3, stride = 2, padding = 1),
            nn.BatchNorm2d(num_features=16),# same shape as input 
            nn.ReLU(),
            #nn.MaxPool2d(kernel_size=2, stride=2),
            
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
            nn.Dropout2d(p=0.5),
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
        )  
       
        
        # decoder 
        self.decoder=nn.Sequential(
        
            
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
            #nn.Sigmoid(),
        )
    def encoding_fn(self, x):
        encoded = self.encoder(x)
        return encoded
                
    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded