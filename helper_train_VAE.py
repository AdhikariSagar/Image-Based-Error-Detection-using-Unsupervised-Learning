import torch.nn as nn
import torchvision
import torch
import time
import random
import csv
import numpy as np
from pathlib import Path
import torch.nn.functional as F
from sklearn.metrics import roc_curve
from sklearn.metrics import auc, ConfusionMatrixDisplay
from sklearn.metrics import confusion_matrix, classification_report
from matplotlib import pyplot as plt
from helper_utils import load_data


class Trainer():
    def __init__(self, model: object, hyperParams, DataLoderParams):
        
        self.model  = model.cuda()
        self.learning_rate = hyperParams.get('LEARNING_RATE', 1e-3)
        self.image_size = hyperParams.get('IMAGE_SIZE', 1024)
        self.n_epoch =  hyperParams.get('NUM_EPOCHS', 50)
        self.data_class = hyperParams.get('DATA_CLASS')
        self.version = hyperParams.get('VERSION')
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr= self.learning_rate)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, factor=0.01, patience=5, mode='min', verbose=True)
        self.loss_fn = F.mse_loss
        #self.loss_fn = nn.MSELoss()
        self.DataLoderParams = DataLoderParams
        self.train_dataLoader, self.valid_dataLoader, self.test_dataLoader = load_data(self.data_class, size= self.image_size, params = self.DataLoderParams)
        
        self.train_data = {}
        self.test_data = {}
        self.evaluation_metrics = {}
        
        self.save_dir = Path(f'models/{self.data_class} / {self.version}')
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.file_name = 'vae_weights.pth'

    def pixelwise_mse_loss(self, recon_fe, features):
        
        batchsize = features.size(0)
        pixelwise = self.loss_fn(recon_fe, features, reduction='none')
        pixelwise = pixelwise.view(batchsize, -1).sum(axis=1) # sum over pixels/ latentdimension 
        pixelwise = pixelwise.mean() # average over batch dimension
        return pixelwise
        
    def loss_vae(self, recon_fe, features, z_mean, z_log_var):
        """
        calculate loss for vae 
        """
        
        reconstruction_term_weight = 1
        kl_div = -0.5 * torch.sum(1 + z_log_var 
                                      - z_mean**2 
                                      - torch.exp(z_log_var), 
                                      dim=1) # sum over latent dimension
        
        #batchsize = features.size(0)
        
        kl_div = kl_div.mean() # average over batch dimension
        #mse = self.loss_fn(recon_fe, features, reduction='none')
        #mse = mse.view(batchsize, -1).sum(axis=1) # sum over pixels/ latentdimension 
        #mse = mse.mean()  # average over batch dimension
        pixelwise = self.pixelwise_mse_loss(recon_fe, features)
        total_loss = reconstruction_term_weight*pixelwise + kl_div
        return total_loss, pixelwise, kl_div
    
#     def loss_vae_new(self, recon_fe, features, z_mean, z_log_var):
#         """
#         calculate loss for vae 
#         """
                
#         reconstruction_term_weight = 0.1
#         kl_div = -0.5 * torch.sum(1 + z_log_var 
#                                       - z_mean**2 
#                                       - torch.exp(z_log_var), 
#                                       dim=1) # sum over latent dimension
        
#         #batchsize = features.size(0)
        
#         kl_div = kl_div.mean() # average over batch dimension
#         #mse = self.loss_fn(recon_fe, features, reduction='none')
#         #mse = mse.view(batchsize, -1).sum(axis=1) # sum over pixels/ latentdimension 
#         #mse = mse.mean()  # average over batch dimension
#         features = F.interpolate(features, size=(recon_fe.shape[2], recon_fe.shape[3]))
#         pixelwise = F.mse_loss(recon_fe, features)
#         total_loss = reconstruction_term_weight*pixelwise + kl_div
#         return total_loss, pixelwise, kl_div
    
    def plot_reconstruction(self, img, recons):
        """
        Plot the original and reconstructed images during training
        """
        fig, axes = plt.subplots(nrows=2, ncols=5, figsize=(2, 2))
        #fig.subplots_adjust(wspace=0., hspace=0.)

        for j in range(5):
            axes[0][j].imshow(np.squeeze(img[j].detach().cpu().numpy()), cmap='gray')
            axes[0][j].axis('off')

        for j in range(5):
            axes[1][j].imshow(np.squeeze(recons[j].detach().cpu().numpy()), cmap='gray')
            axes[1][j].axis('off')

        fig.tight_layout(pad=0., w_pad=0.1, h_pad=.1)
        plt.show()
    
    def train(self):
        epoch_losses = []
        epoch_val_losses = []
        start_time = time.time() 
        self.model.train()
        for i_epoch in range(self.n_epoch):
            epoch_loss = 0.0
            epoch_loss_val = 0.0
            n = 0
            for features in self.train_dataLoader:
                
                # Zero grad of the optimizer
                self.optimizer.zero_grad()
                # load on GPU
                features = features.cuda()
                encoded, z_mean, z_log_var, decoded = self.model(features)
                loss, mse_loss, kl_div = self.loss_vae(decoded, features, z_mean, z_log_var)
                #loss = self.loss_fn(reconstruction, features)
                # backward pass: compute gradient of the loss with respect to model parameters
                loss.backward()
                epoch_loss += float(loss)
                n += 1
                self.optimizer.step()
            epoch_loss = epoch_loss / n
            epoch_losses.append(epoch_loss)
            if (i_epoch+1)%2==0:
                self.plot_reconstruction(features, decoded)
            
            ###+++++++++++validation++++++++++#####
            m = 0
            for features  in self.valid_dataLoader:
                with torch.no_grad():
                    features = features.cuda()
                    #print('in', img.shape)
                    encoded, z_mean, z_log_var, decoded = self.model(features)
                    total_loss, mse_loss, kl_div = self.loss_vae(decoded, features, z_mean, z_log_var)

                    epoch_loss_val += float(total_loss)
                    m += 1
            epoch_loss_val = epoch_loss_val / m
            epoch_val_losses.append(epoch_loss_val)
            print(f'[{i_epoch+1}/{self.n_epoch}] Train-loss: {epoch_loss:.3f} Val-loss: {epoch_loss_val:.3f}')
            if self.scheduler:
                self.scheduler.step(epoch_val_losses[-1])
            elapsed = (time.time() - start_time)/60
            if i_epoch % 4 == 0:
                print(f'Time elapsed: {elapsed:.2f} min')
            

        elapsed = (time.time() - start_time)/60
        print(f'Total Training Time: {elapsed:.2f} min')
        self.train_data['epoch_train_losses'] = epoch_losses
        self.train_data['epoch_val_losses'] = epoch_val_losses

        fig=plt.figure(figsize=(8, 8))
        plt.plot(epoch_losses, 'y', label = 'Training Loss')
        plt.plot(epoch_val_losses, 'r', label = 'Validation Loss')
        plt.xlabel('epochs')
        plt.ylabel('loss')
        plt.title('Training and validation loss / epoch')
        plt.legend()
        plt.show()
        figname = f'training_loss_{self.version}.png'
        fig.savefig(self.save_dir / figname, dpi=fig.dpi)
        

    
    def compute_train_outputs(self):
        
        train_outputs = []
        for batch_imgs in self.train_dataLoader:
            with torch.no_grad():
                batch_imgs = batch_imgs.cuda()
                for img in batch_imgs:
                    with torch.no_grad():
                        img = img[None, :, :, :]
                        img = img.cuda()
                        _, _, _, reconstruction_img = self.model(img)
                        #img = F.interpolate(img, size=(reconstruction_img.shape[2], reconstruction_img.shape[3]))
                        difference = self.pixelwise_mse_loss(reconstruction_img.cuda(), img)#, reduction="mean")
                        train_outputs.append((img.cpu(), reconstruction_img.cpu(), difference.cpu()))
                    torch.cuda.empty_cache()  # Clear GPU memory

        self.train_data['mean_train_error'] = (sum([i[2] for i in train_outputs]) / len(train_outputs)).cpu().item()
        
        print(f'Mean traning recon. error: {self.train_data["mean_train_error"]}')
        self.train_data['train_outputs'] = train_outputs
        
    def compute_test_outputs(self):
        test_outputs = []
        test_good_outputs = []
        test_anom_outputs = []
        all_labels = []
        for (imgs, labels) in self.test_dataLoader:
            all_labels += labels
            for (img, label) in zip(imgs, labels):
                with torch.no_grad():
                    img = img[None, :, :, :]
                    img = img.cuda()
                    _, _, _, reconstruction_img = self.model(img)
                    #img = F.interpolate(img, size=(reconstruction_img.shape[2], reconstruction_img.shape[3]))
                    difference = self.pixelwise_mse_loss(reconstruction_img.cuda(), img)#, reduction="mean")

                    data = (img, reconstruction_img, difference)
                    test_outputs.append(data)
                    if label:
                        test_anom_outputs.append(data)
                    else:
                        test_good_outputs.append(data)
        print(f'Mean test recon. error anom: {sum([i[2] for i in test_anom_outputs]) / len(test_anom_outputs)}')
        print(f'Mean test recon. error good: {sum([i[2] for i in test_good_outputs]) / len(test_good_outputs)}')
        self.test_data['test_outputs'] = test_outputs
        self.test_data['test_good_outputs'] = test_good_outputs
        self.test_data['test_anom_outputs'] = test_anom_outputs
        self.test_data['labels'] = list(map(lambda x: x.item(), all_labels))
        
    def inspect(self, mode: str = 'train'):
        fig = plt.figure(figsize=(18, 4))
        plt.tight_layout()
        
        if mode == 'train':
            outputs = self.train_data['train_outputs']
        else:
            outputs = self.test_data['test_outputs']
            
        imgs = [t[0].cpu().detach() for t in outputs]
        recon = [t[1].cpu().detach() for t in outputs]
        errors = [t[2].cpu().detach() for t in outputs]
        
        foo = list(zip(imgs, recon, errors))
        random.shuffle(foo)
        imgs, recon, errors = zip(*foo)
        
        MEAN = torch.tensor([0.0524])
        STD = torch.tensor([0.0284])
        for i, item in enumerate(recon):
            if i >= 5: break
            ax = plt.subplot(2, 9, 9+i+1)
            #ax.title.set_text(f'{errors[i].item():.2f}')
            item = item[0]
            img = item * STD[:, None, None] + MEAN[:, None, None]
            plt.imshow(np.array(img).transpose(1, 2, 0), cmap='gray')
            plt.yticks([]);
            plt.axis('off')
            plt.tight_layout(pad=0.1)
        for i, item in enumerate(imgs):
            if i >= 5: break
            plt.subplot(2, 9, i+1)
            item = item[0]
            img = item * STD[:, None, None] + MEAN[:, None, None]
            plt.imshow(np.array(img).transpose(1, 2, 0), cmap='gray')
            
            plt.xticks([])
            plt.yticks([])
            plt.axis('off')
            plt.tight_layout(pad=0.1)
          
        figname = f'origin_recon_{self.version}.png'
        fig.savefig(self.save_dir / figname, dpi=fig.dpi)
    
    def show_error_distribution(self):
        
        threshold = self.train_data['mean_train_error']
        
        train_errors = [t[2].item() for t in self.train_data['train_outputs']]
        test_errors = [t[2].item() for t in self.test_data['test_anom_outputs']]
        train_counts, train_bins = np.histogram(train_errors)
        test_counts, test_bins = np.histogram(test_errors)

        fig = plt.figure()
        # Filling the area between steps
        plt.fill_between(train_bins[:-1], train_counts, step="pre", color='green', alpha=0.2)
        plt.fill_between(test_bins[:-1], test_counts, step="pre", color='red', alpha=0.2)
        plt.step(train_bins[:-1], train_counts, color='green', label='Undefected', alpha=.6)
        plt.step(test_bins[:-1], test_counts, color='red', label='Defected', alpha=.6)
        plt.axvline(x=threshold, label='Threshold', linestyle ='dashed', color='blue')
        plt.xlabel('Reconstruction error')
        plt.ylabel('N Samples')
        plt.legend()
        plt.show()
        figname = f'error_distribution_{self.version}.png'
        fig.savefig(self.save_dir / figname, dpi=fig.dpi)

    def evaluate_ae(self):
        
        threshold = self.train_data['mean_train_error']
        
        test_pred = np.array([t[2].cpu().item() for t in self.test_data['test_outputs']])
        test_pred_labels = [i > threshold for i in test_pred]
        
        fpr, tpr, threshholds = roc_curve(self.test_data['labels'], test_pred_labels)
        auc_score = auc(fpr, tpr)
        self.evaluation_metrics['fpr'] = fpr
        self.evaluation_metrics['tpr'] = tpr
        self.evaluation_metrics['threshholds'] = threshholds
        self.evaluation_metrics['auc_score'] = auc_score
        lw = 2
        fig = plt.figure(figsize=(5.15, 5.15))
        plt.plot(fpr, tpr,
                 lw=lw,
                 label=f"ROC curve (AUC = {auc_score:.2f})",
                 color='red')
        plt.plot([0, 1], [0, 1], lw=lw, color='navy', linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        #plt.title(f"ROC {dataset}/{obj}")
        plt.legend(loc="lower right")
        figname = f'ROC_AUC_{self.version}.png'
        #plt.show()
        plt.savefig(self.save_dir / figname, dpi=fig.dpi)

        conf_matrix = confusion_matrix(self.test_data['labels'], test_pred_labels)
        self.evaluation_metrics['conf_matrix'] = conf_matrix
        disp = ConfusionMatrixDisplay(conf_matrix)#, display_labels=["IO", "NIO"])
        #ConfusionMatrixDisplay.from_predictions(self.test_data['labels'], test_pred_labels)
        plt.title("Confusion Matrix")
        disp.plot()
        figname = f'confusion_matrix_{self.version}.png'
        plt.savefig(self.save_dir / figname, dpi=fig.dpi)
        plt.show()
        
        cl_report = classification_report(self.test_data['labels'], test_pred_labels)
        fig, ax = plt.subplots(figsize=(10, 6))
        fig.suptitle("Classification Report", fontsize=13,x = 0.335, y=1.02)
        ax.text(0.05, 0.95, cl_report, va='center', ha='left', fontfamily='DejaVu Sans', fontsize=10)
        ax.axis('off')
        figname = f'classification_report_{self.version}.png'
        plt.savefig(self.save_dir / figname, bbox_inches='tight')
        #plt.show()
        
    def save_model(self):
        
        file_name = f'variational_autoencoder_weights_{self.version}.pth'
        torch.save(self.model.state_dict(), self.save_dir / file_name)
        
        csv_name =  f'variational_autoencoder_results_{self.version}.csv'
        csv_path = self.save_dir / csv_name
        
        if not csv_path.exists():
            with open(csv_path, 'w', newline='') as csv_file:
                writer = csv.writer(csv_file)
                writer.writerow(["Type", "Key", "Value"])  # Write header row
        with open(csv_path, 'a', newline='') as csv_file:
            writer = csv.writer(csv_file)

            for data_type, data_dict in [
                ("train_data", self.train_data),
                ("test_data", self.test_data),
                ("evaluation_metrics", self.evaluation_metrics)
            ]:
                for key, value in data_dict.items():
                    writer.writerow([data_type, key, value])
        
    def load_model(self, version = "v1"):
        file_name = f'variational_autoencoder_weights_{version}.pth'
        self.model.load_state_dict(torch.load(self.save_dir / file_name))




        
        

            


                
                

            
        
