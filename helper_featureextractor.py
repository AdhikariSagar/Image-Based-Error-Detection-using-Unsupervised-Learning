import torch
from matplotlib import pyplot as plt
from sklearn.manifold import TSNE
from sklearn.utils import shuffle
import numpy as np
import pandas as pd
import seaborn as sns

class FeatureExtractor():
    def __init__(self, encoder: object, train_dataloader, test_dataloader):
        
        self.train_dataloader = train_dataloader
        self.test_dataloader = test_dataloader
        self.encoder = encoder
    
    def extract_train(self) -> np.array:
        f_list = []
        for i_batch, data in enumerate(self.train_dataloader):
            data = data.type(torch.FloatTensor).to('cuda')
            self.encoder.eval()
            with torch.no_grad():
                features = self.encoder.encode(data)
                f_list.append(features)
        return torch.cat(f_list).detach().cpu().numpy()
    
    def extract_test(self) -> tuple:
        labels = []
        f_list = []
        for i_batch, data in enumerate(self.test_dataloader):
            batch_labels = data[1]
            data = data[0].type(torch.FloatTensor).to('cuda')
            
            self.encoder.eval()
            with torch.no_grad():
                features = self.encoder.encode(data)
                f_list.append(features)
            labels.append(batch_labels)
                
        return torch.cat(f_list).detach().cpu().numpy(), torch.cat(labels).detach().cpu().numpy()
    
    def plot_tsne(self):
        embeds, labels = self.extract_test()
        tsne = TSNE(n_components=2, verbose=1, perplexity=30, n_iter=1000)
        embeds, labels = shuffle(embeds, labels)
        tsne_results = tsne.fit_transform(embeds)
        tsne_result_df = pd.DataFrame({'tsne_1': tsne_results[:,0], 'tsne_2': tsne_results[:,1], 'label': labels})
        fig, ax = plt.subplots(1)
        sns.scatterplot(x='tsne_1', y='tsne_2', hue='label', data=tsne_result_df, ax=ax)
        ax.legend(loc = 'upper right')
        plt.show()
