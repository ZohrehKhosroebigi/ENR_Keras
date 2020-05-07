from Read_rawdata_1 import read_rawdata
import numpy as np
import matplotlib.pyplot as plt
class Loadrawdata_show():##  Load data
    #read raw data from files
    @property
    def load(self):
        #self.X_train_orig,self.Y_train_orig,self.X_test_orig,self.Y_test_orig,self.classes =read_rawdata()
        return self.words,self.tags,self.sentences,self.n_words,self.n_tags
#show an image of x_train to user

    def __getitem__(self, idx):
        print(self.X_train_orig[idx].shape)
        plt.imshow(self.X_train_orig[idx])
        plt.show()
        print(f'y= {np.squeeze(self.Y_train_orig[:,idx])}')
    def __str__(self):
        try:
            #print("------------------raw data information--------------------------------")
            return f'X_train_orig shape {self.X_train_orig.shape}\nY_train_orig shape{self.Y_train_orig.shape}\nX_test_orig shape{self.X_test_orig.shape}\nY_test_orig shape {self.Y_test_orig.shape}\nClasses is: {self.classes}\nnumber of training examples= {self.X_train_orig.shape[0]}\nnumber of test examples= {self.X_test_orig.shape[0]}'
        except Exception as err:
            print(err)

    def __repr__(self):
        try:
            return f'{self.sentences}{self.words}{self.tags}{self.n_words}{self.n_tags}'
        except Exception as err:
            print(err)