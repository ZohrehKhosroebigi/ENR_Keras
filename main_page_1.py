# Tensor version is 2.0.0
from writing import Writelogs
from Loadrawdata_show_2 import Loadrawdata_show
from Word_tags_to_idx_3 import NoramlPic
from LSTM_model_5 import CNN_model
from Create_model import Create_model
from Compile_model_6 import Compilemodel
from Train_model_7 import Trainmodel
from Evaluate_model_8 import Evaluate_model
from Save_model_json_2 import SaveModelJson
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.python.framework import ops
#from mini_bach import random_mini_batches
#load raw data from files and show to user
loadrawdata=Loadrawdata_show()
loadrawdata.load
#index of the picture that user wnats to see
#loadrawdata[3]
mywriting = Writelogs()
mywriting.writing(str(loadrawdata))
#print(loadrawdata)
###########################################################
"""
#Normalization of raw data
norm_data=NoramlPic()
norm_data.norm(loadrawdata.load)
mywriting.writing(str(norm_data))
#print(norm_data)
"""
#####################################################
#Word to idx and idx to word
#Preprocessing Sentence to idx
#LSTM
#compile
#train

train_model=Trainmodel()
train_model.trainmodel(norm_data.X_train,norm_data.Y_train, creating_model.model_,epoch=2, batch_size=64)

evaluate_model=Evaluate_model()
evaluate_model.evaluatemodel(norm_data.X_test,norm_data.Y_test,creating_model.model_)

save_model=SaveModelJson()
save_model.save_model_json(creating_model.model_)
