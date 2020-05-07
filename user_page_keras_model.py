# load and evaluate a saved model
from numpy import loadtxt
from tensorflow.keras.models import load_model
import os
from Evaluate_model_8 import Evaluate_model
from Loadrawdata_show_2 import Loadrawdata_show
from Word_tags_to_idx_3 import NoramlPic

# load model
def trainmodel(self, x_train, y_train, model_, epoch, batch_size):
    if not os.path.exists("models_keras"):
        os.mkdir("models_keras")
keras_model = load_model('models_keras/model.h5')
# summarize model.
keras_model.summary()
# load dataset

loadrawdata=Loadrawdata_show()
loadrawdata.load

norm_data=NoramlPic()
norm_data.norm(loadrawdata.load)
# evaluate the model

evaluate_model=Evaluate_model()
evaluate_model.evaluatemodel(norm_data.X_test,norm_data.Y_test,keras_model)