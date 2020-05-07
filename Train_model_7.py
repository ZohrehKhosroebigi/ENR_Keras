from Save_model_json_2 import SaveModelJson
from Save_model_keras import SaveModelKeras
import os
class Trainmodel():
    def trainmodel(self, x_train, y_train,model_, epoch, batch_size):
        if not os.path.exists("models_keras"):
            os.mkdir("models_keras")
        #model_.fit(x=x_train, y=y_train, epochs=epoch, batch_size=batch_size)
        model_.fit(x_train, np.array(y_train), batch_size=batch_size, epochs=epoch, validation_split=0.1, verbose=2)

        #save keras model
        saveTokeras_model = SaveModelKeras()
        saveTokeras_model.save_model_keras(model_)
        #save json model
        saveTojason_model=SaveModelJson()
        saveTojason_model.save_model_json(model_)
        return model_


