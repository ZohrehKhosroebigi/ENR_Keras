from keras.models import Model, Input
from keras.layers import LSTM, Embedding, Dense, TimeDistributed, Dropout, Bidirectional
from keras_contrib.layers import CRF
class CreatingModel():
    def creating_model(self,myobj):

        # Model definition
        input = Input(shape=(myobj.max_len_vector,))
        self.model_ = Embedding(input_dim=myobj.n_words+2, output_dim=EMBEDDING, # n_words + 2 (PAD & UNK)
                          input_length=myobj.max_len_vector, mask_zero=True)(input)  # default: 20-dim embedding
        self.model_ = Bidirectional(LSTM(units=50, return_sequences=True,
                                   recurrent_dropout=0.1))(self.model_)  # variational biLSTM
        self.model_ = TimeDistributed(Dense(50, activation="relu"))(self.model_)  # a dense layer as suggested by neuralNer
        crf = CRF(myobj.n_tags+1)  # CRF layer, n_tags+1(PAD)
        out = crf(self.model_)  # output

        self.model_ = Model(input, out)
        return self.model_











        return self.X,self.X_input
