from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
import numpy as np
class SentenceToIdx():
    def sentence_to_idx(self,myobj, MAX_LEN_vector):
        self.n_tags=myobj.n_tags
        self.max_len_vector=MAX_LEN_vector
        # Convert each sentence from list of Token to list of word_index
        X = [[myobj.word2idx[w[0]] for w in s] for s in myobj.sentences]
        # Padding each sentence to have the same lenght
        X = pad_sequences(maxlen=self.max_len_vector, sequences=X, padding="post", value=myobj.word2idx["PAD"])

        # Convert Tag/Label to tag_index
        Y = [[myobj.tag2idx[w[2]] for w in s] for s in myobj.sentences]
        # Padding each sentence to have the same lenght
        Y = pad_sequences(maxlen=self.max_len_vector, sequences=Y, padding="post", value=myobj.tag2idx["PAD"])

        # One-Hot encode
        Y = [to_categorical(i, num_classes=myobj.n_tags + 1) for i in Y]  # n_tags+1(PAD)

        self.X_train, self.X_test, self.Y_train, self.Y_test = train_test_split(X, Y, test_size=0.1)
        self.X_train.shape, self.X_test.shape, np.array(self.Y_train).shape, np.array(self.Y_test).shape

        print('Raw Sample: ', ' '.join([w[0] for w in myobj.sentences[0]]))
        print('Raw Label: ', ' '.join([w[2] for w in myobj.sentences[0]]))
        print('After processing, sample:', X[0])
        print('After processing, labels:', Y[0])
        return self.X_train,self.X_test,self.Y_train,self.Y_test,self.n_tags,self.max_len_vector

    def __repr__(self):
        return f'{self.X_train}{self.X_test}{self.Y_train}{self.Y_test}{self.n_tags}{self.max_len_vector}'




