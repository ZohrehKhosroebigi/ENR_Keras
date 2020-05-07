from one_hot import convert_to_one_hot
class WordTagsToIdx():
   def word_tags_to_idx(self,myobj):
        self.sentences,words, tags,self.n_words,self.n_tags = myobj

####################################################################
        # Vocabulary Key:word -> Value:token_index
        # The first 2 entries are reserved for PAD and UNK
        self.word2idx = {w: i + 2 for i, w in enumerate(words)}
        self.word2idx["UNK"] = 1  # Unknown words
        self.word2idx["PAD"] = 0  # Padding
        # Vocabulary Key:token_index -> Value:word
        self.idx2word = {i: w for w, i in self.word2idx.items()}
        print("The word Obama is identified by the index: {}".format(self.word2idx["خارجه"]))
###################################################################

        # Vocabulary Key:Label/Tag -> Value:tag_index
        # The first entry is reserved for PAD
        self.tag2idx = {t: i + 1 for i, t in enumerate(tags)}
        self.tag2idx["PAD"] = 0
        # Vocabulary Key:tag_index -> Value:Label/Tag
        self.idx2tag = {i: w for w, i in self.tag2idx.items()}
        print("The labels B-geo(which defines Geopraphical Enitities) is identified by the index: {}".format(
            self.tag2idx["B-Loc"]))
        return self.word2idx,self.idx2word,self.tag2idx,self.idx2tag,self.sentences,self.n_words,self.n_tags

   """def __str__(self):
        #print("-------------------normal data information--------------------")
        return f'X_train shape {self.X_train.shape}\nY_train shape{self.Y_train.shape}\nX_test shape{self.X_test.shape}\nY_test shape {self.Y_train.shape}\nClasses is: {self.classes}\nnumber of training examples= {self.X_train.shape[0]}\nnumber of test examples= {self.X_test.shape[0]}\nnumber of classes={self.len_class}'
"""
   def __repr__(self):
        return f'{self.sentences}{self.word2idx}{self.tag2idx}{self.n_words}{self.n_tags}'



