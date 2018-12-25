from gensim.models import word2vec
from gensim.corpora.dictionary import Dictionary
import os
import pickle
import tensorflow as tf
class Word2Vec_xsh:

    def __init__(self, docpath, modelpath, size, window):
        self.docpath = docpath
        #the name of the dictionary
        self.modelpath = modelpath
        self.size = size
        self.window = window
    def build_dictionary(self, model):
        dict_xsh = Dictionary()
        dict_xsh.doc2bow(model.wv.vocab, allow_update=True)
        indexword = {v: k+1  for k, v in dict_xsh.items()}
        print(len(indexword))
        vectorword = {word:model[word] for word in indexword.keys()}
        #print('vector', vectorword.values())
        #print(len(vectorword))
        file_dictionary = open(self.modelpath,'wb')
        pickle.dump(indexword, file_dictionary)
        pickle.dump(vectorword, file_dictionary)
        file_dictionary.close()
        print("词典存储完毕")
    def get_document(self, docpath):
        text = []

        for dir in os.listdir(docpath):
            for dir_1 in os.listdir(os.path.join(docpath, dir)):
                file1 =open(os.path.join(docpath, dir, dir_1), 'r', encoding='utf-8')
                text_temp = file1.readline().strip().split()
                text.append(text_temp)
                file1.close()

        return text

    def train(self):
        texts_train = self.get_document(self.docpath+'/train')
        texts_test = self.get_document(self.docpath+'/test')
        texts=texts_test+texts_train

        model = word2vec.Word2Vec(texts, self.size, min_count=1, window= self.window)
        #调试
        #model.save(self.modelpath)
        self.build_dictionary(model)
        return model
if __name__ =="__main__":
    word2vec_xsh= Word2Vec_xsh('./data/bingli_exp_result', './cidian_data15_size5.pkl', 128, 5)
    model = word2vec_xsh.train()
    print('咳嗽', model.wv['咳嗽'])



