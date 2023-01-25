#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
The NoiseProbe class designed to run probing with noise experiments, assuming it's working with one of the Conneau et al. probing datasets (https://github.com/facebookresearch/SentEval/tree/main/data/probing)

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this
   list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
"""

import torch, os
import numpy as np
import warnings
import random

from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score
from sklearn import preprocessing
from pytorch_pretrained_bert import BertTokenizer, BertModel

class NoiseProbe:
    def __init__(self,data_file,enc,noise):

        if enc == 'bert_train':
            self.bt = BertTokenizer.from_pretrained('bert-base-uncased')
            self.bert = BertModel.from_pretrained('bert-base-uncased')
            self.bert.eval()
            os.environ['KMP_DUPLICATE_LIB_OK']='True'

        self.model = MLPClassifier(max_iter=1500)

        dataset = data_file
        train_data = dataset["train_sample"]
        test_data = dataset["test_sample"]

        self.x_train, self.y_train, self.x_test, self.y_test = [], [], [], []

        for train in train_data: #train data is a list from the input pickle file, so each train is a dictionary with different keys (embedding, verb, noun, class)
            self.x_train.append(self.introduce_noise(self.embedding(train,enc),enc,noise)) #this is where the sent embeddings are used as training data
            self.y_train.append(train["lab_str"]) 

        for test in test_data:
            self.x_test.append(self.introduce_noise(self.embedding(test,enc),enc,noise))
            self.y_test.append(test["lab_str"])

    def scale_vector(self, embedding, new_norm):
        norm = np.linalg.norm(embedding, ord=2)
        k = (new_norm / norm)  # the key is the old and new norm need to be of the same order, e.g. either both L1 or both L2

        embedding = embedding * k
        return embedding

    def introduce_noise(self,embedding,enc,noise):

        if enc == 'bert_train':
            enc = 'bert'

        size = len(embedding)

        if noise != 'vanilla':

            # Generate Random Vectors
            if noise == 'rvec':
                if enc == 'bert':
                    embedding = np.random.uniform(-5.0826163,1.5603778,size) # vector size within the range of BERT
                elif enc == 'glove':
                    embedding = np.random.uniform(-2.5446498,3.19762,size) # vector size within the range of GLOVE

            # Ablate the Norm Container: randomly generates a new norm for the vector (within scale of task vectors) and then scales the dimension values to match it
            if noise == 'abn':
                if enc == 'bert':
                    norm = np.random.uniform(7.1895742,13.285439) # for BERT range
                elif enc == 'glove':
                    norm = np.random.uniform(2.0041258,8.035887) # for GLOVE range

                embedding = self.scale_vector(embedding,norm)

            # Ablate the Dimension Container: randomly generates a vector, replaces its norm with the old norm and scales dimension values to match it
            if noise == 'abd':
                norm = np.linalg.norm(embedding, ord=2)

                if enc == 'bert':
                    embedding = np.random.uniform(-5.0826163,1.5603778,size)
                if enc == 'glove':
                    embedding = np.random.uniform(-2.5446498,3.19762,size) # for BERT: np.random.uniform(-5.0826163,1.5603778,768); for GLOVE: np.random.uniform(-2.5446498,3.19762,300)

                embedding = self.scale_vector(embedding,norm)

            # Ablate both the Norm and Dimension Container: randomly generate a new norm and a new vector, then scales the dimensions of the new vector to the new norm
            if noise == 'abnd':
                if enc == 'bert':
                    norm = np.random.uniform(7.1895742,13.285439) # for BERT
                    embedding = np.random.uniform(-5.0826163,1.5603778,size) # for BERT
                if enc == 'glove':
                    norm = np.random.uniform(2.0041258,8.035887) # for GLOVE
                    embedding = np.random.uniform(-2.5446498,3.19762,size) # for GLOVE

                embedding = self.scale_vector(embedding,norm)

            # Delete half the vector
            if noise == 'd1h':
                embedding = np.delete(embedding, slice(int(len(embedding)/2))) # delete first half of the embedding
            if noise == 'd2h':
                embedding = np.delete(np.flip(embedding), slice(int(len(embedding)/2))) # delete second half of the embedding

        return embedding

    def embedding(self, p, enc):
        if enc == 'bert':   return p['bert']  # this is to load 'frozen' BERT embeddings
        if enc == 'glove':   return p['glove'] # this is to load 'frozen' GLOVE embeddings
        if enc == 'bert_train':   return self.BERT(p['sent']) # this is to train BERT embeddings
        print("no associated embedding for", enc)
       
    def BERT(self,sent):
        tt = self.bt.tokenize("[CLS] "+sent+" [SEP]")
        it = self.bt.convert_tokens_to_ids(tt)

        with torch.no_grad():   encoded_layers, _ = self.bert(torch.tensor([it]), torch.tensor([[1]*len(tt)]))
        embedding = torch.mean(encoded_layers[11], 1)[0].numpy()
        return embedding # this is the BERT sentence embedding - mean of bert's word embeddings (final layer)
        # if we wanted to do something different than mean pooling, that can be done above here

    def multiclass_roc_auc_score(self, y_test, y_pred, average):
        lb = preprocessing.LabelBinarizer()
        lb.fit(y_test)

        y_test = lb.transform(y_test)
        y_pred = lb.transform(y_pred)

        return roc_auc_score(y_test, y_pred, average=average, multi_class="ovo")

    def calculate_metrics(self, y_test, predictions, probabilities, classes):
        matrix = confusion_matrix(y_test, predictions)
        tp = int(matrix[1][1])
        fn = int(matrix[1][0])
        fp = int(matrix[0][1])
        tn = int(matrix[0][0])

        reversed_matrix = np.array([[tp, fn], [fp, tn]]) 
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            acc = accuracy_score(y_test, predictions)
            prec = precision_score(y_test, predictions, average='macro') #alternative: average='binary'
            rec = recall_score(y_test, predictions, average='macro')
            f1 = f1_score(y_test, predictions, average='macro')
            if classes == 'binary':
                auc_pred = self.multiclass_roc_auc_score(y_test, predictions, average='macro')
                auc_probs = roc_auc_score(y_test, probabilities, average='macro')
            elif classes == 'multiclass':
                auc_pred = self.multiclass_roc_auc_score(y_test, predictions, average='macro')
                auc_probs = roc_auc_score(y_test, probabilities, average='macro', multi_class="ovo")

        return acc, prec, rec, f1, auc_pred, auc_probs, reversed_matrix
    
    def train(self, baseline): #we can pass a list of idioms and this will train the model by using all the train data with that idiomatic phrase
        if baseline:
            pass
        else:
            X,Y = self.x_train,self.y_train
            self.model.fit(X, Y)
        
    def test(self, classes, baseline):
        if baseline:
            X, Y = self.x_test, self.y_test

            Y_pred = np.array(range(len(X)), dtype=str)  # initialises an array of the required size; if a different probing task has integer labels then dtype=int
            labels = ["O", "I"] # note that these will only work with the bigram shift task as the labels for other tasks are different; change the values in this array to evaluate on random prediction baseline on different tasks; see commented line below for top constituent prediction
            # classes = ["ADVP_NP_VP_.", "CC_ADVP_NP_VP_.", "CC_NP_VP_.", "IN_NP_VP_.", "NP_ADVP_VP_.", "NP_NP_VP_.", "NP_PP_.", "NP_VP_.", "OTHER", "PP_NP_VP_.", "RB_NP_VP_.", "SBAR_NP_VP_.", "SBAR_VP_.", "S_CC_S_.", "S_NP_VP_.", "S_VP_.", "VBD_NP_VP_.", "VP_.", "WHADVP_SQ_.", "WHNP_SQ_."]
            for i in range(0, len(X)):  # this generates random predictions (random or 1 or 0 or whatever)
                Y_pred[i] = random.choice(labels)  # random class choice

            Y_probs = np.random.dirichlet(np.ones(len(labels)), size=len(X))[:, 1]  # this generates random probabilities (floats between 0,1) for each possible label given an instance

            acc, prec, rec, f1, auc_pred, auc_probs, _ = self.calculate_metrics(Y, Y_pred, Y_probs, classes)

        else:
            X, Y = self.x_test, self.y_test
            Y_pred = self.model.predict(X) # the trained model's prediction
        
            if classes == "binary":
                Y_probs = self.model.predict_proba(X)[:, 1]
            elif classes == "multiclass":
                Y_probs = self.model.predict_proba(X)

            acc, prec, rec, f1, auc_pred, auc_probs, _ = self.calculate_metrics(Y, Y_pred, Y_probs, classes)

        return [round(acc,4), round(prec,4), round(rec,4), round(f1,4), round(auc_pred,4), round(auc_probs,4)]
        
    def evaluate(self, classes, baseline):
        self.train(baseline)
        print(self.test(classes,baseline))


