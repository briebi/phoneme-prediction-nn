import collections
import os
import re
import torch
import torch.nn as nn
import random
import numpy as np

# learning rate for the optimizer
lr = 0.003


# word length used for creating word lists and test sets
# CHANGE ME: you can it to any number between 4-11 inclusive
wdlen = 5 
lang_list = ['uk', 'gr', 'cs', 'fr'] # uk = english, gr = german, cs = czech, fr = french

# define the language prediction model using PyTorch's nn.Module
class PredictLanguage(nn.Module):
    def __init__(self, lang_list):
        super(PredictLanguage, self).__init__()

        # number of training samples to use
        self.num_to_train = 23000

        # list of languages to predict
        self.lang_list = lang_list

        # map each language to an index
        self.lang2ix = {key: i for (i, key) in enumerate(self.lang_list)}
        
        # initialize dictionaries for phoneme data
        self.phoneme_lists = {lang: [] for lang in lang_list} 
        self.phoneme_frequencies = {lang: collections.defaultdict(int) for lang in lang_list}  

    # create word lists and test sets for each language
    def create_wordlists(self):
        self.wordlist_dict = collections.defaultdict(lambda: [])
        self.test_dict = collections.defaultdict(lambda: [])
        for lang in self.lang_list:
            # file path for words of the specified length
            if os.path.isfile(lang+'_phon_'+ '_len'+str(wdlen)+'.txt'):
                with open(lang+'_phon_'+'_len'+str(wdlen)+'.txt') as f0:
                    for wd in f0.readlines():
                        # remove trailing whitespace
                        wd = wd.rstrip()
                        
                        # split words into training or test set (70% train, 30% test)
                        if random.randint(0,9) > 2: self.wordlist_dict[lang].append(wd)
                        else: self.test_dict[lang].append(wd)
                print('Wordlist and test words of words of length', wdlen, 'successfully created for language', lang)
            else:
                print('Language', lang, 'with path', lang+'_phon_'+'_len'+str(wdlen)+'.txt', 'not found!')

    # extract a unique list of phonemes from word lists
    def get_phoneme_list(self):
        self.phonemes = []
        for lang in list(self.wordlist_dict):
            # add phonemes from training and test sets to the phoneme list
            for wd in self.wordlist_dict[lang]:
                for phoneme in wd:
                    if phoneme not in self.phonemes:
                        self.phonemes.append(phoneme)
            for wd in self.test_dict[lang]:
                for phoneme in wd:
                    if phoneme not in self.phonemes:
                        self.phonemes.append(phoneme)
        print('There are', len(self.phonemes), 'phonemes across the', len(lang_list), 'languages')
        
        # map each phoneme to an index
        self.phoneme2ix = {}
        for i, phoneme in enumerate(self.phonemes):
            self.phoneme2ix[phoneme] = torch.tensor(i)

    def create_embeddings_for_phonemes(self):
        self.phoneme_embeddings = nn.Embedding(len(self.phonemes),len(self.phonemes))

    def set_up_model(self):
        # dimensions for the hidden layers
        self.dim_hid1 = 32
        self.dim_hid2 = 32

        # fully connected layers
        self.input2hidden1 = nn.Linear(len(self.phonemes)*wdlen, self.dim_hid1)
        self.hidden1hidden2 = nn.Linear(self.dim_hid1, self.dim_hid2)
        self.hidden2output = nn.Linear(self.dim_hid2, len(lang_list))
        
        # activation function
        self.sigmoid = nn.Sigmoid()


    # forward pass for model
    def forward(self, input_word):
        # convert each phoneme in the word to its embedding vector
        phoneme_vectors = []
        for phoneme in input_word:
            phoneme_vectors.append(self.phoneme_embeddings(self.phoneme2ix[phoneme]))
        
        # concatenate phoneme vectors to form a single input vector
        vector_for_word = torch.unsqueeze(torch.cat(phoneme_vectors, dim=0), 0)

        # pass through the network layers
        hid1 = self.input2hidden1(vector_for_word)
        hid1 = self.sigmoid(hid1)
        hid2 = self.hidden1hidden2(hid1)
        hid2 = self.sigmoid(hid2)

        # output layer
        return self.hidden2output(hid2)

    # makes phoneme list and frequency counts for each language
    def get_phoneme_lists_and_frequencies(self):
        for lang in self.lang_list:
            # temporary variables to store unique phonemes and their frequencies
            unique_phonemes = list()
            phoneme_freq = collections.defaultdict(int)

            # process each word in the training word list
            for wd in self.wordlist_dict[lang]:
                for phoneme in wd:
                    # updating frequency
                    phoneme_freq[phoneme] += 1
                    # updating list
                    if phoneme not in unique_phonemes:
                        unique_phonemes.append(phoneme)

            # list of unique phonemes in the phoneme_lists dictionary
            self.phoneme_lists[lang] = unique_phonemes
            # list of frequencies
            self.phoneme_frequencies[lang] = phoneme_freq

            print(f'Phoneme list for language {lang}:', self.phoneme_lists[lang])
            print(f'Phoneme frequencies for language {lang}:', dict(phoneme_freq))

def train(model):
    criterion = nn.CrossEntropyLoss()
    optimiser = torch.optim.Adam(model.parameters(), lr=lr)
    num_tested = 0
    num_correct = 0
    ep_loss = 0
    
    for i in range(model.num_to_train):
        # select a random language and word from the training set
        lang = random.choice(lang_list)
        wd = random.choice(model.wordlist_dict[lang])

        # model prediction
        num_tested += 1
        pred = model(wd)
        target_lang_ix = model.lang2ix[lang]
        target = torch.unsqueeze(torch.tensor(target_lang_ix), 0)
        
        # update accuracy
        with torch.no_grad():
            pred_numpy = np.argmax(pred.numpy(), axis=1).tolist()[0]
            if pred_numpy == target_lang_ix:
                num_correct += 1
        
        # calculate loss and back propagate
        loss = criterion(pred, target)
        loss.backward()
        optimiser.step()
        optimiser.zero_grad()
        ep_loss += loss.detach()

        # periodically log progress
        if i % 1000 == 0 and i > 0:
            average_loss = ep_loss / 1000
            ep_loss = 0
            print(i, average_loss, 'training accuracy on last 1000 examples', round(num_correct / num_tested, 4))
            num_tested = 0
            num_correct = 0

def test(model):
    num_tested = 0
    num_correct = 0
    for lang in lang_list:
        for wd in model.test_dict[lang]:
            num_tested += 1
            with torch.no_grad():
                pred = model(wd)
                target_lang_ix = model.lang2ix[lang]
                target = torch.unsqueeze(torch.tensor(target_lang_ix), 0)
                pred_numpy = np.argmax(pred.numpy(), axis=1).tolist()[0]
                if pred_numpy == target_lang_ix:
                    num_correct += 1
                else:
                    print(wd, lang, lang_list[pred_numpy])
        print('Test accuracy on language', lang,  round(num_correct / num_tested, 4))
        num_tested = 0
        num_correct = 0

# intialize teh model
model = PredictLanguage(lang_list)

# run the model functions
model.create_wordlists()
model.get_phoneme_list()
model.create_embeddings_for_phonemes()
model.set_up_model()
train(model)
test(model)

# output phoneme data
model.get_phoneme_lists_and_frequencies()