
import numpy as np
import pandas as pd
import emoji
from Model import Model
import torch
from torch import nn
import nltk
import collections
#read the data for tweet analysis
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
from threading import Thread


data = pd.read_csv("input/text_emotion.csv")

#read the misspelled and corresponding correction of the data
misspell_data = pd.read_csv("input/aspell.txt",sep=":",names=["correction","misspell"])

#import contraction dataset
contractions = pd.read_csv("input/contractions.csv")
cont_dic = dict(zip(contractions.Contraction, contractions.Meaning))

#remove white space on two sides
misspell_data.misspell = misspell_data.misspell.str.strip()

#split all possible misspelled data of a word in a array
misspell_data.misspell = misspell_data.misspell.str.split(" ")

#expand the column in misspell from array to many rows
misspell_data = misspell_data.explode("misspell").reset_index(drop=True)

#remove duplicates
misspell_data.drop_duplicates("misspell",inplace=True)

#create a dic object only for print purpose
miss_corr = dict(zip(misspell_data.misspell, misspell_data.correction))
print({v:miss_corr[v] for v in [list(miss_corr.keys())[k] for k in range(20)]})

'''
this function replace contraction to by actual meaning
'''
def cont_to_meaning(val):
    for x in val.split():
        if x in cont_dic.keys():
            val = val.replace(x, cont_dic[x])
    return val

'''
this function replace misspelled words to the correct one
'''
def misspelled_correction(val):
    for x in val.split():
        if x in miss_corr.keys():
            val = val.replace(x, miss_corr[x])
    return val

'''
this function remove all punctuation
'''
def punctuation(val):
    punctuations = '''()-[]{};:'"\,<>./@#$%^&_~'''

    for x in val.lower():
        if x in punctuations:
            val = val.replace(x, " ")
    return val

'''
This function is the combination of three functions above
'''
def clean_text(val):
    val = misspelled_correction(val)
    val = cont_to_meaning(val)
    #val = p.clean(val)
    val = ' '.join(punctuation(emoji.demojize(val)).split())
    return val


data["clean_content"] = data.content.apply(lambda x : clean_text(x))

print("clean data complete")
'''
prepare to map emotions to integer 
'''
#map each emotion to a integer
sent_to_id  = {"empty":0, "sadness":1,"enthusiasm":2,"neutral":3,"worry":4,"surprise":5,"love":6,"fun":7,"hate":8,"happiness":9,"boredom":10,"relief":11,"anger":12}
data["sentiment_id"] = data['sentiment'].map(sent_to_id)

#convert the Y to matrix from categorical variable
Y = np.ndarray((len(data),len(sent_to_id)),dtype=float)
#Y = [np.zeros(len(sent_to_id), dtype = int) for _ in  range(len(data))]
for i in range(len(data)):
    Y[i][data.sentiment_id[i]] = 1




X_train, X_test, y_train, y_test = train_test_split(data.clean_content,Y, random_state= 2021, test_size=0.2, shuffle=True)

print("train and test data split complete")

sentences = X_train.tolist()
X_train = [None for _ in range(len(sentences))]
uniquewords = collections.Counter()

#Record all unique non-trivial words in training data
for s in sentences:
    tokens = nltk.word_tokenize(s)
    for w in tokens:
        uniquewords[w] += 1
word_bag = list(uniquewords.keys())

#tokenize each sentence in training data
for j,s in enumerate(sentences):
    tokens = nltk.word_tokenize(s)
    bag = np.zeros(len(word_bag), dtype=np.float32)
    for i in range(len(word_bag)):
        if word_bag[i] in tokens:
            bag[i] = 1
    X_train[j] = bag
X_train = np.array(X_train)

print("Tokenize complete")

class Dataset(Dataset):

    def __init__(self):
        self.n_samples = len(X_train)
        self.x_data = X_train
        self.y_data = y_train

    # support indexing such that dataset[i] can be used to get i-th sample
    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    # we can call len(dataset) to return the size
    def __len__(self):
        return self.n_samples

'''
CUDA = torch.cuda.is_available()
if CUDA:
    net = net.cuda()
'''
criterion = nn.CrossEntropyLoss()
learning_rate = 0.0001
epochs = 200
batch_size = 200
num_epochs = 200
hidden_size = 50


correct_train = total_train = 0

dataset = Dataset()
train_loader = DataLoader(dataset=dataset,
                          batch_size=batch_size,
                          shuffle=True,
                          num_workers=4)

input_size = len(X_train[0])
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
net = Model(input_size, hidden_size, len(sent_to_id)).to(device)
optimizer = torch.optim.Adam(net.parameters(), lr = learning_rate)
print("using device", device)
print("start to train")
for epoch in range(num_epochs):
    for i, (words, labels) in enumerate(train_loader):
        words = words.to(device)
        labels = labels.to(dtype=torch.long).to(device)
        outputs = net(words)
        loss = criterion(outputs, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')