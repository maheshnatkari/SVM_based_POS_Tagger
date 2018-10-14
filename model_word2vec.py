import os
import re
import sys

import codecs
import ssfAPI_minimal as sp

import pickle
import nltk

import numpy as np
from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import LabelEncoder
from sklearn.multiclass import OneVsRestClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_validate
from sklearn.externals import joblib
from sklearn.metrics import f1_score

from sklearn.preprocessing import LabelBinarizer
import timeit

from gensim.models import Word2Vec

def load_pkl(loc):
	with open(loc,'rb') as inp:
		return joblib.load(inp)

def hash_word(word):
	return int(int(hashlib.sha256(word.encode('utf-8')).hexdigest(), 16) % 1000*5)

word_features = []

binarize = LabelBinarizer()
tagSet = {'CC': 0, 'JJ': 1, 'NN': 2, 'NNP': 3, 'PRP': 4, 'RB': 5, 'RP': 6, 'SYM': 7, 'NST': 8,'VM': 9, 'DEM': 10, 'UT': 11, 'PSP': 12, 'QF': 13, 'VAUX': 14, 'WQ': 15, 'QC': 16, 'INTF': 17, 'INJ': 18, 'RDP': 19, 'QO': 20, 'CL': 21, 'NNPP': 22, 'ECH': 23, 'INTJ': 24, 'ECH': 25, 'NP': 26, 'CO': 27, 'UNK': 28}

intraindir = "train_set"

traindirlist = os.listdir(intraindir)

trainsentlist = []   # List of test set sentences. Each sentence is stored as a list of list Wlist. Wlist stores word as first value, Current POSTag as second value and context words as "CW1&CW2" as third value.

totaltrainwords = 0  # Stores total number of valid words of test set.

for fname in traindirlist:
    tsentfname = intraindir + "/" + fname
    tsecsent = codecs.open(tsentfname, "r", encoding='utf8')
    testrees  = sp.processFile(tsecsent, treeType='madeDict', nesting=True, ignoreErrors=True)
    for treeIter in testrees.iterkeys():
	tree = testrees[treeIter]
	tokenList = sp.returnTokenList(tree.nodeList)
	size = len(tokenList)
	sentwrdlist = []	
        for j in range(0, size):	# will iterate from firstitem (index == 0) to lastitem (index == size-1)
	    Wlist = []
	    if (j == 0): curword = "FIRST"
	    else:
		curword = tokenList[j-1].lex.strip()
		curtag = tokenList[j-1].tokenType.strip()
	    midword = tokenList[j].lex.strip()
	    midtag = tokenList[j].tokenType.strip()
	    if (j == (size -1)): nextword = "LAST"
	    else:
		nextword = tokenList[j+1].lex.strip()
		nextag = tokenList[j+1].tokenType.strip()
	    contextwrds = curword + "&" + nextword
	    Wlist.append(midword)
	    Wlist.append(midtag)
	    Wlist.append(contextwrds)
	    sentwrdlist.append(Wlist)
	    totaltrainwords +=1
	trainsentlist.append(sentwrdlist)

sents = []
penTags = {}
for sent in trainsentlist:
	sent1 = [[x.encode('utf-8'),y.encode('utf-8')] for x,y,z in sent]
	sents.append([x[0] for x in sent1])
	penTag = [[x.encode('utf-8'),y.encode('utf-8')] for x,y,z in sent]
	penTags.update(dict(penTag))

model = Word2Vec(sents, 5, 3, min_count = 1)

# print(sents[:10])
# words = list(model.wv.vocab)

# print(len(words))
all_words = []
for sentence in sents:
	all_words += sentence
all_words = set(all_words)

X = []
y = []

for word in all_words:
	X.append(model[word])
	y.append(tagSet[penTags[word]])

intestdir = "test_set"

testdirlist = os.listdir(intestdir)

testsentlist = []   # List of test set sentences. Each sentence is stored as a list of list Wlist. Wlist stores word as first value, Current POSTag as second value and context words as "CW1&CW2" as third value.

totaltestwords = 0  # Stores total number of valid words of test set.

for fname in testdirlist:
    tsentfname = intestdir + "/" + fname
    tsecsent = codecs.open(tsentfname, "r", encoding='utf8')
    testrees  = sp.processFile(tsecsent, treeType='madeDict', nesting=True, ignoreErrors=True)
    for treeIter in testrees.iterkeys():
	tree = testrees[treeIter]
	tokenList = sp.returnTokenList(tree.nodeList)
	size = len(tokenList)
	sentwrdlist = []	
        for j in range(0, size):	# will iterate from firstitem (index == 0) to lastitem (index == size-1)
	    Wlist = []
	    if (j == 0): curword = "FIRST"
	    else:
		curword = tokenList[j-1].lex.strip()
		curtag = tokenList[j-1].tokenType.strip()
	    midword = tokenList[j].lex.strip()
	    midtag = tokenList[j].tokenType.strip()
	    if (j == (size -1)): nextword = "LAST"
	    else:
		nextword = tokenList[j+1].lex.strip()
		nextag = tokenList[j+1].tokenType.strip()
	    contextwrds = curword + "&" + nextword
	    Wlist.append(midword)
	    Wlist.append(midtag)
	    Wlist.append(contextwrds)
	    sentwrdlist.append(Wlist)
	    totaltestwords +=1
	testsentlist.append(sentwrdlist)

testsents = []
testpenTags = {}
for testsent in testsentlist:
	testsent1 = [[a.encode('utf-8'),b.encode('utf-8')] for a,b,c in testsent]
	testsents.append([a[0] for a in testsent1])
	testpenTag = [[a.encode('utf-8'),b.encode('utf-8')] for a,b,c in testsent]
	testpenTags.update(dict(testpenTag))

testmodel = Word2Vec(testsents, 5, 3, min_count = 1)

# print(sents[:10])
# words = list(testmodel.wv.vocab)
# print(words)

testall_words = []
for testsentence in testsents:
	testall_words += testsentence
testall_words = set(testall_words)


testX = []
testy = []
for testword in testall_words:
	testX.append(testmodel[testword])
	testy.append(tagSet[testpenTags[testword]])


# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

X_train = X[:1000]
y_train = y[:1000]

X_test = testX
y_test = testy

# print "X_train = ",X_train[:10]
# print "y_train = ",y_train[:10]

# print "X_test = ",X_test[:10]
# print "y_test = ",y_test[:10]

start = timeit.default_timer()

# clf = load_pkl('model1.joblib.pkl')

print('Training the model..')

clf = svm.SVC(kernel='linear', probability=True)


clf.fit(X_train,y_train)

print('Saving the model..')
joblib.dump(clf, 'model1.joblib.pkl', compress=9)

print('Testing the model..')

predicted = clf.predict(X_test)

stop = timeit.default_timer()

# print "Consufion matrix : ",confusion_matrix(y_test, predicted)
print

print "weighted f1_score of the model: ",f1_score(y_test, predicted,average = 'weighted')
# print "Accuracy of the model: ",accuracy_score(y_test, predicted)
print "Time taken for train and test: ",stop - start



# print "predicted = ",predicted