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
from sklearn.preprocessing import LabelEncoder
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import f1_score
import warnings
import sklearn.exceptions
warnings.filterwarnings("ignore", category=sklearn.exceptions.UndefinedMetricWarning)

import timeit

def hash_word(word):
	return int(int(hashlib.sha256(word.encode('utf-8')).hexdigest(), 16) % 1000*5)

word_features = []

tagSet = {'CC': 0, 'JJ': 1, 'NN': 2, 'NNP': 3, 'PRP': 4, 'RB': 5, 'RP': 6, 'SYM': 7, 'NST': 8,'VM': 9, 'DEM': 10, 'UT': 11, 'PSP': 12, 'QF': 13, 'VAUX': 14, 'WQ': 15, 'QC': 16, 'INTF': 17, 'INJ': 18, 'RDP': 19, 'QO': 20, 'CL': 21, 'NNPP': 22, 'ECH': 23, 'INTJ': 24, 'ECH': 25, 'NP': 26, 'CO': 27, 'UNK': 28}
# tagSet = {'CC': 0, 'JJ': 0, 'NN': 0, 'NNP': 0, 'PRP': 0, 'RB': 0, 'RP': 0, 'SYM': 0, 'NST': 0,'VM': 0, 'DEM': 0, 'UT': 1, 'PSP': 1, 'QF': 1, 'VAUX': 1, 'WQ': 1, 'QC': 1, 'INTF': 1, 'INJ': 1, 'RDP': 1, 'QO': 1, 'CL': 1, 'NNPP': 1, 'ECH': 1, 'INTJ': 1, 'ECH': 1, 'NP': 1, 'CO': 1, 'UNK': 1}

inputtraindname = "train_set"
inputvaldname = "train_set"
intestdir = "train_set"

allvalwordpostaghash ={}	# Hash of all words found from Validation set with their taglist found from validation set as value
tagbackgrndscorehash = {} # Stores all tags of validation list and their background scores calculated using all words of validation set

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

i = 0


for sent in testsentlist:
	sent = [[x.encode('utf-8'),y.encode('utf-8')] for x,y,z in sent]
	penn_tags = dict(sent)
	word_list = [x[0] for x in sent]
	word_list_numeric = LabelEncoder().fit_transform(word_list)
	word_hash = dict(zip(word_list, word_list_numeric))
	# print(word_list)
	# print(sent)
	bigrms = list(nltk.bigrams(word_list))
	bigrms_hash = [(word_hash[x],word_hash[y]) for x,y in bigrms]
	trigrm = list(nltk.trigrams(word_list))
	trigrm_hash = [(word_hash[x],word_hash[y],word_hash[z]) for x,y,z in trigrm]
	# print(trigrm_hash)
	# print(bigrms_hash)
	# print(trigrm)

	ind = 0
	for word in word_list:
		word_features.append([])
		word_feature = word_features[-1]

		word_feature.append(word_hash[word])

		b_tags = []

		if(ind>1 and ind<len(bigrms_hash)):
			b_tags.append(bigrms_hash[ind-2][0])
			b_tags.append(bigrms_hash[ind-2][1])
		else:
			b_tags.append(-1)
			b_tags.append(-1)

		if(ind>0 and ind<len(bigrms_hash)):
			b_tags.append(bigrms_hash[ind-1][0])
			b_tags.append(bigrms_hash[ind-1][1])
		else:
			b_tags.append(-1)
			b_tags.append(-1)

		if(ind<len(bigrms_hash)):
			b_tags.append(bigrms_hash[ind][0])
			b_tags.append(bigrms_hash[ind][1])
		else:
			b_tags.append(-1)
			b_tags.append(-1)

		if(ind+1 < len(bigrms_hash) and ind<len(bigrms_hash)):
			b_tags.append(bigrms_hash[ind+1][0])
			b_tags.append(bigrms_hash[ind+1][1])
		else:
			b_tags.append(-1)
			b_tags.append(-1)

		if(ind+2 < len(bigrms_hash) and ind<len(bigrms_hash)):
			b_tags.append(bigrms_hash[ind+2][0])
			b_tags.append(bigrms_hash[ind+2][1])
		else:
			b_tags.append(-1)
			b_tags.append(-1)

		for b in b_tags:
			word_feature.append(b)
		# print(b_tags)


		t_tags = []

		if(ind>1 and ind<len(trigrm_hash)-1):
			t_tags.append(trigrm_hash[ind-2][0])
			t_tags.append(trigrm_hash[ind-2][1])
			t_tags.append(trigrm_hash[ind-2][2])
		else:
			t_tags.append(-1)
			t_tags.append(-1)
			t_tags.append(-1)

		if(ind>1 and ind<len(trigrm_hash)-1):
			t_tags.append(trigrm_hash[ind-1][0])
			t_tags.append(trigrm_hash[ind-1][1])
			t_tags.append(trigrm_hash[ind-1][2])
		else:
			t_tags.append(-1)
			t_tags.append(-1)
			t_tags.append(-1)

		if(ind<len(trigrm_hash)-1):
			t_tags.append(trigrm_hash[ind][0])
			t_tags.append(trigrm_hash[ind][1])
			t_tags.append(trigrm_hash[ind][2])
		else:
			t_tags.append(-1)
			t_tags.append(-1)
			t_tags.append(-1)

		if(ind+1 < len(trigrm_hash)-1 and ind<len(trigrm_hash)-1):
			t_tags.append(trigrm_hash[ind+1][0])
			t_tags.append(trigrm_hash[ind+1][1])
			t_tags.append(trigrm_hash[ind+1][2])
		else:
			t_tags.append(-1)
			t_tags.append(-1)
			t_tags.append(-1)

		if(ind+2 < len(trigrm_hash)-1 and ind<len(trigrm_hash)-1):
			t_tags.append(trigrm_hash[ind+2][0])
			t_tags.append(trigrm_hash[ind+2][1])
			t_tags.append(trigrm_hash[ind+2][2])
		else:
			t_tags.append(-1)
			t_tags.append(-1)
			t_tags.append(-1)

		for t in t_tags:
			word_feature.append(t)
		# print(t_tags)


		word_feature.append(len(word))

		# class label
		try:
			word_feature.append(tagSet[penn_tags[word]])
		except:
			word_feature.append(-1)
	
		ind += 1

	word_features.append(word_feature)
	# i += 1
	# if i == 2:
	# 	break
# print(word_feature)
# for x in word_features:
# 	print x
# 	print ""

X_train = [x[:-1] for x in word_features]
y_train = [x[-1] for x in word_features]

X_test = X_train[50000:]
y_test = y_train[50000:]


X_train = X_train[:5000]
y_train = y_train[:5000]


start = timeit.default_timer()

# print('started')

clf = svm.SVC(kernel='linear', C = 1.0)

# clf = OneVsRestClassifier(svm.SVC(kernel='linear', C = 1.0))

clf.fit(X_train,y_train)

predicted = clf.predict(X_test)

stop = timeit.default_timer()

# print(predicted)
print
print "weighted f1_score of the model: ",f1_score(y_test, predicted,average = 'weighted')
print
# print "Accuracy of the model: ",accuracy_score(y_test, predicted)
# print "Time taken for train and test: ",stop - start

# print stop - start 