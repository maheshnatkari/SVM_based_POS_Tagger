Id:	1			
Name of project:	Implementing SVM based POS tagger on Hindi, Telugu and Bengali ILCI datasets and finding which set of features performed the best			
Broad Category under which it will fall:	POS tagging			
Description of Project:	Its a simple implementation of already existing SVM tool available for download at the website mentioned in reference paper: J. Gimenez and L. Marquez, “SVMTool: A general POS tagger generator based on Support Vector Machines,” in Proceedings of the 4th International Conference on Language Resources and Evaluation, 2004, pp. 43–46.			
Faculty:	Prof. Dipti M. Sharma			
Mentor:	Pratibha Rani			



1. We have created two models using two different set of features.

2. 'model_bi&trigrams.py' is trained and tested using bigram, trigram, word itself and word length features. With these set of features we have got an accuracy of 61% with 10% of the telugu training data.

3. 'model_word2vec.py' is trained and tested using word vector feature of each word using word2vec model. With these set of features we have got an accuracy of 39% with 10% of the telugu training data.