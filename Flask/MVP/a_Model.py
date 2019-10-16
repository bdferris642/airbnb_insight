
"""
This module contains functions that help rank airbnb comment topic vectors by similarity to user input.
That is, when views.py calls `Calculate_similarities`
This function in turn calls helper functions in this module.

The inputs are a string (representing the yourbnb.xyz user's preferences),
a pandas series of the mean topic vectors for each listing,
a dictionary of strings to numbers (that was used to encode the vocabulary in numerical form in the training of the LDA model)
and the LDA model itself. 

More details in the functions themselves
"""

import gensim
from gensim.utils import simple_preprocess
from gensim.parsing.preprocessing import STOPWORDS
from gensim import corpora, models
import nltk
nltk.download('wordnet')
import pickle
import sklearn as sk
import numpy as np

# Path to pickle file for stemmer and lemmatizer. 
# These are used to pre-process the textual input
# Alter path needed!
with open('/Users/bennett/Documents/GitHub/airbnb_insight_repo/airbnb_insight/Flask/MVP/nltk_dict.pkl', 'rb') as fp:
    nltk_dict = pickle.load(fp)

SnowballStemmer = nltk_dict['stemmer']
stemmer = SnowballStemmer("english")
WordNetLemmatizer = nltk_dict['lemmatizer']

def _Elementwise_cosine_similarity(arr, ctv):
	"""
	given an array `arr` of shape  (n_comments x n_topics) and a vector `ctv` of shape (n_topics,)
	returns an array of cosine similarities between ctv and each row of arr

	"""
    return sk.metrics.pairwise.cosine_similarity(np.array(ctv).reshape(1, -1), arr)

def Calculate_similarities(fromUser='Default', listings = None, dictionary=None, model = None, langmod = None, elementwise = False):
	"""
	Inputs: 
	fromUser: a string (from the input page, representing the yourbnb.xyz user's preferences),
	listings: if elementwise=False, a pandas series (n_listings x n_topics) representing the average topic vectors for each listing
			if elementwise = True,  the pandas series is a list-of-list-of-list (n_listings x comments_per_listing x n_topic)
	dictionary: a list-of-str--> list-of-tuple (id, occurrences) dictionary to convert strings to a bag-of-words format
	model: a pre-trained gensim LDA model
	elementwise: boolean. If true, apply a cosine similarity function row-wise to sublists in `listings`

	Outputs: 
	sims: cosine similarities between the user's topic vector and the comment vectors in `listings`
		if elementwise = False, a len(n_listings) list-of-float.
		if elementwise = Truem a list-of-list-of-float (n_listings x comments_per_listing)
	comment_topic_vector: an array of shape(n_topics,). The topic vector encoding the fromUser string
	"""

	# preprocess input text, get the topic vector
	comment_stem_lemma = Preprocess_text(fromUser)
	comment_bow = dictionary.doc2bow(comment_stem_lemma)
	comment_topic_vector = [tup[1] for tup in model[comment_bow]]

	# calculate the similarities between user topic vector and listing topic vectors
	# for each review
	if elementwise:
		sims = listings.apply(lambda x: _Elementwise_cosine_similarity(x, comment_topic_vector))
	# for each topic vector averaged within listings
	else:
		sims = sk.metrics.pairwise.cosine_similarity(np.array(comment_topic_vector).reshape(1, -1), np.vstack(listings))[0]

	if fromUser != 'Default':
		return sims, comment_topic_vector
	else:
		return 'check your input' 

def Preprocess_text(text):
	result = []
	for token in gensim.utils.simple_preprocess(text):
		if token not in gensim.parsing.preprocessing.STOPWORDS and len(token) > 3:
			result.append(lemmatize_stemming(token))
	return result

def lemmatize_stemming(text):
	return stemmer.stem(WordNetLemmatizer().lemmatize(text, pos='v'))


def _Model_sentence(sent, langmod):
	for c in ['.', ',', '!', '?', ';', '-', '&', '(', ')', '$']:
		sent = sent.replace(c, '')
	vs = []
	for w in sent.split(' '):
		try:
			vs.append(langmod[x])
		except:
			continue
	return np.mean(vs, axis=0)

	