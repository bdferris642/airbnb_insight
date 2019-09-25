# def ModelIt(fromUser  = 'Default', births = []):
#  in_month = len(births)
#  print('The number born is %i' % in_month)
#  result = in_month
#  if fromUser != 'Default':
#    return result
#  else:
#  	return 'check your input'
import gensim
from gensim.utils import simple_preprocess
from gensim.parsing.preprocessing import STOPWORDS
from gensim import corpora, models
from nltk.stem import WordNetLemmatizer, SnowballStemmer
import sklearn as sk
import numpy as np

def Calculate_similarities(fromUser='Default', listings = None, dictionary=None, model = None):

	# preprocess this text, get the topic vector
	comment_stem_lemma = Preprocess_text(fromUser)
	comment_bow = dictionary.doc2bow(comment_stem_lemma)
	comment_topic_vector = [tup[1] for tup in model[comment_bow]]

	sims = sk.metrics.pairwise.cosine_similarity(np.array(comment_topic_vector).reshape(1, -1), np.vstack(listings))[0]

	if fromUser != 'Default':
		return sims
	else:
		return 'check your input' 

def Preprocess_text(text):
	result = []
	for token in gensim.utils.simple_preprocess(text):
		if token not in gensim.parsing.preprocessing.STOPWORDS and len(token) > 3:
			result.append(lemmatize_stemming(token))
	return result

stemmer = SnowballStemmer("english")
def lemmatize_stemming(text):
	return stemmer.stem(WordNetLemmatizer().lemmatize(text, pos='v'))