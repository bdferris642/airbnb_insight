import gensim
from gensim.utils import simple_preprocess
from gensim.parsing.preprocessing import STOPWORDS
from nltk.stem import WordNetLemmatizer, SnowballStemmer
from gensim import corpora, models
from nltk.stem import WordNetLemmatizer, SnowballStemmer
from nltk.stem.porter import *

# load model, load n_topics,
# load db
# load listing_topic_vectors, listing_topic_stems
# load dictionary


def match_comment_to_listing_topic(comment='Default',
                                    db= this_db, dictionary=this_dictionary)
    
    comment_corpus = dictionary.doc2bow(preprocess(comment))
    comment_score = get_topic_score(comment=comment_corpus,
        model=lda_model, n_topics=this_model_n_topics, norm=True)


    similarities = np.zeros((this_db.shape[0]))
    for irow, row in enumerate(this_db):
        similarities[irow] = cosine_similarity(comment_score, row.topic_score)


def get_topic_score(comment,
    model, n_topics, norm=True):
    """
    inputs: a bag-of-words representation of a comment (from a corpus)
    a gensim lda model
    a number of topics
    
    outputs: a normalized vector of size n_topics with the scores of each topic
    """
    model_output = np.array(model[comment])
    inds = model_output[:, 0].astype(int)
    scores = model_output[:, 1]
    out = np.zeros((n_topics))
    out[inds] = scores
    if norm:
        out = out / np.linalg.norm(out)
    return out

def cosine_similarity(a, b):
    cos_sim = np.dot(a, b)/(np.linalg.norm(a)*np.linalg.norm(b))
    return cos_sim
