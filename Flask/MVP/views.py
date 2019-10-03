from flask import render_template
from MVP import app
from sqlalchemy import create_engine
from sqlalchemy_utils import database_exists, create_database
import pandas as pd
import psycopg2
from flask import request
from MVP.a_Model import Calculate_similarities, Preprocess_text
import pickle
import numpy as np
from gensim.models import KeyedVectors
import copy as cp

#nlp = KeyedVectors.load_word2vec_format('/Users/bennett/Documents/Flask/Airbnb/MVP/GoogleNews-vectors-negative300.bin', binary=True)
with open('/Users/bennett/Documents/Flask/Airbnb/MVP/csrdb.pkl', 'rb') as fp:
    room_df_orig = pickle.load(fp)

room_df_orig = room_df_orig[room_df_orig.comments_screened.apply(lambda x: len(x) >= 10)]

#room_df = pd.read_json('/Users/bennett/Documents/Flask/Airbnb/MVP/new_clean_scored_room_db.json', orient='columns')
room_df_orig = room_df_orig.dropna(subset = ['room_id', 'mean_topic_score', 'topic_scores'])

with open('/Users/bennett/Documents/Flask/Airbnb/MVP/lda08_st10_sb1000_dict_NEW.pkl', 'rb') as fp:
    lda_dict = pickle.load(fp)

model = lda_dict['model']
topics = lda_dict['topics']
corpus = lda_dict['corpus']
dictionary = lda_dict['dictionary']

img_urls = ["http://cdn.cnn.com/cnnnext/dam/assets/171215133931-01-super-slender-skyscrapers-new-york-restricted.jpg",
          "https://hips.hearstapps.com/hmg-prod.s3.amazonaws.com/images/aerial-view-of-lower-manhattan-nyc-high-res-stock-photography-1567771485.jpg",
          "https://wallpaperbro.com/img/534226.jpg",
          "https://vastphotos.com/files/uploads/photos/10120/high-resolution-brooklyn-bridge-m.jpg"]

@app.route('/')
@app.route('/index')

def index():

  # Here put a title page and a button sending users to '/input'
   return render_template("index.html",
      title = 'Home', user = { 'nickname': 'Bennett' },
      )

@app.route('/input')
def airbnb_comments_input():
   return render_template("input.html", img_url = img_urls[np.random.randint(low=0, high=len(img_urls))])

@app.route('/output')
def listings_output():
  global room_df_orig
  room_df = cp.deepcopy(room_df_orig)
  
  user_input = request.args.get('user_input')

  user_input_min_price = request.args.get('user_input_min_price')
  user_input_max_price = request.args.get('user_input_max_price')
  user_input_n_beds = request.args.get('user_input_n_beds')
  user_input_min_rating = request.args.get('user_input_min_rating')
  numerical_user_inputs = [user_input_min_price, user_input_max_price, user_input_n_beds, user_input_min_rating]
  
  for i, ui in enumerate(numerical_user_inputs):
    if(ui):
      try: 
        numerical_user_inputs[i] = float(ui)
      except:
        render_template("input.html", error_message = 'Enter numerals in numerical categories')

  # OJO do them individually
  user_input_min_price = numerical_user_inputs[0]
  user_input_max_price = numerical_user_inputs[1]
  user_input_n_beds = numerical_user_inputs[2]
  user_input_min_rating = numerical_user_inputs[3]
  numerical_user_inputs = [user_input_min_price, user_input_max_price, user_input_n_beds, user_input_min_rating]

  user_input_entire_checked = request.args.get("user_input_room_type_entire") != None
  user_input_private_checked = request.args.get("user_input_room_type_private") != None
  user_input_shared_checked = request.args.get("user_input_room_type_shared") != None
  room_type_checks = np.array([user_input_entire_checked, user_input_private_checked, user_input_shared_checked])
  room_types = np.array(['Entire home/apt', 'Private room', 'Shared room']) 

  if np.any(room_type_checks):
    room_df = room_df.dropna(subset=['room_type'])
    room_types_allowed = room_types[room_type_checks]
    room_df = room_df[room_df.room_type.isin(room_types_allowed)]


  if (user_input_min_price > 0) or (user_input_max_price > 0):
    room_df = room_df.dropna(subset=['room_price'])
    

  if user_input_min_price:

    room_df = room_df[room_df.room_price >= user_input_min_price]
  if user_input_max_price:
    room_df = room_df[room_df.room_price <= user_input_max_price]

  if user_input_n_beds:
    room_df = room_df.dropna(subset=['beds'])
    room_df = room_df[room_df.beds == user_input_n_beds]

  if user_input_min_rating:
    room_df = room_df.dropna(subset = ['review_scores_rating'])
    room_df = room_df[room_df.review_scores_rating >= user_input_min_rating]

  room_df['similarity'], comment_topic_vector = Calculate_similarities(fromUser=user_input,
                                                  listings = room_df.mean_topic_score, 
                                                  dictionary = dictionary,
                                                  model = model,
                                                  elementwise = False)
  room_df['elementwise_similarity'], comment_topic_vector = Calculate_similarities(fromUser=user_input,
                                                   listings = room_df.topic_scores, 
                                                   dictionary = dictionary,
                                                   model = model,
                                                   elementwise = True)

  sorted_room_df = room_df.sort_values(by = 'similarity', ascending = False)

  # get the list of cosine similarities. Sort by them. Render Template with the Top 3
  trimmed_sorted_room_df = sorted_room_df.iloc[:3]
  trimmed_sorted_room_df['most_similar_screened_comment'] = ['' for i in range(np.shape(trimmed_sorted_room_df)[0])]

  for i in range(np.shape(trimmed_sorted_room_df)[0]):
    ind_most_similar = np.where(trimmed_sorted_room_df.elementwise_similarity.iloc[i] == np.max(trimmed_sorted_room_df.elementwise_similarity.iloc[i]))[0][0]
    trimmed_sorted_room_df['most_similar_screened_comment'].iloc[i] = trimmed_sorted_room_df.comments_screened.iloc[i][ind_most_similar]

  the_result={}
  for i in range(np.shape(trimmed_sorted_room_df)[0]):
    the_result[i] = trimmed_sorted_room_df.iloc[i]

  return render_template("output.html",
                        input_text = user_input,
                        comment_topic_vector= [round(i, 2) for i in comment_topic_vector],
                        the_result = the_result)


