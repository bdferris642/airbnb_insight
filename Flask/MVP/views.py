from flask import render_template
from MVP import app
import pandas as pd
import psycopg2
from flask import request
from MVP.a_Model import Calculate_similarities, Preprocess_text
import pickle
import numpy as np
from gensim.models import KeyedVectors
import copy as cp

# 10 row test
#with open('/home/ubuntu/Airbnb/MVP/csrdb_test.pkl', 'rb') as fp:

# dataframe of listings with at least 10 valid comments inenglish, with at least 10 words)
# and their associated data (incl comments, topic vectors, price, beds, room type, latitude, longitude)
with open('/home/ubuntu/Airbnb/MVP/csrdb_ge10_slim.pkl', 'rb') as fp:
    room_df_orig = pickle.load(fp)

room_df_orig = room_df_orig.dropna(subset = ['room_id', 'mean_topic_score', 'topic_scores'])

# Load the gensim model from a pickle
with open('/home/ubuntu/Airbnb/MVP/lda08_st10_sb1000_dict_NEW.pkl', 'rb') as fp:
    lda_dict = pickle.load(fp)

model = lda_dict['model']
topics = lda_dict['topics']
corpus = lda_dict['corpus']
dictionary = lda_dict['dictionary']

# Links to beauuuutiful NYC images. Alas, deprecated, b/c it was hard to read the text.
img_urls = ["http://cdn.cnn.com/cnnnext/dam/assets/171215133931-01-super-slender-skyscrapers-new-york-restricted.jpg",
          "https://hips.hearstapps.com/hmg-prod.s3.amazonaws.com/images/aerial-view-of-lower-manhattan-nyc-high-res-stock-photography-1567771485.jpg",
          "https://wallpaperbro.com/img/534226.jpg",
          "https://vastphotos.com/files/uploads/photos/10120/high-resolution-brooklyn-bridge-m.jpg"]

@app.route('/')
@app.route('/index')
@app.route('/input')
def airbnb_comments_input():
   return render_template("input.html", img_url = img_urls[np.random.randint(low=0, high=len(img_urls))])

@app.route('/output')
def listings_output():

  # Copy the original df so that it does not shrink with repeated queries
  global room_df_orig
  room_df = cp.deepcopy(room_df_orig)
  
  # get user input
  user_input = request.args.get('user_input')
  user_input_min_price = request.args.get('user_input_min_price')
  user_input_max_price = request.args.get('user_input_max_price')
  user_input_n_beds = request.args.get('user_input_n_beds')
  user_input_min_rating = request.args.get('user_input_min_rating')
  numerical_user_inputs = [user_input_min_price, user_input_max_price, user_input_n_beds, user_input_min_rating]
  
  # convert user input strings to floats
  for i, ui in enumerate(numerical_user_inputs):
    if(ui):
      try: 
        numerical_user_inputs[i] = float(ui)
      except:
        render_template("input.html", error_message = 'Enter numerals in numerical categories')
  user_input_min_price = numerical_user_inputs[0]
  user_input_max_price = numerical_user_inputs[1]
  user_input_n_beds = numerical_user_inputs[2]
  user_input_min_rating = numerical_user_inputs[3]
  numerical_user_inputs = [user_input_min_price, user_input_max_price, user_input_n_beds, user_input_min_rating]

  # Check which filters the user input
  # so that we may drop nans from only those columns that user wants to filter on
  user_input_entire_checked = request.args.get("user_input_room_type_entire") != None
  user_input_private_checked = request.args.get("user_input_room_type_private") != None
  user_input_shared_checked = request.args.get("user_input_room_type_shared") != None
  room_type_checks = np.array([user_input_entire_checked, user_input_private_checked, user_input_shared_checked])
  room_types = np.array(['Entire home/apt', 'Private room', 'Shared room']) 

  if np.any(room_type_checks):
    room_df = room_df.dropna(subset=['room_type'])
    room_types_allowed = room_types[room_type_checks]
    room_df = room_df[room_df.room_type.isin(room_types_allowed)]

  if user_input_min_price or user_input_max_price:
    room_df = room_df.dropna(subset=['room_price'])   

  # Apply hard filters
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


  # Calculate a similarity vector 
  # between the topics extracted from the user input text
  # and all of the listings' average topic vectors
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

  # Sort by them. Render Template with the Top 3
  sorted_room_df = room_df.sort_values(by = 'similarity', ascending = False) 
  trimmed_sorted_room_df = sorted_room_df.iloc[:3]
  trimmed_sorted_room_df['most_similar_screened_comment'] = ['' for i in range(np.shape(trimmed_sorted_room_df)[0])]

  for i in range(np.shape(trimmed_sorted_room_df)[0]):
    ind_most_similar = np.where(trimmed_sorted_room_df.elementwise_similarity.iloc[i] == np.max(trimmed_sorted_room_df.elementwise_similarity.iloc[i]))[0][0]
    trimmed_sorted_room_df['most_similar_screened_comment'].iloc[i] = trimmed_sorted_room_df.comments_screened.iloc[i][ind_most_similar]

  the_result={}
  for i in range(np.shape(trimmed_sorted_room_df)[0]):
    the_result[i] = trimmed_sorted_room_df.iloc[i]

  # construct the link for the google static google map
  # with markers at the latitude and logitude of the top 3 listings.
  maplink = "https://maps.googleapis.com/maps/api/staticmap?center=Grand+Central+Station,New+York,NY&zoom=11&size=500x600"
  maplink+= "&maptype=roadmap&markers=color:red%7Clabel:1%7C"
  maplink+= str(trimmed_sorted_room_df.iloc[0].latitude)
  maplink+= ","
  maplink+= str(trimmed_sorted_room_df.iloc[0].longitude)
  maplink+= "&markers=color:red%7Clabel:2%7C"
  maplink+= str(trimmed_sorted_room_df.iloc[1].latitude)
  maplink+= ","
  maplink+= str(trimmed_sorted_room_df.iloc[1].longitude)
  maplink+= "&markers=color:red%7Clabel:3%7C"
  maplink+= str(trimmed_sorted_room_df.iloc[2].latitude)
  maplink+= ","
  maplink+= str(trimmed_sorted_room_df.iloc[2].longitude)
  maplink+= "&key=AIzaSyDTQObkJZDeUeyHCjmC_iMIlgbEWjiuD-A"

  return render_template("output.html",
                        maplink = maplink,
                        input_text = user_input,
                        comment_topic_vector= [round(i, 2) for i in comment_topic_vector],
                        the_result = the_result)
