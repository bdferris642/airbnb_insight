
I built Yourbnb, an NLP-powered web-app
that mines Airbnb reviews to find listings that align with user preferences.

Web address: www.yourbnb.xyz
Link to google slides presentation: https://bit.ly/2qcMbVI

Approach:
(1) Train a Latent Dirichlet Allocation (LDA) topic model on Airbnb reviews from listings in NYC. <br>
(2) For each listing, get a vector represeting the ratio with which each topic was discussed in the listings' reviews. <br>
Save the Airbnb listing data and LDA model as pickles in the directory: 'airbnb_insight/Flask/MVP/' <br>
(3) Validate the model by checking that the topics discussed within listings are more similar than those between listings. <br>
(4) Deploy a Flask web app (hosted on AWS EC2) that takes a user's input, applies the same LDA model to it, and ranks listings by their similarity to the topics discussed by the user. <br> <br>

Code for steps (1) - (3) are included in the jupyter notebook 'airbnb_insight/Topic Analysis.ipynb' <br>
Code for the Flask app is in: 'airbnb_insight/Flask/MVP/' <br>
	• Python code for the app is: 'airbnb_insight/Flask/MVP/views.py' <br>
	• Helper functions for this app located in: 'airbnb_insight/Flask/MVP/a_Model.py' <br>
	• Input and Output pages: 'airbnb_insight/Flask/MVP/templates/input.html' and 'airbnb_insight/Flask/MVP/templates/output.html' <br> <br>

Airbnb listing and review data were downloaded from http://insideairbnb.com/get-the-data.html <br>
Reviews: reviews.csv.gz under the 'New York City, New York, United States' heading <br>
Listings: listings.csv.gz under the 'New York City, New York, United States' heading <br>

This app has not yet been optimized for running on someone else's machine





