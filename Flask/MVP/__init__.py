from flask import Flask
#from flask.ext.sqlalchemy import SQLAlchemy
app = Flask(__name__)
#app.config['SQLALCHEMY_DATABASE_URI'] = 'postgresql://localhost/[YOUR_DATABASE_NAME]'
#db = SQLAlchemy(app)
from MVP import views

