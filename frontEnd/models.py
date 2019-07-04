#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 11 17:26:42 2019

@author: jeff
"""

from flask_sqlalchemy import SQLAlchemy
import logging as lg
from .views import app
import enum
# Create database connection object
db = SQLAlchemy(app)

class Content(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    description = db.Column(db.String(200), nullable=False)
    gender = db.Column(db.Integer(), nullable=False)

    def __init__(self, description, gender):
        self.description = description
        self.gender = gender

db.create_all()



class Gender(enum.Enum):
    female = 0
    male = 1
    other = 2



#class Content(db.Model):
#    gender = db.Column(db.Enum(Gender), nullable=False)
    
db.metadata.clear()
def init_db():
    db.session.add(Content("THIS IS SPARTAAAAAAA!!!", Gender['male']))
    db.session.add(Content("What's your favorite scary movie?", Gender['female']))