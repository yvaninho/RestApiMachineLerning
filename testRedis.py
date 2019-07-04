#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 26 15:11:11 2019

@author: jeff
"""

import redis 
r = redis.StrictRedis()
listerner = r.pubsub()
listener.sbscribe()