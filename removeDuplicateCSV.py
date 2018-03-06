# -*- coding: utf-8 -*-
"""
Created on Tue Mar  6 10:29:15 2018

@author: mamiruzz
"""

from more_itertools import unique_everseen
with open('1042415.csv','r') as f, open('2.csv','w') as out_file:
    out_file.writelines(unique_everseen(f))