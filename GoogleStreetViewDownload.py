# To change this license header, choose License Headers in Project Properties.
# To change this template file, choose Tools | Templates
# and open the template in the editor.

__author__ = "mamiruzz"
__date__ = "$Feb 20, 2018 9:00:51 PM$"


#!/usr/bin/env python
#
# Save Google Street View images
#
# Based on: https://andrewpwheeler.wordpress.com/2015/12/28/using-python-to-grab-google-street-view-imagery/
#



######################################
import json
import urllib.parse
from urllib.request import urlopen

def decode_address_to_coordinates(address):
        params = {
                'address' : address,
                'sensor' : 'false',
        }  
        url = 'http://maps.google.com/maps/api/geocode/json?' + urllib.parse.urlencode(params)
        response = urlopen(url)
        result = json.load(response)
        try:
                return result['results'][0]['geometry']['location']
        except:
                return None


import urllib
import simplejson

googleGeocodeUrl = 'http://maps.googleapis.com/maps/api/geocode/json?'

def get_coordinates(query, from_sensor=False):
    #query = query.encode('utf-8')
    params = {
        'address': query,
        'sensor': "true" if from_sensor else "false"
    }
    #url = googleGeocodeUrl + urllib.urlencode(params)
    url = googleGeocodeUrl + urllib.parse.urlencode(params)
    #json_response = urllib.urlopen(url)
    json_response = urlopen(url)
    response = simplejson.loads(json_response.read())
    if response['results']:
        location = response['results'][0]['geometry']['location']
        latitude, longitude = location['lat'], location['lng']
        print(query, latitude, longitude)
    else:
        latitude, longitude = None, None
        print(query, "<no results>")
    str_array ="[{}, {}]".format(latitude, longitude)
    return str_array

######################################



import urllib, os, datetime
import secrets
import urllib.request

#myloc = os.path.join(os.getcwd(), datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
myloc = "downloads" 
key = "&key=" + "AIzaSyDtg4UBLsMiVlnvFXy7HueEPLWypTQj2h4" #secrets.api_key

def GetStreet(Add,SaveLoc):
  base = "https://maps.googleapis.com/maps/api/streetview?size=1200x800&location="
  MyUrl = base + Add + key
  fi = Add + ".jpg"
  try:
    os.makedirs(SaveLoc)
  except OSError as e:
    if e.errno != 17:
      raise # not EEXISTS
  #urllib.urlretrieve(MyUrl, os.path.join(SaveLoc,fi))
  urllib.request.urlretrieve(MyUrl, os.path.join(SaveLoc,fi))
  if fi not in Sizes:
    return
  for size in Sizes[fi]:
    if os.path.getsize(os.path.join(SaveLoc, fi)) == size:
      os.remove(os.path.join(SaveLoc, fi))
      return

#Tests = ["34.0543543,-118.2586675",
#	 "34.0542833,-118.2587341"]

#Tests = ["120 Luther Ave, Kent, Ohio 44240"]
##print(decode_address_to_coordinates(Tests))
#
##print(get_coordinates(Tests))
#
#str_ = get_coordinates(Tests)

Tests = ["34.0543543,-118.2586675",
	 "34.0542833,-118.2587341"]

Sizes = dict()

# get sizes of recent images
for subdir, dirs, files in os.walk(os.getcwd()):
  for file in files:
    if not file.endswith(".jpg"):
      continue
    if file not in Sizes:
      Sizes[file] = list()
    Sizes[file].append(os.path.getsize(os.path.join(subdir, file)))


for i in Tests:
  GetStreet(Add=i,SaveLoc=myloc)

# remove empty directories
if len(os.listdir(myloc)) == 0:
  os.rmdir(myloc)




if __name__ == "__main__":
    print("Download complete")
