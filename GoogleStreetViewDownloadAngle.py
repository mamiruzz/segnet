# To change this license header, choose License Headers in Project Properties.
# To change this template file, choose Tools | Templates
# and open the template in the editor.

__author__ = "mamiruzz"
__date__ = "$Feb 20, 2018 9:00:51 PM$"


import urllib, os, datetime
import secrets
import urllib.request

import os


myloc = "test1" 
key = "&key=" + "AIzaSyDtg4UBLsMiVlnvFXy7HueEPLWypTQj2h4" #secrets.api_key
heading = 0

csv_file = myloc+'.csv'

if os.path.isfile(csv_file)!=True:
    with open(myloc+'/'+csv_file, 'a') as f:
        f.write('TripId, Lat, Lng, FileName, LinkLocation\n')


def GetStreetView(Add, SaveLoc, TripId, Lat, Lng, FileNo, Ang):
    import urllib.parse
    query = urllib.parse.quote(Add)
    print(Lat)
    host = 'https://maps.googleapis.com/maps/api/streetview?size=480x360&location=%s%s&heading=%s' % (query, key, Ang)
    MyUrl = str(host)
    print(MyUrl)
    fi = Add + ".jpg"
    fi = str(TripId)+'-'+ str(FileNo) +'.jpg'
    with open(SaveLoc+'/'+csv_file, 'a') as f:
        f.write('{}, {}, {}, {}, {}\n'.format(str(TripId), Lat, Lng, fi, SaveLoc+'/'+fi))
        print('csv')    
    try:
        os.makedirs(SaveLoc)
    except OSError as e:
        if e.errno != 17:
            raise # not EEXISTS
    #urllib.urlretrieve(MyUrl, os.path.join(SaveLoc,fi))
    urllib.request.urlretrieve(MyUrl, os.path.join(SaveLoc, fi))
    
    
    
    if fi not in Sizes:
        return
    for size in Sizes[fi]:
        if os.path.getsize(os.path.join(SaveLoc, fi)) == size:
            os.remove(os.path.join(SaveLoc, fi))
            return
Sizes = dict()

#Tests = ["34.0543543,-118.2586675",
#    "34.0542833,-118.2587341",
#    "41.076711,-81.519083"]


id_value = '1042415'
lat = '41.076676'
lng = '-81.519034'
row = lat +',' + lng
      
row = ''.join( c for c in str(row) if  c not in "[']")
for x in range(0, 360):
    GetStreetView(Add=row, SaveLoc=myloc, TripId = id_value, Lat=lat, Lng = lng, FileNo=x, Ang = x)
    print(x)

#print(results)
if __name__ == "__main__":
    print("Download complete")