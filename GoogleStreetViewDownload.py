# To change this license header, choose License Headers in Project Properties.
# To change this template file, choose Tools | Templates
# and open the template in the editor.

__author__ = "mamiruzz"
__date__ = "$Feb 20, 2018 9:00:51 PM$"


import urllib, os, datetime
import secrets
import urllib.request

import os


myloc = "right" 
key = "&key=" + "AIzaSyDtg4UBLsMiVlnvFXy7HueEPLWypTQj2h4" #secrets.api_key
heading = 270

csv_file = myloc+'.csv'

if os.path.isfile(csv_file)!=True:
    with open(myloc+'/'+csv_file, 'a') as f:
        f.write('TripId, Lat, Lng, FileName, LinkLocation\n')


def GetStreetView(Add, SaveLoc, TripId, Lat, Lng, FileNo):
    import urllib.parse
    query = urllib.parse.quote(Add)
    print(Lat)
    host = 'https://maps.googleapis.com/maps/api/streetview?size=1200x800&location=%s%s&heading=%s' % (query, key, heading)
    MyUrl = str(host)
    print(MyUrl)
    fi = Add + ".jpg"
    fi = str(FileNo) +'.jpg'
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


#Tests = ["34.0543543,-118.2586675",
#    "34.0542833,-118.2587341",
#    "41.076711,-81.519083"]

Sizes = dict()



import csv

results = []



source_file = '10values.csv'
with open(source_file, newline='') as myFile:  
    reader = csv.reader(myFile)
    lat_lng = ""
    firstline = True
    i = 0
    for row in reader:
        if firstline:    #skip first line
            firstline = False
            continue
        results.append(str(row[0]))
        id_value = row[0]
        lat = str(row[2])
        lng = str(row[3])
        row = row[2] +',' + row[3]
        
        row = ''.join( c for c in str(row) if  c not in "[']")
        GetStreetView(Add=row, SaveLoc=myloc, TripId = id_value, Lat=lat, Lng = lng, FileNo=i)
        print(i)
        if row[0] in (None, ","):
             break
        i+=1

#print(results)
if __name__ == "__main__":
    print("Download complete")