
"""
   Module for eurocontrol format 
NOTE:
- 'Time (UTC) at which the point was crossed' for time;
- 'Altitude in flight levels at which the point was crossed' for flight level;
- 'Latitude in decimal degrees' for latitude;
- 'Longitude in decimal degrees' for longitude.
- Sequence number (chronological order of crossed waypoint)
"""

import argparse
import datetime
import os
import numpy as np
import pandas as pd
import uuid
from collections import OrderedDict
import random
import csv

# Columns name of the .csv header according to the standard Eurocontrol template.
FINAL_COLUMNS_NAMES = ['UAS id', 
   'UAS Relative time',
   'x',
   'y', 
   'target angle',
   'linear velocity', 
   'angular velocity']


COLUMNS_NAMES = ['ECTRL ID',
                'Sequence Number',
                'Time Over',
                'Flight Level', # Actually is Z coordinate
                'Latitude',     # Actually is Y coordinate
                'Longitude']    # Actually is X coordinate


"""   OLD version
COLUMNS_NAMES2 = ['id',
                'time',
                'x',
                'y', 
                'z',     
               #  'th', #NA
               #  "tv", #NA
               #  "rv", #NA
               ] 

"""

FILE_DIR = "eurocontrol/"

def generate_flight_id():
    # Generate a unique ID for a flight.

    id = uuid.uuid1()
    flight_id = id.fields[0]

    return flight_id

def flights_points( data, file_name):
    # 'data' is an ordered list represented in this way --> [ [flights_IDs], [number_waypoints_sequence], 
    #    [crossingg_waypoints_times], [flights_levels], [Latitudes], [Longitutes] ].
    # This method take 'data' as input and put it into a .csv file (by creating it) according to the Eurocontrol standard template.

    
    header = True
    d = {}
    fields = len(COLUMNS_NAMES)
    for flight in data:
        for field in range(fields):
            d[COLUMNS_NAMES[field]] = flight[field]
        print(d)
        df = pd.DataFrame.from_dict(d)
        df = df[COLUMNS_NAMES]
        mode = 'w' if header==True else 'a'
        df.to_csv(FILE_DIR+file_name, encoding='utf-8', 
        mode=mode, header=header, index=False)

def data_to_csv(data,filename,header=True):

   d={}
   fields = len(COLUMNS_NAMES)
   for row in data:
        for field in range(fields):
            if COLUMNS_NAMES[field] not in d:
               d[COLUMNS_NAMES[field]] = [row[field]]
            else:
               d[COLUMNS_NAMES[field]].append( row[field])

   df = pd.DataFrame.from_dict(d)
   mode = 'w' if header==True else 'a'
   df.to_csv(filename, encoding='utf-8', mode=mode, header=header, index=False)



def extract_waypoints_from_flights_points_csv( file_name):
    # Read a .csv file (according to the Eurocontrol standard template) and return a dictionary in which each key is a flight ID and
    # the corresponding values are the Z,Y,X coordinates of the crossed waypoints for that considered flight. 

    n_names = len(COLUMNS_NAMES)

    file = pd.read_csv(file_name, header=0)
    flights_and_coords = [[] for i in range(4)] # 4 = flightID + Z + Y + X

    flights_and_coords[0] = file[COLUMNS_NAMES[0]].values
    flights_and_coords[1] = file[COLUMNS_NAMES[3]].values
    flights_and_coords[2] = file[COLUMNS_NAMES[4]].values
    flights_and_coords[3] = file[COLUMNS_NAMES[5]].values

    flights_IDs = list(OrderedDict.fromkeys(flights_and_coords[0]))
    n_flights = len(flights_IDs)
    occurrennces_per_flight = [list(flights_and_coords[0]).count(flights_IDs[i]) for i in range(n_flights)]

    previous_occurrence = 0
    current_occurrence = 0
    flights_and_coords_dict = {}
    for i in range(n_flights):
        current_occurrence += occurrennces_per_flight[i]
        flights_and_coords_dict[flights_IDs[i]] = [flights_and_coords[1][previous_occurrence:current_occurrence],
                                                   flights_and_coords[2][previous_occurrence:current_occurrence],
                                                   flights_and_coords[3][previous_occurrence:current_occurrence]]
        previous_occurrence = current_occurrence

    return flights_and_coords_dict


def create_eurocontrol_file(trajs,filename,header = True):
   if(trajs is None): raise Exception("Invalid input (None)")
   if( trajs == [] or trajs[0] is None or 
      trajs[0] == []  or trajs[0][0] is None or 
      trajs[0][0] == [] ): 
      raise Exception("Invalid input (No input)")
   dimensions = len(trajs[0][0])
   if(dimensions <2 or dimensions >3): raise Exception("Only 2D or 3D, received", dimensions)
   offset=3
   print("Found",len(trajs),"trajectories with dimensions of num.:", dimensions)
   with open(filename,"w",newline="") as fout:
      wr = csv.writer(fout, delimiter=",")
      wr.writerow(FINAL_COLUMNS_NAMES) #(header)
      for traj in trajs:
         id = generate_flight_id()
         n_points = len(traj)
         for i in range(n_points):
            # Time over is N/A right now
            row = [id,i]
            for field in range(0,len(FINAL_COLUMNS_NAMES)):
               if(field < 2):
                  continue               
               elif field == 2:
                  row.append(traj[i][0])
               elif field == 3:
                  row.append(traj[i][1])
               elif field == 4 and dimensions==3:
                  row.append(traj[i][2])

            wr.writerow(row)


def convert_df_to_eurocontrol_format(df,uasId):
   if "z_pos" in df.columns:
      df.pop("z_pos")
   df["UAS id"] = [uasId]*len(df.index)
   df[ 'UAS Relative time' ] = df.index
   df = df.rename(columns={"x_pos":"x","y_pos":"y"})
   return df

# def create_eurocontrol_file(trajs,dimensions,filename,header = True):
#    if(dimensions <2 or dimensions >3): raise Exception("Only 2D or 3D")
#    d = {}
#    offset=3
#    for traj in trajs:
#       id = generate_flight_id()
#       for i in range(len(traj)):
#          # Time over is N/A right now
#          row = [id,i,None]
#          for field in range(0,len(fields)):
#             if(field < 3):
#                value = row[field]               
#             elif(dimensions==2 and field == 3):
#                value = None
#             elif field == 3:
#                value = traj[i][2]
#             elif field == 4:
#                value = traj[i][1]
#             elif field == 5:
#                value = traj[i][0]

#             if COLUMNS_NAMES[field] not in d:
#                d[COLUMNS_NAMES2[field]] = [value]
#             else:
#                d[COLUMNS_NAMES2[field]].append( value)

#    df = pd.DataFrame.from_dict(d)
#    mode = 'w' if header==True else 'a'
#    df.to_csv(FILE_DIR + filename, encoding='utf-8', mode=mode, header=header, index=False)


if __name__ == "__main__":

   

   parser = argparse.ArgumentParser(description='Eurocontrol converter')

   parser.add_argument('-i', type=str, required=True,
        help='input folder of trajs (default: %(default)s)')

   parser.add_argument('-o', type=str,required=True, 
        help='Output folder path (default: %(default)s)')


   args = parser.parse_args()
   
   trajectories = []
   concatDf = pd.DataFrame(columns=["UAS id","UAS relative time","x","y"])
   for idx,t_filename in enumerate(os.listdir(args.i)):
      if(t_filename[-4:] == ".csv" ):
         df = pd.read_csv( os.path.join(args.i, t_filename),delimiter=",",index_col="index")
            # print(df)
         trajectories.append( df.to_numpy() )
      elif(t_filename[-4:] == ".npy"):
         trajectories = np.load(os.path.join( args.i,t_filename) )
      else:
         raise Exception("invalid file in dir")
      # trajectories = [d1,d2]
      print(t_filename,":",trajectories[0][0:2],"...")
      # create_eurocontrol_file(trajectories,os.path.join(args.i, "euro"+t[0:-4]+".csv") )
      newDf = convert_df_to_eurocontrol_format(df,int(t_filename.split("traj")[1].replace(".csv","")))
      concatDf = pd.concat([newDf,concatDf])
   
   concatDf.to_csv(os.path.join(args.o,
      "euroMerged-"+str(datetime.datetime.now().strftime('%Y-%m-%d--%H-%M'))+".csv")) 
   # data_to_csv(data,"test.csv")