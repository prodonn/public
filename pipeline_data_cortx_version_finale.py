# -*- coding: utf-8 -*-
"""
Created on Sun Dec  5 15:46:31 2021

@author: Olivier
"""

import argparse
import pandas as pd
import os 
import glob
import gc
import numpy as np # linear algebra
import boto3

from sklearn import preprocessing
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler


region='eu-central-1'

session = boto3.Session(
    region_name=region
)
s3 = session.resource('s3')

s3client = boto3.client('s3')
    
BUCKET_NAME = 'recrutementdatascience'

def get_body(file_name) :
   return s3.Bucket(BUCKET_NAME).Object(file_name).get()['Body']

def mk_tmp_dir(tmp_path) :
   try:
      os.mkdir(tmp_path)
      print(f"Directory {tmp_path} has been created successfully")
   except OSError as error:
      print(error)
      print(f"Directory {tmp_path} can not be created")

def rm_tmp_dir(tmp_path) :
   try:
      for file_path in glob.glob('{tmp_path}/*.csv'):
         os.remove(file_path)
      os.rmdir(tmp_path)
      print(f"Directory {tmp_path} has been removed successfully")
   except OSError as error:
      print(error)
      print(f"Directory {tmp_path} can not be removed")
       
def remove_local_file(tmp_path) :
   try:
      os.remove(tmp_path)
      print(f"File {tmp_path} has been removed successfully")
   except OSError as error:
      print(error)
      print(f"File {tmp_path} can not be removed") 

def upload_s3(local_file, s3_file):
   try:
      s3client.upload_file(local_file, BUCKET_NAME, s3_file)
      print(f"Upload of the {s3_file} on S3 Successful")
      return True
   except FileNotFoundError:
      print(f"The file {local_file} was not found")
      return False

def dump_s3(df, csv_name, dir_s3, dir_tmp) :
   # Load the results on s3
   vib_data = Data_Polymorphe(f'{dir_s3}/{csv_name}', f'{dir_tmp}/{csv_name}')
   vib_data.df = df
   print(f'Size of the vib data {csv_name} : {vib_data.df.shape}')
   print(f'Sample of the vib data {csv_name} : {vib_data.df.head()}')
   #Upload to s3
   return vib_data.upload_s3_file()
    
def erase_from_memory(l_df) :
   del l_df
   gc.collect()

class Data_Polymorphe :
    
   def __init__(self, s3_path, local_path):
     self.local_path = local_path
     self.s3_path = s3_path
     self.df = pd.DataFrame()

   def set_df_from_s3(self, chunk=False, index=0, chunksize=10000000) :
      try:
         print(f'Load of the data from s3 :{self.s3_path}')
         #Load and order the milling_modes file
         if chunk :
            self.df = pd.read_csv(get_body(self.s3_path), chunksize=chunksize, index_col=index)
         else :
            self.df = pd.read_csv(get_body(self.s3_path), index_col=index)
      except FileNotFoundError as error :
         print(error)
         print(f"Sorry, the csv file {self.s3_path} does not exist")
      finally :
         #print('Size of the df ', self.df.head())
         return self.df

   def upload_s3_file(self):
      self.df.to_csv(self.local_path)
      result = upload_s3(self.local_path, self.s3_path)
      #Suppression du fichier temporaire
      remove_local_file(self.local_path)
      erase_from_memory([self.df])
      return result


def consol_prog_vib_CD(source_milling_modes, source_vib_D, source_vib_C, l_month, dir_tmp, dir_CD_s3, prefix_DC_prog='vib_DC_prog_') : 
   """
   Join vibration C, D and prog/modes then partition data by month.

   param source_milling_modes: The path of your prog/modes data CSV on the s3 bucket, such as 'milling_modes.csv'.
   param source_vib_D: The path of your vib D data CSV on the s3 bucket.
   param source_vib_C: The path of your vib C data CSV on the s3 bucket.
   param l_month: The list of month values, such as ['2019-09', '2019-10'].
   param dir_tmp: The name of your local temporary directory, such as 'tmp'.
   param dir_CD_s3: The directory on the s3 bucket where output is written, such as 'vib_CD'.
   prefix_DC_prog: The prefix of the output file.
   """
   #Prepare tmp directory
   mk_tmp_dir(dir_tmp)
   
   print("*"*20)
   print('Load of the prog/mode, vib_D and vib_C files.')
   
   #Load the sources and order the dataframes
   #Create prog_mode data object
   prog_data = Data_Polymorphe(source_milling_modes, None) 
   prog_df = prog_data.set_df_from_s3().sort_index()

   #Load and order the vib D file
   vib_D_data = Data_Polymorphe(source_vib_D, None) 
   vib_D_df = vib_D_data.set_df_from_s3().sort_index()

   #Load and order the vib C file
   vib_C_data = Data_Polymorphe(source_vib_C, None) 
   vib_C_df = vib_C_data.set_df_from_s3().sort_index()

   if (not vib_D_df.empty and not prog_df.empty) and not vib_C_df.empty :
      print('Join of the data')
      #Join the vib_D and prog_mode data by date (s).
      vib_D_prog_df =  vib_D_df.join(prog_df, how = 'outer')
      erase_from_memory([vib_D_df, prog_df])
      # Fill the null programmes and modes with the last non null
      vib_D_prog_df[['programme', 'mode' ]] = vib_D_prog_df[['programme', 'mode' ]].fillna(method ='ffill')

      #Join the vib_C, vib D and prog_mode data by date.
      vib_DC_prog_df = vib_D_prog_df.join(vib_C_df, how = 'left')
      erase_from_memory([vib_D_prog_df, vib_C_df])
      # Fill the null values3 with the last non null to use it as a mean/min
      vib_DC_prog_df['values3'] = vib_DC_prog_df['values3'].fillna( method ='ffill')

      #On éclate le résultat en batch mensuels qui seront utilisés pour les jointures avec vib_AB
      print('create tmp files partitionned by month')
      for month in l_month :
         vib_month_df = vib_DC_prog_df.filter(like =  month, axis=0)
         #Upload of the object feature data to s3
         dump_s3(vib_month_df, f'{prefix_DC_prog}{month}.csv', dir_CD_s3, dir_tmp)
         
      # Free memory
      erase_from_memory([vib_month_df,vib_DC_prog_df])
      rm_tmp_dir(dir_tmp)
      return True
   else :
      return False
   


def create_month_batches_vib_AB(source_vib_AB, dir_AB_s3, l_month, dir_tmp, prefix_AB='vib_AB_') :
   """
   partition data vib AB by month

   param source_vib_AB:The path of your vib AB data  on the s3 bucket.
   param l_month: The list of month values, such as ['2019-09', '2019-10'].
   param dir_tmp: The name of your local temporary directory, such as 'tmp'.
   param dir_AB_s3: The directory on the s3 bucket where output is written, such as 'vib_CD'.
   prefix_AB_prog: The prefix of the output file.
   """
   #Prepare tmp directory
   mk_tmp_dir(dir_tmp)
   print("*"*20)
   print('Load of the vib_AB file, and explode it in a bunch of chunks.')
   #Load and order the vib AB file
   vib_AB_data = Data_Polymorphe(source_vib_AB, None) 
   chunks_AB = vib_AB_data.set_df_from_s3(chunk=True)
   
   nb_dumps = 0
   vib_AB_df = pd.DataFrame()
   for i, chunk in enumerate(chunks_AB):
      print(f'create tmp files partitionned by month : chunk {i}')
      n_month = 0
      #On éclate le résultat en batch mensuels qui seront utilisés pour les jointures avec vib_AB
      for m, month in enumerate(l_month[nb_dumps:]) :

         tmp_df = chunk.filter(like = month, axis=0) 
         
         if tmp_df.shape[0] != 0 :
            
            if n_month > 0 :
               print(f"dump_s3 for month {l_month[nb_dumps]}")
               #Upload of the object feature data to s3
               dump_s3(vib_AB_df, f'{prefix_AB}{l_month[nb_dumps]}.csv', dir_AB_s3, dir_tmp)
               #Reset dataframe
               vib_AB_df = pd.DataFrame()
               nb_dumps += 1
            n_month += 1
            vib_AB_df=vib_AB_df.append(tmp_df)  
         else :
            if n_month > 0 :
               break
   # Ecriture du dernier chunk
   print(f"dump_s3 for month {l_month[-1]}")

   #Création de l'objet qui contiendra les vib_AB data
   dump_s3(vib_AB_df, f'{prefix_AB}{l_month[-1]}.csv', dir_AB_s3, dir_tmp)
   # Free memory
   erase_from_memory([vib_AB_df])
   rm_tmp_dir(dir_tmp)         




def consolidate_vib(dir_tmp, dir_CD_s3, dir_AB_s3, dir_cons_s3, l_month, prefix_AB='vib_AB_', prefix_CD_prog='vib_DC_prog_', prefix_consolidate='vib_ABCD_') :
   """
   Join vibration CD and AB partitionned  by month.

   param l_month: The list of month values, such as ['2019-09', '2019-10'].
   param dir_tmp: The name of your local temporary directory, such as 'tmp'. 
   param dir_cons_s3: The directory on the s3 bucket where output is written.
   param dir_CD_s3: The directory on the s3 bucket where vib_CD is read.
   param dir_AB_s3: The directory on the s3 bucket where output is read.
   prefix_AB: The prefix of the vib_AB file.
   prefix_CD_prog: The prefix of the vib_CD file.
   prefix_consolidate: The prefix of the output file.
   """
   #Prepare tmp directory
   mk_tmp_dir(dir_tmp)
   #Lets join the monthly partitioned data between vib_CD_prog and vib_AB
   print('*'*20)
   print('load and join the vib files partitionned by month')
   for month in l_month :
      #Load and order the CD_prog vib file   
      vib_CD_prog_data = Data_Polymorphe(f'{dir_CD_s3}/{prefix_CD_prog}{month}.csv', None) 
      vib_CD_prog_df = vib_CD_prog_data.set_df_from_s3(index=False).\
       rename(columns = {'Unnamed: 0':'date_DC'}).sort_values(by=['date_DC'])
      print(f'load and join the resulting vib_DC_prog file for month {month} on s3 ')
      print(vib_CD_prog_df.dropna().head())
      #Load and order the vib_AB file
      vib_AB_data = Data_Polymorphe(f'{dir_AB_s3}/{prefix_AB}{month}.csv', None) 
      vib_AB_df = vib_AB_data.set_df_from_s3(index=False).\
       rename(columns = {'Unnamed: 0':'date_AB'}).sort_values(by=['date_AB'])
      print(f'load and join the resulting vib_AB file for month {month} on s3 ')
      print(vib_AB_df.dropna().head())

      if not vib_CD_prog_df.empty and not vib_AB_df.empty :

         csv_name = f'{prefix_consolidate}{month}.csv' 
         vib_AB_df['date_AB_s'] = vib_AB_df['date_AB'].str.slice(stop=19)
         vib_ABCD_df = vib_AB_df.merge(vib_CD_prog_df, left_on='date_AB_s', right_on='date_DC')
         #Création de l'objet qui contiendra les vib_AB data
         dump_s3(vib_ABCD_df, f'{prefix_consolidate}{month}.csv', dir_cons_s3, dir_tmp)
         erase_from_memory([vib_ABCD_df])
      # Free memory
      erase_from_memory([vib_AB_df, vib_CD_prog_df])
   rm_tmp_dir(dir_tmp)  

def clean_data(source_consolidate, dir_cons_s3, l_month, dir_clean_s3, dir_tmp, prefix_clean='vib_clean_') : 
  
   """
   Join vibration CD and AB partitionned  by month.

   param source_consolidate:The path of your input data  on the s3 bucket.
   param l_month: The list of month values, such as ['2019-09', '2019-10'].
   param dir_tmp: The name of your local temporary directory, such as 'tmp'. 
   param dir_clean_s3: The directory on the s3 bucket where output is written.
   param dir_cons_s3: The directory on the s3 bucket where vib_CD is read.
   prefix_clean: The prefix of the output file.
   """
   #Prepare tmp directory
   mk_tmp_dir(dir_tmp)
   for month in l_month :
      print("*"*20)
      print('Load of the consolidate data.')
      #Create input data object
      conso_data = Data_Polymorphe(f'{dir_cons_s3}/{source_consolidate}{month}.csv', None) 
      conso_df = conso_data.set_df_from_s3()
      if not conso_df.empty :
         print(f'Size of the consolidated data : {conso_df.shape}')
         print('Drop the null values')
         clean_df = conso_df.dropna()
         #Upload of the object clean data to s3
         dump_s3(clean_df, f'{prefix_clean}{month}.csv', dir_clean_s3, dir_tmp)
         erase_from_memory([clean_df])
   # Free memory
   rm_tmp_dir(dir_tmp)


def from_data_to_features(path_milling_modes, source_clean, dir_clean_s3, l_month, dir_feature_s3, dir_tmp, prefix_feature='vib_feature_') : 
   """
   Prepare features partitionned  by month.

   param path_milling_modes: The path of your prog/modes data CSV on the s3 bucket, such as 'milling_modes.csv'.
   param source_clean :The name of your input data  on the s3 bucket.
   param l_month: The list of month values, such as ['2019-09', '2019-10'].
   param dir_tmp: The name of your local temporary directory, such as 'tmp'. 
   param dir_clean_s3: The directory on the s3 bucket where input is written.
   param dir_feature_s3: The directory on the s3 bucket where vib_CD is read.
   prefix_feature: The prefix of the output file.
   """
    
   #Prepare tmp directory
   mk_tmp_dir(dir_tmp)
   #Train OneHotEncoding of the prog_mode field
   milling_data = Data_Polymorphe(path_milling_modes, None) 
   milling_df = milling_data.set_df_from_s3()
   ohe_prog_mode = OneHotEncoder(handle_unknown='ignore')
   ohe_prog_mode.fit(milling_df[['programme','mode']])
   #Construction de la liste des modes
   list_prog_mode = [f'prog_mode_{i}' for i in range(8)] 

   for m, month in enumerate(l_month) :
      print("*"*20)
      print(f'Load of the clean data for the month {month}.')
      #Load and order the milling_modes file
      #Create input data object
      clean_data = Data_Polymorphe(f'{dir_clean_s3}/{source_clean}{month}.csv', None) 
      clean_df = clean_data.set_df_from_s3()
      # On ne garde que les colonnes qui seront utiles pour les features
      clean_df = clean_df[['date_AB', 'values1', 'values2', 'values3', 'values4','programme','mode']]
      print(f'Size of the clean data : {clean_df.shape}')
      print(f'Columns of the clean data : {clean_df.columns}')

      if not clean_df.empty :
         if m == 0 : 
            #Training of the Standard scaling of the values 1 to 4
            scaler_values = StandardScaler()
            scaler_values.fit(clean_df[['values1', 'values2', 'values3', 'values4']])   

         #Tranform the one hot column prog_mode  
         clean_df[list_prog_mode] = ohe_prog_mode.transform(clean_df[['programme', 'mode']]).toarray()
         #transform the standard scaling
         clean_df[['norm_values1', 'norm_values2', 'norm_values3', 'norm_values4']] = \
         scaler_values.transform(clean_df[['values1', 'values2', 'values3', 'values4']])

         #build the object feature data
         feature_df = clean_df[['date_AB', 'norm_values1', 'norm_values2', 'norm_values3', 'norm_values4']\
                                 + list_prog_mode]

         #Upload of the object feature data to s3
         dump_s3(feature_df, f'{prefix_feature}{month}.csv', dir_feature_s3, dir_tmp)

      else :
         if m == 0 :
            print('The first file is missing')
            print('This stage of the pipeline is dead !')
            break
   # Free memory
   rm_tmp_dir(dir_tmp)

    
if __name__ == "__main__":
    
   #I will partition data by month
   l_month = ['2018-10', '2018-11', '2018-12', '2019-01', '2019-02', \
           '2019-03', '2019-04', '2019-05', '2019-06', '2019-07', \
           '2019-08', '2019-09', '2019-10']
    

   dir_tmp = 'tmp'

   #Stage 0 : join C, D et prog 
   source_milling_modes = 'milling_modes.csv'
   source_vib_D = 'vibration_axis_D.csv'
   source_vib_C = 'vibration_axis_C.csv'
   dir_CD_s3 = 'tmp_CD'

   #consol_prog_vib_CD(source_milling_modes, source_vib_D, source_vib_C, l_month, dir_tmp, dir_CD_s3)

   #Stage 1 : explode vib_AB
   source_vib_AB = 'vibration_axis_A_axis_B.csv'
   dir_AB_s3 = 'tmp_AB'

   #create_month_batches_vib_AB(source_vib_AB, dir_AB_s3, l_month, dir_tmp) 

   #Stage 2 : join vib_AB and vib_DC
   dir_cons_s3 = 'consolidate'

   #consolidate_vib(dir_tmp, dir_CD_s3, dir_AB_s3, dir_cons_s3, l_month) 

   #Stage 3 : join vib_AB and vib_DC
   dir_clean_s3 = 'clean'
   source_consolidate = 'vib_ABCD_'

   #clean_data(source_consolidate, dir_cons_s3, l_month, dir_clean_s3, dir_tmp)

   #Stage 4 : prepare features
   dir_feature_s3 = 'feature'
   source_clean = 'vib_clean_'

   from_data_to_features(source_milling_modes, source_clean, dir_clean_s3, l_month, dir_feature_s3, dir_tmp)

