import pandas as pd  #pandas for using dataframe and reading csv 
import numpy as np   #numpy for vector operations and basic maths 
import urllib        #for url stuff
import re            #for processing regular expressions
import datetime      #for datetime operations
import calendar      #for calendar for datetime operations
import scipy         #for other dependancies
from sklearn.cluster import KMeans # for doing K-means clustering
from haversine import haversine # for calculating haversine distance
import math          #for basic maths operations
import os                # for os commands
import nltk
from nltk.corpus import stopwords
import string
import xgboost as xgb
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn import ensemble, metrics, model_selection, naive_bayes

train_df = pd.read_csv('input/train.tsv', sep='\t')

def cat_split(row):
		try:
			text = row
			txt1, txt2, txt3 = text.split('/')
			return txt1, txt2, txt3
		except:
			return np.nan, np.nan, np.nan


train_df["cat_1"], train_df["cat_2"], train_df["cat_3"] = zip(*train_df.category_name.apply(lambda val: cat_split(val)))

keys = train_df.cat_1.unique().tolist()
keys = list(set(keys))
values = list(range(keys.__len__()))
cat1_dict = dict(zip(keys, values))

keys2 = train_df.cat_2.unique().tolist()
keys2 = list(set(keys2))
values2 = list(range(keys2.__len__()))
cat2_dict = dict(zip(keys2, values2))

keys3 = train_df.cat_3.unique().tolist()
keys3 = list(set(keys3))
values3 = list(range(keys3.__len__()))
cat3_dict = dict(zip(keys3, values3))

def cat_lab(row,cat1_dict = cat1_dict, cat2_dict = cat2_dict, cat3_dict = cat3_dict):
		"""function to give cat label for cat1/2/3"""
		txt1 = row['cat_1']
		txt2 = row['cat_2']
		txt3 = row['cat_3']
		return cat1_dict[txt1], cat2_dict[txt2], cat3_dict[txt3]
		
train_df["cat_1_label"], train_df["cat_2_label"], train_df["cat_3_lable"] = zip(*train_df.apply(lambda val: cat_lab(val), axis =1))

def if_catname(row):
		"""function to give if brand name is there or not"""
		if row == row:
			return 1
		else:
			return 0
		
train_df['if_cat'] = train_df.category_name.apply(lambda row : if_catname(row))

def if_brand(row):
		"""function to give if brand name is there or not"""
		if row == row:
			return 1
		else:
			return 0
		
train_df['if_brand'] = train_df.brand_name.apply(lambda row : if_brand(row))

keys = train_df.brand_name.dropna().unique()
values = list(range(keys.__len__()))
brand_dict = dict(zip(keys, values))

def brand_label(row):
		"""function to assign brand label"""
		try:
			return brand_dict[row]
		except:
			return np.nan

train_df['brand_label'] = train_df.brand_name.apply(lambda row: brand_label(row))

def if_description(row):
		"""function to say if description is present or not"""
		if row == 'No description yet':
			a = 0
		else:
			a = 1
		return a

train_df['is_description'] = train_df.item_description.apply(lambda row : if_description(row))
train_df = train_df.loc[train_df.item_description == train_df.item_description]
train_df = train_df.loc[train_df.name == train_df.name]

tfidf_vec = TfidfVectorizer(stop_words='english', ngram_range=(1,1))
full_tfidf = tfidf_vec.fit_transform(train_df['item_description'].values.tolist())
train_tfidf = tfidf_vec.transform(train_df['item_description'].values.tolist())

n_comp = 40
svd_obj = TruncatedSVD(n_components=n_comp, algorithm='arpack')
svd_obj.fit(full_tfidf)

train_svd = pd.DataFrame(svd_obj.transform(train_tfidf))
train_svd.columns = ['svd_item_'+str(i) for i in range(n_comp)]

train_df = pd.concat([train_df, train_svd], axis=1)
train_df = train_df.loc[train_df.item_description == train_df.item_description]
train_df = train_df.loc[train_df.name == train_df.name]
train_df = train_df.loc[train_df.item_description == train_df.item_description]
train_df = train_df.loc[train_df.name == train_df.name]

tfidf_vec = TfidfVectorizer(stop_words='english', ngram_range=(1,1))
full_tfidf = tfidf_vec.fit_transform(train_df['name'].values.tolist())
train_tfidf = tfidf_vec.transform(train_df['name'].values.tolist())

n_comp = 40
svd_obj = TruncatedSVD(n_components=n_comp, algorithm='arpack')
svd_obj.fit(full_tfidf)

train_svd = pd.DataFrame(svd_obj.transform(train_tfidf))
train_svd.columns = ['svd_name_'+str(i) for i in range(n_comp)]

train_df = pd.concat([train_df, train_svd], axis=1)
train_df.fillna(0, inplace=True)
train = train_df.copy()

do_not_use_for_training = ['cat_1','test_id','cat_2','cat_3','train_id','name', 'category_name', 'brand_name', 'price', 'item_description']
feature_names = [f for f in train.columns if f not in do_not_use_for_training]
y = np.log(train['price'].values + 1)


def format_test_data(dict_data):
	test_df = pd.DataFrame([dict_data], columns = dict_data.keys())
	test_df["cat_1"], test_df["cat_2"], test_df["cat_3"] = zip(*test_df.category_name.apply(lambda val: cat_split(val)))
	test_df["cat_1_label"], test_df["cat_2_label"], test_df["cat_3_lable"] = zip(*test_df.apply(lambda val: cat_lab(val), axis =1))
	test_df['if_cat'] = test_df.category_name.apply(lambda row : if_catname(row))
	test_df['if_brand'] = test_df.brand_name.apply(lambda row : if_brand(row))
	test_df['brand_label'] = test_df.brand_name.apply(lambda row: brand_label(row))
	test_df['is_description'] = test_df.item_description.apply(lambda row : if_description(row))
	test_df = test_df.loc[test_df.item_description == test_df.item_description]	
	test_df = test_df.loc[test_df.name == test_df.name]
	test_tfidf = tfidf_vec.transform(test_df['item_description'].values.tolist())

	
	test_svd = pd.DataFrame(svd_obj.transform(test_tfidf))
	test_svd.columns = ['svd_item_'+str(i) for i in range(n_comp)]	
	test_df = pd.concat([test_df, test_svd], axis=1)

	test_df = test_df.loc[test_df.item_description == test_df.item_description]
	test_df = test_df.loc[test_df.name == test_df.name]

	
	test_df = test_df.loc[test_df.item_description == test_df.item_description]
	
	test_df = test_df.loc[test_df.name == test_df.name]

	
	test_tfidf = tfidf_vec.transform(test_df['name'].values.tolist())

	test_svd = pd.DataFrame(svd_obj.transform(test_tfidf))
		
	
	test_svd.columns = ['svd_name_'+str(i) for i in range(n_comp)]
	
	test_df = pd.concat([test_df, test_svd], axis=1)


	
	
	test_df.fillna(0, inplace=True)
	
	test = test_df.copy()
	dtest = xgb.DMatrix(test[feature_names].values)
	return test, dtest
