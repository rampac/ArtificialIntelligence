# -*- coding: utf-8 -*-
"""
Created on Sun Oct 28 17:24:00 2018

@author: RPACHIANNAN
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import IsolationForest

from flask import Flask
from flask import jsonify
from flask import request
from flask_cors import CORS
app = Flask(__name__)
CORS(app)

@app.route('/postdata', methods = ['POST'])
def postUserData():
 userData=pd.DataFrame.from_dict(request.get_json(), orient='columns')
 outputDF=runModel(userData)
 return jsonify(outputDF)

@app.route('/fetchPDdata', methods = ['GET'])
def fetchPDData():
 #userData=pd.DataFrame.from_dict(request.get_json(), orient='columns')
 outputDF=fetchModelData()
 return jsonify(outputDF)
 
@app.route('/runPDdata', methods = ['GET'])
def runPDData():
 #userData=pd.DataFrame.from_dict(request.get_json(), orient='columns')
 outputDF=runModel(pd.DataFrame())
 return jsonify(outputDF)
 
def fetchModelData():
	try:
		from configparser import ConfigParser
	except ImportError:
		from ConfigParser import ConfigParser

	# instantiate
	config = ConfigParser()

	# parse existing file
	config.read('FILE.ini')

	# read values from a section
	input_file_path = config.get('configurations', 'input_file_path')
	input_file_name = config.get('configurations', 'input_file_name')
	modelData_df = pd.read_excel(input_file_path + input_file_name)
	
	return modelData_df.to_json(orient='records')
 
def runModel(userData):
	try:
		from configparser import ConfigParser
	except ImportError:
		from ConfigParser import ConfigParser

	# instantiate
	config = ConfigParser()

	# parse existing file
	config.read('FILE.ini')

	# read values from a section
	input_file_path = config.get('configurations', 'input_file_path')
	input_file_name = config.get('configurations', 'input_file_name')
	if userData.shape[0]>0:
		#userData.columns = userData.iloc[0]
		#userData = userData.iloc[1:]
		vAR_df = userData
	else:
		vAR_df = pd.read_excel(input_file_path + input_file_name)
	
	print(vAR_df)
	modified_Company_Series = vAR_df[config.get('configurations', 'feature1')].fillna('000000')
	modified_Trading_Partner_Series = vAR_df[config.get('configurations', 'feature2')].fillna('000000')
	modified_GL_Account_Series = vAR_df[config.get('configurations', 'feature3')].fillna('000000')
	modified_Amount_Series = vAR_df[config.get('configurations', 'feature4')].fillna('000000')
	modified_Customer_Series = vAR_df[config.get('configurations', 'feature5')].fillna('NULL')
	modified_Vendor_Series = vAR_df[config.get('configurations', 'feature6')].fillna('NULL')
	modified_TT_Series = vAR_df[config.get('configurations', 'feature7')].fillna('NULL')

	modified_Company = pd.DataFrame(modified_Company_Series)

	modified_Trading_Partner = pd.DataFrame(modified_Trading_Partner_Series.apply(np.int64))
	modified_GL_Account = pd.DataFrame(modified_GL_Account_Series.apply(np.int64))
	modified_Amount = pd.DataFrame(modified_Amount_Series.apply(np.int64))
	modified_Customer = pd.DataFrame(modified_Customer_Series)
	modified_Vendor = pd.DataFrame(modified_Vendor_Series)
	modified_TT = pd.DataFrame(modified_TT_Series)

	vAR_df2 = vAR_df.merge(modified_Customer,left_index=True, right_index=True)

	vAR_df3 = vAR_df2.merge(modified_Vendor,left_index=True, right_index=True)

	vAR_df4 = vAR_df3.merge(modified_TT,left_index=True, right_index=True)
	vAR_df5 = vAR_df4.merge(modified_Company,left_index=True, right_index=True)
	vAR_df6 = vAR_df5.merge(modified_Trading_Partner,left_index=True, right_index=True)
	vAR_df7 = vAR_df6.merge(modified_GL_Account,left_index=True, right_index=True)
	vAR_df8 = vAR_df7.merge(modified_Amount,left_index=True, right_index=True)
	
	required_columns_df = vAR_df8[['Company_y','Trading_Partner_y','Customer_y','Vendor_y','GL_Account_y','Amount_y','Trasaction_Type_y']]
	labelEncoder = LabelEncoder()

	col_Customer_Conversion = labelEncoder.fit_transform(required_columns_df.iloc[0:,2])
	col_Customer_Conversion_df = pd.DataFrame(col_Customer_Conversion,columns={'Customer_Converted'})
	col_Vendor_Conversion = labelEncoder.fit_transform(required_columns_df.iloc[0:,3])
	col_Vendor_Conversion_df = pd.DataFrame(col_Vendor_Conversion,columns={'Vendor_Converted'})
	col_TT_Conversion = labelEncoder.fit_transform(required_columns_df.iloc[0:,6])
	col_TT_Conversion_df = pd.DataFrame(col_TT_Conversion,columns={'Transaction_Type_Converted'})
	required_columns_df = required_columns_df.reset_index(drop=True)
	merge_df1 = required_columns_df.merge(col_Customer_Conversion_df,left_index=True, right_index=True)
	merge_df2 = merge_df1.merge(col_Vendor_Conversion_df,left_index=True, right_index=True)
	merge_df3 = merge_df2.merge(col_TT_Conversion_df,left_index=True, right_index=True)

	final_columns_df = merge_df3[['Company_y','Trading_Partner_y','Customer_Converted','Vendor_Converted','GL_Account_y','Amount_y','Transaction_Type_Converted']]
	#print(final_columns_df[final_columns_df.isnull().any(1)])

	#final_columns_df.to_csv('ramesh_test.xls', sep='\t')

	clf = IsolationForest( max_samples=498,
	random_state=np.random.RandomState(42))
	clf.fit(final_columns_df)

	scores_pred = clf.decision_function(final_columns_df)

	resultDf = pd.DataFrame(columns=['Valid Record'])
	counter =0
	for n in range(len(scores_pred)):
		if (scores_pred[n] >=0.1):
			resultDf.loc[n] = 'N'
		else:
			resultDf.loc[n] = 'Y' 
	
	vAR_df = vAR_df.reset_index()

	final_dataset = vAR_df.merge(resultDf,left_index=True, right_index=True)

	#output_file_path = config.get('configurations', 'output_file_path')
	#output_file_name = config.get('configurations', 'output_file_name')

	#final_dataset.to_csv(output_file_path + output_file_name, sep='\t')
	print(final_dataset.to_json(orient='records'))
	return final_dataset.to_json(orient='records')
	
if __name__ == '__main__':
 app.run()