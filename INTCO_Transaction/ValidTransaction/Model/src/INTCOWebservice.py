# -*- coding: utf-8 -*-
"""
Created on Sun Oct 28 17:24:00 2018

@author: RPACHIANNAN
"""
# import pandas library 
import pandas as pd
# import numpy library
import numpy as np
# import LabelEncoder for tranforming char data
from sklearn.preprocessing import LabelEncoder
# import IsolationForest for anomaly detection
from sklearn.ensemble import IsolationForest

# import Flask for exposing python service
from flask import Flask
# import jsonify for jsonifying the data
from flask import jsonify
# import request for reaing JSON data from request
from flask import request
# import flask_cors for allowing CORS enablement
from flask_cors import CORS
app = Flask(__name__)

# request handling for while posting model data for Valid Transaction
CORS(app)
@app.route('/postModel1data', methods = ['POST'])
def postUserDataForModel1():
 userData=pd.DataFrame.from_dict(request.get_json(), orient='columns')
 outputDF=runModel1(userData)
 return jsonify(outputDF)

# request handling for pulling pre defined model data for Valid Transaction 
@app.route('/fetchPDModel1data', methods = ['GET'])
def fetchPDDataForModel1():
 outputDF=fetchDataForValidTr()
 return jsonify(outputDF)

# request handling for running pre defined model data for Valid Transaction  
@app.route('/runPDModel1data', methods = ['GET'])
def runPDDataForModel1():
 outputDF=runModel1(pd.DataFrame())
 return jsonify(outputDF)

# request handling for while posting model data for Inter company trasaction
CORS(app)
@app.route('/postModel2data', methods = ['POST'])
def postUserDataForModel2():
 userData=pd.DataFrame.from_dict(request.get_json(), orient='columns')
 outputDF=runModel2(userData)
 return jsonify(outputDF)

# request handling for pulling pre defined model data for Inter company Transaction  
@app.route('/fetchPDModel2data', methods = ['GET'])
def fetchPDDataForModel2():
 outputDF=fetchModel2Data()
 return jsonify(outputDF)

# request handling for running pre defined model data for Inter company Transaction  
@app.route('/runPDModel2data', methods = ['GET'])
def runPDDataForModel2():
 outputDF=runModel2(pd.DataFrame())
 return jsonify(outputDF)

# request handling for while posting model data for Matching Transaction 
CORS(app) 
@app.route('/postModel3data', methods = ['POST'])
def postUserDataForModel3():
 userData=pd.DataFrame.from_dict(request.get_json(), orient='columns')
 outputDF=runModel3(userData)
 return jsonify(outputDF)

# request handling for pulling pre defined model data for Matching Transaction  
@app.route('/fetchPDModel3data', methods = ['GET'])
def fetchPDDataForModel3():
 outputDF=fetchModel3Data()
 return jsonify(outputDF)

# request handling for running pre defined model data for Matching Transaction  
@app.route('/runPDModel3data', methods = ['GET'])
def runPDDataForModel3():
 outputDF=runModel3(pd.DataFrame())
 return jsonify(outputDF)

# request handling for while posting model data for Booking Transaction 
CORS(app) 
@app.route('/postModel4data', methods = ['POST'])
def postUserDataForModel4():
 userData=pd.DataFrame.from_dict(request.get_json(), orient='columns')
 outputDF=runModel4(userData)
 return jsonify(outputDF)

# request handling for pulling pre defined model data for Booking Transaction  
@app.route('/fetchPDModel4data', methods = ['GET'])
def fetchPDDataForModel4():
 outputDF=fetchModel4Data()
 return jsonify(outputDF)

# request handling for running pre defined model data for Booking Transaction  
@app.route('/runPDModel4data', methods = ['GET'])
def runPDDataForModel4():
 outputDF=runModel4(pd.DataFrame())
 return jsonify(outputDF)

# request handling for while posting model data for Eliminating Transaction 
CORS(app) 
@app.route('/postModel5data', methods = ['POST'])
def postUserDataForModel5():
 userData=pd.DataFrame.from_dict(request.get_json(), orient='columns')
 outputDF=runModel5(userData)
 return jsonify(outputDF)

# request handling for pulling pre defined model data for Eliminating Transaction  
@app.route('/fetchPDModel5data', methods = ['GET'])
def fetchPDDataForModel5():
 outputDF=fetchModel5Data()
 return jsonify(outputDF)

# request handling for running pre defined model data for Eliminating Transaction  
@app.route('/runPDModel5data', methods = ['GET'])
def runPDDataForModel5():
 outputDF=runModel5(pd.DataFrame())
 return jsonify(outputDF)

CORS(app)
@app.route('/postModel6data', methods = ['POST'])
def postUserDataForModel6():
 userData=pd.DataFrame.from_dict(request.get_json(), orient='columns')
 outputDF=runModel6(userData)
 return jsonify(outputDF)

@app.route('/fetchPDModel6data', methods = ['GET'])
def fetchPDDataForModel6():
 outputDF=fetchModel6Data()
 return jsonify(outputDF)
 
@app.route('/runPDModel6data', methods = ['GET'])
def runPDDataForModel6():
 outputDF=runModel6(pd.DataFrame())
 return jsonify(outputDF)

 
def fetchDataForValidTr():
	try:
		from configparser import ConfigParser
	except ImportError:
		from ConfigParser import ConfigParser

	# instantiate
	config = ConfigParser()

	# parse existing file
	config.read('C:\\Ramesh\\AI\\INTCO_Transaction\\ValidTransaction\\Model\\configuration\\FILE.ini')

	# read values from a section
	input_file_path = config.get('configurations', 'input_file_path')
	input_file_name = config.get('configurations', 'input_file_name')
	modelData_df = pd.read_excel(input_file_path + input_file_name)
	
	return modelData_df.to_json(orient='records')
 
def runModel1(userData):
	try:
		from configparser import ConfigParser
	except ImportError:
		from ConfigParser import ConfigParser

	# instantiate
	config = ConfigParser()

	# parse existing file
	config.read('C:\\Ramesh\\AI\\INTCO_Transaction\\ValidTransaction\\Model\\configuration\\FILE.ini')

	# read values from a section
	input_file_path = config.get('configurations', 'input_file_path')
	input_file_name = config.get('configurations', 'input_file_name')
	if userData.shape[0]>0:
		vAR_df = userData
	else:
		vAR_df = pd.read_excel(input_file_path + input_file_name)
	
	# if columns have any empty / null data, fill it with either 000000 / NULL
	modified_Company_Series = vAR_df[config.get('configurations', 'feature1')].fillna('000000')
	modified_Trading_Partner_Series = vAR_df[config.get('configurations', 'feature2')].fillna('000000')
	modified_GL_Account_Series = vAR_df[config.get('configurations', 'feature3')].fillna('000000')
	modified_Amount_Series = vAR_df[config.get('configurations', 'feature4')].fillna('000000')
	modified_Customer_Series = vAR_df[config.get('configurations', 'feature5')].fillna('NULL')
	modified_Vendor_Series = vAR_df[config.get('configurations', 'feature6')].fillna('NULL')
	modified_TT_Series = vAR_df[config.get('configurations', 'feature7')].fillna('NULL')

	modified_Company = pd.DataFrame(modified_Company_Series)
	# Convert into dataframes
	modified_Trading_Partner = pd.DataFrame(modified_Trading_Partner_Series.apply(np.int64))
	modified_GL_Account = pd.DataFrame(modified_GL_Account_Series.apply(np.int64))
	modified_Amount = pd.DataFrame(modified_Amount_Series.apply(np.int64))
	modified_Customer = pd.DataFrame(modified_Customer_Series)
	modified_Vendor = pd.DataFrame(modified_Vendor_Series)
	modified_TT = pd.DataFrame(modified_TT_Series)
	#mege with mian dataframe
	vAR_df2 = vAR_df.merge(modified_Customer,left_index=True, right_index=True)

	vAR_df3 = vAR_df2.merge(modified_Vendor,left_index=True, right_index=True)

	vAR_df4 = vAR_df3.merge(modified_TT,left_index=True, right_index=True)
	vAR_df5 = vAR_df4.merge(modified_Company,left_index=True, right_index=True)
	vAR_df6 = vAR_df5.merge(modified_Trading_Partner,left_index=True, right_index=True)
	vAR_df7 = vAR_df6.merge(modified_GL_Account,left_index=True, right_index=True)
	vAR_df8 = vAR_df7.merge(modified_Amount,left_index=True, right_index=True)
	
	# pull only requird columns
	required_columns_df = vAR_df8[['Company_y','Trading_Partner_y','Customer_y','Vendor_y','GL_Account_y','Amount_y','Trasaction_Type_y']]
	labelEncoder = LabelEncoder()
	
	#encode the label data
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

	#merge converted data into dataframe
	final_columns_df = merge_df3[['Company_y','Trading_Partner_y','Customer_Converted','Vendor_Converted','GL_Account_y','Amount_y','Transaction_Type_Converted']]
	#print(final_columns_df[final_columns_df.isnull().any(1)])

	clf = IsolationForest( max_samples=498,
	random_state=np.random.RandomState(42))
	#run model
	clf.fit(final_columns_df)
	
	#predict the score
	scores_pred = clf.decision_function(final_columns_df)
	
	# label it based the prediction score
	resultDf = pd.DataFrame(columns=['Valid Record1'])
	counter =0
	for n in range(len(scores_pred)):
		if (scores_pred[n] >=0.1):
			resultDf.loc[n] = 'N'
		else:
			resultDf.loc[n] = 'Y' 
	
	vAR_df = vAR_df.reset_index(drop=True)

	final_dataset = vAR_df.merge(resultDf,left_index=True, right_index=True)
	#write the outcome to the file
	output_file_path = config.get('configurations', 'output_file_path')
	output_file_name = config.get('configurations', 'output_file_name')
	final_dataset.to_csv(output_file_path+output_file_name, sep='\t')
	#final_dataset.to_csv('C:\\Ramesh\\AI\\INTCO_Transaction\\ValidTransaction\\ModelOutcome\\ValidTransactionResult.xls', sep='\t')
	
	# return the dataframe as json
	return final_dataset.to_json(orient='records')

def fetchModel2Data():
	try:
		from configparser import ConfigParser
	except ImportError:
		from ConfigParser import ConfigParser

	# instantiate
	config = ConfigParser()

	# parse existing file
	config.read('C:\\Ramesh\\AI\\INTCO_Transaction\\INTCTransaction\\Model\\configuration\\FILE.ini')

	# read values from a section
	input_file_path = config.get('configurations', 'input_file_path')
	input_file_name = config.get('configurations', 'input_file_name')
	modelData_df = pd.read_excel(input_file_path + input_file_name)
	
	return modelData_df.to_json(orient='records')
 
def runModel2(userData):
	try:
		from configparser import ConfigParser
	except ImportError:
		from ConfigParser import ConfigParser

	# instantiate
	config = ConfigParser()

	# parse existing file
	config.read('C:\\Ramesh\\AI\\INTCO_Transaction\\INTCTransaction\\Model\\configuration\\FILE.ini')

	# read values from a section
	input_file_path = config.get('configurations', 'input_file_path')
	input_file_name = config.get('configurations', 'input_file_name')
	if userData.shape[0]>0:
		#userData.columns = userData.iloc[0]
		#userData = userData.iloc[1:]
		vAR_df = userData
	else:
		vAR_df = pd.read_excel(input_file_path + input_file_name)
	
	# if columns have any empty / null data, fill it with either 000000 / NULL
	modified_Company_Series = vAR_df[config.get('configurations', 'feature1')].fillna('000000')
	modified_Trading_Partner_Series = vAR_df[config.get('configurations', 'feature2')].fillna('000000')
	modified_GL_Account_Series = vAR_df[config.get('configurations', 'feature3')].fillna('000000')
	modified_Amount_Series = vAR_df[config.get('configurations', 'feature4')].fillna('000000')
	modified_Customer_Series = vAR_df[config.get('configurations', 'feature5')].fillna('NULL')
	modified_Vendor_Series = vAR_df[config.get('configurations', 'feature6')].fillna('NULL')
	modified_TT_Series = vAR_df[config.get('configurations', 'feature7')].fillna('NULL')

	modified_Company = pd.DataFrame(modified_Company_Series)
	# Convert into dataframes
	modified_Trading_Partner = pd.DataFrame(modified_Trading_Partner_Series.apply(np.int64))
	modified_GL_Account = pd.DataFrame(modified_GL_Account_Series.apply(np.int64))
	modified_Amount = pd.DataFrame(modified_Amount_Series.apply(np.int64))
	modified_Customer = pd.DataFrame(modified_Customer_Series)
	modified_Vendor = pd.DataFrame(modified_Vendor_Series)
	modified_TT = pd.DataFrame(modified_TT_Series)
	#mege with mian dataframe
	vAR_df2 = vAR_df.merge(modified_Customer,left_index=True, right_index=True)

	vAR_df3 = vAR_df2.merge(modified_Vendor,left_index=True, right_index=True)

	vAR_df4 = vAR_df3.merge(modified_TT,left_index=True, right_index=True)
	vAR_df5 = vAR_df4.merge(modified_Company,left_index=True, right_index=True)
	vAR_df6 = vAR_df5.merge(modified_Trading_Partner,left_index=True, right_index=True)
	vAR_df7 = vAR_df6.merge(modified_GL_Account,left_index=True, right_index=True)
	vAR_df8 = vAR_df7.merge(modified_Amount,left_index=True, right_index=True)
	
	# pull only requird columns
	required_columns_df = vAR_df8[['Company_y','Trading_Partner_y','Customer_y','Vendor_y','GL_Account_y','Amount_y','Trasaction_Type_y']]
	labelEncoder = LabelEncoder()
	
	#encode the label data
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

	#merge converted data into dataframe
	final_columns_df = merge_df3[['Company_y','Trading_Partner_y','Customer_Converted','Vendor_Converted','GL_Account_y','Amount_y','Transaction_Type_Converted']]
	#print(final_columns_df[final_columns_df.isnull().any(1)])

	clf = IsolationForest( max_samples=498,
	random_state=np.random.RandomState(42))
	#run model
	clf.fit(final_columns_df)
	
	#predict the score
	scores_pred = clf.decision_function(final_columns_df)
	
	# label it based the prediction score
	resultDf = pd.DataFrame(columns=['Valid Record2'])
	counter =0
	for n in range(len(scores_pred)):
		if (scores_pred[n] >=0.1):
			resultDf.loc[n] = 'N'
		else:
			resultDf.loc[n] = 'Y' 
	
	vAR_df = vAR_df.reset_index(drop=True)

	final_dataset = vAR_df.merge(resultDf,left_index=True, right_index=True)
	#write the outcome to the file
	output_file_path = config.get('configurations', 'output_file_path')
	output_file_name = config.get('configurations', 'output_file_name')
	final_dataset.to_csv(output_file_path+output_file_name, sep='\t')
	#final_dataset.to_csv('C:\\Ramesh\\AI\\INTCO_Transaction\\INTCTransaction\\ModelOutcome\\INTCTransactionResult.xls', sep='\t')

	return final_dataset.to_json(orient='records')

def fetchModel3Data():
	try:
		from configparser import ConfigParser
	except ImportError:
		from ConfigParser import ConfigParser

	# instantiate
	config = ConfigParser()

	# parse existing file
	config.read('C:\\Ramesh\\AI\\INTCO_Transaction\\MATCHTransaction\\Model\\configuration\\FILE.ini')

	# read values from a section
	input_file_path = config.get('configurations', 'input_file_path')
	input_file_name = config.get('configurations', 'input_file_name')
	modelData_df = pd.read_excel(input_file_path + input_file_name)
	
	return modelData_df.to_json(orient='records')
 
def runModel3(userData):
	try:
		from configparser import ConfigParser
	except ImportError:
		from ConfigParser import ConfigParser

	# instantiate
	config = ConfigParser()

	# parse existing file
	config.read('C:\\Ramesh\\AI\\INTCO_Transaction\\MATCHTransaction\\Model\\configuration\\FILE.ini')

	# read values from a section
	input_file_path = config.get('configurations', 'input_file_path')
	input_file_name = config.get('configurations', 'input_file_name')
	if userData.shape[0]>0:
		#userData.columns = userData.iloc[0]
		#userData = userData.iloc[1:]
		vAR_df = userData
	else:
		vAR_df = pd.read_excel(input_file_path + input_file_name)
	
	# if columns have any empty / null data, fill it with either 000000 / NULL
	modified_Company_Series = vAR_df[config.get('configurations', 'feature1')].fillna('000000')
	modified_Trading_Partner_Series = vAR_df[config.get('configurations', 'feature2')].fillna('000000')
	modified_GL_Account_Series = vAR_df[config.get('configurations', 'feature3')].fillna('000000')
	modified_Amount_Series = vAR_df[config.get('configurations', 'feature4')].fillna('000000')
	modified_Customer_Series = vAR_df[config.get('configurations', 'feature5')].fillna('NULL')
	modified_Vendor_Series = vAR_df[config.get('configurations', 'feature6')].fillna('NULL')
	modified_TT_Series = vAR_df[config.get('configurations', 'feature7')].fillna('NULL')

	modified_Company = pd.DataFrame(modified_Company_Series)
	# Convert into dataframes
	modified_Trading_Partner = pd.DataFrame(modified_Trading_Partner_Series.apply(np.int64))
	modified_GL_Account = pd.DataFrame(modified_GL_Account_Series.apply(np.int64))
	modified_Amount = pd.DataFrame(modified_Amount_Series.apply(np.int64))
	modified_Customer = pd.DataFrame(modified_Customer_Series)
	modified_Vendor = pd.DataFrame(modified_Vendor_Series)
	modified_TT = pd.DataFrame(modified_TT_Series)
	#mege with mian dataframe
	vAR_df2 = vAR_df.merge(modified_Customer,left_index=True, right_index=True)

	vAR_df3 = vAR_df2.merge(modified_Vendor,left_index=True, right_index=True)

	vAR_df4 = vAR_df3.merge(modified_TT,left_index=True, right_index=True)
	vAR_df5 = vAR_df4.merge(modified_Company,left_index=True, right_index=True)
	vAR_df6 = vAR_df5.merge(modified_Trading_Partner,left_index=True, right_index=True)
	vAR_df7 = vAR_df6.merge(modified_GL_Account,left_index=True, right_index=True)
	vAR_df8 = vAR_df7.merge(modified_Amount,left_index=True, right_index=True)
	
	# pull only requird columns
	required_columns_df = vAR_df8[['Company_y','Trading_Partner_y','Customer_y','Vendor_y','GL_Account_y','Amount_y','Trasaction_Type_y']]
	labelEncoder = LabelEncoder()
	
	#encode the label data
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

	#merge converted data into dataframe
	final_columns_df = merge_df3[['Company_y','Trading_Partner_y','Customer_Converted','Vendor_Converted','GL_Account_y','Amount_y','Transaction_Type_Converted']]
	#print(final_columns_df[final_columns_df.isnull().any(1)])

	clf = IsolationForest( max_samples=498,
	random_state=np.random.RandomState(42))
	#run model
	clf.fit(final_columns_df)
	
	#predict the score
	scores_pred = clf.decision_function(final_columns_df)
	
	# label it based the prediction score
	resultDf = pd.DataFrame(columns=['Valid Record3'])
	counter =0
	for n in range(len(scores_pred)):
		if (scores_pred[n] >=0.1):
			resultDf.loc[n] = 'N'
		else:
			resultDf.loc[n] = 'Y' 
	
	vAR_df = vAR_df.reset_index(drop=True)

	final_dataset = vAR_df.merge(resultDf,left_index=True, right_index=True)
	#write the outcome to the file
	output_file_path = config.get('configurations', 'output_file_path')
	output_file_name = config.get('configurations', 'output_file_name')
	final_dataset.to_csv(output_file_path+output_file_name, sep='\t')
	#final_dataset.to_csv('C:\\Ramesh\\AI\\INTCO_Transaction\\MATCHTransaction\\ModelOutcome\\MATCHTransactionResult.xls', sep='\t')

	return final_dataset.to_json(orient='records')	
	
def fetchModel4Data():
	try:
		from configparser import ConfigParser
	except ImportError:
		from ConfigParser import ConfigParser

	# instantiate
	config = ConfigParser()

	# parse existing file
	config.read('C:\\Ramesh\\AI\\INTCO_Transaction\\BOOKTransaction\\Model\\configuration\\FILE.ini')

	# read values from a section
	input_file_path = config.get('configurations', 'input_file_path')
	input_file_name = config.get('configurations', 'input_file_name')
	modelData_df = pd.read_excel(input_file_path + input_file_name)
	
	return modelData_df.to_json(orient='records')
 
def runModel4(userData):
	try:
		from configparser import ConfigParser
	except ImportError:
		from ConfigParser import ConfigParser

	# instantiate
	config = ConfigParser()

	# parse existing file
	config.read('C:\\Ramesh\\AI\\INTCO_Transaction\\BOOKTransaction\\Model\\configuration\\FILE.ini')

	# read values from a section
	input_file_path = config.get('configurations', 'input_file_path')
	input_file_name = config.get('configurations', 'input_file_name')
	if userData.shape[0]>0:
		#userData.columns = userData.iloc[0]
		#userData = userData.iloc[1:]
		vAR_df = userData
	else:
		vAR_df = pd.read_excel(input_file_path + input_file_name)
	
	# if columns have any empty / null data, fill it with either 000000 / NULL
	modified_Company_Series = vAR_df[config.get('configurations', 'feature1')].fillna('000000')
	modified_Trading_Partner_Series = vAR_df[config.get('configurations', 'feature2')].fillna('000000')
	modified_GL_Account_Series = vAR_df[config.get('configurations', 'feature3')].fillna('000000')
	modified_Amount_Series = vAR_df[config.get('configurations', 'feature4')].fillna('000000')
	modified_Customer_Series = vAR_df[config.get('configurations', 'feature5')].fillna('NULL')
	modified_Vendor_Series = vAR_df[config.get('configurations', 'feature6')].fillna('NULL')
	modified_TT_Series = vAR_df[config.get('configurations', 'feature7')].fillna('NULL')

	modified_Company = pd.DataFrame(modified_Company_Series)
	# Convert into dataframes
	modified_Trading_Partner = pd.DataFrame(modified_Trading_Partner_Series.apply(np.int64))
	modified_GL_Account = pd.DataFrame(modified_GL_Account_Series.apply(np.int64))
	modified_Amount = pd.DataFrame(modified_Amount_Series.apply(np.int64))
	modified_Customer = pd.DataFrame(modified_Customer_Series)
	modified_Vendor = pd.DataFrame(modified_Vendor_Series)
	modified_TT = pd.DataFrame(modified_TT_Series)
	#mege with mian dataframe
	vAR_df2 = vAR_df.merge(modified_Customer,left_index=True, right_index=True)

	vAR_df3 = vAR_df2.merge(modified_Vendor,left_index=True, right_index=True)

	vAR_df4 = vAR_df3.merge(modified_TT,left_index=True, right_index=True)
	vAR_df5 = vAR_df4.merge(modified_Company,left_index=True, right_index=True)
	vAR_df6 = vAR_df5.merge(modified_Trading_Partner,left_index=True, right_index=True)
	vAR_df7 = vAR_df6.merge(modified_GL_Account,left_index=True, right_index=True)
	vAR_df8 = vAR_df7.merge(modified_Amount,left_index=True, right_index=True)
	
	# pull only requird columns
	required_columns_df = vAR_df8[['Company_y','Trading_Partner_y','Customer_y','Vendor_y','GL_Account_y','Amount_y','Trasaction_Type_y']]
	labelEncoder = LabelEncoder()
	
	#encode the label data
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

	#merge converted data into dataframe
	final_columns_df = merge_df3[['Company_y','Trading_Partner_y','Customer_Converted','Vendor_Converted','GL_Account_y','Amount_y','Transaction_Type_Converted']]
	#print(final_columns_df[final_columns_df.isnull().any(1)])

	clf = IsolationForest( max_samples=498,
	random_state=np.random.RandomState(42))
	#run model
	clf.fit(final_columns_df)
	
	#predict the score
	scores_pred = clf.decision_function(final_columns_df)
	
	# label it based the prediction score
	resultDf = pd.DataFrame(columns=['Valid Record4'])
	counter =0
	for n in range(len(scores_pred)):
		if (scores_pred[n] >=0.1):
			resultDf.loc[n] = 'N'
		else:
			resultDf.loc[n] = 'Y' 
	
	vAR_df = vAR_df.reset_index(drop=True)

	final_dataset = vAR_df.merge(resultDf,left_index=True, right_index=True)
	#write the outcome to the file
	output_file_path = config.get('configurations', 'output_file_path')
	output_file_name = config.get('configurations', 'output_file_name')
	final_dataset.to_csv(output_file_path+output_file_name, sep='\t')
	print(final_dataset)
	return final_dataset.to_json(orient='records')	
	
def fetchModel5Data():
	try:
		from configparser import ConfigParser
	except ImportError:
		from ConfigParser import ConfigParser

	# instantiate
	config = ConfigParser()

	# parse existing file
	config.read('C:\\Ramesh\\AI\\INTCO_Transaction\\ELIMTransaction\\Model\\configuration\\FILE.ini')

	# read values from a section
	input_file_path = config.get('configurations', 'input_file_path')
	input_file_name = config.get('configurations', 'input_file_name')
	modelData_df = pd.read_excel(input_file_path + input_file_name)
	
	return modelData_df.to_json(orient='records')
 
def runModel5(userData):
	try:
		from configparser import ConfigParser
	except ImportError:
		from ConfigParser import ConfigParser

	# instantiate
	config = ConfigParser()

	# parse existing file
	config.read('C:\\Ramesh\\AI\\INTCO_Transaction\\ELIMTransaction\\Model\\configuration\\FILE.ini')

	# read values from a section
	input_file_path = config.get('configurations', 'input_file_path')
	input_file_name = config.get('configurations', 'input_file_name')
	if userData.shape[0]>0:
		#userData.columns = userData.iloc[0]
		#userData = userData.iloc[1:]
		vAR_df = userData
	else:
		vAR_df = pd.read_excel(input_file_path + input_file_name)
	
	# if columns have any empty / null data, fill it with either 000000 / NULL
	modified_Company_Series = vAR_df[config.get('configurations', 'feature1')].fillna('000000')
	modified_Trading_Partner_Series = vAR_df[config.get('configurations', 'feature2')].fillna('000000')
	modified_GL_Account_Series = vAR_df[config.get('configurations', 'feature3')].fillna('000000')
	modified_Amount_Series = vAR_df[config.get('configurations', 'feature4')].fillna('000000')
	modified_Customer_Series = vAR_df[config.get('configurations', 'feature5')].fillna('NULL')
	modified_Vendor_Series = vAR_df[config.get('configurations', 'feature6')].fillna('NULL')
	modified_TT_Series = vAR_df[config.get('configurations', 'feature7')].fillna('NULL')

	modified_Company = pd.DataFrame(modified_Company_Series)
	# Convert into dataframes
	modified_Trading_Partner = pd.DataFrame(modified_Trading_Partner_Series.apply(np.int64))
	modified_GL_Account = pd.DataFrame(modified_GL_Account_Series.apply(np.int64))
	modified_Amount = pd.DataFrame(modified_Amount_Series.apply(np.int64))
	modified_Customer = pd.DataFrame(modified_Customer_Series)
	modified_Vendor = pd.DataFrame(modified_Vendor_Series)
	modified_TT = pd.DataFrame(modified_TT_Series)
	#mege with mian dataframe
	vAR_df2 = vAR_df.merge(modified_Customer,left_index=True, right_index=True)

	vAR_df3 = vAR_df2.merge(modified_Vendor,left_index=True, right_index=True)

	vAR_df4 = vAR_df3.merge(modified_TT,left_index=True, right_index=True)
	vAR_df5 = vAR_df4.merge(modified_Company,left_index=True, right_index=True)
	vAR_df6 = vAR_df5.merge(modified_Trading_Partner,left_index=True, right_index=True)
	vAR_df7 = vAR_df6.merge(modified_GL_Account,left_index=True, right_index=True)
	vAR_df8 = vAR_df7.merge(modified_Amount,left_index=True, right_index=True)
	
	# pull only requird columns
	required_columns_df = vAR_df8[['Company_y','Trading_Partner_y','Customer_y','Vendor_y','GL_Account_y','Amount_y','Trasaction_Type_y']]
	labelEncoder = LabelEncoder()
	
	#encode the label data
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

	#merge converted data into dataframe
	final_columns_df = merge_df3[['Company_y','Trading_Partner_y','Customer_Converted','Vendor_Converted','GL_Account_y','Amount_y','Transaction_Type_Converted']]
	#print(final_columns_df[final_columns_df.isnull().any(1)])

	clf = IsolationForest( max_samples=498,
	random_state=np.random.RandomState(42))
	#run model
	clf.fit(final_columns_df)
	
	#predict the score
	scores_pred = clf.decision_function(final_columns_df)
	
	# label it based the prediction score
	resultDf = pd.DataFrame(columns=['Valid Record5'])
	counter =0
	for n in range(len(scores_pred)):
		if (scores_pred[n] >=0.1):
			resultDf.loc[n] = 'N'
		else:
			resultDf.loc[n] = 'Y' 
	
	vAR_df = vAR_df.reset_index(drop=True)

	final_dataset = vAR_df.merge(resultDf,left_index=True, right_index=True)
	#write the outcome to the file
	output_file_path = config.get('configurations', 'output_file_path')
	output_file_name = config.get('configurations', 'output_file_name')
	final_dataset.to_csv(output_file_path+output_file_name, sep='\t')
	#final_dataset.to_csv('C:\\Ramesh\\AI\\INTCO_Transaction\\ELIMTransaction\\ModelOutcome\\ELIMTransactionResult.xls', sep='\t')
	
	return final_dataset.to_json(orient='records')	

def fetchModel6Data():
	try:
		from configparser import ConfigParser
	except ImportError:
		from ConfigParser import ConfigParser

	# instantiate
	config = ConfigParser()

	# parse existing file
	config.read('C:\\Ramesh\\AI\\INTCO_Transaction\\RunAllTransactions\\Model\\configuration\\FILE.ini')

	# read values from a section
	input_file_path = config.get('configurations', 'input_file_path')
	input_file_name = config.get('configurations', 'input_file_name')
	print(input_file_path + input_file_name)
	modelData_df = pd.read_excel(input_file_path + input_file_name)
	
	print(modelData_df.to_json(orient='records'))
	return modelData_df.to_json(orient='records')
 
def runModel6(userData):
	df1 = runModel1(userData)
	df2 = runModel2(pd.read_json(df1))
	df3 = runModel3(pd.read_json(df2))
	df4 = runModel4(pd.read_json(df3))
	df5 = runModel5(pd.read_json(df4))
	
	return df5
if __name__ == '__main__':
 app.run()