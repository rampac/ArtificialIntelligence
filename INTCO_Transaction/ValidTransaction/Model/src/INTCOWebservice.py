# -*- coding: utf-8 -*-
"""
Created on Sun Oct 28 17:24:00 2018

@author: RPACHIANNAN
"""
# import pandas library 
import pandas as pd
# import numpy library
import numpy as np
# import time for embedding timestamp in filename 
import time
# import LabelEncoder for tranforming char data
from sklearn.preprocessing import LabelEncoder
# import IsolationForest for anomaly detection
from sklearn.ensemble import IsolationForest

# import Flask for exposing python service
from flask import Flask
# import jsonify for jsonifying the data
from flask import jsonify
# import request for reading JSON data from request
from flask import request
# import flask_cors for  CORS enablement
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
	# read values from a section
	input_file_path = getConfig('PROBLEM_1_CONFIGURATION', 'INPUT_FILE_PATH')
	input_file_name = getConfig('PROBLEM_1_CONFIGURATION', 'INPUT_FILE_NAME')
	modelData_df = getTrainingData(input_file_path + input_file_name)

	return modelData_df.to_json(orient='records')
 
def runModel1(userData):
	'''
		Collecting train and test data
	'''
	vAR_df = collectTrainingdataForModel1(userData)
		
	#vAR_df_test = getTestData(input_file_path + input_file_name)	
	
	'''
		Preparing the data for Model
	'''
	
	final_dataset = prepareAndRunModel(vAR_df, 1)
	
	'''
		Provisioning the data by exporting to a file and displaying in the screen
	'''
	output_file_path = getConfig('PROBLEM_1_CONFIGURATION', 'OUTPUT_FILE_PATH')
	output_file_name = getConfig('PROBLEM_1_CONFIGURATION', 'OUTPUT_FILE_NAME')
	final_dataset.to_csv(output_file_path+getTimeString()+output_file_name, sep='\t')
	
	# return the dataframe as json
	return final_dataset.to_json(orient='records')

def fetchModel2Data():

	# read values from a section
	input_file_path = getConfig('PROBLEM_2_CONFIGURATION', 'INPUT_FILE_PATH')
	input_file_name = getConfig('PROBLEM_2_CONFIGURATION', 'INPUT_FILE_NAME')
	modelData_df = getTrainingData(input_file_path + input_file_name)
	
	return modelData_df.to_json(orient='records')
 
def runModel2(userData):
	'''
		Collecting train and test data
	'''
	vAR_df = collectTrainingdataForModel2(userData)
		
	#vAR_df_test = getTestData(input_file_path + input_file_name)
	
	'''
		Preparing the data for Model
	'''	
	final_dataset = prepareAndRunModel(vAR_df, 2)
	'''
		Provisioning the data by exporting to a file and displaying on the screen
	'''
	output_file_path = getConfig('PROBLEM_2_CONFIGURATION', 'OUTPUT_FILE_PATH')
	output_file_name = getConfig('PROBLEM_2_CONFIGURATION', 'OUTPUT_FILE_NAME')
	final_dataset.to_csv(output_file_path+getTimeString()+output_file_name, sep='\t')

	return final_dataset.to_json(orient='records')

def fetchModel3Data():

	# read values from a section
	input_file_path = getConfig('PROBLEM_3_CONFIGURATION', 'INPUT_FILE_PATH')
	input_file_name = getConfig('PROBLEM_3_CONFIGURATION', 'INPUT_FILE_NAME')
	modelData_df = getTrainingData(input_file_path + input_file_name)
	
	return modelData_df.to_json(orient='records')
 
def runModel3(userData):
	'''
		Collecting train and test data
	'''
	vAR_df = collectTrainingdataForModel3(userData)
		
	#vAR_df_test = getTestData(input_file_path + input_file_name)
	'''
		Preparing the data for Model
	'''	
	final_dataset = prepareAndRunModel(vAR_df, 3)
	'''
		Provisioning the data by exporting to a file and displaying on the screen
	'''
	output_file_path = getConfig('PROBLEM_3_CONFIGURATION', 'OUTPUT_FILE_PATH')
	output_file_name = getConfig('PROBLEM_3_CONFIGURATION', 'OUTPUT_FILE_NAME')
	final_dataset.to_csv(output_file_path+getTimeString()+output_file_name, sep='\t')

	return final_dataset.to_json(orient='records')	
	
def fetchModel4Data():

	# read values from a section
	input_file_path = getConfig('PROBLEM_4_CONFIGURATION', 'INPUT_FILE_PATH')
	input_file_name = getConfig('PROBLEM_4_CONFIGURATION', 'INPUT_FILE_NAME')
	modelData_df = getTrainingData(input_file_path + input_file_name)
	
	return modelData_df.to_json(orient='records')
 
def runModel4(userData):
	'''
		Collecting train and test data
	'''
	vAR_df = collectTrainingdataForModel4(userData)
		
	#vAR_df_test = getTestData(input_file_path + input_file_name)
	'''
		Preparing the data for Model
	'''	
	final_dataset = prepareAndRunModel(vAR_df, 4)
	'''
		Provisioning the data by exporting to a file and displaying on the screen
	'''
	output_file_path = getConfig('PROBLEM_4_CONFIGURATION', 'OUTPUT_FILE_PATH')
	output_file_name = getConfig('PROBLEM_4_CONFIGURATION', 'OUTPUT_FILE_NAME')
	final_dataset.to_csv(output_file_path+getTimeString()+output_file_name, sep='\t')
	return final_dataset.to_json(orient='records')	
	
def fetchModel5Data():

	# read values from a section
	input_file_path = getConfig('PROBLEM_5_CONFIGURATION', 'INPUT_FILE_PATH')
	input_file_name = getConfig('PROBLEM_5_CONFIGURATION', 'INPUT_FILE_NAME')
	modelData_df = getTrainingData(input_file_path + input_file_name)
	
	return modelData_df.to_json(orient='records')
 
def runModel5(userData):
	'''
		Collecting train and test data
	'''
	vAR_df = collectTrainingdataForModel5(userData)
		
	#vAR_df_test = getTestData(input_file_path + input_file_name)
	'''
		Preparing the data for Model
	'''	
	final_dataset = prepareAndRunModel(vAR_df, 5)
	'''
		Provisioning the data by exporting to a file and displaying on the screen
	'''
	output_file_path = getConfig('PROBLEM_5_CONFIGURATION', 'OUTPUT_FILE_PATH')
	output_file_name = getConfig('PROBLEM_5_CONFIGURATION', 'OUTPUT_FILE_NAME')
	
	final_dataset.to_csv(output_file_path+getTimeString()+output_file_name, sep='\t')
	
	return final_dataset.to_json(orient='records')	

def fetchModel6Data():

	# read values from a section
	input_file_path = getConfig('PROBLEM_6_CONFIGURATION', 'INPUT_FILE_PATH')
	input_file_name = getConfig('PROBLEM_6_CONFIGURATION', 'INPUT_FILE_NAME')
	modelData_df = getTrainingData(input_file_path + input_file_name)
	
	return modelData_df.to_json(orient='records')

# function to run all models and consolidate the transactions
def runModel6(userData):
	vAR_df1 = runModel1(userData)
	vAR_df2 = runModel2(pd.read_json(vAR_df1))
	vAR_df3 = runModel3(pd.read_json(vAR_df2))
	vAR_df4 = runModel4(pd.read_json(vAR_df3))
	vAR_df5 = runModel5(pd.read_json(vAR_df4))
	
	#write the outcome to the file
	output_file_path = getConfig('PROBLEM_6_CONFIGURATION', 'OUTPUT_FILE_PATH')
	output_file_name = getConfig('PROBLEM_6_CONFIGURATION', 'OUTPUT_FILE_NAME')
	
	#vAR_df5.to_csv(output_file_path+output_file_name, sep='\t')
	'''
		Provisioning the data by exporting to a file and displaying on the screen
	'''
	return vAR_df5

#utility funcion to get the config	
def getConfig(section, key):
	try:
		from configparser import ConfigParser
	except ImportError:
		from ConfigParser import ConfigParser

	# instantiate
	config = ConfigParser()
	config.read('C:\\Ramesh\\AI\\INTCO_Transaction\\configuration\\CONFIGURATION.ini')

	return config.get(section, key)

'''
	Function to collect Traning data
'''
def getTrainingData(dataPath):
	modelData = aggregateTrainingData(dataPath)
	return modelData
'''
	Function to aggregate Traning data
'''	
def aggregateTrainingData(dataPath):
	fileData = collectDataFromFile(dataPath)
	hadoopData = collectDataFromHadoop()
	SAPData = collectDataFromSAP()
	oracleData = collectDataFromOracle()
	mySQLData = collectDataFromMySQL()
	
	aggregatedDF = None
	
	if fileData is not None:
		aggregatedDF = fileData
	if hadoopData is not None:
		aggregatedDF = aggregatedDF.merge(hadoopData,left_index=True, right_index=True)
	if SAPData is not None:
		aggregatedDF = aggregatedDF.merge(SAPData,left_index=True, right_index=True)	
	if oracleData is not None:
		aggregatedDF = aggregatedDF.merge(oracleData,left_index=True, right_index=True)			
	if mySQLData is not None:
		aggregatedDF = aggregatedDF.merge(mySQLData,left_index=True, right_index=True)		
	
	return fileData
'''
	Function to get Traning data 
'''	
def collectDataFromFile(dataPath):
	vAR_df = pd.read_excel(dataPath)
	return vAR_df
'''
	Function to get Traning data from Hadoop file system(HDFS)
'''		
def collectDataFromHadoop():
	return None
'''
	Function to get Traning data from SAP
'''		
def collectDataFromSAP():
	return None
'''
	Function to get Traning data from Oracle
'''		
def collectDataFromOracle():
	return None
'''
	Function to get Traning data from MySQL
'''		
def collectDataFromMySQL():
	return None
	
'''
	Function to get Test data
'''		
def getTestData(dataPath):
	modelData = aggregateTestData(dataPath)
	return modelData
'''
	Function to aggregate Test data
'''		
def aggregateTestData(dataPath):
	fileData = collectTestDataFromFile(dataPath)
	hadoopData = collectTestDataFromHadoop()
	SAPData = collectTestDataFromSAP()
	oracleData = collectTestDataFromOracle()
	mySQLData = collectTestDataFromMySQL()
	
	aggregatedDF = None
	
	if fileData is not None:
		aggregatedDF = fileData
	if hadoopData is not None:
		aggregatedDF = aggregatedDF.merge(hadoopData,left_index=True, right_index=True)
	if SAPData is not None:
		aggregatedDF = aggregatedDF.merge(SAPData,left_index=True, right_index=True)	
	if oracleData is not None:
		aggregatedDF = aggregatedDF.merge(oracleData,left_index=True, right_index=True)			
	if mySQLData is not None:
		aggregatedDF = aggregatedDF.merge(mySQLData,left_index=True, right_index=True)		
	
	return fileData
'''
	Function to get Traning data from file
'''	
def collectTestDataFromFile(dataPath):
	vAR_df = pd.read_excel(dataPath)
	return vAR_df
'''
	Function to get Traning data from Hadoop file system(HDFS)
'''		
def collectTestDataFromHadoop():
	return None
'''
	Function to get Traning data from SAP
'''			
def collectTestDataFromSAP():
	return None
'''
	Function to get Traning data from Oracle
'''			
def collectTestDataFromOracle():
	return None
'''
	Function to get Traning data from MySQL
'''			
def collectTestDataFromMySQL():
	return None	
	
def collectTrainingdataForModel1(userData):
	input_file_path = getConfig('PROBLEM_1_CONFIGURATION', 'INPUT_FILE_PATH')
	input_file_name = getConfig('PROBLEM_1_CONFIGURATION', 'INPUT_FILE_NAME')
	
	if userData.shape[0]>0:
		return userData
	else:
		return getTrainingData(input_file_path + input_file_name)	

def collectTrainingdataForModel2(userData):
	input_file_path = getConfig('PROBLEM_2_CONFIGURATION', 'INPUT_FILE_PATH')
	input_file_name = getConfig('PROBLEM_2_CONFIGURATION', 'INPUT_FILE_NAME')
	
	if userData.shape[0]>0:
		return userData
	else:
		return getTrainingData(input_file_path + input_file_name)		

def collectTrainingdataForModel3(userData):
	input_file_path = getConfig('PROBLEM_3_CONFIGURATION', 'INPUT_FILE_PATH')
	input_file_name = getConfig('PROBLEM_3_CONFIGURATION', 'INPUT_FILE_NAME')
	
	if userData.shape[0]>0:
		return userData
	else:
		return getTrainingData(input_file_path + input_file_name)	

def collectTrainingdataForModel4(userData):
	input_file_path = getConfig('PROBLEM_4_CONFIGURATION', 'INPUT_FILE_PATH')
	input_file_name = getConfig('PROBLEM_4_CONFIGURATION', 'INPUT_FILE_NAME')
	
	if userData.shape[0]>0:
		return userData
	else:
		return getTrainingData(input_file_path + input_file_name)	

def collectTrainingdataForModel5(userData):
	input_file_path = getConfig('PROBLEM_5_CONFIGURATION', 'INPUT_FILE_PATH')
	input_file_name = getConfig('PROBLEM_5_CONFIGURATION', 'INPUT_FILE_NAME')
	
	if userData.shape[0]>0:
		return userData
	else:
		return getTrainingData(input_file_path + input_file_name)	

'''
	Args:
        param1: the input DataFrame.
        param2: Model number to run.

    Returns:
        the dataframe which should be returned to the called
'''
def prepareAndRunModel(vAR_df, number):
	# if columns have any empty / null data, fill it with either 000000 / NULL
	modified_Company_Series = vAR_df[getConfig('PROBLEM_{}_CONFIGURATION'.format(number), 'FEATURE1')].fillna('000000')
	modified_Trading_Partner_Series = vAR_df[getConfig('PROBLEM_{}_CONFIGURATION'.format(number), 'FEATURE2')].fillna('000000')
	modified_GL_Account_Series = vAR_df[getConfig('PROBLEM_{}_CONFIGURATION'.format(number), 'FEATURE3')].fillna('000000')
	modified_Amount_Series = vAR_df[getConfig('PROBLEM_{}_CONFIGURATION'.format(number), 'FEATURE4')].fillna('000000')
	modified_Customer_Series = vAR_df[getConfig('PROBLEM_{}_CONFIGURATION'.format(number), 'FEATURE5')].fillna('NULL')
	modified_Vendor_Series = vAR_df[getConfig('PROBLEM_{}_CONFIGURATION'.format(number), 'FEATURE6')].fillna('NULL')
	modified_TT_Series = vAR_df[getConfig('PROBLEM_{}_CONFIGURATION'.format(number), 'FEATURE7')].fillna('NULL')

	modified_Company = pd.DataFrame(modified_Company_Series)
	# Convert into dataframes
	modified_Trading_Partner = pd.DataFrame(modified_Trading_Partner_Series.apply(np.int64))
	modified_GL_Account = pd.DataFrame(modified_GL_Account_Series.apply(np.int64))
	modified_Amount = pd.DataFrame(modified_Amount_Series.apply(np.int64))
	modified_Customer = pd.DataFrame(modified_Customer_Series)
	modified_Vendor = pd.DataFrame(modified_Vendor_Series)
	modified_TT = pd.DataFrame(modified_TT_Series)
	#merge with main dataframe
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

	vAR_clf = IsolationForest( max_samples=498,
	random_state=np.random.RandomState(42))
	#run model
	vAR_clf.fit(final_columns_df)
	
	#predict the score
	scores_pred = vAR_clf.decision_function(final_columns_df)
	
	# label it based the prediction score
	resultDf = pd.DataFrame(columns=['Valid Record{}'.format(number)])
	counter =0
	for n in range(len(scores_pred)):
		if (scores_pred[n] >=0.1):
			resultDf.loc[n] = 'N'
		else:
			resultDf.loc[n] = 'Y' 
	
	vAR_df = vAR_df.reset_index(drop=True)

	final_dataset = vAR_df.merge(resultDf,left_index=True, right_index=True)
	return final_dataset
def getTimeString():
	return time.strftime("%Y%m%d-%H%M%S_")
if __name__ == '__main__':
 app.run()