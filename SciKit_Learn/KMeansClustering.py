# import libraries
import pandas as vAR_pd
from sklearn.preprocessing import LabelEncoder
from sklearn.cluster import KMeans

# read excel file with data
var_df = vAR_pd.read_excel("C:/Ramesh/AI/2_AI Boot Camp - Problem2_Pair_Intercompany_Transaction_Train.xlsx")

#  initialize LabelEncoder()
var_le = LabelEncoder()

# convert Char to numeric
var_transaction_type_conversion = var_le.fit_transform(var_df.iloc[:,7])

var_transaction_type_conversion_df = vAR_pd.DataFrame(var_transaction_type_conversion, columns={'Transaction_Type_Converted'})

var_data_category_conversion = var_le.fit_transform(var_df.iloc[:,9])

var_data_category_conversion_df = vAR_pd.DataFrame(var_data_category_conversion, columns={'Data_Category_Converted'})

var_doc_data_conversion = var_le.fit_transform(var_df.iloc[:,3])

var_doc_data_conversion_df = vAR_pd.DataFrame(var_doc_data_conversion, columns={'Document_Data_Converted'})

# merge newly formed columns
var_df1 = var_df.merge(var_transaction_type_conversion_df, left_index=True, right_index=True)
var_df2 = var_df1.merge(var_data_category_conversion_df, left_index=True, right_index=True)
var_df3 = var_df2.merge(var_doc_data_conversion_df, left_index=True, right_index=True)

var_df4 = var_df3[['Company', 'Document_Date', 'Trading_Partner', 'Transaction_Type']]

var_model = KMeans(n_clusters=12, random_state=0)

var_model.fit(var_df4)

var_class = var_model.labels_
var_class_def = vAR_pd.DataFrame(var_class, columns={'New_Group'})

var_model.predict(var_df4)

var_df6 = vAR_pd.read_excel("C:/Ramesh/AI/2_AI Boot Camp - Problem2_Pair_Intercompany_Transaction_Test_Converted.xlsx")

var_df7 = var_df6[['Company', 'Document_Date','Trading_Partner','Transaction_Type']]

var_features_test = var_df7

var_labels_prediction = var_model.predict(var_features_test)

var_labels_prediction = vAR_pd.DataFrame(var_labels_prediction, columns={'predicted_inter_transaction_pair'})

var_df8 = vAR_pd.read_excel('C:/Ramesh/AI/2_AI Boot Camp - Problem2_Pair_Intercompany_Transaction_Train_Actual.xlsx')

var_df9 = var_df8.merge(var_labels_prediction, left_index=True, right_index=True)

var_df10 = var_df9.to_excel('C:/Ramesh/AI/Predicted_Results.xlsx')

var_df11 = vAR_pd.read_excel('C:/Ramesh/AI/Predicted_Results.xlsx')














