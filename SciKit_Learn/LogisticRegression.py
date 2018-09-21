import pandas as vAR_pd
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression

vAR_df = vAR_pd.read_excel('C:/Users/rpachiannan/PycharmProjects/AI_GIT/ArtificialIntelligence/Data/1 - Intercompany_Transaction_Train.xlsx')

vAR_le = LabelEncoder()
vAR_Transaction_Type_Conversion = vAR_le.fit_transform(vAR_df.iloc[:,7])

#print(vAR_Transaction_Type_Conversion)

vAR_Transaction_Type_Conversion_df = vAR_pd.DataFrame(vAR_Transaction_Type_Conversion,columns={'Transaction_Type_Converted'})

# print(vAR_Transaction_Type_Conversion_df)

vAR_Data_Category_Conversion = vAR_le.fit_transform(vAR_df.iloc[:,9])

#print(vAR_Data_Category_Conversion)

vAR_Data_Category_Conversion_df = vAR_pd.DataFrame(vAR_Data_Category_Conversion,columns={'Data_Category_Converted'})

#print(vAR_Data_Category_Conversion_df)

vAR_df1 = vAR_df.merge(vAR_Transaction_Type_Conversion_df,left_index=True, right_index=True)
vAR_df2 = vAR_df1.merge(vAR_Data_Category_Conversion_df,left_index=True, right_index=True)

#vAR_Features = vAR_df2[['Company','Trading_Partner','Transaction_Type_Converted','Data_Category_Converted']]
#vAR_Labels = vAR_df.iloc[:,12]


vAR_Features_train = vAR_df2[['Company','Trading_Partner','Transaction_Type_Converted','Data_Category_Converted']]

# print(vAR_Features_train)

vAR_Label_train = vAR_df.iloc[:,12]

# print(vAR_Label_train)

model = LogisticRegression()
model.fit(vAR_Features_train,vAR_Label_train)

model.predict(vAR_Features_train)

vAR_df3 = vAR_pd.read_excel('C:/Users/rpachiannan/PycharmProjects/AI_GIT/ArtificialIntelligence/Data/1 - Intercompany_Transaction_Test.xlsx')
vAR_Transaction_Type_Conversion_test = vAR_le.fit_transform(vAR_df3.iloc[:,3])
vAR_Transaction_Type_Conversion_test_df = vAR_pd.DataFrame(vAR_Transaction_Type_Conversion_test,columns={'Transaction_Type_Converted'})
vAR_Data_Category_Conversion_test = vAR_le.fit_transform(vAR_df.iloc[:,4])
vAR_Data_Category_Conversion_test_df = vAR_pd.DataFrame(vAR_Data_Category_Conversion_test,columns={'Data_Category_Converted'})
vAR_df4 = vAR_df3.merge(vAR_Transaction_Type_Conversion_test_df,left_index=True, right_index=True)
vAR_df5 = vAR_df4.merge(vAR_Data_Category_Conversion_test_df,left_index=True, right_index=True)

vAR_Features_test = vAR_df5[['Company','Trading_Partner','Transaction_Type_Converted','Data_Category_Converted']]

vAR_Labels_Pred = model.predict(vAR_Features_test)

# print(vAR_Labels_Pred)

vAR_Labels_Pred = vAR_pd.DataFrame(vAR_Labels_Pred,columns={'Predicted_Inter_Transaction_Type'})
#vAR_Features_test = vAR_Labels_Pred.sort()

vAR_df6 = vAR_pd.read_excel('C:/Users/rpachiannan/PycharmProjects/AI_GIT/ArtificialIntelligence/Data/1 - Intercompany_Transaction_Test.xlsx')
vAR_df7 = vAR_df6.merge(vAR_Labels_Pred,left_index=True, right_index=True)
vAR_df8 = vAR_df7.to_excel('C:/Users/rpachiannan/PycharmProjects/AI_GIT/ArtificialIntelligence/Data/Problem1_Predicted_Results.xlsx')

vAR_df9 = vAR_pd.read_excel('C:/Users/rpachiannan/PycharmProjects/AI_GIT/ArtificialIntelligence/Data/Problem1_Predicted_Results.xlsx')
print(vAR_df9)