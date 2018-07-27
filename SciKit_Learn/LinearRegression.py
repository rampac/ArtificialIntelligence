# Step 1 - Import the Required Libraries

#Our Model Implementation needs the Following Libraries:

#Sklearn: Sklearn is the Machine Learning Library which is used for numerical & scientific computations.

#Pandas: Pandas is a library used for data manipulation and analysis. 

#In Our Implementation. we are using it for Importing the Data file & Creating Dataframes (Stores the Data).

import pandas as vAR_pd
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression

#>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>#

# Step 2 - Import Training Data

#Next step after importing all libraries is getting the Training data imported. 

#We are importing the Clustering data stored in our local system with the use of Pandas library.

vAR_df = vAR_pd.read_excel('C:/AI_Bootcamp/Problem4/Problem4_Intercompany_Transaction_Weighted_Accuracy_Intercompany_Transcation_Train_Converted.xlsx')
vAR_df.head(3)

#>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>#

# Step 3 – Convert Categorical Data into Numerical Values using Label Encoder 

# Next Step of the Implementation is Convertion of Categorical Data into Numerical Values & Feature Selection for Clustering.

vAR_Features_Train = vAR_df[['Company_Code','Trading_Partner','Trading_Partner_Country','Transaction_Type','Data_Category']]
vAR_Labels_Train = vAR_df.iloc[:,12]

#>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>#

# Step 4 – Train the Model

# Training the data means Making the model to Learn, understand & recognize the Pattern in the data.

vAR_model = LinearRegression()
vAR_model.fit(vAR_Features_Train,vAR_Labels_Train)

#>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>#

# Step 5 – Review Learning Algorithm

# We Review the Algorithm as to see how it has learned from the Features we Provided
vAR_model.predict(vAR_Features_Train).astype(int)

#>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>#

# Step 6 - Import Test Data

# Importing the Test Data is to check how the data used on the Model Performs

vAR_df6 = vAR_pd.read_excel('C:/AI_Bootcamp/Problem4/Problem4_Intercompany_Transaction_Weighted_Accuracy_Intercompany_Transcation_Test_Converted.xlsx')
vAR_df6.head(5)
vAR_Features_Test = vAR_df6[['Company_Code','Trading_Partner','Trading_Partner_Country','Transaction_Type','Data_Category']]

#>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>#

# Step 7 – Running Model on Test Data

# Running the Model on Test Data is to use the imported test data to Prodict our Outcome

vAR_Labels_Pred = vAR_model.predict(vAR_Features_Test).astype(int)

#>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>#

# Step 8 – Review Model Outcome

# We check the Output of Model i.e the Prediction it has made on the test data

vAR_Labels_Pred = vAR_pd.DataFrame(vAR_Labels_Pred,columns={'Predicted_Inter_Transaction_Weighted_Accuracy'})
#vAR_Features_test = vAR_Features_test.sort()

#>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>#

# Step 9 - Write Model Outcome to File

# Write the Model Output to an excel file for analysis.

vAR_df7 = vAR_pd.read_excel('C:/AI_Bootcamp/Problem4/Problem4_Intercompany_Transaction_Weighted_Accuracy_Intercompany_Transcation_Test_Actual.xlsx',)
vAR_df8 = vAR_df7.iloc[:,:-1]
vAR_df10 = vAR_df8.merge(vAR_Labels_Pred,left_index=True, right_index=True)
vAR_df11 = vAR_df10.to_excel('C:/AI_Bootcamp/Problem4/Problem4_Predicted_Results.xlsx')

#>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>#

# Step 10 - To open and view the file outcome

# Open the Written File &amp; Check the Outcome as Shown. Execute to View the data

vAR_df12 = vAR_pd.read_excel('C:/AI_Bootcamp/Problem4/Problem4_Predicted_Results.xlsx')
vAR_df12

#>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>#