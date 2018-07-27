# Import the Required Libraries

import pandas as vAR_pd
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression

# Import Training Data

vAR_df = vAR_pd.read_excel('C:/Users/rpachiannan/PycharmProjects/AI_GIT/ArtificialIntelligence/data/4 - Intercompany_Transaction_Weighted_Accuracy_Intercompany_Transcation_Train_Converted.xlsx')
print(vAR_df.head(3))

# Convert Categorical Data into Numerical Values using Label Encoder

# Convertion of Categorical Data into Numerical Values & Feature Selection for Clustering.

vAR_Features_Train = vAR_df[['Company_Code','Trading_Partner','Trading_Partner_Country','Transaction_Type','Data_Category']]
vAR_Labels_Train = vAR_df.iloc[:,12]

# Train the Model(Making the model to Learn, understand & recognize the Pattern in the data.)

vAR_model = LinearRegression()
vAR_model.fit(vAR_Features_Train,vAR_Labels_Train)

# Review Learning Algorithm

# We Review the Algorithm as to see how it has learned from the Features we Provided
vAR_model.predict(vAR_Features_Train).astype(int)

# Import Test Data to check how the data used on the Model Performs

vAR_df6 = vAR_pd.read_excel('C:/Users/rpachiannan/PycharmProjects/AI_GIT/ArtificialIntelligence/data/4 - Intercompany_Transaction_Weighted_Accuracy_Intercompany_Transcation_Test_Converted.xlsx')
vAR_df6.head(5)
vAR_Features_Test = vAR_df6[['Company_Code','Trading_Partner','Trading_Partner_Country','Transaction_Type','Data_Category']]

# Run Model on Test Data to use the imported test data to Prodict our Outcome

vAR_Labels_Pred = vAR_model.predict(vAR_Features_Test).astype(int)

# Review Model Outcome

# We check the Output of Model i.e the Prediction it has made on the test data

vAR_Labels_Pred = vAR_pd.DataFrame(vAR_Labels_Pred,columns={'Predicted_Inter_Transaction_Weighted_Accuracy'})
#vAR_Features_test = vAR_Features_test.sort()

# Write Model Outcome to File

vAR_df7 = vAR_pd.read_excel('C:/Users/rpachiannan/PycharmProjects/AI_GIT/ArtificialIntelligence/data/4 - Intercompany_Transaction_Weighted_Accuracy_Intercompany_Transcation_Test_Converted.xlsx',)
vAR_df8 = vAR_df7.iloc[:,:-1]
vAR_df10 = vAR_df8.merge(vAR_Labels_Pred,left_index=True, right_index=True)
vAR_df11 = vAR_df10.to_excel('C:/Users/rpachiannan/PycharmProjects/AI_GIT/ArtificialIntelligence/data/Problem4_Predicted_Results.xlsx')

# Step 10 - To open and view the file outcome

# Open the Written File &amp; Check the Outcome as Shown. Execute to View the data

vAR_df12 = vAR_pd.read_excel('C:/Users/rpachiannan/PycharmProjects/AI_GIT/ArtificialIntelligence/data/Problem4_Predicted_Results.xlsx')

