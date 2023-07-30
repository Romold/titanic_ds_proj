import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
import seaborn as sns 

training = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")

training['train_test'] = 1 # adding a new columnn and filling it's rows with number 1 
test['train_test'] = 0 

test['Survived'] = np.NaN

all_data = pd.concat([training , test]) #concatinate these two dataframes 

training.info() #Information about each of the columns 
print(training.describe()) #to take a look at the SDs 

#then we need to look at numeric and categorical values separately (numeric with histos and categoricals with value counts)

df_num = training[['Age' , 'SibSp' , 'Parch' , 'Fare']] #9:43 
df_cat = training[['Survived' , 'Pclass' , 'Sex' , 'Ticket' , 'Cabin' , 'Embarked']]

for i in df_num.columns:
    plt.hist(df_num[i])
    plt.title(i)
    plt.show() #plotting histogram for every Column in the df_num dataframe
    
print(df_num.corr()) #for correlations 
sns.heatmap(df_num.corr())

#Now we look at how survival rates differ amongs these group 

pd.pivot_table(training , index = 'Survived' , values = ['Age' , 'SibSp' , 'Parch' , 'Fare'])

#Now we do the analysis for the categorical variables using barplots
for i in df_cat.columns:
    sns.barplot(x = df_cat[i].value_counts().index , y=df_cat[i].value_counts()).set_title(i)
    plt.show()

#update the comments for the below lines of code 
print(pd.pivot_table(training , index = 'Survived' , columns='Pclass' , values = 'Ticket' , aggfunc='count'))
print('\n')
print(pd.pivot_table(training , index = 'Survived' , columns= 'Sex' , values = 'Ticket' , aggfunc='count' ))
print('\n')
print(pd.pivot_table(training , index = 'Survived' , columns = 'Embarked' , values = 'Ticket' , aggfunc='count'))

