## Wrapper Methods

#In this project, you'll analyze data from a survey conducted by Fabio Mendoza Palechor and Alexis de la Hoz Manotas that asked people about their eating habits and weight. The data was obtained from the [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/Estimation+of+obesity+levels+based+on+eating+habits+and+physical+condition+). Categorical variables were changed to numerical ones in order to facilitate analysis.

#First, you'll fit a logistic regression model to try to predict whether survey respondents are obese based on their answers to questions in the survey. After that, you'll use three different wrapper methods to choose a smaller feature subset.

#You'll use sequential forward selection, sequential backward floating selection, and recursive feature elimination. After implementing each wrapper method, you'll evaluate the model accuracy on the resulting smaller feature subsets and compare that with the model accuracy using all available features.


# Import libraries
import pandas as pd
from sklearn.linear_model import LogisticRegression
from mlxtend.feature_selection import SequentialFeatureSelector as SFS
from mlxtend.plotting import plot_sequential_feature_selection as plot_sfs
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import RFE


## Evaluating a Logistic Regression Model

#The data set `obesity` contains 18 predictor variables. Here's a brief description of them.

#* `Gender` is `1` if a respondent is male and `0` if a respondent is female.
#* `Age` is a respondent's age in years.
#* `family_history_with_overweight` is `1` if a respondent has family member who is or was overweight, `0` if not.
#* `FAVC` is `1` if a respondent eats high caloric food frequently, `0` if not.
#* `FCVC` is `1` if a respondent usually eats vegetables in their meals, `0` if not.
#* `NCP` represents how many main meals a respondent has daily (`0` for 1-2 meals, `1` for 3 meals, and `2` for more than 3 meals).
#* `CAEC` represents how much food a respondent eats between meals on a scale of `0` to `3`.
#* `SMOKE` is `1` if a respondent smokes, `0` if not.
#* `CH2O` represents how much water a respondent drinks on a scale of `0` to `2`.
#* `SCC` is `1` if a respondent monitors their caloric intake, `0` if not.
#* `FAF` represents how much physical activity a respondent does on a scale of `0` to `3`.
#* `TUE` represents how much time a respondent spends looking at devices with screens on a scale of `0` to `2`.
#* `CALC` represents how often a respondent drinks alcohol on a scale of `0` to `3`.
#* `Automobile`, `Bike`, `Motorbike`, `Public_Transportation`, and `Walking` indicate a respondent's primary mode of transportation. Their primary mode of transportation is indicated by a `1` and the other columns will contain a `0`.

#The outcome variable, `NObeyesdad`, is a `1` if a patient is obese and a `0` if not.

#Use the `.head()` method and inspect the data.

# https://archive.ics.uci.edu/ml/datasets/Estimation+of+obesity+levels+based+on+eating+habits+and+physical+condition+

# Load the data
obesity = pd.read_csv("obesity.csv")

# Inspect the data
obesity.head()

### Split the data into `X` and `y`

# Split the data into predictor variables and an outcome variable
X = obesity.drop(["NObeyesdad"], axis=1)
y = obesity.NObeyesdad

#Create a logistic regression model called `lr`. Include the parameter `max_iter=1000` to make sure that the model will converge when you try to fit it.
# Create a logistic regression model
lr = LogisticRegression(max_iter=1000)

# Fit the logistic regression model
lr.fit(X, y)

# Print the accuracy of the model
print(lr.score(X,y))

#output
0.7659876835622927

## Sequential Forward Selection

#Now that you've created a logistic regression model and evaluated its performance, you're ready to do some feature selection. 

#Create a sequential forward selection model called `sfs`. 
#* Be sure to set the `estimator` parameter to `lr` and set the `forward` and `floating` parameters to the appropriate values. 
#* Also use the parameters `k_features=9`, `scoring='accuracy'`, and `cv=0`.

# Create a sequential forward selection model
sfs = SFS(lr, 
          k_features=6, 
          forward=True, 
          floating=False, 
          scoring='accuracy',
          cv=0)

# Fit the sequential forward selection model to X and y
sfs.fit(X, y)

# Inspect the results of sequential forward selection
print(sfs.subsets_[6])

# See which features sequential forward selection chose
print(sfs.subsets_[6]['feature_names'])

# Print the model accuracy after doing sequential forward selection
print(sfs.subsets_[6]['avg_score'])

#output
('Age', 'family_history_with_overweight', 'FAVC', 'CAEC', 'SCC', 'FAF')
0.7693036475603979


# Plot the model accuracy as a function of the number of features used
plot_sfs(sfs.get_metric_dict())
plt.show()
#output
its graph



# Create a sequential backward selection model
sbs = SFS(lr, 
          k_features=7, 
          forward=False, 
          floating=False, 
          scoring='accuracy',
          cv=0)

# Fit the sequential backward selection model to X and y
sbs.fit(X, y)


# Inspect the results of sequential backward selection
print(sbs.subsets_[7])

# See which features sequential backward selection chose
print(sbs.subsets_[7]['feature_names'])

# Print the model accuracy after doing sequential backward selection
print(sbs.subsets_[7]['avg_score'])

# Plot the model accuracy as a function of the number of features used
plot_sfs(sbs.get_metric_dict())
plt.show()




# Get feature names
features = X.columns

# Standardize the data
X = pd.DataFrame(StandardScaler().fit_transform(X))

# Create a recursive feature elimination model
rfe = RFE(estimator=lr, n_features_to_select=6)

# Fit the recursive feature elimination model to X and y
rfe.fit(X, y)

### Inspect chosen features

#Now that you've fit the RFE model you can evaluate the results. Create a list of chosen feature names and call it `rfe_features`. You can use a list comprehension and filter the features in `zip(features, rfe.support_)` based on whether their support is `True` (meaning the model kept them) or `False` (meaning the model eliminated them).

#Hint: `[f for (f, support) in zip(features, rfe.support_) if support]` will produce the desired list of feature names.

# See which features recursive feature elimination chose
rfe_features = [f for (f, support) in zip(features, rfe.support_) if support]
print(rfe_features)
#output
['Age', 'family_history_with_overweight', 'FAVC', 'CAEC', 'SCC', 'Automobile']


# Print the model accuracy after doing recursive feature elimination
print(rfe.score(X, y))

#output
0.757934628138323

