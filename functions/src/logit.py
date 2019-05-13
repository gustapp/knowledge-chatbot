#%%
# Parameters
# religion       : judaism
# religion       : christian
# cause_of_death : crucifixion 
target_rel = 'cause_of_death'
label = 'crucifixion'

#%%
# Load data
import pandas as pd

df = pd.read_csv('./full_topzera.csv', index_col=0)
df.head()

#%%
# Replace target tail
def replace_target(item, label=label):
    if item == label:
        return 1
    else:
        return 0

df[target_rel] = df[target_rel].apply(replace_target)
target = df.pop(target_rel)

df.head()

#%%
# Encode labels to categorical features
from sklearn.preprocessing import LabelEncoder

intrp_label = []
for column in df:
    le = LabelEncoder()
    df[column] = le.fit_transform(df[column])
    intrp_label += map(lambda x: '{}:{}'.format(column, x), list(le.classes_))

df.head()

# #%%
# # Split target
# target = df.pop(target_rel)

#%%
# Encode one hot 
import numpy as np
from sklearn.preprocessing import OneHotEncoder

ohc = OneHotEncoder()
out = ohc.fit_transform(df)

#%%
# Full set
X = out
y = target

#%%
from sklearn.metrics import mean_squared_error, accuracy_score
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

#%%
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
logreg = LogisticRegression()
logreg.fit(X_train, y_train)

#%%
y_pred = logreg.predict(X_test)
print('Accuracy of logistic regression classifier on test set: {:.2f}'.format(logreg.score(X_test, y_test)))

#%%
from sklearn.metrics import confusion_matrix
confusion_matrix = confusion_matrix(y_test, y_pred)
print(confusion_matrix)

#%%
# Feature Importance (why jesus religion is judaism?)
weights = logreg.coef_
labels = intrp_label

exp_df = pd.DataFrame(data={'labels': labels, 'weights': weights[0]})
exp_df.sort_values('weights', inplace=True, ascending=False)
exp_df.head(3)