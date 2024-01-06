import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, accuracy_score

df = pd.read_csv('train.csv') 
df.drop(['bdate', 'has_mobile', 'graduation', 'langs', 'life_main', 'people_main', 'city', 'last_seen', 'career_start', 'career_end', 'occupation_name'], axis = 1, inplace = True)

df['education_form'].fillna(0, inplace = True)
def edu_form(form):
    if form == 'Full-time':
        return 1
    elif form == 'Distance Learning':
        return 2
    elif form == 'Part-time':
        return 3
    elif form == 'External':
        return 4
    return 0
df['education_form'] = df['education_form'].apply(edu_form)

def get_edu_status(status):
    if status == "Undergraduate applicant":
        return 1
    elif status == "Student (Bachelor's)":
        return 2
    elif status == "Alumnus (Bachelor's)":
        return 3
    elif status == "Student (Master's)":
        return 4
    elif status == "Alumnus (Master's)":
        return 5
    elif status == "Student (Specialist)":
        return 6
    elif status == "Alumnus (Specialist)":
        return 7
    elif status == "Phd":
        return 8
    elif status == "Candidate of Sciences":
        return 9
    return 0
df['education_status'] = df['education_status'].apply(get_edu_status)

def get_occupation(row):
    if row == 'school':
        return 1
    elif row == 'work':
        return 2
    return 3
df['occupation_type'] = df['occupation_type'].apply(get_occupation)

X = df.drop('result', axis = 1)
y = df['result']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3)

sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

classifier = KNeighborsClassifier(n_neighbors = 3)
classifier.fit(X_train, y_train)


y_pred = classifier.predict(X_test)
percent = accuracy_score(y_test, y_pred) * 100

print(percent)

'''
df2 = pd.read_csv('test.csv') 
ID = df2['id']
result = pd.DataFrame({'id' : ID, 'result' : y_pred})
result.to_csv('result', index = False)'''