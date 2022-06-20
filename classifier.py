import matplotlib.pyplot as plt
import seaborn as sns

import pandas as pd  

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score


def create_dataframes(remove_weight=False, file=None):
  # reading dataset
  if file:
    dataset = pd.read_csv(file)
  else:
    dataset = pd.read_csv('diabetes.csv')

  # defining features(x) and the target(y)
  if remove_weight:
      x = pd.DataFrame(dataset.iloc[:,1:9])
  else:
    x = pd.DataFrame(dataset.iloc[:,:-1])
  y = pd.DataFrame(dataset.iloc[:,-1])

  return x, y


def split_dataset(x, y, test_size=0.2):
  # splitting the dataset into train and test 
  # using sklearn
  x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=test_size)
  return x_train, x_test, y_train, y_test


def create_classifier(n_estimators=100):
  # building the random forest classifier

  classifier = RandomForestClassifier(n_estimators=n_estimators)

  # predicting values using the random forest
  classifier.fit(x_train, y_train)
  y_pred = classifier.predict(x_test)
  return classifier, y_pred


def feature_selection(classifier, x):
  # feature selection
  # ie. Finding out important features
  feature_imp = pd.Series(classifier.feature_importances_, index=x.columns).sort_values(ascending=True)
  return feature_imp


def plot_confusion_matrix(confusion_matrix, msg="Confusion Matrix"):
    fig, ax = plt.subplots(figsize=(3,3))
    ax.matshow(confusion_matrix, cmap=plt.cm.Oranges, alpha=0.3)
    for i in range(confusion_matrix.shape[0]):
        for j in range(confusion_matrix.shape[1]):
            ax.text(x=j, y=i,s=confusion_matrix[i, j], va='center', ha='center', size='xx-large')
    
    plt.xlabel('Predictions', fontsize=18)
    plt.ylabel('Actuals', fontsize=18)
    plt.title(msg, fontsize=18)
    plt.show()


def classification_reports(y_test, y_pred):
  confusion_matrix_test = confusion_matrix(y_test, y_pred)

  _classification_report = classification_report(y_test, y_pred)
  _accuracy_score = accuracy_score(y_test, y_pred)

  print(f'------- Confusion Matrix -------')
  plot_confusion_matrix(confusion_matrix_test)
  print(f'------- Reports -------')
  print(_classification_report)
  print(f'------- Accuracy Score -------')
  print(_accuracy_score)


def visualize_important_features(feature_imp):
  # visualizing the important features
  # creating bar graph
  sns.barplot(x=feature_imp, y=feature_imp.index)
  plt.xlabel('Feature Importance Score')
  plt.ylabel('Features')
  plt.title('Visualizing Important Features')
  plt.show()


def random_forest_classifier(n_decision_trees=100, remove_weight=False):
  classifier, y_pred = create_classifier(n_estimators=n_decision_trees)
  feature_imp = feature_selection(classifier, x)

  print(f'*'*50)
  print(f'------- Printing important features -------')
  print(feature_imp)
  print(f'*'*50)
  print(f'------- Printing classification reports -------')
  classification_reports(y_test, y_pred)
  print(f'*'*50)
  print(f'------- Visualizing important features -------')
  visualize_important_features(feature_imp)
  print(f'*'*50)


def analyze_classifier_with_variable_dts():
    # Analyzing Random Forest results with multiple sized decision trees

    print('Algorithm starts for 10 decision trees\n \n')
    random_forest_classifier(10)
    print('\n \n Algorithm Ends for 10 decision trees')

    print('#'*100)

    print('Algorithm starts for 50 decision trees\n \n')
    random_forest_classifier(50)
    print('\n \n Algorithm Ends for 50 decision trees')
    print('#'*200)

    print('Algorithm starts for 100 decision trees\n \n')
    random_forest_classifier(100)
    print('\n \n Algorithm Ends for 100 decision trees')


def predict_result_from_realworld():
    # predict from user input data
    input_data = [1,131,64,14,415,23.7,0.389,21]
    i_d = [3,93,60,31,0,42.5,0.315,33]
    # actual class = 0

    x,y = create_dataframes()
    x_train, x_test, y_train, y_test = split_dataset(x, y, test_size=0.2)
    classifier, y_pred = create_classifier()

    predicted_class = classifier.predict([i_d])

    print('Actual class => ', 0)
    print('Predicted class => ', predicted_class[0])


if __name__ == '__main__':
    print('Program Started')
    x,y = create_dataframes()
    x_train, x_test, y_train, y_test = split_dataset(x, y, test_size=0.1)

    analyze_classifier_with_variable_dts()
    predict_result_from_realworld()
    print('Program Ended')
