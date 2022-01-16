# Import Dependencies
import yaml
from joblib import dump, load
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
# Naive Bayes Approach
from sklearn.naive_bayes import MultinomialNB
# Trees Approach
from sklearn.tree import DecisionTreeClassifier
# Ensemble Approach
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
import seaborn as sn
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons

class DiseasePrediction:
    def __init__(self, model_name=None):
        try:
            with open('./config.yaml', 'r') as f:
                self.config = yaml.safe_load(f)
        except Exception as e:
            print("Error reading Config file...")

        # Verbose
        self.verbose = self.config['verbose']
        # load data TRAINING
        self.train_features, self.train_labels, self.train_df = self._load_train_dataset()
        # load data TESTING
        self.test_features, self.test_labels, self.test_df = self._load_test_dataset()
        # Feature Correlation in Training Data
        self._feature_correlation(data_frame=self.train_df, show_fig=False)
        # model name
        self.model_name = model_name
        # Model Save Path
        self.model_save_path = self.config['model_save_path']

    # Function to Load Train Dataset
    def _load_train_dataset(self):
        df_train = pd.read_csv(self.config['dataset']['training_data_path'])
        cols = df_train.columns
        cols = cols[:-2]
        train_features = df_train[cols]
        train_labels = df_train['prognosis']

        # Check for data sanity
        #assert (len(train_features.iloc[0]) == 132)
        #assert (len(train_labels) == train_features.shape[0])

        if self.verbose:
            print("Length of Training Data: ", df_train.shape)
            print("Training Features: ", train_features.shape)
            print("Training Labels: ", train_labels.shape)
        return train_features, train_labels, df_train

    # Load testing dataset from the csv file
    def _load_test_dataset(self):
        df_test = pd.read_csv(self.config['dataset']['test_data_path'])
        cols = df_test.columns
        cols = cols[:-1]
        test_features = df_test[cols]
        test_labels = df_test['prognosis']

        # Check for data correctness
        #assert (len(test_features.iloc[0]) == 132)
        #assert (len(test_labels) == test_features.shape[0])

        #if self.verbose:
        #    print("Length of Test Data: ", df_test.shape)
        #    print("Test Features: ", test_features.shape)
        #    print("Test Labels: ", test_labels.shape)
        return test_features, test_labels, df_test

    # Features Correlation
    def _feature_correlation(self, data_frame=None, show_fig=False):
        # Get Feature Correlation
        corr = data_frame.corr()
        sn.heatmap(corr, square=True, annot=False, cmap="YlGnBu")
        plt.title("Feature Correlation")
        plt.tight_layout()
        if show_fig:
            plt.show()
        plt.savefig('feature_correlation.png')

    # Dataset Train Validation Split
    def _train_val_split(self):
        X_train, X_val, y_train, y_val = train_test_split(self.train_features, self.train_labels,
                                                          test_size=self.config['dataset']['validation_size'],
                                                          random_state=self.config['random_state'])
        #if self.verbose:
        #    print("Number of Training Features: {0}\tNumber of Training Labels: {1}".format(len(X_train), len(y_train)))
        #    print("Number of Validation Features: {0}\tNumber of Validation Labels: {1}".format(len(X_val), len(y_val)))
        return X_train, y_train, X_val, y_val

    # model selection
    def select_model(self):
        if self.model_name == 'mnb':
            self.clf = MultinomialNB()
        elif self.model_name == 'decision_tree':
            self.clf = DecisionTreeClassifier(criterion=self.config['model']['decision_tree']['criterion'])
        elif self.model_name == 'random_forest':
            self.clf = RandomForestClassifier(n_estimators=self.config['model']['random_forest']['n_estimators'])
        elif self.model_name == 'gradient_boost':
            self.clf = GradientBoostingClassifier(n_estimators=self.config['model']['gradient_boost']['n_estimators'],
                                                  criterion=self.config['model']['gradient_boost']['criterion'])
        return self.clf

    # training the model
    def train_model(self):
        # get data
        #x_val,y_val -from testing data
        X_train, y_train, X_val, y_val = self._train_val_split()
        classifier = self.select_model()
        # train the model!!!!!!!
        classifier = classifier.fit(X_train, y_train)
        # trained model evaluation on testing data
        confidence = classifier.score(X_val, y_val)
        # validation data prediction
        y_pred = classifier.predict(X_val)
        # Model Validation Accuracy!!! as parameters: data and prediction
        accuracy = accuracy_score(y_val, y_pred)
        # Model Confusion Matrix; from data given for testing and the prediction data
        conf_mat = confusion_matrix(y_val, y_pred)
        # Model Classification Report
        clf_report = classification_report(y_val, y_pred)
        # Model Cross Validation Score
        score = cross_val_score(classifier, X_val, y_val, cv=3)

        if self.verbose:
            print('\nTraining Accuracy: ', confidence)
            print('\nValidation Prediction: ', y_pred)
            print('\nValidation Accuracy: ', accuracy)
            print('\nValidation Confusion Matrix: \n', conf_mat)
            print('\nCross Validation Score: \n', score)
            print('\nClassification Report: \n', clf_report)

        # save trained model
        dump(classifier, str(self.model_save_path + self.model_name + ".joblib"))

    # Function to Make Predictions on Test Data
    def make_prediction(self, saved_model_name=None, test_data=None):
        try:
            # Load Trained Model
            clf = load(str(self.model_save_path + saved_model_name + ".joblib"))
        except Exception as e:
            print("Model not found...")

        if test_data is not None:
            result = clf.predict(test_data)
            return result
        else:
            result = clf.predict(self.test_features)
        accuracy = accuracy_score(self.test_labels, result)
        clf_report = classification_report(self.test_labels, result)
        return accuracy, clf_report

    def comparison(self):
        name = ["Naive Bayes",
                "Decision Tree",
                "Random Forest",
                "Gradient Boost"]
        classifiers = [
            MultinomialNB(),
            DecisionTreeClassifier(criterion=self.config['model']['decision_tree']['criterion']),
            RandomForestClassifier(n_estimators=self.config['model']['random_forest']['n_estimators']),
            GradientBoostingClassifier(n_estimators=self.config['model']['gradient_boost']['n_estimators'],
                                                  criterion=self.config['model']['gradient_boost']['criterion'])
            ]



def chooseModel(current_model_name):
    dp = DiseasePrediction(model_name=current_model_name)
    # train the Model
    dp.train_model()
    # get Model Performance on Test Data
    test_accuracy, classification_report = dp.make_prediction(saved_model_name=current_model_name)
    print("Model Test Accuracy:", test_accuracy)
    print("Test Data Classification Report: \n", classification_report)
    return test_accuracy



if __name__ == "__main__":
    #mnb naive bayes
    #decision_tree
    #random_forest
    #gradient_boost
    #mnb_accuracy=chooseModel('mnb')
    decision_tree_accuracy = chooseModel('decision_tree')
    #random_forest_accuracy=chooseModel('random_forest')
    #gradient_boost_accuracy=chooseModel('gradient_boost')
    #print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
   # print("mnb accuracy:",mnb_accuracy)
   # print("decision tree accuracy",decision_tree_accuracy)
   # print("random forest accuracy",random_forest_accuracy)
   # print("gradient boost accuracy",gradient_boost_accuracy)








    # Model Currently Training
    #current_model_name = 'random_forest'
    # Instantiate the Class
    #dp = DiseasePrediction(model_name=current_model_name)
    # Train the Model
    #dp.train_model()
    # Get Model Performance on Test Data
    #test_accuracy, classification_report = dp.make_prediction(saved_model_name=current_model_name)
    #print("Model Test Accuracy:", test_accuracy)
    #print("Test Data Classification Report: \n", classification_report)
