import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import os
import matplotlib.pyplot as plt


class get_data():
        
    def __init__(self, based_on=None) -> None:
        self.generate_data()
        self.based_on = based_on
        # self.merge_data()
        # self.create_X_y(q=5)
        # self.split(q=0.2,random_state=42)
        
                
    def generate_data(self):
        # Get the current working directory
        current_directory = os.getcwd()

        # Get the parent directory of the current directory
        parent_directory = os.path.dirname(current_directory)

        # Get the path of the project directory by going two levels up
        project_directory = os.path.abspath(os.path.join(parent_directory, ".."))

        self.df_features = pd.read_csv(os.path.join(project_directory, 'data/features/features_interpolated.csv'), encoding='latin-1', engine='python')
        self.df_features.drop(columns=['Unnamed: 0'], inplace=True)
        self.immigration = pd.read_csv(os.path.join(project_directory, 'data/labels/OECD_acquisition_data_interpolated.csv'), encoding='latin-1', engine='python')
        self.immigration.drop(columns=['Unnamed: 0'], inplace=True)
        
        
    def merge_data(self,vis=False):
        if self.based_on!=None:
            # merge two dataset together
            self.df_merged = pd.merge(self.df_features, self.immigration, left_on=['Country', 'Year'], right_on=[self.based_on, 'Year'])
            self.df_merged.drop(columns=[self.based_on], inplace=True)
        else:
            # Merge the first two datasets based on country and year
            self.df_merged  = pd.merge(self.df_features, self.immigration, left_on=["Country", "Year"], right_on=["COU", "Year"])

            # Merge the third dataset based on country and year
            self.df_merged  = pd.merge(self.df_features, self.df_merged , left_on=["Country", "Year"], right_on=["CO2", "Year"])

            if not vis:
                self.df_merged.drop(columns=["Country_x","Country_y","COU", "CO2"], inplace=True)
        
        
    def create_X_y(self,q=5,mode="class"):
        y = self.df_merged['Value']  # Target variable
        
        if self.based_on!=None:
            if self.based_on == "CO2":
                X = self.df_merged.drop(['Value','Country',"COU"], axis=1)  # Features
            else:
                X = self.df_merged.drop(['Value','Country',"CO2"], axis=1)

            # Normalize the data between 0 and 1
            scaler = MinMaxScaler()
            self.normalized_features = scaler.fit_transform(X)

        else:                
            # Normalize the data between 0 and 1
            scaler = MinMaxScaler()
            self.normalized_features = scaler.fit_transform(self.df_merged.drop(columns=["Value"]))
            
        if mode=='class':
            # Perform binning to split the values into classes based on the distribution
            self.class_labels = pd.qcut(y, q=q, labels=False,duplicates='drop')
        else:
            self.class_labels = scaler.fit_transform(y.values.reshape(-1, 1))
            
            
    def split(self,test_size=0.2, random_state=42):
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.normalized_features, 
                                                                        self.class_labels, 
                                                                        test_size=test_size, 
                                                                        random_state=random_state)
        
    def vis_bins(self):
        if self.mode=='class':
            # Compute the bin frequencies
            bin_counts = np.bincount(self.class_labels)

            # Plot the bar plot of the bin frequencies
            plt.bar(range(len(bin_counts)), bin_counts)
            plt.xlabel('Bins')
            plt.ylabel('Frequency')
            plt.title('Distribution of Bins')
            plt.show()
        else:
            print("Not a classification model")
        
    