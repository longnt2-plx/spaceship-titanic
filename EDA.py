import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from ydata_profiling import ProfileReport

from sklearn.preprocessing import LabelEncoder

df = pd.read_csv('data/train.csv')

class Preprocessor:
    def __init__(self):
        pass
        # self.encoder = {}


    def fit_transform(self, df):
        df = self.group_passenger(df)
        df = self.group_cabin(df)

        df.drop('Name', axis=1, inplace=True)

        df.set_index('PassengerId', inplace=True)

        df['Age'] = df['Age'].fillna(df['Age'].mean())
        for column in df.columns:
            if df[column].dtype == 'object' or df[column].dtype == 'bool':
                pass
            else:
                df[column] = df[column].fillna(0)

        df['Total_Spend'] = df['RoomService'] + df['FoodCourt'] + df['ShoppingMall'] + df['Spa'] + df['VRDeck']

        # for column in df.columns:
        #     if df[column].dtype == 'object' or df[column].dtype == 'bool':
        #         # self.encoder[column] = LabelEncoder()
        #         df[column] = LabelEncoder().fit_transform(df[column])

        return df
    
    def group_passenger(self, df):
        df['Passenger_Group'] = df['PassengerId'].apply(lambda x: x.split('_')[0])
        return df
    
    def group_cabin(self, df):
        cabin_df = df['Cabin'].str.split('/', expand=True)

        df['Cabin_Deck'] = cabin_df[0]
        df['Cabin_Number'] = cabin_df[1]
        df['Cabin_Side'] = cabin_df[2]
        
        # print(df[['Cabin', 'Cabin_Deck', 'Cabin_Number', 'Cabin_Side']])
        
        df.drop('Cabin', axis=1, inplace=True)
        
        return df

preprocessor = Preprocessor()
df = preprocessor.fit_transform(df)

profile = ProfileReport(df, title='Pandas Profiling Report')

profile.to_file("output.html")

