from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix

def train_model(df):
    #Model Training: Linear Regression 
    data = df.drop(['match'], axis = 1)
    target = df['match']
    X_train, X_test, y_train, y_test = train_test_split(data, target, test_size = 0.25)
    X_train = X_train.values
    X_test = X_test.values
    y_train = y_train.values
    y_test = y_test.values
    model = LinearRegression()
    
    model.fit(X_train, y_train)
    predicted = model.predict(X_test)
    scores = model.score(X_test, y_test)
    expected = y_test   
    for p in range(len(predicted)):
        if predicted[p] > 0.7:
            predicted[p] = 1
        else:
            predicted[p] = 0   

    return [predicted, expected]

def cm(expected_overall, predicted_overall):
    #finding confusion matrix
    cm_overall = confusion_matrix(expected_overall, predicted_overall)
    TP = cm_overall[0][0]
    FP = cm_overall[0][1]
    FN = cm_overall[1][0]
    TN = cm_overall[1][1]
    
    #calculating FPR and FNR
    overall_FPR = FP / (FP + TN)
    overall_FNR = FN / (FN + TP)
    return [overall_FPR, overall_FNR]

def main():
    df = pd.read_csv('Speed_Dating_Data.csv')

    #Data cleaning and data preparation
    undergrad1 = pd.get_dummies(df['undergra'], drop_first = False)
    from1 = pd.get_dummies(df['from'], drop_first=False)
    career1 = pd.get_dummies(df['career'], drop_first=False)
    df = df.drop(['iid', 'id', 'idg', 'condtn', 'wave', 'round', 'position', 'positin1', 'order', 'partner', 'pid', 'tuition', 'field', 'zipcode', 'income', 'mn_sat', 'int_corr', 'samerace', 'goal', 'date', 'attr4_1', 'sinc4_1', 'intel4_1', 'fun4_1', 'amb4_1', 'shar4_1'], axis = 1)
    df = pd.concat([df, undergrad1, from1, career1], axis = 1)
    df.fillna(0, inplace = True)
    df = df.drop(['undergra', 'from', 'career'], axis = 1)

    #Created data frames that seperate values based on gender and race
    df_female = df[df['gender'] == 0]
    df_male = df[df['gender'] == 1]
    df_black = df[df['race'] == 1]
    df_white = df[df['race'] == 2]
    df_latino = df[df['race'] == 3]
    df_asian = df[df['race'] == 4]
    df_other = df[df['race'] == 6]
    df_blackfemale = df_female[df_female['race'] == 1]
    df_whitemale = df_male[df_male['race'] == 2]

    #Overall dataframe: training model and finding confusion matrix
    train_df = train_model(df)
    predicted_overall, expected_overall = train_df[0], train_df[1]
    cm_overall = cm(expected_overall, predicted_overall)
    FPR_overall, FNR_overall = cm_overall[0], cm_overall[1]
    print("RMSE: %s" % np.sqrt(np.mean((predicted_overall-expected_overall) ** 2))) 
    print("Overall FPR and FNR: " + str(FPR_overall) + ", " + str(FNR_overall))

    #Dataframe only including females
    train_female = train_model(df_female)
    predicted_female, expected_female = train_female[0], train_female[1]
    cm_female = cm(expected_female, predicted_female)
    FPR_female, FNR_female = cm_female[0], cm_female[1]
    print("Female FPR and FNR: " + str(FPR_female) + ", " + str(FNR_female))

    #Dataframe only including males
    train_male = train_model(df_male)
    predicted_male, expected_male = train_male[0], train_male[1]
    cm_male = cm(expected_male, predicted_male)
    FPR_male, FNR_male = cm_male[0], cm_male[1]
    print("Male FPR and FNR: " + str(FPR_male) + ", " + str(FNR_male))

    #Dataframe only including African Americans
    train_black = train_model(df_black)
    predicted_black, expected_black = train_black[0], train_black[1]
    cm_black= cm(expected_black, predicted_black)
    FPR_black, FNR_black = cm_black[0], cm_black[1]
    print("Black FPR and FNR: " + str(FPR_black) + ", " + str(FNR_black))

    #Dataframe only including European/Caucasian
    train_white = train_model(df_white)
    predicted_white, expected_white = train_white[0], train_white[1]
    cm_white= cm(expected_white, predicted_white)
    FPR_white, FNR_white = cm_white[0], cm_white[1]
    print("White FPR and FNR: " + str(FPR_white) + ", " + str(FNR_white))

    #Dataframe only including Latino/Hispanics
    train_latino = train_model(df_latino)
    predicted_latino, expected_latino = train_latino[0], train_latino[1]
    cm_latino= cm(expected_latino, predicted_latino)
    FPR_latino, FNR_latino = cm_latino[0], cm_latino[1]
    print("Latino FPR and FNR: " + str(FPR_latino) + ", " + str(FNR_latino))

    #Dataframe only including Asians
    train_asian = train_model(df_asian)
    predicted_asian, expected_asian = train_asian[0], train_asian[1]
    cm_asian = cm(expected_asian, predicted_asian)
    FPR_asian, FNR_asian = cm_asian[0], cm_asian[1]
    print("Asian FPR and FNR: " + str(FPR_asian) + ", " + str(FNR_asian))

    #Dataframe including people who indicated other for their race
    train_other = train_model(df_other)
    predicted_other, expected_other = train_other[0], train_other[1]
    cm_other = cm(expected_other, predicted_other)
    FPR_other, FNR_other = cm_other[0], cm_other[1]
    print("Other FPR and FNR: " + str(FPR_other) + ", " + str(FNR_other))

    #Dataframe only including Black Females
    train_bf = train_model(df_blackfemale)
    predicted_bf, expected_bf = train_bf[0], train_bf[1]
    cm_bf = cm(expected_bf, predicted_bf)
    FPR_bf, FNR_bf = cm_bf[0], cm_bf[1]
    print("Black Female FPR and FNR: " + str(FPR_bf) + ", " + str(FNR_bf))

    #Dataframe only including White Males
    train_wm = train_model(df_whitemale)
    predicted_wm, expected_wm = train_wm[0], train_wm[1]
    cm_wm = cm(expected_wm, predicted_wm)
    FPR_wm, FNR_wm = cm_wm[0], cm_wm[1]
    print("White Male FPR and FNR: " + str(FPR_wm) + ", " + str(FNR_wm))

if __name__=="__main__":
    main()
