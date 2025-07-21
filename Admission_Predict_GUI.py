from tkinter import *
import numpy
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn import metrics
from tkinter import *

#Linear Regression Model
def getModel():
    df = pd.read_csv("Admission_Predict.csv")
    df = df.drop("Serial No.",axis=1)
    outcome = df["Chance of Admit "]
    df = df.drop("Chance of Admit ",axis=1)
    xtrain,xtest,ytrain,ytest = train_test_split(df,outcome,test_size=0.2,random_state=42)
    linearRegression = LinearRegression()
    linearRegression.fit(xtrain,ytrain)
    return linearRegression


#Model Calling Function
def predict_Chance(window,gre,toefl,uni,sop,lor,cgpa,research):
    values = numpy.array([float(gre.get()),float(toefl.get()),float(uni.get()),float(sop.get()),float(lor.get()),float(cgpa.get()),float(research.get())])
    values = values.reshape(1,-1)
    model = getModel()
    predicted_chance = model.predict(values)[0] * 100
    predicted_chance = round(predicted_chance,2)

    result_label = Label(window,text=f"The Chances of Admission is : {predicted_chance} %",font=('ariel',10,'bold'))
    result_label.grid(row=12,column=1)


#Main
if __name__ == "__main__":
    window = Tk()
    window.minsize(500,350)
    window.maxsize(500,350)
    window.title("Admission Prediction")

    gre_score = StringVar()
    toefl_score = StringVar()
    uni_rating = StringVar()
    sop = StringVar()
    lor = StringVar()
    cgpa = StringVar()
    research = StringVar()

    title_Label = Label(window,text="Admission Prediction Model",font=('algerian',10))
    title_Label.grid(row=1,column=1)
    gre_score_label = Label(window,text="GRE Score : ",font=('ariel',10,'bold'))
    gre_score_label.grid(row=3,column=0)
    gre_score_input = Entry(window,textvariable=gre_score,font=('ariel',10,'bold'))
    gre_score_input.grid(row=3,column=1)
    toefl_score_label = Label(window,text="TOEFL Score : ",font=('ariel',10,'bold'))
    toefl_score_label.grid(row=4,column=0)
    toefl_score_input = Entry(window,textvariable=toefl_score,font=('ariel',10,'bold'))
    toefl_score_input.grid(row=4,column=1)
    uni_rating_label = Label(window,text="University Rating : ",font=('ariel',10,'bold'))
    uni_rating_label.grid(row=5,column=0)
    uni_rating_input = Entry(window,textvariable=uni_rating,font=('ariel',10,'bold'))
    uni_rating_input.grid(row=5,column=1)
    sop_label = Label(window,text="SOP : ",font=('ariel',10,'bold'))
    sop_label.grid(row=6,column=0)
    sop_input = Entry(window,textvariable=sop,font=('ariel',10,'bold'))
    sop_input.grid(row=6,column=1)
    lor_label = Label(window,text="LOR : ",font=('ariel',10,'bold'))
    lor_label.grid(row=7,column=0)
    lor_input = Entry(window,textvariable=lor,font=('ariel',10,'bold'))
    lor_input.grid(row=7,column=1)
    cgpa_label = Label(window,text="CGPA : ",font=('ariel',10,'bold'))
    cgpa_label.grid(row=8,column=0)
    cgpa_input = Entry(window,textvariable=cgpa,font=('ariel',10,'bold'))
    cgpa_input.grid(row=8,column=1)
    research_label = Label(window,text="Research : ",font=('ariel',10,'bold'))
    research_label.grid(row=9,column=0)
    research_input = Entry(window,textvariable=research,font=('ariel',10,'bold'))
    research_input.grid(row=9,column=1)
    calculate_button = Button(window,text="Calculate",command=lambda:predict_Chance(window,gre_score,toefl_score,uni_rating,sop,lor,cgpa,research),font=('ariel',10,'bold'))
    calculate_button.grid(row=10,column=1)

    window.mainloop()
