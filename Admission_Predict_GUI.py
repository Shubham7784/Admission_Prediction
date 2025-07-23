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
    result_label.pack()


root = Tk()
root.title("Admission Prediction")
root.geometry("400x450")
root.configure(bg="#f0f0f0")

title = Label(root, text="ADMISSION PREDICTION MODEL", font=("Helvetica", 14, "bold"), bg="#f0f0f0")
title.pack(pady=10)

form_frame = Frame(root, bg="#f0f0f0")
form_frame.pack()

def create_label_entry(frame, label_text):
    row = Frame(frame, bg="#f0f0f0")
    row.pack(pady=5)
    label = Label(row, text=label_text, width=20, anchor="w", font=("Helvetica", 10), bg="#f0f0f0")
    entry = Entry(row, width=20, font=("Helvetica", 10))
    label.pack(side=LEFT)
    entry.pack(side=RIGHT)
    return entry

entry_gre = create_label_entry(form_frame, "GRE Score:")
entry_toefl = create_label_entry(form_frame, "TOEFL Score:")
entry_rating = create_label_entry(form_frame, "University Rating:")
entry_sop = create_label_entry(form_frame, "SOP:")
entry_lor = create_label_entry(form_frame, "LOR:")
entry_cgpa = create_label_entry(form_frame, "CGPA:")
entry_research = create_label_entry(form_frame, "Research (0 or 1):")

calculate_btn = Button(root, text="Calculate", command=lambda:predict_Chance(root,entry_gre,entry_toefl,entry_rating,entry_sop,entry_lor,entry_cgpa,entry_research),
                          font=("Helvetica", 10, "bold"), bg="#4CAF50", fg="white", padx=10, pady=5)
calculate_btn.pack(pady=20)

result_label = Label(root, text="", font=("Helvetica", 12), bg="#f0f0f0", fg="blue")
result_label.pack()

root.mainloop()
