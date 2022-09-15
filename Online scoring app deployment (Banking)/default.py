from urllib import request
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

import pdfkit as pdf
import sqlite3

def export(df):
    df.to_html('temp.html') 
    nom = input('export filename')
    df.to_csv(nom+'.csv')
    df.to_excel(nom+'.xlsx')
    pdf.from_file('temp.html', nom+'.pdf')
    
def nonzero(data):
    l = []
    print('Non-zero values rate by column\n')
    for i in data.columns:
        k=str(i)
        t = (1.-data[k].isnull().sum()/len(data))*10000//1/100
        print('For the column', i,', the rate is :', t, '%')

def highlight_neg(cell):
    if type(cell) != str and cell < 0 :
        return 'background: red; color:black'
    
def highlight_zero(cell):
    if type(cell) != str and cell == 0 :
        return 'background: red; color:black'

def plot_nan(df, figsize=(8,8), show_null=True):
    sns.set(palette='muted', color_codes=True, style='white')
    df_nan = df.isna().sum().sort_values(ascending=False)
    if show_null == False: df_nan = df_nan[df_nan != 0]
    plt.figure(figsize=figsize)
    try: plt.title('Percentage of NaN values per column in dataframe %s' %df.name)
    except: plt.title('Percentage of NaN values per column')
    sns.despine()
    fig = sns.barplot(x=df_nan.values/df.shape[0]*100, y=df_nan.index)
    if len(df_nan)>1:
        fig.axes.grid(True, axis='x')
        
def plot_zero(df, figsize=(8,8), show_null=True):
    sns.set(palette='muted', color_codes=True, style='white')
    df_zero = df.isnull().sum().sort_values(ascending=False)
    if show_null == False: df_zero = df_zero[df_zero != 0]
    plt.figure(figsize=figsize)
    try: plt.title('Percentage of null values per column in dataframe %s' %df.name)
    except: plt.title('Percentage of null values per column')
    sns.despine()
    fig = sns.barplot(x=df_zero.values/df.shape[0]*100, y=df_zero.index)
    if len(df_zero)>1:
        fig.axes.grid(True, axis='x')