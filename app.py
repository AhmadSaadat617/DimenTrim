#Importing the core package
#import plotly
import streamlit as st
import pandas as pd
import numpy as np
#from plotly import express as px 
import sklearn
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
#Changing Page Configuration 
st.set_page_config(page_title="DimenTrim",page_icon="Logo3.JPG",layout="centered")

#Setting title for our dashboard
st.title("DimenTrim")
# Apply CSS to center-align the text
st.markdown("<style>h1 {text-align: center;}</style>", unsafe_allow_html=True)


#Taking the file as input from user
file=pd.read_csv(st.file_uploader("Upload Your File Below:-"))

#Displaying the uploaded file as a dataframe
st.dataframe(file)

#Basic information about the uploaded file
st.markdown(f"The file contains **{file.shape[0]} Rows** and **{file.shape[1]} Columns**")

#Extracting numerical columns 
numerical_columns = file[file.select_dtypes(include=['number']).columns]
categorical_columns = file[file.select_dtypes(include=['object', 'category']).columns]

#Dimensionality Reduction technique
choice=st.radio("**Which Dimensionality Reduction technique do you want to use:**",["Principal Component Analysis (PCA)","t-Distributed Stochiatic Embedding (t-SNE)"],index=None)

if choice=="Principal Component Analysis (PCA)":
    #Creating a PCA model for making the plot
    pca1=PCA(n_components=numerical_columns.shape[1])
    pca1.fit(numerical_columns)
    #Cumulative Variance explained plot
    st.markdown("##### **:red[Plot showing the Cumulative Variance explained versus the number of Principal Components]**")
    cumulative_var_explained=np.cumsum(pca1.explained_variance_ratio_)*100
    pc=np.arange(1,numerical_columns.shape[1]+1)
    fig=plt.figure()
    sns.lineplot(x=pc,y=cumulative_var_explained,markers=True)
    plt.xlabel("Number of Principal Components")
    plt.ylabel("Cumulative Variance Explained")
    plt.grid()
    st.pyplot(fig)
    
    
    #Displaying information for the user
    count=1
    for i in cumulative_var_explained:
        if i>80:
            break
        else:
            count+=1
    
    #Creating a PCA model for fitting on data
    st.subheader("**Reduced Dataset**",divider=True)
    n_comp=count+1
    pca2=PCA(n_components=n_comp)
    pca_data=pca2.fit_transform(numerical_columns)
    pca_data=pd.DataFrame(pca_data,columns=[f"PC{k}" for k in range(1,n_comp+1)])
    st.dataframe(pca_data)
    
    #Plotting the reduced data
    fig=plt.figure()
    sns.scatterplot(x=pca_data["PC1"],y=pca_data["PC2"],data=pca_data)
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    st.pyplot(fig)
    
if choice=="t-Distributed Stochiatic Embedding (t-SNE)":
    #Creating a slider for perplexity and no. of iterations
    a=st.slider("Select a value of Perplexity",5,50,step=5)
    b=st.slider("Select the number of iterations", 1000,5000,step=500)
    #Creating a t-SNE model
    tsne=TSNE(n_components=2,perplexity=a,n_iter=b)
    tsne_data=tsne.fit_transform(numerical_columns)
    tsne_data=pd.DataFrame(tsne_data,columns=["Column 1","Column 2"])
    st.dataframe(tsne_data)
    
    #Plotting the scatter plot
    fig=plt.figure()
    sns.scatterplot(x="Column 1",y="Column 2",data=tsne_data)
    st.pyplot(fig)
