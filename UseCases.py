#!/usr/bin/env python
# coding: utf-8

# # Google Playstore Case Study

# In this module you’ll be learning data visualisation with the help of a case study. This will enable you to understand how visualisation aids you in solving business problems. 

# **Problem Statement**
# 
# The team at Google Play Store wants to develop a feature that would enable them to boost visibility for the most promising apps. Now, this analysis would require a preliminary understanding of the features that define a well-performing app. You can ask questions like:
# - Does a higher size or price necessarily mean that an app would perform better than the other apps? 
# - Or does a higher number of installs give a clear picture of which app would have a better rating than others?
# 

# 
# 
# ### Session 1 - Introduction to Data Visualisation

# In[4]:


#import the libraries
import numpy as np
import pandas as pd


# In[6]:


#read the dataset and check the first five rows
inp0 = pd.read_csv('googleplaystore_v2.csv', on_bad_lines='skip')
inp0.head()


# In[7]:


#Check the shape of the dataframe
inp0.shape


# ### Data Handling and Cleaning

# The first few steps involve making sure that there are no __missing values__ or __incorrect data types__ before we proceed to the analysis stage. These aforementioned problems are handled as follows:
# 
#  - For Missing Values: Some common techniques to treat this issue are
#     - Dropping the rows containing the missing values
#     - Imputing the missing values
#     - Keep the missing values if they don't affect the analysis
#  
#     
#  - Incorrect Data Types:
#     - Clean certain values 
#     - Clean and convert an entire column
#  

# In[8]:


#Check the datatypes of all the columns of the dataframe
inp0.info()


# #### Missing Value Treatment

# In[9]:


#Check the number of null values in the columns
inp0.isnull().sum()


# Handling missing values for rating
#  - Ratings is the target variable
#  - drop the records

# In[11]:


#Drop the rows having null values in the Rating field
inp1 = inp0[~inp0.Rating.isnull()]


#Check the shape of the dataframe
inp1.shape


# In[12]:


# Check the number of nulls in the Rating field again to cross-verify
inp1.Rating.isnull().sum()


# In[9]:


#Question
#Check the number of nulls in the dataframe again and find the total number of null values
inp1.isnull().sum()


# In[13]:


#Inspect the nulls in the Android Version column
inp1[inp1['Android Ver'].isnull()]


# In[15]:


#Drop the row having shifted values
inp1 = inp1[~(inp1['Android Ver'].isnull() & (inp1.Category == '1.9'))]

#Check the nulls againin Android version column to cross-verify
inp1[inp1['Android Ver'].isnull()]


# Imputing Missing Values
# 
# - For numerical variables use mean and median
# - For categorical variables use mode

# In[16]:


#Check the most common value in the Android version column
inp1['Android Ver'].value_counts()


# In[18]:


#Fill up the nulls in the Android Version column with the above value
inp1['Android Ver'].mode()[0]
inp1['Android Ver'] = inp1['Android Ver'].fillna(inp1['Android Ver'].mode()[0])


# In[19]:


#Check the nulls in the Android version column again to cross-verify
inp1['Android Ver'].isnull().sum()


# In[20]:


#Check the nulls in the entire dataframe again
inp1.isnull().sum()


# In[21]:


#Check the most common value in the Current version column
inp1['Current Ver'].value_counts()


# In[23]:


#Replace the nulls in the Current version column with the above value
inp1['Current Ver'].mode()[0]
inp1['Current Ver'] = inp1['Current Ver'].fillna(inp1['Current Ver'].mode()[0])


# In[24]:


# Question : Check the most common value in the Current version column again
inp1['Current Ver'].value_counts()


# #### Handling Incorrect Data Types 

# In[25]:


#Check the datatypes of all the columns 
inp1.dtypes


# In[26]:


#Question - Try calculating the average price of all apps having the Android version as "4.1 and up" 

inp1.Price.value_counts('Android Ver'==4.1 and up)


# In[27]:


#Analyse the Price column to check the issue
inp1.Price.value_counts()


# In[28]:


#Write the function to make the changes

inp1.Price = inp1.Price.apply(lambda x: 0 if x=='0' else float(x[1:]))


# In[30]:


#Verify the dtype of Price once again
inp1.Price.dtypes
inp1.Price.value_counts()


# In[31]:


#Analyse the Reviews column
inp1.Reviews.value_counts()


# In[33]:


#Change the dtype of this column
inp1.Reviews = inp1.Reviews.astype('int32')

#Check the quantitative spread of this dataframe

inp1.Reviews.describe()


# In[34]:


#Analyse the Installs Column

inp1.Installs.head()


# In[35]:


#Question Clean the Installs Column and find the approximate number of apps at the 50th percentile.
def clean_installs(val):
    return int(val.replace(",","").replace("+",""))
type(clean_installs("3,000+"))
inp1.Installs = inp1.Installs.apply(clean_installs)
inp1.Installs.describe()


# #### Sanity Checks

# The data that we have needs to make sense and therefore you can perform certain sanity checks on them to ensure they are factually correct as well. Some sanity checks can be:
# 
# - Rating is between 1 and 5 for all the apps.
# - Number of Reviews is less than or equal to the number of Installs.
# - Free Apps shouldn’t have a price greater than 0.
# 

# In[36]:


#Perform the sanity checks on the Reviews column
inp1 = inp1[inp1.Reviews <= inp1.Installs]


# In[37]:


#perform the sanity checks on prices of free apps 
inp1[(inp1.Type == 'Free') & (inp1.Price>0)]


# #### Outliers Analysis Using Boxplot

# Now you need to start identifying and removing extreme values or __outliers__ from our dataset. These values can tilt our analysis and often provide us with a biased perspective of the data available. This is where you’ll start utilising visualisation to achieve your tasks. And the best visualisation to use here would be the box plot. Boxplots are one of the best ways of analysing the spread of a numeric variable
# 
# 
# Using a box plot you can identify the outliers as follows:

# ![BoxPlots to Identify Outliers](images\Boxplot.png)

# - Outliers in data can arise due to genuine reasons or because of dubious entries. In the latter case, you should go ahead and remove such entries immediately. Use a boxplot to observe, analyse and remove them.
# - In the former case, you should determine whether or not removing them would add value to your analysis procedure.

# - You can create a box plot directly from pandas dataframe or the matplotlib way as you learnt in the previous session. Check out their official documentation here:
#    - https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.boxplot.html
#    - https://matplotlib.org/3.1.1/api/_as_gen/matplotlib.pyplot.boxplot.html

# In[38]:


#import the plotting libraries

import matplotlib.pyplot as plt


# In[39]:


#Create a box plot for the price column
plt.boxplot(inp1.Price)


# In[40]:


#Check the apps with price more than 200
inp1[inp1.Price>200]


# In[41]:


#Clean the Price column
inp1 = inp1[inp1.Price<=10]


# In[42]:


#Create a box plot for paid apps
inp1 = inp1[inp1.Price<=10]
plt.boxplot(inp1.Price)
plt.show()


# In[43]:


#Check the apps with price more than 30
inp1 = inp1[inp1.Price<=30]
plt.boxplot(inp1.Price)
plt.show()


# In[44]:


#Clean the Price column again
inp1 = inp1[inp1.Price<=30]
plt.boxplot(inp1.Price)
plt.show()


# ### Histograms
# 
# Histograms can also be used in conjuction with boxplots for data cleaning and data handling purposes. You can use it to check the spread of a numeric variable. Histograms generally work by bucketing the entire range of values that a particular variable takes to specific __bins__. After that, it uses vertical bars to denote the total number of records in a specific bin, which is also known as its __frequency__.
# 

# ![Histogram](images\Histogram.png)

# You can adjust the number of bins to improve its granularity

# ![Bins change](images\Granular.png)

# You'll be using plt.hist() to plot a histogram. Check out its official documentation:https://matplotlib.org/3.1.1/api/_as_gen/matplotlib.pyplot.hist.html

# In[45]:


#Create a histogram of the Reviews
plt.hist(inp1.Reviews)
plt.show()


# In[46]:


#Create a boxplot of the Reviews column
plt.boxplot(inp1.Reviews)
plt.show()


# In[47]:


#Check records with 1 million reviews
inp1[inp1.Reviews > 10000000]


# In[48]:


#Drop the above records
inp1 = inp1[inp1.Reviews <= 10000000]
inp1.shape


# In[49]:


#Question - Create a histogram again and check the peaks
plt.hist(inp1.Reviews)
plt.show()


# In[50]:


#Question - Create a box plot for the Installs column and report back the IQR
plt.boxplot(inp1.Installs)
plt.show()


# In[51]:


#Question - CLean the Installs by removing all the apps having more than or equal to 100 million installs

inp1 = inp1[inp1.Installs <= 100000000]
inp1.Installs.shape


# In[100]:


#Plot a histogram for Size as well.
plt.hist(inp1.Size)
plt.show()


# In[52]:


#Question - Create a boxplot for the Size column and report back the median value
plt.boxplot(inp1.Size)
plt.show()


# ### Session 2 - Data Visualisation with Seaborn

# Seaborn is Python library to create statistical graphs easily. It is built on top of matplotlib and closely integrated with pandas.
# 
# _Functionalities of Seaborn_ :
# 
# - Dataset oriented API
# - Analysing univariate and bivariate distributions
# - Automatic estimation and plotting of  linear regression models
# - Convenient views for complex datasets
# - Concise control over style
# - Colour palettes
# 

# In[ ]:


#import the necessary libraries


# #### Distribution Plots

# A distribution plot is pretty similar to the histogram functionality in matplotlib. Instead of a frequency plot, it plots an approximate probability density for that rating bucket. And the curve (or the __KDE__) that gets drawn over the distribution is the approximate probability density curve. 
# 
# The following is an example of a distribution plot. Notice that now instead of frequency on the left axis, it has the density for each bin or bucket.

# ![Distplot](images\Distplot.png)

# You'll be using sns.distplot for plotting a distribution plot. Check out its official documentation: https://seaborn.pydata.org/generated/seaborn.distplot.html

# In[62]:


#Create a distribution plot for rating
import warnings
warnings.filterwarnings("ignore")
inp1.Rating.plot.hist()
plt.show()

import seaborn as sns
sns.distplot(inp1.Rating,bins=15,vertical=False)
plt.show()


# In[63]:


#Change the number of bins

sns.distplot(inp1.Rating,bins=15,vertical=False)
plt.show()


# In[64]:


#Change the colour of bins to green
sns.distplot(inp1.Rating,bins=15,vertical=False,color='Green')
plt.show()


# In[68]:


#Apply matplotlib functionalities
sns.distplot(inp1.Rating,bins=20,color='Blue')
plt.title("Distribution of App Ratings",fontsize=12)
plt.show()


# #### Styling Options
# 
# One of the biggest advantages of using Seaborn is that you can retain its aesthetic properties and also the Matplotlib functionalities to perform additional customisations. Before we continue with our case study analysis, let’s study some styling options that are available in Seaborn.

# -  Check out the official documentation:https://seaborn.pydata.org/generated/seaborn.set_style.html

# In[76]:


#Check all the styling options
sns.set_style('darkgrid')
sns.distplot(inp1.Rating,bins=20,color='Green')
plt.title("Distribution of Apps Rating", fontsize=12)
plt.show()


# In[72]:


#Change the number of bins to 20
plt.style.available


# #### Pie-Chart and Bar Chart

# For analysing how a numeric variable changes across several categories of a categorical variable you utilise either a pie chart or a box plot

# For example, if you want to visualise the responses of a marketing campaign, you can use the following views:

# ![PieChart](images\pie.png)

# ![barChart](images\bar.png)

# - You'll be using the pandas method of plotting both a pie chart and a bar chart. Check out their official documentations:
#    - https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.plot.bar.html
#    - https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.plot.pie.html

# In[77]:


#Analyse the Content Rating column
inp1['Content Rating'].value_counts()


# In[79]:


#Remove the rows with values which are less represented 
inp1 = inp1[~inp1['Content Rating'].isin(['Adults only 18+','Unrated'])]
inp1['Content Rating'].value_counts()


# In[80]:


#Reset the index
inp1.reset_index(inplace=True,drop=True)


# In[81]:


#Check the apps belonging to different categories of Content Rating 
inp1['Content Rating'].value_counts()


# In[82]:


#Plot a pie chart
inp1['Content Rating'].value_counts().plot.pie()
plt.show()


# In[83]:


#Plot a bar chart
inp1['Content Rating'].value_counts().plot.bar()
plt.show()


# In[86]:


#Question - Plot a bar plot for checking the 4th highest Android version type
inp1['Android Ver'].value_counts()


# #### Scatter Plots

# Scatterplots are perhaps one of the most commonly used as well one of the most powerful visualisations you can use in the field of machine learning. They are pretty crucial in revealing relationships between the data points and you can generally deduce some sort of trends in the data with the help of a scatter plot. 

# ![Scatterplot](images\scatter.png)

# - They're pretty useful in regression problems to check whether a linear trend exists in the data or not. For example, in the image below, creating a linear model in the first case makes far more sense since a clear straight line trend is visible.

# ![Scatterplot-Reg](images\regression3.png)

# - Also, they help in observing __naturally occuring clusters__. In the following image, the marks of students in Maths and Biology has been plotted.You can clearly group the students to 4 clusters now. Cluster 1 are students who score very well in Biology but very poorly in Maths, Cluster 2 are students who score equally well in both the subjects and so on.

# ![Scatter-Clusters](images\Clusters.png)

# **Note**: You'll be studying about both Regression and Clustering in greater detail in the machine learning modules

# You'll be using **sns.jointplot()** for creating a scatter plot. Check out its documentation:
# https://seaborn.pydata.org/generated/seaborn.jointplot.html

# In[87]:


###Size vs Rating

##Plot a scatter-plot in the matplotlib way between Size and Rating
plt.scatter(inp1.Size,inp1.Rating)
plt.show()


# In[89]:


### Plot the same thing now using a jointplot
sns.set_style("dark")
sns.jointplot(inp1.Size,inp1.Rating)
plt.show()


# In[90]:


## Plot a jointplot for Price and Rating
sns.set_style("dark")
sns.jointplot(inp1.Price,inp1.Rating)
plt.show()


# **Reg Plots**
# 
# - These are an extension to the jointplots, where a regression line is added to the view 

# In[92]:


##Plot a reg plot for Price and Rating and observe the trend
sns.jointplot(inp1.Price,inp1.Rating,kind='reg')
plt.show()


# In[97]:


## Question - Plot a reg plot for Price and Rating again for only the paid apps.

sns.jointplot(inp1.Price,inp1.Rating,kind='reg')
plt.show()


# **Pair Plots**

#  - When you have several numeric variables, making multiple scatter plots becomes rather tedious. Therefore, a pair plot visualisation is preferred where all the scatter plots are in a single view in the form of a matrix
#  - For the non-diagonal views, it plots a **scatter plot** between 2 numeric variables
#  - For the diagonal views, it plots a **histogram**

# Pair Plots help in identifying the trends between a target variable and the predictor variables pretty quickly. For example, say you want to predict how your company’s profits are affected by three different factors. In order to choose which you created a pair plot containing profits and the three different factors as the variables. Here are the scatterplots of profits vs the three variables that you obtained from the pair plot.

# ![Pairplots](images\pairplots2.png)

# It is clearly visible that the left-most factor is the most prominently related to the profits, given how linearly scattered the points are and how randomly scattered the rest two factors are.

# You'll be using **sns.pairplot()** for this visualisation. Check out its official documentation:https://seaborn.pydata.org/generated/seaborn.pairplot.html

# In[100]:


## Create a pair plot for Reviews, Size, Price and Rating
sns.set_style('dark')
sns.pairplot(inp1[['Reviews','Size','Price','Rating']])
plt.show()


# **Bar Charts Revisited**

# - Here, you'll be using bar charts once again, this time using the **sns.barplot()** function. Check out its official documentation:https://seaborn.pydata.org/generated/seaborn.barplot.html
# - You can modify the **estimator** parameter to change the aggregation value of your barplot

# In[103]:


##Plot a bar plot of Content Rating vs Average Rating 
inp1.groupby(['Content Rating'])['Rating'].mean().plot.bar()
plt.show()


# In[104]:


##Plot the bar plot again with Median Rating
inp1.groupby(['Content Rating'])['Rating'].median().plot.bar()
plt.show()


# In[106]:


##Plot the above bar plot using the estimator parameter
sns.barplot(data=inp1,x='Content Rating',y='Rating',estimator=np.median)
plt.show()


# In[107]:


##Plot the bar plot with only the 5th percentile of Ratings
sns.barplot(data=inp1,x='Content Rating',y='Rating',estimator=lambda x:np.quantile(x,0.05))
plt.show()


# In[108]:


##Question - Plot the bar plot with the minimum Rating
sns.barplot(data=inp1, x="Content Rating", y="Rating", estimator = np.min)
plt.show()


# __Box Plots Revisited__
# 
# - Apart from outlier analysis, box plots are great at comparing the spread and analysing a numerical variable across several categories
# - Here you'll be using **sns.boxplot()** function to plot the visualisation. Check out its documentation: https://seaborn.pydata.org/generated/seaborn.boxplot.html
# 
# 
# 

# In[109]:


##Plot a box plot of Rating vs Content Rating
plt.figure(figsize=[6,7])
sns.boxplot(inp1['Content Rating'],inp1.Rating)
plt.show()


# In[110]:


##Question - Plot a box plot for the Rating column only
plt.figure(figsize=[6,7])
sns.boxplot(inp1.Rating)
plt.show()


# In[113]:


##Question - Plot a box plot of Ratings across the 4 most popular Genres
plt.figure(figsize=[10,30])
sns.boxplot(inp1['Rating'],inp1.Genres)
plt.show()


# #### Heat Maps

# Heat mapsutilise the concept of using colours and colour intensities to visualise a range of values. You must have seen heat maps in cricket or football broadcasts on television to denote the players’ areas of strength and weakness.

# ![HeatMap](images\heatmap1.png)

# - In python, you can create a heat map whenever you have a rectangular grid or table of numbers analysing any two features

# ![heatmap2](images\heatmap2.png)

# - You'll be using **sns.heatmap()** to plot the visualisation. Checkout its official documentation :https://seaborn.pydata.org/generated/seaborn.heatmap.html

# # Ratings vs Size vs Content Rating
# 
# ##Prepare buckets for the Size column using pd.qcut
# inp1['Size_Bucket']=pd.cut(inp1.Size,[0,0.2,0.4,0.6,0.8,1],['VL','L','M','H','VH'])
# inp1.head()                                                   
# 

# In[128]:


##Create a pivot table for Size_buckets and Content Rating with values set to Rating
pd.pivot_table(data=inp1,index='Content Rating',columns='Size_Bucket',values='Rating')


# In[122]:


##Change the aggregation to median
pd.pivot_table(data=inp1,index='Content Rating',columns='Size_Bucket',values='Rating',aggfunc=np.median)


# In[129]:


##Change the aggregation to 20th percentile
pd.pivot_table(data=inp1,index='Content Rating',columns='Size_Bucket',values='Rating',aggfunc=lambda x:np.quantile(x,0.2))


# In[ ]:


##Store the pivot table in a separate variable


# In[131]:


##Plot a heat map
sns.heatmap('Rating','Content Rating')


# In[ ]:


##Apply customisations


# In[ ]:


##Question - Replace Content Rating with Review_buckets in the above heat map
##Keep the aggregation at minimum value for Rating


# ### Session 3: Additional Visualisations

# #### Line Plots

# - A line plot tries to observe trends using time dependent data.
# -  For this part, you'll be using **pd.to_datetime()** function. Check out its documentation:https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.to_datetime.html
# 

# In[132]:


## Extract the month from the Last Updated Date
inp1['Last Updated'].head()


# In[ ]:


## Find the average Rating across all the months


# In[ ]:


## Plot a line graph


# #### Stacked Bar Charts

# - A stacked bar chart breaks down each bar of the bar chart on the basis of a different category
# - For example, for the Campaign Response bar chart you saw earlier, the stacked bar chart is also showing the Gender bifurcation as well

# ![Stacked](images\stacked.png)

# In[ ]:


## Create a pivot table for Content Rating and updated Month with the values set to Installs


# In[ ]:


##Store the table in a separate variable


# In[ ]:


##Plot the stacked bar chart.


# In[ ]:


##Plot the stacked bar chart again wrt to the proportions.


# #### Plotly

# Plotly is a Python library used for creating interactive visual charts. You can take a look at how you can use it to create aesthetic looking plots with a lot of user-friendly functionalities like hover, zoom, etc.

# Check out this link for installation and documentation:https://plot.ly/python/getting-started/

# In[ ]:


#Install plotly


# In[ ]:


#Take the table you want to plot in a separate variable


# In[ ]:


#Import the plotly libraries


# In[ ]:


#Prepare the plot

