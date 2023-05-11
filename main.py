import pandas as pd  # reading the project
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import ydata_profiling
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

path = "/Users/ivymbunda/PycharmProjects/House_sales_king_county/venv/kc_house_data.csv"
df = pd.read_csv(path)
# lets clean the data
# checking for missing values
print(df.isnull().sum())
# fill in the missing values with median
df.fillna(df.median(), inplace=True)

# check for duplicates
print(df.duplicated().sum())

# remove any duplicates
df.drop_duplicates(inplace=True)

# lets check for any outliers
plt.boxplot(df['price'])


# lets remove the outliers
Q1 = df['price'].quantile(0.25)
Q3 = df['price'].quantile(0.75)
IQR = Q3 - Q1
data = df[(df['price'] > (Q1 - 1.5 * IQR)) & (df['price'] < (Q3 + 1.5 * IQR))]

# lets check for data types
print(df.dtypes)

df['date'] = pd.to_datetime(df['date'])
df['zipcode'] = df['zipcode'].astype(str)

# lets normalize the data
df['price'] = np.log(df['price'])

# lets check for multicollinearity
corr_matrix = df.corr().abs()
upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
to_drop = [column for column in upper.columns if any(upper[column] > 0.8)]
df.drop(to_drop, axis=1, inplace=True)

# lets remove unnecessary features that are not useful in predicting the market price
df.drop(['id', 'date', 'yr_built', 'yr_renovated'], axis=1, inplace=True)

# Relationship between the square footage and sale price
plt.scatter(df['sqft_living'], df['price'])
plt.title("Relationship Between Square Footage And The Sale Price")
plt.xlabel('Square Footage')
plt.ylabel('Sale Price')
plt.show()

# lets create a box plot to create a distribution of sale price in the data set
sns.boxplot(x=data['price'])
plt.title('Distribution of the sale price')
plt.xlabel('Sale Price')
plt.show()

# Histogram to show the distribution of the square footage in the data
plt.hist(data['sqft_living'], bins=50)
plt.title('Distribution of Square Footage')
plt.xlabel('Square Footage')
plt.ylabel('Frequency')
plt.show()

# Bar Plot
neighborhood_counts = df['zipcode'].value_counts().head(15)
plt.bar(neighborhood_counts.index, neighborhood_counts.values)
plt.title('Top 10 Neighborhoods by Count')
plt.xlabel('Zipcode')
plt.ylabel('Count')
plt.show()

# Heat Map
corr = data.corr()
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(corr, mask=mask, cmap='coolwarm', center=0, square=True, annot=True, fmt='.2f')
plt.title('Correlation Heatmap between different features in the dataset')
plt.show()

# Regression model to predict sale prices
# Define the features and target variable
X = df[['bedrooms', 'bathrooms', 'sqft_living', 'floors', 'waterfront', 'view', 'condition', 'grade', 'sqft_above', 'sqft_basement', 'yr_built', 'yr_renovated', 'zipcode']]
y = df['price']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Fit the linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions on the testing set
y_pred = model.predict(X_test)

# Evaluate the model performance
r2 = r2_score(y_test, y_pred)
print('R-squared:', r2)

my_report = ydata_profiling.ProfileReport(df)
my_report.to_file('my_report.html')
