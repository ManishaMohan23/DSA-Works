from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import joblib
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np

df = pd.read_csv('YouTubeProcessed.csv')
df=df.drop(['comments_disabled','ratings_disabled'],axis=1)

# Encoding Categorical Columns
le = LabelEncoder()
df['category'] = le.fit_transform(df['category'])


# Create a scaler for other columns excluding 'dislikes'
scaler= StandardScaler()
scaled_features = scaler.fit_transform(df[['view_count', 'likes', 'comment_count']])
df[['view_count', 'likes', 'comment_count']] = scaled_features

print(df.columns)

# Train-Test Split
df_train = df[df['published_year'] < 2022]
df_test = df[df['published_year'] == 2022]
df_train = df_train.drop('published_year', axis=1)
df_test = df_test.drop('published_year', axis=1)
X_train = df_train.drop('dislikes',axis=1)
y_train = df_train['dislikes']
# Splitting into Train and Validation Sets
X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

# Initializing and Training RandomForestRegressor
model = RandomForestRegressor(random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)
print("Mean Absolute Error:", mae)
print("Mean Squared Error:", mse)
print("Root Mean Squared Error:", rmse)
print("R-squared:", r2)

# Save the Trained Model and Preprocessing Objects
joblib.dump(model, 'model.pkl')
joblib.dump(le, 'label_encoder.pkl')
joblib.dump(scaler, 'standard_scaler.pkl')