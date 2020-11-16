import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.feature_selection import SelectKBest
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler



#Data Preparation
def wrangle(df):
    df = df.copy()
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.drop(columns=["DailyAveragePressure", "DailyAverageWindspeed", "DailyDepartureFromNormalAverageTemperature", "DailyPrecipitation"], axis=1)

    return df

carlsbad_temps = pd.read_csv("/home/mrrumpf/build/planb/trainingdata/planb.csv")
print(carlsbad_temps.shape)
print(carlsbad_temps.columns)
print(carlsbad_temps.info())
print(carlsbad_temps.isnull().sum())


model_data = wrangle(carlsbad_temps)
print(model_data.shape)
print(model_data.columns)
print(model_data.info())

#Display visual
corrMatrix = model_data.corr()
sns.heatmap(corrMatrix, annot=True)
plt.show()

# define target & features
target = "DailyAverageTemp"
y = model_data[target]
x = model_data[["DailyMaxTemp", "DailyMinTemp"]]
xtrain, xval, ytrain, yval = train_test_split(x, y, test_size=0.25, random_state=42)
print(xtrain.shape)
print(xval.shape)
print(ytrain.shape)
print(yval.shape)

# establish a baseline to beat with this model
ypred = [ytrain.mean()] * len(ytrain)
print("Baseline MAE: ", round(mean_squared_error(ytrain, ypred), 5))

# classification & regression algo
forest = make_pipeline(
    SelectKBest(k="all"),
    StandardScaler(),
    RandomForestRegressor(
        n_estimators=100,
        max_depth=50,
        random_state=77,
        n_jobs=-1
    )
)
print("hi")
forest.fit(xtrain, ytrain)
print("hi")

#print(ypred)
print(yval)
# avg % error
errors = abs(ypred - yval)
print("hi")
mape = 100 * (errors/ytrain)
print("hi")
accuracy = 100 - np.mean(mape)
print("hi")
print("Random Forest Model: ", round(accuracy, 2), "%")