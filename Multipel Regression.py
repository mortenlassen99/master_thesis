import numpy as np
import pandas as pd
import math 
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold, cross_val_score, cross_val_predict


file_placering = '/Users/mortenlassen/Desktop/Data_v7.xlsx'

df_data = pd.read_excel(file_placering, sheet_name='Combined')


X = df_data[['Ejerlejlighed','Antal værelser','Enhedsareal - Beboelse','Enhedsareal - Erhverv','Kælderareal','Familieoverdragelse','Fritidsbolig','Landbrugsbolig','Energimærke','Opførelsesår','Omtilbygningsår','Grundareal','Postnr']]
y = df_data['Handelspris']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


pipeline = Pipeline([
    ('linear', LinearRegression())
])

k = 5
kfold = KFold(n_splits=k, shuffle=True, random_state=40)

resultater = cross_val_score(pipeline, X, y, cv=kfold, scoring='neg_mean_squared_error')

y_prædiktion = cross_val_predict(pipeline, X, y, cv=kfold)

mape = np.mean(np.abs((y - y_prædiktion) / y)) * 100

gennemsnit_mse = np.mean(resultater)
gennemsnit_rmse = math.sqrt(-gennemsnit_mse)
formateret_mse = f"{-gennemsnit_mse:.0f}"
formateret_rmse = f"{gennemsnit_rmse:.0f}"


print(f"Gennemsnitlig MSE for K-Fold Cross-Validation: {formateret_mse}")
print(f"Gennemsnitlig RMSE for K-Fold Cross-Validation: {formateret_rmse}")
print(f"MAPE for K-Fold Cross-Validation: {mape:.8f}%")



pipeline.fit(X_train, y_train)

skæringspunkt = pipeline.named_steps['linear'].intercept_
koefficienter = pipeline.named_steps['linear'].coef_

formateret_skæringspunkt = f"Skærningspunktet: {skæringspunkt:.0f}"
formateret_koefficienter = [f"{coef:.0f}" for coef in koefficienter]
print(formateret_skæringspunkt)
print(f"Hældningskoefficienter: {formateret_koefficienter}")



observation_1000 = X.iloc[999]

print("Information for observation 1000:")
print(observation_1000)

observation_1000_omformet = observation_1000.values.reshape(1, -1) if hasattr(observation_1000, 'values') else observation_1000
prædiktion = pipeline.predict(observation_1000_omformet)

formateret_prædiktion = f"{prædiktion[0]:.2f}"

print(f"Prædiktion af pris for observation 1000: {formateret_prædiktion}")

