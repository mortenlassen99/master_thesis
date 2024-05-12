import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV, KFold, cross_val_score, cross_val_predict
from sklearn.model_selection import train_test_split


file_placering = '/Users/mortenlassen/Desktop/Data_v7.xlsx'

df_data = pd.read_excel(file_placering, sheet_name='Combined')


X = df_data[['Ejerlejlighed','Antal værelser','Enhedsareal - Beboelse','Enhedsareal - Erhverv','Kælderareal','Familieoverdragelse','Fritidsbolig','Landbrugsbolig','Energimærke','Opførelsesår','Omtilbygningsår','Grundareal','Postnr']]
y = df_data['Handelspris']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)  


alpha_søgning = {
    'ridge__alpha': np.logspace(0, 6, 1000)
}

pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('ridge', Ridge(max_iter=1000000))
])

k = 5 
kfold = KFold(n_splits=k, shuffle=True, random_state=40)

grid_search = GridSearchCV(pipeline, alpha_søgning, scoring='neg_mean_squared_error', cv=kfold)

resultater = cross_val_score(grid_search, X, y, scoring='neg_mean_squared_error', cv=kfold)

y_prædiktion = cross_val_predict(pipeline, X, y, cv=kfold)

mape = np.mean(np.abs((y - y_prædiktion) / y)) * 100

gennemsnit_mse = np.mean(resultater)
gennemsnit_rmse = math.sqrt(-gennemsnit_mse)
formateret_mse = f"{-gennemsnit_mse:.0f}"
formateret_rmse = f"{gennemsnit_rmse:.0f}"

print(f"Gennemsnitlig MSE for K-Fold Cross-Validation: {formateret_mse}")
print(f"Gennemsnitlig RMSE for K-Fold Cross-Validation: {formateret_rmse}")
print(f"MAPE for K-Fold Cross-Validation: {mape:.8f}%")


grid_search.fit(X_train, y_train)

bedste_alpha = grid_search.best_params_['ridge__alpha']
formateret_alpha = f"Bedste_alpha: {bedste_alpha:.0f}"

print(formateret_alpha)

optimeret_ridge = Ridge(alpha=bedste_alpha)

optimeret_ridge.fit(X_train, y_train)

skæringspunkt = optimeret_ridge.intercept_
koefficienter = optimeret_ridge.coef_

formateret_skæringspunkt = f"Skærningspunktet: {skæringspunkt:.0f}"
formateret_koefficienter = [f"{coef:.0f}" for coef in koefficienter]
print(formateret_skæringspunkt)
print(f"Hældningskoefficienter: {formateret_koefficienter}")


gennemsnitlig_resultat_for_alphaer = grid_search.cv_results_['mean_test_score']

gennemsnitlig_resultat_for_alphaer = -gennemsnitlig_resultat_for_alphaer

alpha_værdier = grid_search.cv_results_['param_ridge__alpha'].data
bedste_alpha = grid_search.best_params_['ridge__alpha']

alpha_værdier_millions = [x / 1e6 for x in alpha_værdier]
gennemsnitlig_resultat_for_alphaer_millions = [x / 1e6 for x in gennemsnitlig_resultat_for_alphaer]
bedste_alpha_millions = bedste_alpha / 1e6

plt.figure(figsize=(10, 6))
plt.plot(alpha_værdier_millions, gennemsnitlig_resultat_for_alphaer_millions, label='MSE baseret på cross-validation')

plt.axvline(bedste_alpha_millions, color='red', linestyle='--', label=f'Bedste lambda: {bedste_alpha:.0f}')

plt.xscale('linear') 
plt.xlabel('Lambda (i millioner)')
plt.ylabel('Mean Squared Error (i millioner)')
plt.legend()

plt.gca().get_yaxis().set_major_formatter(ticker.FuncFormatter(lambda x, p: format(int(x))))

plt.figure(figsize=(10, 6))
plt.plot(alpha_værdier, gennemsnitlig_resultat_for_alphaer_millions, label='MSE baseret på cross-validation')
plt.axvline(bedste_alpha, color='red', linestyle='--', label=f'Bedste lambda: {bedste_alpha:.0f}')

plt.xlim(0, 10000)
plt.xlabel('Lambda')
plt.ylabel('Mean Squared Error (i millioner)')
plt.legend()
plt.xticks(rotation=45)  
plt.gca().get_yaxis().set_major_formatter(ticker.FuncFormatter(lambda x, p: format(int(x))))
plt.show()

