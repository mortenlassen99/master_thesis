pip install keras-tuner
import os
import keras
import shutil
import numpy as np
import pandas as pd
import tensorflow as tf
import keras_tuner as kt
import matplotlib.pyplot as plt
from math import sqrt
from keras.optimizers import Adam
from keras.models import Sequential
from keras.regularizers import l1_l2
from sklearn.pipeline import Pipeline
from keras.layers import Dense, Dropout
from sklearn.model_selection import KFold
from keras_tuner.tuners import RandomSearch
from sklearn.compose import ColumnTransformer
from tensorflow.keras.models import load_model
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.layers import Dense, Dropout
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import ModelCheckpoint

file_placering = '/Users/mortenlassen/Desktop/Data_v7.xlsx'

df_data = pd.read_excel(file_placering, sheet_name='Combined')


X = df_data[['Ejerlejlighed','Antal værelser','Enhedsareal - Beboelse','Enhedsareal - Erhverv','Kælderareal','Familieoverdragelse','Fritidsbolig','Landbrugsbolig','Energimærke','Opførelsesår','Omtilbygningsår','Grundareal','Postnr']]
y = df_data['Handelspris']

numeriske_variable = X.select_dtypes(include=['int64', 'float64']).columns

numerisk_transformation = Pipeline(steps=[
    ('scaler', StandardScaler())
])

transformation_før_testing = ColumnTransformer(
    transformers=[
        ('num', numerisk_transformation, numeriske_variable),
    ])

X_forberedt = transformation_før_testing.fit_transform(X)

antal_variable = X_forberedt.shape[1]

X_train, X_test, y_train, y_test = train_test_split(X_forberedt, y, test_size=0.2, random_state=20)


def build_model(hp):
    model = Sequential()
    model.add(Dense(hp.Int('input_units', min_value=13, max_value=416, step=13),
                    input_shape=(antal_variable,),
                    activation='relu', 
                    kernel_regularizer=l1_l2(
                        l1=hp.Float('l1', min_value=0.01, max_value=0.25, step=0.01),
                        l2=hp.Float('l2', min_value=0.01, max_value=0.25, step=0.01))))
    
    for i in range(hp.Int('n_layers', 1, 10)):  #
        model.add(Dense(hp.Int(f'dense_{i}_units', min_value=13, max_value=416, step=32), 
                        activation='relu', 
                        kernel_regularizer=l1_l2(
                            l1=hp.Float(f'l1_{i}', min_value=0.01, max_value=0.1, step=0.01), 
                            l2=hp.Float(f'l2_{i}', min_value=0.01, max_value=0.1, step=0.01))))
        model.add(Dropout(hp.Float('dropout', 0, 0.5, step=0.01)))
    
    model.add(Dense(1, activation=None)) 

    model.compile(optimizer=Adam(learning_rate=hp.Float('learning_rate', min_value=1e-5, max_value=1e-1, sampling='log')),
                  loss='mean_squared_error',
                  metrics=['mean_squared_error'])

    return model

directory = 'my_dir/intro_to_kt'
if os.path.exists(directory):
    shutil.rmtree(directory)

hyperparameter = kt.HyperParameters()
hyperparameter.Int('units', min_value=13, max_value=416, step=13)
hyperparameter.Float('dropout', min_value=0.0, max_value=0.5, step=0.01)
hyperparameter.Float('learning_rate', min_value=1e-5, max_value=1e-1, sampling='LOG')

tuning_modellen = RandomSearch(
    build_model,
    objective='val_mean_squared_error',
    max_trials=100,  
    executions_per_trial=1,  
    directory='my_dir',
    project_name='intro_to_kt',
)

bedste_resultat = float('inf')
bedste_model = None
bedste_forsøg_id = None

k = 5
kf = KFold(n_splits=k, shuffle=True)

for fold, (træning_index, validering_index) in enumerate(kf.split(X_forberedt)):
    print(f'Running fold {fold + 1}/{k}')
    X_træning_k, X_validering_k = X_forberedt[træning_index], X_forberedt[validering_index]
    y_træning_k, y_validering_k = y.iloc[træning_index], y.iloc[validering_index]

    tuning_modellen.search(X_træning_k, y_træning_k,
                 epochs=100,
                 batch_size=200,
                 validation_data=(X_validering_k, y_validering_k))

    bedste_fold_resultat = tuning_modellen.oracle.get_best_trials(num_trials=1)[0].score

    if bedste_fold_resultat < bedste_resultat:
        bedste_resultat = bedste_fold_resultat
        bedste_forsøg_id = tuning_modellen.oracle.get_best_trials(num_trials=1)[0].trial_id
        
        bedste_model = tuning_modellen.get_best_models(num_models=1)[0]

bedste_model.save('path_to_save_model')

def optimal_model():
    model = Sequential()
    model.add(Dense(299, input_shape=(antal_variable,), activation='relu', 
                    kernel_regularizer=l1_l2(l1=0.19, l2=0.09)))
    
    layers = [
        (205, 0.06999999999999999, 0.060000000000000005),
        (141, 0.09, 0.03),
        (45, 0.09999999999999999, 0.060000000000000005),
        (45, 0.04, 0.09),
        (237, 0.03, 0.060000000000000005),
        (237, 0.09999999999999999, 0.01),
        (397, 0.01, 0.05),
        (77, 0.09999999999999999, 0.09999999999999999),
        (237, 0.09, 0.01)
    ]

    for units, l1, l2 in layers:
        model.add(Dense(units, activation='relu', kernel_regularizer=l1_l2(l1=l1, l2=l2)))
        model.add(Dropout(0.2))

    model.add(Dense(1, activation=None))  

    model.compile(optimizer=Adam(learning_rate=0.00016065713800105893),
                  loss='mean_squared_error',
                  metrics=['mean_squared_error'])

    return model

bedste_hyperparameters_sidst_run = tuning_modellen.get_best_hyperparameters()[0]

modellen = build_model(bedste_hyperparameters_sidst_run)

optimal_model = optimal_model()

checkpoint = ModelCheckpoint(
    'best_model.h5',          
    monitor='val_loss',       
    verbose=1,                
    save_best_only=True,      
    mode='min',               
    save_weights_only=False   
)

model_fittet = optimal_model.fit(
    X_forberedt,
    y,
    epochs=150,
    validation_split=0.2,
    batch_size=200,
    callbacks=[checkpoint]    
)

plt.figure(figsize=(14, 10))
plt.plot(model_fittet.history['loss'])
plt.plot(model_fittet.history['val_loss'])
plt.ylabel('Loss function')
plt.xlabel('Antal epochs')
plt.legend(['Træning', 'Test'], loc='upper right')
plt.show()

prædiktioner = modellen.predict(X_forberedt)

y_værdier = y.values
y_værdier = y_værdier.reshape(-1, 1)

mape = np.mean(100 * np.abs((y_værdier - prædiktioner) / y_værdier))

print(f"MAPE baseret på den bedste model: {mape:.2f}%")

y_værdier_i_millioner = y_værdier / 1e6
predictions_i_millioner = prædiktioner / 1e6

plt.figure(figsize=(14, 10))
plt.scatter(y_værdier_i_millioner, predictions_i_millioner, alpha=0.3)
plt.xlabel('Faktiske værdier (millioner)')
plt.ylabel('Estimerede værdier (millioner)')

plt.plot([y_værdier_i_millioner.min(), y_værdier_i_millioner.max()], 
         [y_værdier_i_millioner.min(), y_værdier_i_millioner.max()], 'k--', lw=2)
plt.show()
