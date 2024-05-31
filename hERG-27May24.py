# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
#First we extract data from dataset
#Although Colab can mount your GoogleDrive it is a pain and actually quicker to access files from your Github account.
import pandas as pd
# df = pd.read_excel("Supplementary.xlsx")
df = pd.read_excel("https://github.com/vjeo0114/hERG-May-2024/blob/main/Supplementary.xlsx")
print(df.info())
df.head(2)

#Add a molecule column and make sure RDkt can convert all SMILES
from rdkit import Chem, DataStructs
from rdkit.Chem import PandasTools, AllChem
PandasTools.AddMoleculeColumnToFrame(df,'SMILES','Molecule')
df[["SMILES","Molecule"]].head(1)
#Check for smiles that rdkit can't convert to molecule. If sum = 0 then they are all OK
df.Molecule.isna().sum()
#Define function to generate fp's from SMILES
#Here we are producing a Morgan FP with radius 2 and calling for 1024 bits.
import numpy as np
def mol2fp(mol):
    fp = AllChem.GetHashedMorganFingerprint(mol, 2, nBits=1024)
    ar = np.zeros((1,), dtype=np.int8)
    DataStructs.ConvertToNumpyArray(fp, ar)
    return ar
# Demonstrate that the function "mol2fp" is working with a single SMILES.
fp =mol2fp(Chem.MolFromSmiles(df.loc[1,"SMILES"]))
fp
# Now use the mol2fp function to genereate fingerprints for all rdkit molecule objects.
# Here we are creating a new column called FPs in the dataframe df and applying the mol2fp function to it.
df["FPs"] = df.Molecule.apply(mol2fp)
df.head(2)
# Take a few molecule objects from our dataframe
mol_list = df['Molecule'].head(3).tolist()
# Now we can draw these and inspect them.
from rdkit import Chem
from rdkit.Chem import Draw
from rdkit.Chem.Draw import IPythonConsole
from rdkit.Chem import Descriptors
from rdkit.Chem import AllChem
from rdkit import DataStructs
img = Draw.MolsToGridImage(mol_list, molsPerRow=3)
img
# Now take the second molecule, 2,5-dimethoxy-4-chloroamphetamine, which we will call Dimethoxy4C for short.
Dimethoxy4C = df['Molecule'][1]

# Here we generate a fingerprint for Dimethoxy4C and display each "bit" that is "on"
bi = {}

fp = AllChem.GetMorganFingerprintAsBitVect(Dimethoxy4C, 2, nBits=1024, bitInfo=bi)
fp_arr = np.zeros((1,))
DataStructs.ConvertToNumpyArray(fp, fp_arr)
np.nonzero(fp_arr)
list(fp.GetOnBits())

# Here we draw each part of Dimethoxy4C corresponding to the "on" bits.
prints = [(Dimethoxy4C, x, bi) for x in fp.GetOnBits()]
Draw.DrawMorganBits(prints, molsPerRow=4, legends=[str(x) for x in fp.GetOnBits()])

# Now we want to extract the fingerprints into seperate columns for modelling.
fp_df = df["FPs"].apply(pd.Series)
fp_df.head(2)
print(fp_df.head())
#Bring back the outcome column
fp_df.insert(1024, "pIC50", df["pIC50"])
fp_df.head(2)
# Bring back the name column for later use
fp_df.insert(0, "NAME", df["NAME"])
fp_df.head(2)

#Load data from pandas dataframe
x = fp_df.iloc[:,1:1025]
y = fp_df['pIC50'].values
x.head(2)

y.shape
# sklearn has a nice tool for setting up training and testing sets in data modelling projects.
# Here we are splitting the data into 85% training data and 15% test data where x indicates the set of descriptor variables
# and y indicates the set of target values i.e. logA in this case.

from sklearn.ensemble import RandomForestRegressor
from sklearn.datasets import make_regression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn import set_config

xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=.15)
# Since the pIC50 is a scalar variable (non-binary) we want to do some kind of regression to model it.
#https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html

rfr = RandomForestRegressor(random_state=1)
rfr.fit(xtrain, ytrain)

print("Training Performance Statistics")
print("-------------------------------")
score = rfr.score(xtrain, ytrain)
print("R-squared:", score)

ypred = rfr.predict(xtrain)

mse = mean_squared_error(ytrain, ypred)
print("MSE: ", mse)
print("RMSE: ", mse**(1/2.0))

import numpy as np
import matplotlib.pyplot as plt
x = ytest
y = rfr.predict(xtest)
a, b = np.polyfit(x, y, 1)

#Find axis dimensions for scatter plot
import math
max_dim = math.ceil(max(x.max(),y.max())) +1
min_dim = math.floor(min(x.min(), y.min())) -1

plt.scatter(x, y)
plt.ylim([min_dim, max_dim])
plt.xlim([min_dim, max_dim])
plt.plot(x, a*x+b)
plt.title("pIC50 Prediction")
plt.xlabel('Test Set pIC50 Values')
plt.ylabel('Predicted pIC50 Values')
# The optional line below saves the figure to file as an eps vector graphic as required in many journals.
# If you can't open in try using Acrobat, it will be able to read the postcript.
plt.savefig('hERG-Practice.eps', format='eps')

print("Testset Prediction Performance")
print("------------------------------")
correlation = np.corrcoef(x, y)[0,1]
print("Correlation:", correlation)
#R-square for Test Set results (above is R-square for training results)
rsquare = correlation**2
print("Testset Rsquare:", rsquare)

# joblib can be used to save and load trained models.
# This is called pickelling the model so they are called dot pkl files.
import joblib
joblib.dump(rfr, 'hERG-Practice.pkl')

# load the model
model = joblib.load('hERG-Practice.pkl')

# Here we use the mol2fp function from above to convert a smiles string into a morgan fingerprint with radius 2.
test_smi = ("CCCNCCCCC")
m = mol2fp(Chem.MolFromSmiles(test_smi))
m

# Finally, we use model.predict to output a predicted value for pIC50.
model.predict([m])