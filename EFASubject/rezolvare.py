import numpy as np
import pandas as pd
import factor_analyzer as fa
from factor_analyzer import calculate_bartlett_sphericity, calculate_kmo

vot_categorii_sex = pd.read_csv("DateIN/Vot_Categorii_Sex.csv")
coduri_localitati = pd.read_csv("DateIN/Coduri_Localitati.csv")
b_df = vot_categorii_sex.drop(["Siruta"], axis=1)
merged_df = vot_categorii_sex.merge(coduri_localitati, how="outer", on="Siruta")
merged_df.drop(["Localitate_X", "Localitate_y"], inplace=True, axis=1)
merged_df.rename(columns={"Localitate_x" : "Localitate"}, inplace=True)

print(coduri_localitati.head())
print(vot_categorii_sex.head())

vot_categorii_sex.fillna(vot_categorii_sex.select_dtypes(include='number').mean(), inplace=True)

vot_categorii_sex["Categorie"] = vot_categorii_sex.select_dtypes(include='number').idxmin(axis=1)

df_cerinta1 = vot_categorii_sex[["Siruta", "Localitate", "Categorie"]]

df_cerinta1.to_csv("DateOUT/cerinta1.csv", index=False)

print(merged_df.head())
merged_df.drop(["Siruta", "Localitate"], inplace=True, axis=1)
grouped_df = merged_df.groupby(["Judet"]).mean()

grouped_df.to_csv("DateOUT/cerinta2.csv", index=True)

print(grouped_df.head())

##Trecand la treburi mai serioase
#Pregatire date pentru EFA
b_df.set_index("Localitate", inplace=True)
print(b_df.head())

chi_square, p_val = calculate_bartlett_sphericity(b_df) #p val < 0.05
kmo_per_var, kmo_total = calculate_kmo(b_df) # kmo_total > 0.6
print("p value: ", p_val, ", kmo:", kmo_total)

analyzer = fa.FactorAnalyzer(n_factors=b_df.shape[1], rotation=None)

analyzer.fit(b_df)
eigenvalues, v = analyzer.get_eigenvalues()

n_factors = np.sum(eigenvalues > 1)

print("Numar optim de factori: ", n_factors)

fa_no_rot = fa.FactorAnalyzer(n_factors=n_factors, rotation=None)
scores = fa_no_rot.fit_transform(b_df)

pd.DataFrame(scores).to_csv("DateOUT/factor_analyzer.csv", index=False)
print(scores.shape)#coordonatele fiecarei instante in spatiul factorilor noi