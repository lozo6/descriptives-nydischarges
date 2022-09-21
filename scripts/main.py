import pandas as pd
import numpy as np
import researchpy as rp
from tableone import TableOne
from statsmodels.formula.api import ols


sparcs = pd.read_csv('https://health.data.ny.gov/resource/gnzp-ekau.csv')

df = sparcs.dropna(axis=0)

df.shape # (950, 34) 940 rows and 34 columns after droping all empty rows

df.columns

used_columns = ['age_group', 'gender', 'race', 'ethnicity', 'length_of_stay', 'type_of_admission', 'apr_drg_code', 'ccs_diagnosis_code', 'apr_severity_of_illness_code']

for n in used_columns:
    print(f'{n} has {df[n].dtypes} dtype')

print(df.sample(10))

# prints out a summary of description of data in table
print(df.describe())

# finding the relationship between gender and apr_drg_code
gender_drg = ols('apr_drg_code ~ gender + 1', df).fit()
print(gender_drg.summary())

# finding the relationship between type of admission and length of stay
length_admission = ols('length_of_stay ~ type_of_admission', df).fit()
print(length_admission.summary())

df_columns = ['age_group', 'gender', 'race', 'ethnicity', 'length_of_stay', 'type_of_admission', 'apr_drg_code', 'ccs_diagnosis_code', 'apr_severity_of_illness_code']
df_categories = ['age_group', 'gender', 'race', 'ethnicity', 'type_of_admission', 'apr_severity_of_illness_code']
df_groupby = ['age_group']

df_table = TableOne(df, columns=df_columns, categorical=df_categories, groupby=df_groupby, pval=False)
print(df_table.tabulate())
df_table.to_csv('data/SPARCS_2.csv')

# descriptive data analytics using researchpy
rp.codebook(df)
rp.summary_cont(df[['length_of_stay', 'ccs_diagnosis_code', 'total_costs']])
rp.summary_cat(df[['age_group', 'gender', 'race', 'ethnicity', 'apr_severity_of_illness_code']])