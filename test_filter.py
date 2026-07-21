import pandas as pd
import json

df = pd.read_pickle(r'd:\DATASET\PROJECT\ksp_prototype_data\ksp_cleaned_prototype.pkl')
with open('crime_glossary.json') as f:
    glossary = json.load(f)

mask = pd.Series(True, index=df.index)
mask &= df['District_Name'].str.lower().str.contains('ballari')
mask &= df['FIR_YEAR'].astype(str).str.contains('2016')
mask &= df['CrimeHead_Name'].isin(glossary['theft_robbery'])

unit_cases = df[mask]
print("Found:", len(unit_cases))

context_cases = []
for _, r in unit_cases.head(5).iterrows():
    s = f"- Filter Match in {r.get('UnitName')} ({r.get('FIR_YEAR')}): {r.get('CrimeHead_Name')} - {str(r.get('Description', ''))[:200]}. Victims: {r.get('Male', 0)} M, {r.get('Female', 0)} F. IO: {r.get('IOName', 'Unknown')} (KGID: {r.get('KGID', 'Unknown')})"
    context_cases.append(s)

print("Context cases:", context_cases)
