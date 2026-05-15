"""
Creates smaller dataset for embedding by randomly selecting some number of unique patients
and storing the only the examples associated with those patients in new files in data/
"""

import pandas as pd
import numpy as np
import json

np.random.seed(42)

admissions_df = pd.read_csv('data/admissions_summary.csv')

all_patient_ids = admissions_df['subject_id'].unique()
rand_patient_ids = np.random.choice(all_patient_ids, 500)
remaining_patient_ids = np.setdiff1d(all_patient_ids, rand_patient_ids)
test_patient_ids = np.random.choice(remaining_patient_ids, 500, replace=False)

admissions_df_small = admissions_df[admissions_df['subject_id'].isin(rand_patient_ids)]
print(admissions_df_small.shape)
admissions_df_small.to_csv('data/admissions_summary_small.csv', index=False)

pairs_df = pd.read_json('data/temporal_pairs.json')
pairs_df_cut = pairs_df[pairs_df['subject_id'].isin(rand_patient_ids)]
print(pairs_df_cut.shape)
pairs_df_cut.to_json('data/temporal_pairs_small.json', orient='records', indent=2)

notes_df = pd.read_csv('data/notes_with_icd.csv')
notes_df_cut = notes_df[notes_df['subject_id'].isin(rand_patient_ids)]
print(notes_df_cut.shape)
notes_df_cut.to_csv('data/notes_with_icd_small.csv', index=False)

notes_df_test = notes_df[notes_df['subject_id'].isin(test_patient_ids)]
print(notes_df_test.shape)
notes_df_test.to_csv('data/notes_with_icd_small_test.csv')