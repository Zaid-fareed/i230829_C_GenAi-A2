# Error analysis - finetuned on test

Category counts: {'missed_decision': 26, 'evidence_only': 28, 'false_decision': 50, 'correct': 101, 'na_confusion': 1}
Decisive predictions with no cited sentence: 131

## E1 [false_decision] patient sigir-201410, trial NCT02097186
- criterion (exclusion): Pregnancy
- expert: **not applicable**, evidence []  | model: **not excluded**, evidence []
- expert evidence text: {}
- model evidence text: {}
- raw output: `{"label": "not excluded", "evidence": []}`

## E2 [false_decision] patient sigir-201411, trial NCT01526382
- criterion (exclusion): Patients were moribund.
- expert: **not enough information**, evidence [0, 1, 2, 3, 4]  | model: **not excluded**, evidence []
- expert evidence text: {0: 'A 40-year-old woman with no past medical history presents to the ER with excruciating pain in her right arm that had started 1 hour prior to her admission.', 1: 'She denies trauma.', 2: 'On examination she is pale and in moderate discomfort, as well as tachypneic and tachycardic.', 3: 'Her body temperature is normal and her blood pressure is 80/60.', 4: 'Her right arm has no discoloration or movement limitation.'}
- model evidence text: {}
- raw output: `{"label": "not excluded", "evidence": []}`

## E3 [false_decision] patient sigir-201417, trial NCT01745731
- criterion (exclusion): BMI> 40 kg/m2.
- expert: **not enough information**, evidence []  | model: **not excluded**, evidence []
- expert evidence text: {}
- model evidence text: {}
- raw output: `{"label": "not excluded", "evidence": []}`

## E4 [evidence_only] patient sigir-201410, trial NCT02097186
- criterion (inclusion): Patient willing to give full informed consent for participation
- expert: **included**, evidence [3]  | model: **included**, evidence [0]
- expert evidence text: {3: 'The patient will provide informed consent, and will comply with the trial protocol without any practical issues.'}
- model evidence text: {0: 'A physician is called to see a 67-year-old woman who underwent cardiac catheterization via the right femoral artery earlier in the morning.'}
- raw output: `{"label": "included", "evidence": [0]}`

## E5 [evidence_only] patient sigir-201411, trial NCT01723137
- criterion (exclusion): Less than 18 years of age
- expert: **not excluded**, evidence [0]  | model: **not excluded**, evidence []
- expert evidence text: {0: 'A 40-year-old woman with no past medical history presents to the ER with excruciating pain in her right arm that had started 1 hour prior to her admission.'}
- model evidence text: {}
- raw output: `{"label": "not excluded", "evidence": []}`

## E6 [evidence_only] patient sigir-201417, trial NCT01745731
- criterion (inclusion): Patients with liver space occupying lesion (LOE) that require extended hepatic resection.
- expert: **not enough information**, evidence [3]  | model: **not enough information**, evidence []
- expert evidence text: {3: 'Exploratory laparotomy reveals a ruptured liver abscess, which is then surgically drained.'}
- model evidence text: {}
- raw output: `{"label": "not enough information", "evidence": []}`

## E7 [missed_decision] patient sigir-201410, trial NCT02097186
- criterion (inclusion): Age greater than 18 years
- expert: **included**, evidence [0]  | model: **not enough information**, evidence []
- expert evidence text: {0: 'A physician is called to see a 67-year-old woman who underwent cardiac catheterization via the right femoral artery earlier in the morning.'}
- model evidence text: {}
- raw output: `{"label": "not enough information", "evidence": []}`

## E8 [missed_decision] patient sigir-201411, trial NCT01526382
- criterion (inclusion): Patients were included if they were diagnosed with Shock, Acute respiratory distress syndrome (ARDS), or both.
- expert: **included**, evidence [2, 3]  | model: **not enough information**, evidence []
- expert evidence text: {2: 'On examination she is pale and in moderate discomfort, as well as tachypneic and tachycardic.', 3: 'Her body temperature is normal and her blood pressure is 80/60.'}
- model evidence text: {}
- raw output: `{"label": "not enough information", "evidence": []}`

## E9 [missed_decision] patient sigir-201417, trial NCT01745731
- criterion (inclusion): Patients of both sexes aged ≥ 18 years.
- expert: **included**, evidence [0]  | model: **not enough information**, evidence []
- expert evidence text: {0: 'A 48-year-old white male with history of common variable immunodeficiency (CVID) with acute abdominal pain, fever, dehydration, HR of 132 bpm, BP 80/40.'}
- model evidence text: {}
- raw output: `{"label": "not enough information", "evidence": []}`

## E10 [na_confusion] patient sigir-201516, trial NCT00839124
- criterion (inclusion): --Negative pregnancy test for females
- expert: **not applicable**, evidence []  | model: **not enough information**, evidence []
- expert evidence text: {}
- model evidence text: {}
- raw output: `{"label": "not enough information", "evidence": []}`
