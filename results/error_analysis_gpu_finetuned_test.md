# Error analysis - finetuned on test

Category counts: {'correct': 111, 'missed_decision': 13, 'polarity_flip': 3, 'false_decision': 42, 'evidence_only': 35, 'na_confusion': 2}
Decisive predictions with no cited sentence: 130

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

## E4 [evidence_only] patient sigir-201410, trial NCT02264964
- criterion (exclusion): Had been performed central venous puncture or catheterization before.
- expert: **not excluded**, evidence [0]  | model: **not excluded**, evidence []
- expert evidence text: {0: 'A physician is called to see a 67-year-old woman who underwent cardiac catheterization via the right femoral artery earlier in the morning.'}
- model evidence text: {}
- raw output: `{"label": "not excluded", "evidence": []}`

## E5 [evidence_only] patient sigir-201411, trial NCT01526382
- criterion (inclusion): bilateral pulmonary infiltrates or a chest radiograph consistent with edema;
- expert: **not enough information**, evidence []  | model: **not enough information**, evidence [0, 1, 2, 3, 4]
- expert evidence text: {}
- model evidence text: {0: 'A 40-year-old woman with no past medical history presents to the ER with excruciating pain in her right arm that had started 1 hour prior to her admission.', 1: 'She denies trauma.', 2: 'On examination she is pale and in moderate discomfort, as well as tachypneic and tachycardic.', 3: 'Her body temperature is normal and her blood pressure is 80/60.', 4: 'Her right arm has no discoloration or movement limitation.'}
- raw output: `{"label": "not enough information", "evidence": [0, 1, 2, 3, 4]}`

## E6 [evidence_only] patient sigir-201417, trial NCT01745731
- criterion (inclusion): Patients with liver space occupying lesion (LOE) that require extended hepatic resection.
- expert: **not enough information**, evidence [3]  | model: **not enough information**, evidence []
- expert evidence text: {3: 'Exploratory laparotomy reveals a ruptured liver abscess, which is then surgically drained.'}
- model evidence text: {}
- raw output: `{"label": "not enough information", "evidence": []}`

## E7 [missed_decision] patient sigir-201410, trial NCT02097186
- criterion (inclusion): Patients undergoing elective carotid endarterectomy
- expert: **not included**, evidence []  | model: **not enough information**, evidence []
- expert evidence text: {}
- model evidence text: {}
- raw output: `{"label": "not enough information", "evidence": []}`

## E8 [missed_decision] patient sigir-201411, trial NCT01526382
- criterion (inclusion): Patients were included if they were diagnosed with Shock, Acute respiratory distress syndrome (ARDS), or both.
- expert: **included**, evidence [2, 3]  | model: **not enough information**, evidence []
- expert evidence text: {2: 'On examination she is pale and in moderate discomfort, as well as tachypneic and tachycardic.', 3: 'Her body temperature is normal and her blood pressure is 80/60.'}
- model evidence text: {}
- raw output: `{"label": "not enough information", "evidence": []}`

## E9 [missed_decision] patient sigir-201421, trial NCT00997100
- criterion (inclusion): Present with active SLE disease with at least one of the following symptoms: i) Arthritis - > 2 joints with pain and signs of inflammation (i.e. tenderness, swelling, or effusion) ii) Inflammatory-type skin rash iii) Oral ulcers
- expert: **included**, evidence [0, 1]  | model: **not enough information**, evidence []
- expert evidence text: {0: 'A 21-year-old female is evaluated for progressive arthralgias and malaise.', 1: 'On examination she is found to have alopecia, a rash mainly distributed on the bridge of her nose and her cheeks, a delicate non-palpable purpura on her calves, and swelling and tenderness of her wrists and ankles.'}
- model evidence text: {}
- raw output: `{"label": "not enough information", "evidence": []}`

## E10 [polarity_flip] patient sigir-201410, trial NCT02097186
- criterion (inclusion): Patients undergoing open abdominal aortic aneurysm repair
- expert: **not included**, evidence []  | model: **included**, evidence [0]
- expert evidence text: {}
- model evidence text: {0: 'A physician is called to see a 67-year-old woman who underwent cardiac catheterization via the right femoral artery earlier in the morning.'}
- raw output: `{"label": "included", "evidence": [0]}`

## E11 [na_confusion] patient sigir-201422, trial NCT00236912
- criterion (inclusion): If female, using birth control
- expert: **not applicable**, evidence [3]  | model: **not enough information**, evidence []
- expert evidence text: {3: 'She has had no sexual partners and her menses are regular.'}
- model evidence text: {}
- raw output: `{"label": "not enough information", "evidence": []}`

## E12 [na_confusion] patient sigir-201513, trial NCT01255670
- criterion (inclusion): female patients have adequate birth-control method
- expert: **not applicable**, evidence []  | model: **not enough information**, evidence []
- expert evidence text: {}
- model evidence text: {}
- raw output: `{"label": "not enough information", "evidence": []}`
