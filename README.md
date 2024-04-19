## Patient Trajectory Forecasting

### Overview
Currently focused on dataset preparation for patient trajectory forecasting.

### Sources of Inspiration
Drawing inspiration from various repositories:

1. **Clinical-GAN by vigi30**
   - Source: [GitHub](https://github.com/vigi30/Clinical-GAN/blob/5f32201fda798ca91129f22a9c52820d2aa3e414/process_data.py)

2. **LIG-Doctor by jfrjunio**
   - Source: [GitHub](https://github.com/jfrjunio/LIG-Doctor/blob/master/preprocess_mimiciii.py)

3. **DoctorAI by mp2893**
   - Source: [GitHub](https://github.com/mp2893/doctorai/blob/master/process_mimic.py)

### Stratification Strategy
# This tasks is pending...
Planning to implement stratification using:
- **MIMIC-IV Splits Generator by JoakimEdin**
  - Source: [GitHub](https://github.com/JoakimEdin/medical-coding-reproducibility/blob/main/prepare_data/generate_mimiciv_splits.py)

**Finished on the afternoon of april 16th :** 

- [x] Load MIMIC data
- [x] Clean data
- [x] Map ICD data to CCS and CCSR
- [x] Trim codes assigned per visit based on a threshold
- [x] Build the data
- [x] Remove codes with occurrence less than a certain threshold
- [x] Save the data before formatting based on the task
- [x] Prepare data for Trajectory Forecasting
- [x] Remove certain codes from output for different data formats
- [x] Store files for Trajectory Forecasting
- [x] Prepare data for Sequential disease prediction
- [x] Remove certain codes from output for Sequential disease prediction
- [x] Add test branch
- [x] Clean the code
- [x] Implement the function to add special tokens
- [ ] Making tests before merge
