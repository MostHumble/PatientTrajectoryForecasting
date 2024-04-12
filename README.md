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
Planning to implement stratification using:
- **MIMIC-IV Splits Generator by JoakimEdin**
  - Source: [GitHub](https://github.com/JoakimEdin/medical-coding-reproducibility/blob/main/prepare_data/generate_mimiciv_splits.py)

More details on the tasks I'm checking: 

- [x] Load MIMIC data
- [x] Clean data
- [x] Map ICD data to CCS and CCSR
- [x] Trim codes assigned per visit based on a threshold
- [x] Build the data
- [x] Remove codes with occurrence less than a certain threshold
- [ ] Save the data before formatting based on the task
- [ ] Prepare data for Trajectory Forecasting
- [ ] Remove certain codes from output for different data formats
- [ ] Store files for Trajectory Forecasting
- [ ] Prepare data for Sequential disease prediction
- [ ] Remove certain codes from output for Sequential disease prediction
