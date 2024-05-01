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

**todos:**
- [ ] Implement mapk, and recall@k metrics
- [ ] Add function to save data splits, to increase iteration speed
- [ ] look into how to map admissions ids to "medical reports" for future work on bio-cliclical bert
