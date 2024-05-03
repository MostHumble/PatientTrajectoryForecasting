## Patient Trajectory Forecasting

### Overview
Currenly investingating the impact of positionnal embeddings (PE). The main issue that could be problematic is the prediction of the EOV tokens, if the model is unable to predict an appropriate position for the EOV we will have problems with the number of diagnosis codes predicted, as we currently stop the prediction when `pred_tok_id == eov_Id`

**todos:**
- [x] Implement map@k
- [x] Add option for positionnal embeddings

**Next steps**
- [ ] investigate other potential metrics ( mainly looking into recommender systems ) 
- [ ] look into how to map admissions ids to "medical reports" for future work on bio-cliclical bert
