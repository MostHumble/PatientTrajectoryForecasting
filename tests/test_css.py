import pandas as pd


def check_icd_ccs_pr_map(procedure_file: pd.DataFrame, CCSRPCS_file: pd.DataFrame, threshold : int = 15):
    """
    check if all ICD-10-PCS codes in the procedure file are present in the CCSRPCS file

    Args:
    procedure_file: pd.DataFrame
    CCSRPCS_file: pd.DataFrame
    threshold: int
    Returns:
    bool
    """
    return len(set(procedure_file[(procedure_file['icd_version']==10)].icd_code) - (set(CCSRPCS_file["'ICD-10-PCS'"].str.strip("'")))) < threshold 
    
def check_icd_ccs_dx_map(diagnosis_file: pd.DataFrame, CCSDX_file: pd.DataFrame, threshold : int = 15):
    """
    check if all ICD-10-CM codes in the diagnosis file are present in the CCSDX file

    Args:
    diagnosis_file: pd.DataFrame
    CCSDX_file: pd.DataFrame
    threshold: int
    Returns:
    bool
    """
    return len(set(diagnosis_file[(diagnosis_file['icd_version']==10) - (diagnosis_file['icd_code'].str.len()==7)].icd_code).issubset(set(CCSDX_file["'ICD-10-CM CODE'"].str.strip("'")))) < threshold 