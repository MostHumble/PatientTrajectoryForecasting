import os
import gzip
import pickle
import pandas as pd
from itertools import chain
from collections import defaultdict, Counter
from typing import Dict, Optional, Tuple, List, Set, Union

def get_drugs_from_mimic_file(fileName :str, choice : Optional[str] ='ndc') -> Tuple[Dict[str, str], Dict[int, list]]:
    """
    Extracts drug information from a MIMIC file.

    Args:
        fileName (str): The path to the MIMIC file.
        choice (str, optional): The choice of drug code to extract. Defaults to 'ndc'.

    Returns:
        tuple: A tuple containing two dictionaries:
            - drugDescription: A dictionary mapping drug codes to their descriptions.
            - mapping: A dictionary mapping hospital admission IDs to lists of drug codes.

    Raises:
        Exception: If an error occurs while processing the MIMIC file.
    """
    mapping = {}
    drugDescription = {}
    mimicFile = gzip.open(fileName, 'r')  # subject_id,hadm_id,gsn,ndc,drug
    mimicFile.readline()
    try:
        for line in mimicFile:
            tokens = line.decode('utf-8').strip().split(',')
            hadm_id = int(tokens[1])
            if choice =='ndc':
                drug_code = tokens[12]
            else:
                drug_code = tokens[11]
            drug_code = drug_code.strip()
            drug_code = 'DR'+'_'+drug_code
            if hadm_id in mapping:
                mapping[hadm_id].append(drug_code.strip())
            else:
                mapping[hadm_id]=[drug_code.strip()]
            if drug_code not in drugDescription:
                drugDescription[drug_code] = tokens[9]
    except Exception as e:
        print(e)
    mimicFile.close()
    return drugDescription, mapping

def get_ICDs_from_mimic_file(fileName: str, isdiagnosis: bool = True) -> Dict[int, List[str]]:
    """
    Retrieves ICD codes from a MIMIC file.

    Args:
        fileName (str): The path to the MIMIC file.
        isdiagnosis (bool, optional): Specifies whether to retrieve diagnosis codes (True) or procedure codes (False). 
                                      Defaults to True.

    Returns:
        dict: A dictionary mapping hospital admission IDs (hadm_id) to a list of corresponding ICD codes.

    """

    mapping = {}
    mimicFile = gzip.open(fileName, 'r')    
    number_of_null_ICD9_codes = 0
    number_of_null_ICD10_codes = 0

    mimicFile.readline() # ignore the header
    for line in mimicFile: 
        tokens = line.decode('utf-8').strip().split(',')
        #print(tokens)
        hadm_id = int(tokens[1])
        # depending on the file, the ICD code is in different columns (3 for diagnosis, 4 for procedures)
        if (isdiagnosis and len(tokens[3]) == 0) or (not isdiagnosis and len(tokens[4]) == 0):
            if isdiagnosis:
                if (tokens[4] =='9'):
                    # ignore diagnoses where ICD9_code is null
                    number_of_null_ICD9_codes += 1
                else:
                    number_of_null_ICD10_codes += 1
                continue
            else:
                if (tokens[5] =='9'):
                    # ignore diagnoses where ICD9_code is null
                    number_of_null_ICD9_codes += 1
                else:
                    number_of_null_ICD10_codes += 1
                continue
                
        if isdiagnosis:
            ICD_code = tokens[3]
        else:
            ICD_code = tokens[4] 

        if ICD_code.find("\"") != -1:
            ICD_code = ICD_code[1:-1].strip()  # toss off quotes and proceed

        # since diagnosis and procedure ICD9 codes have intersections, a prefix is necessary for disambiguation
        if isdiagnosis:
            ICD_code = 'D' + tokens[4]+ '_' +ICD_code
        else:
            ICD_code = 'P' + tokens[5] + '_' + ICD_code

        if hadm_id in mapping:
            mapping[hadm_id].append(ICD_code.strip())
        else:
            mapping[hadm_id]= [ICD_code.strip()]  

    mimicFile.close()
    print ('-Number of null ICD9 codes in file ' + fileName + ': ' + str(number_of_null_ICD9_codes))
    print ('-Number of null ICD10 codes in file ' + fileName + ': ' + str(number_of_null_ICD10_codes))
    return mapping

def load_mimic_data(admission_file : str, diagnosis_file : str, procedure_file : str, prescription_file : str, choice : Optional[str] = 'ndc', **kw) \
    -> Tuple[Dict[int,List[int]], Dict[int,List[int]], Dict[int,List[int]], Dict[int,List[int]], Dict[str, str]]:
    """
    Loads MIMIC data and returns various mappings.

    Args:
    - choice (Optional[str]): The choice of drug mapping. Defaults to 'ndc'.

    Returns:
    - subject_idAdmMap (dict): A dictionary mapping subject_id to a list of admission IDs.
    - admDxMap (dict): A dictionary mapping admission IDs to diagnosis codes.
    - admPxMap (dict): A dictionary mapping admission IDs to procedure codes.
    - admDrugMap (dict): A dictionary mapping admission IDs to drug codes.
    - drugDescription (dict): A dictionary mapping drug codes to drug descriptions.
    """
    print ('Building subject_id-admission mapping, admission-date mapping')

    subject_idAdmMap = {}
    infd = gzip.open(admission_file, 'r')
    infd.readline()
    for line in infd:
        tokens = line.decode('utf-8').strip().split(',')
        subject_id = int(tokens[0])
        hadm_id = int(tokens[1])
        if subject_id in subject_idAdmMap: 
            subject_idAdmMap[subject_id].add(hadm_id)
        else: 
            subject_idAdmMap[subject_id] = set()
            subject_idAdmMap[subject_id].add(hadm_id)
    for subject_id in subject_idAdmMap.keys():
        subject_idAdmMap[subject_id] = list(subject_idAdmMap[subject_id])  
    infd.close()

    print ('Building admission-diagnosis mapping')
    admDxMap = get_ICDs_from_mimic_file(diagnosis_file)

    print ('Building admission-procedure mapping')
    admPxMap = get_ICDs_from_mimic_file(procedure_file, isdiagnosis=False)

    print ('Building admission-drug mapping')
    drugDescription, admDrugMap = get_drugs_from_mimic_file(prescription_file, choice)
    return subject_idAdmMap,admDxMap,admPxMap,admDrugMap,drugDescription

def list_avg_visit(dic: Dict[int, List[int]]) -> float:
    a =[len(intList) for k,intList in dic.items()]
    return sum(a)/len(a)

def countCodes(*dicts: Dict[int, List[str]]) -> int:
    all_values = [value for dic in dicts for value in dic.values()]
    code_counts = Counter(code for sublist in all_values for code in sublist)
    return len(code_counts)

def display(pidAdmMap,admDxMap,admPxMap,admDrugMap):
    print(f" Total Number of patients {len(pidAdmMap)}")
    print(f" Total Number of admissions {len(admDxMap)}")
    print(f" Average number of admissions per patient {list_avg_visit(pidAdmMap)}")
    print(f" Total Number of diagnosis code {countCodes(admDxMap)}")
    print(f" Total Number of procedure code {countCodes(admPxMap)}")
    print(f" Total Number of drug code {countCodes(admDrugMap)}")
    print(f" Total Number of codes {countCodes(admPxMap) +countCodes(admDxMap)+countCodes(admDrugMap) }")
    print(f" average Number of procedure code per visit {list_avg_visit(admPxMap):.2f}")
    print(f" average Number of diagnosis code per visit {list_avg_visit(admDxMap):.2f}")
    print(f" average Number of Drug code per visit {list_avg_visit(admDrugMap)}")

def updateAdmCodeList(subject_idAdmMap: Dict[int, List[int]], admDxMap:  Dict[int, List[str]],
                       admPxMap : Dict[int, List[str]], admDrugMap :  Dict[int, List[str]]) \
      -> Tuple[Dict[int, List[str]], Dict[int, List[str]], Dict[int, List[str]]]:
    """
    Update the admission code lists for each admission ID (to take into account deleted elements).

    Args:
        subject_idAdmMap (dict): A dictionary mapping subject IDs to a list of admission IDs.
        admDxMap (dict): A dictionary mapping admission IDs to diagnosis codes.
        admPxMap (dict): A dictionary mapping admission IDs to procedure codes.
        admDrugMap (dict): A dictionary mapping admission IDs to drug codes.

    Returns:
        tuple: A tuple containing three dictionaries:
            - adDx: A dictionary mapping admission IDs to diagnosis codes.
            - adPx: A dictionary mapping admission IDs to procedure codes.
            - adDrug: A dictionary mapping admission IDs to drug codes.
    """
    adDx = {}
    adPx = {}
    adDrug = {}
    for subject_id, admIdList in subject_idAdmMap.items():
        for admId in admIdList:
            adDx[admId] = admDxMap[admId]
            adPx[admId] = admPxMap[admId]
            adDrug[admId] = admDrugMap[admId]
            
    return adDx, adPx, adDrug

def clean_data(subject_idAdmMap : Dict[int, List[int]], admDxMap : Dict[int, List[str]],
                admPxMap : Dict[int, List[str]], admDrugMap : Dict[int, List[str]], min_admissions_threshold : int = 2) \
      -> Tuple[Dict[int, List[int]], Dict[int, List[str]], Dict[int, List[str]], Dict[int, List[str]]]:
    """
    Cleans the data by removing patient records that do not have all three medical codes for an admission
    and removing patients who made less than a specified number of admissions.

    Args:
        subject_idAdmMap (dict): A dictionary mapping subject IDs to a list of admission IDs.
        admDxMap (dict): A dictionary mapping admission IDs to diagnostic codes.
        admPxMap (dict): A dictionary mapping admission IDs to procedure codes.
        admDrugMap (dict): A dictionary mapping admission IDs to drug codes.
        min_admissions_threshold (int, optional): The minimum number of admissions required for a patient to be included. Defaults to 2.

    Returns:
        tuple: A tuple containing the updated subject_idAdmMap, adDx, adPx, and adDrug dictionaries.
    """
    print("Cleaning data...")
    subDelList = []

    print("Removing patient records who do not have all three medical codes for an admission")
    for subject_id, hadm_ids in subject_idAdmMap.items():
        for hadm_id in hadm_ids:
            if hadm_id not in admDxMap.keys():
                subDelList.append(subject_id)
            if hadm_id not in admPxMap.keys():
                subDelList.append(subject_id)
            if hadm_id not in admDrugMap.keys():
                subDelList.append(subject_id)

    subDelList = list(set(subDelList))

    for subject_id_to_rm in subDelList:
        del subject_idAdmMap[subject_id_to_rm]


    adDx, adPx, adDrug = updateAdmCodeList(subject_idAdmMap, admDxMap, admPxMap, admDrugMap)

    print(f"Removing patients who made less than {min_admissions_threshold} admissions")
    pidMap = {}
    adm = []
    subDelList = []
    subject_idAdmMap1 = subject_idAdmMap
    for pid, admIdList in subject_idAdmMap.items():
        if len(admIdList) < min_admissions_threshold:
            subDelList.append(pid)
            continue

    for i in subDelList:
        del subject_idAdmMap[i]

    adDx, adPx, adDrug = updateAdmCodeList(subject_idAdmMap, adDx, adPx, adDrug)
    display(subject_idAdmMap, adDx, adPx, adDrug)
    return subject_idAdmMap, adDx, adPx, adDrug

def create_CCS_CCSR_mapping(CCSRDX_file : str, CCSRPCS_file : str, CCSDX_file : str, CCSPX_file : str, dump : bool = True) -> Dict[str, str]:
    """
    Creates a mapping of ICD-10 diagnosis and procedure codes to CCS tokens.

    Args:
        CCSRDX_file (str): The file path of the CCSRDX file containing ICD-10 diagnosis codes and CCS categories.
        CCSRPCS_file (str): The file path of the CCSRPCS file containing ICD-10 procedure codes and CCS categories.
        CCSDX_file (str): The file path of the CCSDX file containing ICD-9 diagnosis codes and CCS categories.
        CCSPX_file (str): The file path of the CCSPX file containing ICD-9 procedure codes and CCS categories.
        dump (bool, optional): Whether to dump the mapping dictionaries to pickle files. Defaults to True.

    Returns:
        dict: A dictionary mapping CCS codes to their descriptions.
    """
    # This part creates an ICD-10 Diagnosis, Procedures map to CCS token list
    df = pd.read_csv(CCSRDX_file)
    a = df[["'ICD-10-CM CODE'", "'CCSR CATEGORY 1'", "'CCSR CATEGORY 2'", "'CCSR CATEGORY 3'", "'CCSR CATEGORY 4'", "'CCSR CATEGORY 5'", "'CCSR CATEGORY 6'"]]

    a = a.map(lambda x: str(x)[1:-1])

    a = a.set_index("'ICD-10-CM CODE'").T.to_dict('list')
    # remove null values
    for key, value in a.items():
        newValue = []
        value = list(filter(lambda x: x.strip(), value))
        for value in value:
            newValue.append('D10_' + value)
        a[key] = newValue

    b = {}
    for key in a.keys():
        new_key = 'D10_' + key
        b[new_key] = a[key]

    df = pd.read_csv(CCSRPCS_file, on_bad_lines='skip')
    df = df[["'ICD-10-PCS'", "'PRCCSR'"]]
    df = df.map(lambda x: str(x)[1:-1])
    df = df.set_index("'ICD-10-PCS'").T.to_dict('list')

    for key, value in df.items():
        newValue = []
        value = list(filter(lambda x: x.strip(), value))
        for value in value:
            newValue.append('P10_' + value)
        df[key] = newValue

    for key in df.keys():
        new_key = 'P10_' + key
        b[new_key] = df[key]

    # ICD-9 diagnosis code and prescription to CCS
    ccsTOdescription_Map = {}
    dxref_ccs_file = open(CCSDX_file, 'r')
    dxref_ccs_file.readline()  # note
    dxref_ccs_file.readline()  # header
    dxref_ccs_file.readline()  # null
    for line in dxref_ccs_file:
        tokens = line.strip().split(',')
        b['D9_' + str(tokens[0][1:-1]).strip()] = 'D9_' + str(tokens[1][1:-1]).strip()
        ccsTOdescription_Map['D9_' + str(tokens[1][1:-1]).strip()] = str(tokens[2][1:-1]).strip()
    dxref_ccs_file.close()

    dxprref_ccs_file = open(CCSPX_file, 'r')
    dxprref_ccs_file.readline()  # note
    dxprref_ccs_file.readline()  # header
    dxprref_ccs_file.readline()  # null
    for line in dxprref_ccs_file:
        tokens = line.strip().split(',')
        b['P9_' + str(tokens[0][1:-1]).strip()] = 'P9_' + str(tokens[1][1:-1]).strip()
        ccsTOdescription_Map['P9_' + str(tokens[1][1:-1]).strip()] = str(tokens[2][1:-1]).strip()
    dxprref_ccs_file.close()

    if dump:
        pickle.dump(b, open('ICD_9_10_to_CSS', 'wb'), -1)
        pickle.dump(ccsTOdescription_Map, open('ccs_to_description_dictionary', 'wb'), -1)
    print('Total ICD to CCS entries: ' + str(len(b)))
    print('Total CCS codes/descriptions: ' + str(len(ccsTOdescription_Map)))

    v1 = []
    for v in b.values():
        for val in v:
            v1.append(val)
    v1 = list(set(v1))
    print("Total number of unique codes (Diag + Proc):", len(v1))

    return ccsTOdescription_Map

def map_ccsr_description(filename: str, cat: str = 'Diag') -> Dict[str, str]:
    """
    Maps CCSR (Clinical Classifications Software Refined) category codes to their descriptions.

    Args:
        filename (str): The path to the Excel file containing the CCSR categories.
        cat (str, optional): The category type ('Diag' or 'Proc'). Defaults to 'Diag'.

    Returns:
        Dict[str, str]: A dictionary mapping CCSR category codes to their descriptions.
    """
    if cat == 'Diag':
        padStr = 'D10_'
    else:
        padStr = 'P10_'
    df = pd.read_excel(filename, sheet_name="CCSR_Categories", skiprows=1)
    if type != 'Diag':
        df = df[:-1]
    codeDescription = df[["CCSR Category", "CCSR Category Description"]]
    codeDescription = codeDescription.map(lambda x: padStr + str(x))
    codeDescription = codeDescription.set_index("CCSR Category").T.to_dict('list')
    for key, value in codeDescription.items():
        newValue = value[0][4:]
        codeDescription[key] = newValue

    return codeDescription

def convValuestoList(codeDic : Dict[str, str]) -> Dict[str, List[str]]:
    for key, value in codeDic.items():
        codeDic[key] =  [value]
    return codeDic

def map_ICD_to_CCSR(mapping : Dict[int, List[int]]) -> Tuple[Dict[int, List[str]], List[str], Set[str]]:
    """
    Maps ICD codes to CCSR codes based on a given mapping.

    Args:
        mapping (Dict[int, List[int]]): A dictionary containing the mapping of hadm_id to a list of ICD codes.

    Returns:
        Tuple[Dict[int, List[str]], List[str], Set[str]]: A tuple containing the following:
            - CodesToInternalMap: A dictionary mapping hadm_id to a list of CCSR codes.
            - missingCodes: A list of ICD codes that could not be mapped to CCSR codes.
            - set_of_used_codes: A set of ICD codes that were successfully mapped to CCSR codes.
    """
    
    icdTOCCS_Map = pickle.load(open('ICD_9_10_to_CSS','rb'))
    CodesToInternalMap = {}
    missingCodes = []
    set_of_used_codes = set()
    number_of_codes_missing = 0
    countICD9=0
    countICD10 =0
    for (hadm_id, ICDs_List) in mapping.items():
        for ICD in ICDs_List:
            if ICD.startswith('D10_'):
                padStr = 'D10_'
            elif ICD.startswith('D9_'):
                padStr = 'D9_'
            elif ICD.startswith('P10_'):
                padStr = 'P10_'    
            elif ICD.startswith('P9_'):
                padStr = 'P9_'  
            else:
                print("Wrong coding format")

            try:

                CCS_code = icdTOCCS_Map[ICD]

                if hadm_id in CodesToInternalMap:
                    if(isinstance(CCS_code, str)): 
                        CodesToInternalMap[hadm_id].append(CCS_code)
                    else:
                        for code in CCS_code:
                            CodesToInternalMap[hadm_id].append(code)
                        
                else:
                    if(isinstance(CCS_code, str)): 
                        CodesToInternalMap[hadm_id] = [CCS_code]
                    else:
                        for i in range(len(CCS_code)):
                            if i==0:
                                CodesToInternalMap[hadm_id] = [CCS_code[i]]
                            else:
                                CodesToInternalMap[hadm_id].append(CCS_code[i])
                                
                            
                set_of_used_codes.add(ICD)

            except KeyError:
                missingCodes.append(ICD)
                number_of_codes_missing +=1


            
    print(f"total number of ICD9 codes used {countICD9} and ICD10 codes: {countICD10}")  
    print ('-Total number (complete set) of ICD9+ICD10 codes (diag + proc): ' + str(len(set(icdTOCCS_Map.keys()))))
    print ('-Total number of ICD codes actually used: ' + str(len(set_of_used_codes)))
    print ('-Total number of ICD codes missing in the admissions list: ' , number_of_codes_missing)
    
    return CodesToInternalMap,missingCodes,set_of_used_codes

def min_max_codes(dic):
    countCode = []
    for codes in dic.values():
        countCode.append(len(codes))    
                
    return min(countCode),max(countCode)

def display_code_stats(adDx ,adPx,adDrug):
    print(f" Total Number of diagnosis code {countCodes(adDx)}")
    print(f" Total Number of procedure code {countCodes(adPx)}")
    print(f" Total Number of drug code {countCodes(adDrug)}")
    print(f" Total Number of unique  D,P codes {countCodes(adDx,adPx) }")
    print(f" Total Number of all codes {countCodes(adDx,adPx,adDrug) }")


    print(f" average Number of procedure code per visit {list_avg_visit(adPx)}")
    print(f" average Number of diagnosis code per visit {list_avg_visit(adDx)}")
    print(f" average Number of drug code per visit {list_avg_visit(adDrug)}")

    print(f" Min. and max. Number of diagnosis code per admission {min_max_codes(adDx)}")
    print(f" Min. and max. Number of procedure code  per admission{min_max_codes(adPx)}")
    print(f" Min. and max. Number of drug code  per admission {min_max_codes(adDrug)}")

def icd_mapping(CCSRDX_file: str, CCSRPCS_file: str, CCSDX_file: str, CCSPX_file: str, D_CCSR_Ref_file: str, P_CCSR_Ref_file: str,\
                 adDx: Dict[int, List[int]], adPx: Dict[int, List[int]], adDrug: Dict[int, List[int]],\
                      drugDescription: Dict[str, str], **kw) -> Tuple[Dict[int, List[int]], Dict[int, List[int]], Dict[str, str]]:
    """
    Maps ICD codes to CCS and CCSR codes and returns the mapped diagnosis codes, procedure codes, and code descriptions.

    Args:
    - CCSRDX_file (str): Path to the CCSRDX file.
    - CCSRPCS_file (str): Path to the CCSRPCS file.
    - CCSDX_file (str): Path to the CCSDX file.
    - CCSPX_file (str): Path to the CCSPX file.
    - D_CCSR_Ref_file (str): Path to the D_CCSR_Ref file.
    - P_CCSR_Ref_file (str): Path to the P_CCSR_Ref file.
    - adDx (Dict[int, List[int]]): Dictionary containing the diagnosis codes.
    - adPx (Dict[int, List[int]]): Dictionary containing the procedure codes.
    - adDrug (Dict[int, List[int]]): Dictionary containing the drug codes.
    - drugDescription (Dict[str, str]): Dictionary containing the descriptions of drug codes.

    Returns:
    - Tuple[Dict[int, List[int]], Dict[int, List[int]], Dict[str, str]]: A tuple containing the mapped diagnosis codes, procedure codes, and code descriptions.
    """
    # creating mappint between all ICD codes to CCS and CCSR mapping
    ccsTOdescription_Map = create_CCS_CCSR_mapping(CCSRDX_file,CCSRPCS_file,CCSDX_file,CCSPX_file)
    # getting the description of all codes
    DxcodeDescription = map_ccsr_description(D_CCSR_Ref_file)
    PxcodeDescription = map_ccsr_description(P_CCSR_Ref_file, cat = 'Proc')
    codeDescription ={**DxcodeDescription ,**PxcodeDescription }
    codeDescription ={**codeDescription , **convValuestoList(ccsTOdescription_Map), **drugDescription}
    # mapping diagnois codes
    adDx,missingDxCodes,set_of_used_codes1 = map_ICD_to_CCSR(adDx)
    # mapping procedure codes
    adPx,missingPxCodes,set_of_used_codes2 = map_ICD_to_CCSR(adPx)
    codeDescription['SOH'] = 'Start of history'
    codeDescription['EOH'] = 'End of history'
    codeDescription['BOV'] = 'Beginning of visit'
    codeDescription['EOV'] = 'End of visit'
    codeDescription['BOS'] = 'Beginning of sequence'
    codeDescription['PAD'] = 'Padding'
    display_code_stats(adDx,adPx,adDrug)
    return adDx,adPx,codeDescription


def trim(adDx  : Dict[int,List[int]], adPx  : Dict[int,List[int]], adDrug  : Dict[int,List[int]], max_dx : int, max_px : int, max_drg: int)\
      -> Tuple[Dict[int,List[int]], Dict[int,List[int]], Dict[int,List[int]]]:
    """
    Trims the diagnosis, procedure, and medication codes for each visit.

    Args:
    adDx (dict): A dictionary containing admission IDs as keys and diagnosis codes as values.
    adPx (dict): A dictionary containing admission IDs as keys and procedure codes as values.
    adDrug (dict): A dictionary containing admission IDs as keys and medication codes as values.
    max_dx (int): The maximum number of diagnosis codes to keep for each admission.
    max_px (int): The maximum number of procedure codes to keep for each admission.
    max_drg (int): The maximum number of medication codes to keep for each admission.

    Returns:
    tuple: A tuple containing the trimmed dictionaries for diagnosis codes, procedure codes, and medication codes.
    """

    print("Trimming the diagnosis, procedure, and medication codes for each visit")
    
    for admission, DiagCodes in adDx.items():
        adDx[admission] = DiagCodes[:max_dx]
        
    for admission, ProcCodes in adPx.items():
        adPx[admission] = ProcCodes[:max_px]
        
    for admission, DrugCodes in adDrug.items():
        adDrug[admission] = DrugCodes[:max_drg]
        
    display_code_stats(adDx, adPx, adDrug)
    return adDx, adPx, adDrug

def build_data(subject_idAdmMap : Dict[int, List[int]], adDx: Dict[int, List[int]], adPx: Dict[int, List[int]], adDrug: Dict[int, List[int]], minVisits: int = 2) -> Tuple[List[List[List[int]]], Dict[str, int]]:
    """
    Builds the data for patient trajectory forecasting.

    Args:
        subject_idAdmMap (dict): A dictionary mapping subject IDs to admission IDs.
        adDx (dict): A dictionary mapping admission IDs to diagnosis codes.
        adPx (dict): A dictionary mapping admission IDs to procedure codes.
        adDrug (dict): A dictionary mapping admission IDs to drug codes.
        minVisits (int, optional): The minimum number of visits required for a patient. Defaults to 2.

    Returns:
        Tuple[List[List[List[int]]], Dict[str, int]]: A tuple containing the processed patient sequences and the code types dictionary.
    """
    
    adPx, adDx, adDrug = map(lambda d: defaultdict(list, d), (adPx, adDx, adDrug)) # add default [] for missing values

    print(f'Building admission-Visits mapping & filtering patients with less than {minVisits} ')
    pidSeqMap = {}
    
    skipped = 0 
    for subject_id, admIdList in subject_idAdmMap.items():
        if len(admIdList) < minVisits: 
            skipped += 1
            continue # skip patients with less than minVisits ( default 1 )
        sortedList = [(adDx[admId], adPx[admId], adDrug[admId]) for admId in admIdList]
        
        pidSeqMap[subject_id] = sortedList
        
    adPx, adDx, adDrug = map(dict, (adPx, adDx, adDrug))  # remove default [] behavior to not break something

    print(f'{skipped} subjects were removed')
    print('Building subject-id, diagnosis, procedure, drugs mapping')
    subject_ids = []
    dates = []
    seqs = []
    ages = []
    for subject_id, visits in pidSeqMap.items():
        subject_ids.append(subject_id)
        diagnose = []
        procedure = []
        drugs = []
        date = []
        seq = []
        for visit in visits:
            joined = list(dict.fromkeys(chain.from_iterable(visit))) # dict.fromkeys used as an ordered set function
            seq.append(joined)
        seqs.append(seq)

    print('Converting Strings Codes into unique integer, and making types')
    types = {}
    newSeqs = []
    for patient in seqs:
        newPatient = []
        #print("patient",patient)
        for visit in patient:
            #print("visit",visit)
            newVisit = []
            for code in visit:
                #print("code",code)
                if code in types:
                    newVisit.append(types[code])
                else:
                    types[code] = len(types)
                    newVisit.append(types[code])
                    #print("newVisit",newVisit)
            newPatient.append(newVisit)
        newSeqs.append(newPatient)
    return newSeqs, types

def remove_code(currentSeqs : List[List[List[int]]], types, threshold :int = 5) -> Tuple[List[List[List[int]]], Dict[str, int], Dict[int, str]]:
    """
    Removes infrequent codes from the given sequences.

    Args:
        currentSeqs (List[List[List[int]]]): The input sequences containing codes.
        types (Dict[str, int]): A dictionary mapping code types to their corresponding integer values.
        threshold (int, optional): The threshold value for removing infrequent codes. Codes with a count less than or equal to this threshold will be removed. Defaults to 5.

    Returns:
        Tuple[List[List[List[int]]], Dict[str, int], Dict[int, str]]: A tuple containing the updated sequences, the updated types dictionary, and the reverse types dictionary.
    """
    countCode = Counter()
    
    for visits in currentSeqs:
        for visit in visits:
            countCode.update(visit)
            
    codes = [key for key, value in countCode.items() if value <= threshold]
    
    print(f" Total number of codes removed: {len(codes)}  ")
    print(f" Total number of  unique codes : {len(countCode)}  ")

    reverseTypes = {v:k for k,v in types.items()}

    # List of codes like : D9_660...
    types = defaultdict(lambda: len(types), {"PAD": 0,"BOH":1 ,"BOS": 2, "BOV": 3, "EOV": 4, "EOH": 5})

    # Recreates a new mapping while taking into consideration the removed tokens
    updatedSeqs = [[[types[reverseTypes[code]] for code in visit if code not in codes] for visit in patient] for patient in currentSeqs]
    
    reverseTypes = {v:k for k,v in types.items()}

    return updatedSeqs, dict(types), reverseTypes

def generate_code_types(reverseTypes: Dict[int, str], outFile : str = 'outputData/originalData/') -> Dict[str, int] :
    """
    Generate code types based on reverse types dictionary.

    Args:.
    - reverseTypes (Dict[int, str]): A dictionary containing reverse types.
    - outFile (str): The name of the output file

    Returns:
    - codeType (Dict[str, int]): A dictionary containing the generated code types.
    """
    ICD_9_10_to_CSS = pickle.load(open('ICD_9_10_to_CSS', 'rb'))
    codeType = {}
    countD = 0
    countP = 0
    countDr = 0
    countT = 0

    for keys, values in reverseTypes.items():
        found = 0
        if keys not in codeType:
            if values.startswith('DR_'):
                found = 1
                codeType[keys] = 'DR'
                countDr = countDr + 1
            elif values == 'PAD' or values == 'BOH' or values == "BOS" or values == 'BOV' or values == 'EOV' or values == 'EOH':
                found = 1
                codeType[keys] = 'T'
                countT = countT + 1
            else:
                for k, v in ICD_9_10_to_CSS.items():
                    if values in v:
                        found = 1
                        if keys not in codeType:
                            if k.startswith('D'):
                                codeType[keys] = 'D'
                                countD = countD + 1
                            elif k.startswith('P'):
                                codeType[keys] = 'P'
                                countP = countP + 1
            if found == 0:
                print(keys, values)

    print(f'Number of Diagnosis codes: {countD}, Procedure codes: {countP}, Drug codes: {countDr}, special codes: {countT}')
    pickle.dump(codeType, open(os.path.join(outFile , 'data.code_types'), 'wb'), -1)

    return codeType

def save_files(updatedSeqs : List[List[List[int]]], types : Dict[str, int], codeDescription : Dict[str, str], outFile : str = 'outputData/originalData/'):
    """
    Save the updated sequences, types, and code description to files.

    Args:
    updatedSeqs (List[List[List[int]]]): The updated sequences to be saved.
    types: The types to be saved.
    codeDescription (str): The code description to be saved.
    outpath (str, optional): The output path where the files will be saved. Defaults to 'outputData/originalData/'.
    """

    if not os.path.exists(outFile):
        os.makedirs(outFile)
    
    pickle.dump(updatedSeqs, open(outFile + 'data.seqs', 'wb'), -1)
    pickle.dump(types, open(outFile + 'data.types', 'wb'), -1)
    pickle.dump(codeDescription, open(outFile + 'data.description', 'wb'), -1)