import asyncio
import gzip
import os
import pickle
import statistics
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from itertools import chain, takewhile
from typing import Dict, List, Optional, Set, Tuple

import pandas as pd


def get_drugs_from_mimic_file(
    fileName: str, choice: str
) -> Tuple[Dict[str, str], Dict[int, list]]:
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
    mimicFile = gzip.open(fileName, "r")  # subject_id,hadm_id,gsn,ndc,drug
    mimicFile.readline()
    try:
        for line in mimicFile:
            tokens = line.decode("utf-8").strip().split(",")
            hadm_id = int(tokens[1])
            if choice == "ndc":
                drug_code = tokens[3]
            else:
                drug_code = tokens[2]
            drug_code = drug_code.strip()
            drug_code = "DR" + "_" + drug_code
            if hadm_id in mapping:
                mapping[hadm_id].append(drug_code.strip())
            else:
                mapping[hadm_id] = [drug_code.strip()]
            if drug_code not in drugDescription:
                drugDescription[drug_code] = tokens[4]
    except Exception as e:
        print(e)
    mimicFile.close()
    return drugDescription, mapping


def get_ICDs_from_mimic_file(
    fileName: str, isdiagnosis: bool = True
) -> Dict[int, List[str]]:
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
    mimicFile = gzip.open(fileName, "r")
    number_of_null_ICD9_codes = 0
    number_of_null_ICD10_codes = 0

    mimicFile.readline()  # ignore the header
    for line in mimicFile:
        tokens = line.decode("utf-8").strip().split(",")
        # print(tokens)
        hadm_id = int(tokens[1])
        # depending on the file, the ICD code is in different columns (3 for diagnosis, 4 for procedures)
        if (isdiagnosis and len(tokens[3]) == 0) or (
            not isdiagnosis and len(tokens[4]) == 0
        ):
            if isdiagnosis:
                if tokens[4] == "9":
                    # ignore diagnoses where ICD9_code is null
                    number_of_null_ICD9_codes += 1
                else:
                    number_of_null_ICD10_codes += 1
                continue
            else:
                if tokens[5] == "9":
                    # ignore diagnoses where ICD9_code is null
                    number_of_null_ICD9_codes += 1
                else:
                    number_of_null_ICD10_codes += 1
                continue

        if isdiagnosis:
            ICD_code = tokens[3]
        else:
            ICD_code = tokens[4]

        if ICD_code.find('"') != -1:
            ICD_code = ICD_code[1:-1].strip()  # toss off quotes and proceed

        # since diagnosis and procedure ICD9 codes have intersections, a prefix is necessary for disambiguation
        if isdiagnosis:
            ICD_code = "D" + tokens[4] + "_" + ICD_code
        else:
            ICD_code = "P" + tokens[5] + "_" + ICD_code

        if hadm_id in mapping:
            mapping[hadm_id].append(ICD_code.strip())
        else:
            mapping[hadm_id] = [ICD_code.strip()]

    mimicFile.close()
    print(
        "-Number of null ICD9 codes in file "
        + fileName
        + ": "
        + str(number_of_null_ICD9_codes)
    )
    print(
        "-Number of null ICD10 codes in file "
        + fileName
        + ": "
        + str(number_of_null_ICD10_codes)
    )
    return mapping


def prepare_note(
    file_name: str = "../physionet.org/files/mimic-iv-note/2.2/note/discharge.csv.gz",
    save_path="../mimic-iv-2.2/note/discharge.csv.gz",
):
    notes = pd.read_csv(file_name)
    notes.drop(
        ["note_id", "charttime", "storetime", "note_type", "note_seq"],
        axis=1,
        inplace=True,
    )
    if len(notes.hadm_id.unique()) == len(notes):  # thus can be used as an id
        print("unique hadm_id can be used as index")
        notes.set_index("hadm_id", inplace=True)
    else:
        raise Exception(
            "filter_notes notes needs this formatting to work, please check what went wrong"
        )
    notes.to_csv(save_path, index="hadm_id", compression="gzip")
    return notes


async def load_notes(note_file, index_col, load_note):
    """
    Load notes from a csv file
    """
    if load_note:
        print("loading notes, this may take a while...")
        notes = pd.read_csv(note_file, index_col=index_col)
        print("notes loaded")
        return notes
    return None


async def build_subject_id_admissions(admission_file: str):

    print("Building subject_id-admission mapping, admission-date mapping")

    subject_id_adm_map = {}
    infd = gzip.open(admission_file, "r")
    infd.readline()
    for line in infd:
        tokens = line.decode("utf-8").strip().split(",")
        subject_id = int(tokens[0])
        hadm_id = int(tokens[1])
        if subject_id in subject_id_adm_map:
            subject_id_adm_map[subject_id].add(hadm_id)
        else:
            subject_id_adm_map[subject_id] = set()
            subject_id_adm_map[subject_id].add(hadm_id)
    for subject_id in subject_id_adm_map.keys():
        subject_id_adm_map[subject_id] = list(subject_id_adm_map[subject_id])
    infd.close()

    print("Done building subject_id-admission mapping")
    return subject_id_adm_map


async def build_admissions_diagnosis(diagnosis_file):
    print("Building admission-diagnosis mapping")
    admDxMap = get_ICDs_from_mimic_file(diagnosis_file)
    print("Done building admission-diagnosis mapping")
    return admDxMap


async def build_admissions_procedure(procedure_file):
    print("Building admission-procedure mapping")
    admPxMap = get_ICDs_from_mimic_file(procedure_file, isdiagnosis=False)
    print("Done building admission-procedure mapping")
    return admPxMap


async def build_admissions_drug(prescription_file, choice):
    print("Building admission-drug mapping")
    drugDescription, admDrugMap = get_drugs_from_mimic_file(prescription_file, choice)
    print("Done building admission-drug mapping")

    return drugDescription, admDrugMap


async def load_mimic_data(
    admission_file: str,
    diagnosis_file: str,
    procedure_file: str,
    prescription_file: str,
    note_file: str,
    choice: Optional[str] = "ndc",
    load_note=True,
    index_col="hadm_id",
    **kw,
) -> Tuple[
    Dict[int, List[int]],
    Dict[int, List[int]],
    Dict[int, List[int]],
    Dict[int, List[int]],
    Dict[str, str],
    Optional[pd.DataFrame],
]:
    """
    Loads MIMIC data and returns various mappings.

    Args:
    - choice (Optional[str]): The choice of drug mapping. Defaults to 'ndc'.

    Returns:
    - subject_id_adm_map (dict): A dictionary mapping subject_id to a list of admission IDs.
    - admDxMap (dict): A dictionary mapping admission IDs to diagnosis codes.
    - admPxMap (dict): A dictionary mapping admission IDs to procedure codes.
    - admDrugMap (dict): A dictionary mapping admission IDs to drug codes.
    - drugDescription (dict): A dictionary mapping drug codes to drug descriptions.
    - note_file (str): The path to the note file.
    - note_file (bool): Whether to load the notes file.
    """

    subject_id_adm_map_task = asyncio.create_task(
        build_subject_id_admissions(admission_file)
    )

    admDxMap_task = asyncio.create_task(build_admissions_diagnosis(diagnosis_file))
    admPxMap_task = asyncio.create_task(build_admissions_procedure(procedure_file))
    admDrugMap_task = asyncio.create_task(
        build_admissions_drug(prescription_file, choice)
    )
    notes_task = asyncio.create_task(load_notes(note_file, index_col, load_note))

    (subject_id_adm_map, admDxMap, admPxMap, (drugDescription, admDrugMap), notes) = (
        await asyncio.gather(
            subject_id_adm_map_task,
            admDxMap_task,
            admPxMap_task,
            admDrugMap_task,
            notes_task,
        )
    )

    return subject_id_adm_map, admDxMap, admPxMap, admDrugMap, drugDescription, notes


def mean_std(dic):
    codes = [len(codes_adm) for codes_adm in dic.values()]
    mean = statistics.fmean(codes)
    std_dev = statistics.stdev(codes)
    return mean, std_dev


def print_mean_std(dic, name):
    mean, std_dev = mean_std(dic)
    print(f"- Average Number of {name} per visit: {mean:.2f} +- {std_dev:.2f}")


def countCodes(*dicts: Dict[int, List[str]]) -> int:
    all_values = [value for dic in dicts for value in dic.values()]
    code_counts = Counter(code for sublist in all_values for code in sublist)
    return len(code_counts)


def update_adm_code_list(
    subject_idAdmMap: Dict[int, List[int]],
    admDxMap: Dict[int, List[str]],
    admPxMap: Dict[int, List[str]],
    admDrugMap: Dict[int, List[str]],
) -> Tuple[Dict[int, List[str]], Dict[int, List[str]], Dict[int, List[str]]]:
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


def clean_data(
    subject_id_adm_map: Dict[int, List[int]],
    adm_dx_map: Dict[int, List[str]],
    adm_px_map: Dict[int, List[str]],
    adm_drug_map: Dict[int, List[str]],
    min_visits: int = 2,
) -> Tuple[
    Dict[int, List[int]],
    Dict[int, List[str]],
    Dict[int, List[str]],
    Dict[int, List[str]],
]:
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
    subject_del_list = []
    could_not_saved = 0
    average_visits_saved = []
    print(
        "Removing patient records who do not have all three medical codes for an admission"
    )
    for subject_id, hadm_ids in subject_id_adm_map.items():
        flag = False
        for i, hadm_id in enumerate(hadm_ids):
            if hadm_id not in adm_dx_map.keys():
                flag = True
            if hadm_id not in adm_px_map.keys():
                flag = True
            if hadm_id not in adm_drug_map.keys():
                flag = True
            if flag:
                if i > 1:
                    average_visits_saved.append(i + 1)
                    subject_id_adm_map[subject_id] = hadm_ids[:i]
                else:
                    could_not_saved += 1
                    subject_del_list.append(subject_id)
                break

    # todo: check if this is usefull now
    subject_del_list = list(set(subject_del_list))

    for subject_id_to_rm in subject_del_list:
        del subject_id_adm_map[subject_id_to_rm]

    adm_dx_map, adm_px_map, adm_drug_map = update_adm_code_list(
        subject_id_adm_map, adm_dx_map, adm_px_map, adm_drug_map
    )

    print(f"Removing patients who made less than {min_visits} admissions")
    subject_del_list = []
    for pid, admIdList in subject_id_adm_map.items():
        if len(admIdList) < min_visits:
            subject_del_list.append(pid)
            continue

    for i in subject_del_list:
        del subject_id_adm_map[i]

    adm_dx_map, adm_px_map, adm_drug_map = update_adm_code_list(
        subject_id_adm_map, adm_dx_map, adm_px_map, adm_drug_map
    )
    display_code_stats(adm_dx_map, adm_px_map, adm_drug_map)
    return subject_id_adm_map, adm_dx_map, adm_px_map, adm_drug_map


def create_CCS_CCSR_mapping(
    CCSRDX_file: str,
    CCSRPCS_file: str,
    CCSDX_file: str,
    CCSPX_file: str,
    dump: bool = True,
) -> Dict[str, str]:
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
    # This part creates an ICD-10 Diagnosis, Procedures map to CCS
    b = {}

    df = pd.read_csv(CCSRDX_file, low_memory=False)
    df = df[
        [
            "'ICD-10-CM CODE'",
            "'CCSR CATEGORY 1'",
            "'CCSR CATEGORY 2'",
            "'CCSR CATEGORY 3'",
            "'CCSR CATEGORY 4'",
            "'CCSR CATEGORY 5'",
            "'CCSR CATEGORY 6'",
        ]
    ]
    df = df.map(lambda x: str(x)[1:-1])
    df = df.set_index("'ICD-10-CM CODE'").T.to_dict("list")
    # remove null values
    for key, value in df.items():
        value = ["D10_" + x.strip("'").strip() for x in value if x.strip()]
        b["D10_" + key.strip("'")] = value

    # ICD-10 procedure code and prescription to CCS

    df = (
        pd.read_csv(CCSRPCS_file)[["'ICD-10-PCS'", "'PRCCSR'"]]
        .set_index("'ICD-10-PCS'")
        .T.to_dict("list")
    )
    for key, value in df.items():
        value = ["P10_" + x.strip("'").strip() for x in value if x.strip()]
        b["P10_" + key.strip("'")] = value

    # ICD-9 diagnosis code and prescription to CCS
    ccsTOdescription_Map = {}
    dxref_ccs_file = open(CCSDX_file, "r")
    dxref_ccs_file.readline()  # note
    dxref_ccs_file.readline()  # header
    dxref_ccs_file.readline()  # null
    for line in dxref_ccs_file:
        tokens = line.strip().split(",")
        b["D9_" + str(tokens[0][1:-1]).strip()] = "D9_" + str(tokens[1][1:-1]).strip()
        ccsTOdescription_Map["D9_" + str(tokens[1][1:-1]).strip()] = str(
            tokens[2][1:-1]
        ).strip()
    dxref_ccs_file.close()

    dxprref_ccs_file = open(CCSPX_file, "r")
    dxprref_ccs_file.readline()  # note
    dxprref_ccs_file.readline()  # header
    dxprref_ccs_file.readline()  # null
    for line in dxprref_ccs_file:
        tokens = line.strip().split(",")
        b["P9_" + str(tokens[0][1:-1]).strip()] = "P9_" + str(tokens[1][1:-1]).strip()
        ccsTOdescription_Map["P9_" + str(tokens[1][1:-1]).strip()] = str(
            tokens[2][1:-1]
        ).strip()

    dxprref_ccs_file.close()

    if dump:
        pickle.dump(b, open("ICD_9_10_to_CSS", "wb"), -1)
        pickle.dump(
            ccsTOdescription_Map, open("ccs_to_description_dictionary", "wb"), -1
        )
    print("Total ICD to CCS entries: " + str(len(b)))
    print("Total CCS codes/descriptions: " + str(len(ccsTOdescription_Map)))

    v1 = []
    for v in b.values():
        for val in v:
            v1.append(val)
    v1 = list(set(v1))
    print("Total number of unique codes (Diag + Proc):", len(v1))

    return ccsTOdescription_Map


def map_ccsr_description(filename: str, cat: str = "Diag") -> Dict[str, str]:
    """
    Maps CCSR (Clinical Classifications Software Refined) category codes to their descriptions.

    Args:
        filename (str): The path to the Excel file containing the CCSR categories.
        cat (str, optional): The category type ('Diag' or 'Proc'). Defaults to 'Diag'.

    Returns:
        Dict[str, str]: A dictionary mapping CCSR category codes to their descriptions.
    """
    if cat == "Diag":
        padStr = "D10_"
        codeDescription = pd.read_excel(
            filename, sheet_name="CCSR_Categories", skiprows=1
        )[["CCSR Category", "CCSR Category Description"]]

    else:
        padStr = "P10_"
        codeDescription = pd.read_excel(
            filename, sheet_name="CCSR Categories", skiprows=1
        )[["CCSR Category", "CCSR Category Description"]]

    # Identify duplicate values in 'CCSR Category' column
    codeDescription.dropna(inplace=True)
    duplicate_mask = codeDescription["CCSR Category"].duplicated(keep=False)
    # Filter the DataFrame to include only rows with duplicate 'CCSR Category'
    duplicates_df = codeDescription[duplicate_mask]

    # For each duplicate 'CCSR Category', keep the row with the longest 'CCSR Category Description'
    longest_descriptions_df = duplicates_df.loc[
        duplicates_df.groupby("CCSR Category")["CCSR Category Description"].apply(
            lambda x: x.str.len().idxmax()
        )
    ]
    non_duplicates_df = codeDescription[~duplicate_mask]
    codeDescription = pd.concat(
        [non_duplicates_df, longest_descriptions_df]
    ).sort_index()
    codeDescription["CCSR Category"] = codeDescription["CCSR Category"].map(
        lambda x: padStr + str(x)
    )
    codeDescription = codeDescription.set_index("CCSR Category").T.to_dict("list")

    return codeDescription


def convValuestoList(codeDic: Dict[str, str]) -> Dict[str, List[str]]:
    for key, value in codeDic.items():
        codeDic[key] = [value]
    return codeDic


def map_ICD_to_CCSR(
    mapping: Dict[int, List[int]]
) -> Tuple[Dict[int, List[str]], List[str], Set[str]]:
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

    icdTOCCS_Map = pickle.load(open("ICD_9_10_to_CSS", "rb"))
    CodesToInternalMap = {}
    missingCodes = []
    number_of_codes_missing = 0
    countICD9 = 0
    countICD10 = 0
    for hadm_id, ICDs_List in mapping.items():
        for ICD in ICDs_List:

            if ICD.startswith("D9_"):
                countICD9 += 1
            elif ICD.startswith("P10_"):
                countICD10 += 1
            elif ICD.startswith("D10_"):
                countICD10 += 1
            elif ICD.startswith("P9_"):
                countICD9 += 1
            else:
                print("Wrong coding format")
            try:

                CCS_code = icdTOCCS_Map[ICD]

                if hadm_id in CodesToInternalMap:
                    if isinstance(CCS_code, str):
                        CodesToInternalMap[hadm_id].append(CCS_code)
                    else:
                        for code in CCS_code:
                            CodesToInternalMap[hadm_id].append(code)

                else:
                    if isinstance(CCS_code, str):
                        CodesToInternalMap[hadm_id] = [CCS_code]
                    else:
                        for i in range(len(CCS_code)):
                            if i == 0:
                                CodesToInternalMap[hadm_id] = [CCS_code[i]]
                            else:
                                CodesToInternalMap[hadm_id].append(CCS_code[i])

            except KeyError:
                missingCodes.append(ICD)
                number_of_codes_missing += 1

    print(
        f"- Total number of ICD9 codes used {countICD9} and ICD10 codes: {countICD10}"
    )
    print(
        "- Total number (complete set) of ICD9+ICD10 codes (diag + proc): "
        + str(len(set(icdTOCCS_Map.keys())))
    )
    print(
        "- Total number of ICD codes missing in the admissions list: ",
        number_of_codes_missing,
    )

    return CodesToInternalMap, missingCodes


def min_max_codes(dic):
    countCode = []
    for codes in dic.values():
        countCode.append(len(codes))

    return min(countCode), max(countCode)


def display_code_stats(adDx, adPx, adDrug):
    print(f"- Total Number of diagnosis code {countCodes(adDx)}")
    print(f"- Total Number of procedure code {countCodes(adPx)}")
    print(f"- Total Number of drug code {countCodes(adDrug)}")
    print(f"- Total Number of all codes {countCodes(adDx,adPx,adDrug) }")

    print_mean_std(adPx, "procedure codes")
    print_mean_std(adDx, "diagnosis codes")
    print_mean_std(adDrug, "Drug codes")

    print(
        f"- Min. and max. Number of diagnosis code per admission {min_max_codes(adDx)}"
    )
    print(
        f"- Min. and max. Number of procedure code  per admission{min_max_codes(adPx)}"
    )
    print(f"- Min. and max. Number of drug code  per admission {min_max_codes(adDrug)}")


def icd_mapping(
    CCSRDX_file: str,
    CCSRPCS_file: str,
    CCSDX_file: str,
    CCSPX_file: str,
    D_CCSR_Ref_file: str,
    P_CCSR_Ref_file: str,
    adDx: Dict[int, List[int]],
    adPx: Dict[int, List[int]],
    adDrug: Dict[int, List[int]],
    drugDescription: Dict[str, str],
    **kw,
) -> Tuple[Dict[int, List[int]], Dict[int, List[int]], Dict[str, str]]:
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
    ccsTOdescription_Map = create_CCS_CCSR_mapping(
        CCSRDX_file, CCSRPCS_file, CCSDX_file, CCSPX_file
    )
    # getting the description of all codes
    print("creating diagnosis code description...")
    DxcodeDescription = map_ccsr_description(D_CCSR_Ref_file)
    print("creating procedure code description...")
    PxcodeDescription = map_ccsr_description(P_CCSR_Ref_file, cat="Proc")
    codeDescription = {**DxcodeDescription, **PxcodeDescription}
    codeDescription = {
        **codeDescription,
        **convValuestoList(ccsTOdescription_Map),
        **drugDescription,
    }
    # mapping diagnois codes
    print("addmision diagnosis codes...")
    adDx, missingDxCodes = map_ICD_to_CCSR(adDx)
    # mapping procedure codes
    print("addmision procedure codes...")
    adPx, missingPxCodes = map_ICD_to_CCSR(adPx)
    codeDescription["SOH"] = "Start of history"
    codeDescription["EOH"] = "End of history"
    codeDescription["BOV"] = "Beginning of visit"
    codeDescription["EOV"] = "End of visit"
    codeDescription["BOS"] = "Beginning of sequence"
    codeDescription["PAD"] = "Padding"

    display_code_stats(adDx, adPx, adDrug)

    return adDx, adPx, codeDescription, missingPxCodes, missingDxCodes


def trim(
    adDx: Dict[int, List[int]],
    adPx: Dict[int, List[int]],
    adDrug: Dict[int, List[int]],
    max_dx: int,
    max_px: int,
    max_drg: int,
) -> Tuple[Dict[int, List[int]], Dict[int, List[int]], Dict[int, List[int]]]:
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


def filter_subjects(
    subject_id_adm_map: Dict[int, List[int]], min_visits: int = 2
) -> Dict[int, List[int]]:
    """
    Filters subjects who've made less that min_visits visits

    Args:
        subject_id_adm_map (Dict[int, List[int]]): A dictionary mapping subject IDs to a list of admission IDs.
        min_visits (int, optional): The minimum number of visits required for a patient. Defaults to 2.

    Returns:
        Dict[int, List[int]]: A dictionary containing the filtered subject IDs and admission IDs.
    """
    subject_id_adm_map_ = {}
    filtred = 0
    for subject_id, adm_id_list in subject_id_adm_map.items():
        if len(adm_id_list) < min_visits:
            filtred += 1
            continue
        subject_id_adm_map_[subject_id] = adm_id_list

    print(f"Number of patients with less than {min_visits} visits: {filtred}")
    return subject_id_adm_map_


def filter_notes(
    notes: pd.DataFrame, subject_id_adm_map: Dict[int, List[int]], min_visits: int = 2
) -> Tuple[pd.DataFrame, Dict[int, List[int]]]:
    """
    Filters the notes dataframe based on the given subject_id_hadm_id_map and a minimum number of visits.

    Args:
        notes (pandas.DataFrame): The dataframe containing the notes.
        subject_id_adm_map (dict): A dictionary mapping subject IDs to a list of associated HADM IDs.
        min_visits (int): The minimum number of visits required for a subject to be included.

    Returns:
        filtered_notes (pandas.DataFrame): The filtered notes dataframe.
        subject_id_adm_map_ (dict): The filtered subject_id_hadm_id_map dictionary.

    """
    print(
        f"filtering notes where the subject has made less than {min_visits} successive visits..."
    )
    subject_id_adm_map_ = {}
    subjects_to_rm = 0
    visits_to_rm = 0
    for subject_id, hadm_ids in subject_id_adm_map.items():
        temp_df = notes[notes["subject_id"] == subject_id]
        if set(temp_df.index).issuperset(set(hadm_ids)):
            subject_id_adm_map_[subject_id] = hadm_ids
        else:
            temp_hadm_ids = list(takewhile(lambda x: x in set(temp_df.index), hadm_ids))
            try:
                # add one more visit, which will be the last one, as we do not feed the notes to the decoder.
                temp_hadm_ids.append(hadm_ids[len(temp_hadm_ids)])
            except IndexError:
                pass
            if len(temp_hadm_ids) > min_visits:
                subject_id_adm_map_[subject_id] = temp_hadm_ids
            else:
                subjects_to_rm += 1
                visits_to_rm += len(hadm_ids)
    print(
        f"found {subjects_to_rm} subjects and {visits_to_rm} visits that need to be removed"
    )
    return subject_id_adm_map_


def build_data(
    subject_id_adm_map: Dict[int, List[int]],
    adDx: Dict[int, List[int]],
    adPx: Dict[int, List[int]],
    adDrug: Dict[int, List[int]],
) -> Tuple[List[List[List[int]]], Dict[str, int]]:
    """
    Builds the data for patient trajectory forecasting.

    Args:
        subject_id_adm_map (dict): A dictionary mapping subject IDs to admission IDs.
        adDx (dict): A dictionary mapping admission IDs to diagnosis codes.
        adPx (dict): A dictionary mapping admission IDs to procedure codes.
        adDrug (dict): A dictionary mapping admission IDs to drug codes.
        minVisits (int, optional): The minimum number of visits required for a patient. Defaults to 2.

    Returns:
        Tuple[List[List[List[int]]], Dict[str, int]]: A tuple containing the processed patient sequences and the code types dictionary.
    """

    adPx, adDx, adDrug = map(
        lambda d: defaultdict(list, d), (adPx, adDx, adDrug)
    )  # add default [] for missing values

    sid_seq_map = {}

    print("Building subject-id, diagnosis, procedure, drugs mapping")

    for subject_id, adm_id_list in subject_id_adm_map.items():

        sid_seq_map[subject_id] = [
            (adDx[adm_id], adPx[adm_id], adDrug[adm_id]) for adm_id in adm_id_list
        ]

    seqs = []
    for _, visits in sid_seq_map.items():
        seq = []
        for visit in visits:
            # chain.from_iterable flattens the list of lists
            # dict.fromkeys used as an ordered set function
            # list() used to convert the dict_keys object to a list
            joined = list(dict.fromkeys(chain.from_iterable(visit)))
            seq.append(joined)
        seqs.append(seq)
    # to_review
    print("Converting Strings Codes into unique integer, and making types")
    codes_to_ids = defaultdict(lambda: len(codes_to_ids))
    new_seqs = []
    for patient in seqs:
        new_patient = []
        for visit in patient:
            new_visit = []
            for code in visit:
                new_visit.append(codes_to_ids[code])
            new_patient.append(new_visit)
        new_seqs.append(new_patient)
    return (
        new_seqs,
        dict(codes_to_ids),
    )


def remove_code(
    currentSeqs: List[List[List[int]]],
    tokens_to_ids: Dict[str, int],
    threshold: int = 5,
) -> Tuple[List[List[List[int]]], Dict[str, int], Dict[int, str]]:
    """
    Removes infrequent codes from the given sequences.

    Args:
        currentSeqs (List[List[List[int]]]): The input sequences containing codes.
        tokens_to_ids (Dict[str, int]): A dictionary mapping code types to their corresponding integer values.
        threshold (int, optional): The threshold value for removing infrequent codes. Codes with a count less than or equal to this threshold will be removed. Defaults to 5.

    Returns:
        Tuple[List[List[List[int]]], Dict[str, int], Dict[int, str]]: A tuple containing the updated sequences, the updated types dictionary, and the reverse types dictionary.
    """
    countCode = Counter()

    for visits in currentSeqs:
        for visit in visits:
            countCode.update(visit)

    codes = [key for key, value in countCode.items() if value <= threshold]

    print(f"Total number of codes removed: {len(codes)}")
    print(f"Total number of unique codes: {len(countCode)}")

    ids_to_tokens = {v: k for k, v in tokens_to_ids.items()}

    # Recreate a new mapping while taking into consideration the removed tokens
    tokens_to_ids = defaultdict(
        lambda: len(tokens_to_ids),
        {"PAD": 0, "BOH": 1, "BOS": 2, "BOV": 3, "EOV": 4, "EOH": 5},
    )

    def process_patient(patient: List[List[int]]) -> List[List[int]]:
        return [
            [tokens_to_ids[ids_to_tokens[code]] for code in visit if code not in codes]
            for visit in patient
        ]

    with ThreadPoolExecutor() as executor:
        updatedSeqs = list(executor.map(process_patient, currentSeqs))

    ids_to_tokens = {v: k for k, v in tokens_to_ids.items()}

    return updatedSeqs, dict(tokens_to_ids), ids_to_tokens

    # List of codes like : D9_660...


def generate_code_types(
    reverseTypes: Dict[int, str], outFile: str = "outputData/originalData/"
) -> Dict[str, int]:
    """
    Generate code types based on reverse types dictionary.

    Args:.
    - reverseTypes (Dict[int, str]): A dictionary containing reverse types.
    - outFile (str): The name of the output file

    Returns:
    - codeType (Dict[str, int]): A dictionary containing the generated code types.
    """
    ICD_9_10_to_CSS = pickle.load(open("ICD_9_10_to_CSS", "rb"))
    codeType = {}
    countD = 0
    countP = 0
    countDr = 0
    countT = 0

    for keys, values in reverseTypes.items():
        found = 0
        if keys not in codeType:
            if values.startswith("DR_"):
                found = 1
                codeType[keys] = "DR"
                countDr = countDr + 1
            elif (
                values == "PAD"
                or values == "BOH"
                or values == "BOS"
                or values == "BOV"
                or values == "EOV"
                or values == "EOH"
            ):
                found = 1
                codeType[keys] = "T"
                countT = countT + 1
            else:
                for k, v in ICD_9_10_to_CSS.items():
                    if values in v:
                        found = 1
                        if keys not in codeType:
                            if k.startswith("D"):
                                codeType[keys] = "D"
                                countD = countD + 1
                            elif k.startswith("P"):
                                codeType[keys] = "P"
                                countP = countP + 1
            if found == 0:
                print(keys, values)

    print(
        f"Number of Diagnosis codes: {countD}, Procedure codes: {countP}, Drug codes: {countDr}, special codes: {countT}"
    )
    pickle.dump(codeType, open(os.path.join(outFile, "ids_to_types.pkl"), "wb"), -1)

    return codeType


def save_files(
    patients_visits_sequences: List[List[List[int]]],
    tokens_ids_map: Dict[str, int],
    ids_to_types: Dict[str, str],
    subject_id_hadm_map: Optional[Dict[int, List[int]]] = None,
    outFile: str = "outputData/originalData/",
):
    """
    Save the patients_visits_sequences, tokens_ids_map,
    code description, and subject id to admissions list mapping to the output directory.

    Args:
    updatedSeqs (List[List[List[int]]]): The updated sequences to be saved.
    tokens_ids_map: The types to be saved.
    codeDescription (str): The code description to be saved.
    subject_id_hadm_map (dict, optional): The subject ID to admission ID mapping. Defaults to None.
    outpath (str, optional): The output path where the files will be saved. Defaults to 'outputData/originalData/'.
    """

    if not os.path.exists(outFile):
        os.makedirs(outFile)

    pickle.dump(
        patients_visits_sequences,
        open(outFile + "patients_visits_sequences.pkl", "wb"),
        -1,
    )
    pickle.dump(tokens_ids_map, open(outFile + "tokens_to_ids.pkl", "wb"), -1)
    pickle.dump(ids_to_types, open(outFile + "ids_to_types.pkl", "wb"), -1)
    if subject_id_hadm_map:
        pickle.dump(
            subject_id_hadm_map, open(outFile + "subject_id_hadm_map.pkl", "wb"), -1
        )
