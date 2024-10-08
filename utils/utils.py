import os
import pickle
from typing import Dict, List, Tuple


def load_data(
    path: str = "outputData/originalData/",
    updated_ids_to_types: bool = False,
    train: bool = False,
    processed_data: bool = False,
    with_notes: bool = False,
    reindexed: bool = False,
) -> Tuple[List[List[List[int]]], Dict[str, int], Dict[str, int], Dict[int, str]]:
    """
    Load data from the specified file.

    Args:
    - path (str): The path to the file containing the data.

    Returns:
    - Tuple[List[List[List[int]]], Dict[str, ], Dict[str, int], Dict[int, str]]: A tuple containing the loaded data.
        - The first element is a list of sequences, where each sequence is a list of events, and each event is a list of integers.
        - The second element is a dictionary mapping event types to their corresponding codes.
        - The third element is a dictionary mapping event codes to their corresponding types.
        - The fourth element is a dictionary mapping event codes to their corresponding types (reversed mapping).
    """

    if train:

        source_target_sequences = pickle.load(
            open(os.path.join(path, "source_target_sequences.pkl"), "rb")
        )
        ids_to_types_map = pickle.load(
            open(os.path.join(path, "ids_to_types.pkl"), "rb")
        )
        tokens_to_ids_map = pickle.load(
            open(os.path.join(path, "tokens_to_ids.pkl"), "rb")
        )
        ids_to_tokens_map = pickle.load(
            open(os.path.join(path, "ids_to_tokens.pkl"), "rb")
        )
        if with_notes:
            subject_id_hadm_map = pickle.load(
                open(os.path.join(path, "subject_id_hadm_map.pkl"), "rb")
            )
        if updated_ids_to_types:
            updated_ids_to_types = pickle.load(
                open(os.path.join(path, "updated_ids_to_types.pkl"), "rb")
            )
            reverse_up_ids_to_types = pickle.load(
                open(os.path.join(path, "reverse_up_ids_to_types.pkl"), "rb")
            )
            if with_notes:
                return (
                    source_target_sequences,
                    ids_to_types_map,
                    tokens_to_ids_map,
                    updated_ids_to_types,
                    reverse_up_ids_to_types,
                    subject_id_hadm_map,
                )
            return (
                source_target_sequences,
                ids_to_types_map,
                tokens_to_ids_map,
                updated_ids_to_types,
                reverse_up_ids_to_types,
            )
        if with_notes:
            return (
                source_target_sequences,
                ids_to_types_map,
                tokens_to_ids_map,
                ids_to_tokens_map,
                subject_id_hadm_map,
            )
        return (
            source_target_sequences,
            ids_to_types_map,
            tokens_to_ids_map,
            ids_to_tokens_map,
        )

    if processed_data:

        source_sequences = pickle.load(
            open(os.path.join(path, "source_sequences.pkl"), "rb")
        )

        target_sequences = pickle.load(
            open(os.path.join(path, "target_sequences.pkl"), "rb")
        )

        try:
            old_to_new_ids_source = pickle.load(
                open(os.path.join(path, "old_to_new_ids_source.pkl"), "rb")
            )
        except Exception as e:
            print("old_to_new_ids_source.pkl not found", e)
            print(
                "old_to_new_ids_source file not availble, mapping is the same as the old one"
            )
            old_to_new_ids_source = None

        try:
            old_to_new_ids_target = pickle.load(
                open(os.path.join(path, "old_to_new_ids_target.pkl"), "rb")
            )
        except Exception as e:
            print("old_to_new_ids_target.pkl not found", e)
            print(
                "new_to_old_ids_target file not availble, mapping is the same as the old one"
            )
            old_to_new_ids_target = None
        try:
            source_tokens_to_ids = pickle.load(
                open(os.path.join(path, "source_tokens_to_ids.pkl"), "rb")
            )
        except Exception as e:
            print("source_tokens_to_ids.pkl not found", e)
            source_tokens_to_ids = None
        try:
            target_tokens_to_ids = pickle.load(
                open(os.path.join(path, "target_tokens_to_ids.pkl"), "rb")
            )
        except Exception as e:
            print("target_tokens_to_ids.pkl not found", e)
            target_tokens_to_ids = None
        if reindexed:
            try:
                hospital_ids_source = pickle.load(
                    open(os.path.join(path, "hospital_ids_source_reindexed.pkl"), "rb")
                )
            except Exception as e:
                print("hospital_ids_source_reindexed.pkl not found", e)
                hospital_ids_source = None
            return (
                source_sequences,
                target_sequences,
                source_tokens_to_ids,
                target_tokens_to_ids,
                old_to_new_ids_source,
                old_to_new_ids_target,
                hospital_ids_source,
            )
        try:
            hospital_ids_source = pickle.load(
                open(os.path.join(path, "hospital_ids_source.pkl"), "rb")
            )
        except Exception as e:
            print("hospital_ids_source.pkl not found", e)
            hospital_ids_source = None

        return (
            source_sequences,
            target_sequences,
            source_tokens_to_ids,
            target_tokens_to_ids,
            old_to_new_ids_source,
            old_to_new_ids_target,
            hospital_ids_source,
        )

    else:

        patients_visits_sequences = pickle.load(
            open(os.path.join(path, "patients_visits_sequences.pkl"), "rb")
        )
        tokens_to_ids_map = pickle.load(
            open(os.path.join(path, "tokens_to_ids.pkl"), "rb")
        )
        ids_to_types_map = pickle.load(
            open(os.path.join(path, "ids_to_types.pkl"), "rb")
        )
        try:
            subject_id_hadm_id_map_notes = pickle.load(
                open(os.path.join(path, "subject_id_hadm_map.pkl"), "rb")
            )
        except Exception as e:
            print("subject_id_hadm_map.pkl not found", e)
            subject_id_hadm_id_map_notes = None

        return (
            patients_visits_sequences,
            tokens_to_ids_map,
            ids_to_types_map,
            subject_id_hadm_id_map_notes,
        )


def get_paths(
    config: dict,
    strategy: str = None,
    predict_procedure: bool = False,
    predict_drugs: bool = False,
    train: bool = False,
    processed_data=False,
    with_notes: bool = False,
) -> dict:
    """
    creates relevant paths according the given parameters

    Args:
        config (dict): contains the loaded paths.yaml
        strategy (str): strategy of training to adopt
        predict_procedure (str): whether to load the files that contain procedure in the target sequences
        predict_drugs (str): whether to load the files that contain drugs in the target sequences
        train (bool): whether to return only the training path
        processed_data (bool): whether to return only the processed data path
        prepared_note_file (bool): whether to return the note file path

    Retruns:
        paths (dict): dictonnary that contrains all relevant paths
    """

    if train and strategy is None:
        strategy = strategy.upper()
        if strategy not in ("TF", "SDP"):
            raise ValueError("Wrong strategy, must choose either TF, SDP")

    output_files = config["output_files"]
    if with_notes:
        suffix = "_with_notes/"
    else:
        suffix = "/"
    paths = {}
    output_file = config["output_path"]
    root = config["root_path"]
    mimic_iv_path = os.path.join(root, config["mimic_iv_path"])
    ccs_file_path = os.path.join(root, config["ccs_path"])

    if train:
        if predict_procedure and predict_drugs:
            paths["train_data_path"] = os.path.join(
                root,
                output_file,
                strategy,
                output_files["all_outputs_file"][:-1] + suffix,
            )
        if predict_procedure and not (predict_drugs):
            paths["train_data_path"] = os.path.join(
                root,
                output_file,
                strategy,
                output_files["diagnosis_procedure_output_file"][:-1] + suffix,
            )
        else:
            paths["train_data_path"] = os.path.join(
                root,
                output_file,
                strategy,
                output_files["diagnosis_output_file"][:-1] + suffix,
            )
        if processed_data:
            paths["processed_data_path"] = os.path.join(
                root, paths["train_data_path"], "processed_data" + suffix
            )
        return paths

    for mimic_file, path in config["mimic_files"].items():
        paths[mimic_file] = os.path.join(mimic_iv_path, path)

    for ccs_file, path in config["css_files"].items():
        paths[ccs_file] = os.path.join(ccs_file_path, path)

    paths["note_file"] = os.path.join(root, config["mimic_iv_notes_path"])
    return paths


def store_files(
    source_target_sequences: List[Tuple[List[int], List[int]]] = None,
    ids_to_types_map: Dict[int, str] = None,
    tokens_to_ids_map: Dict[str, int] = None,
    ids_to_tokens_map: Dict[int, str] = None,
    updated_ids_to_types: Dict[int, int] = None,
    patients_visits_sequences=None,
    source_tokens_to_ids: Dict[str, int] = None,
    target_tokens_to_ids: Dict[str, int] = None,
    types=None,
    code_to_description_map=None,
    train=False,
    processed_data=False,
    old_to_new_ids_source=None,
    old_to_new_ids_target=None,
    source_sequences=None,
    target_sequences=None,
    subject_id_hadm_map=None,
    hospital_ids_source=None,
    output_file: str = "outputData/originalData/",
    **kw,
):
    """
    I can't remember what this function does :p
    """
    if not os.path.exists(output_file):
        os.makedirs(output_file)

    print(f"dumping in {os.path.join(output_file)}")

    if train:
        if updated_ids_to_types is not None:
            pickle.dump(
                updated_ids_to_types,
                open(os.path.join(output_file, "updated_ids_to_types.pkl"), "wb"),
                -1,
            )
            reverse_up_ids_to_types = {v: k for k, v in updated_ids_to_types.items()}
            pickle.dump(
                reverse_up_ids_to_types,
                open(os.path.join(output_file, "reverse_up_ids_to_types.pkl"), "wb"),
                -1,
            )
        if source_target_sequences is not None:
            pickle.dump(
                source_target_sequences,
                open(os.path.join(output_file, "source_target_sequences.pkl"), "wb"),
                -1,
            )
        if ids_to_types_map is not None:
            pickle.dump(
                ids_to_types_map,
                open(os.path.join(output_file, "ids_to_types.pkl"), "wb"),
                -1,
            )
        if tokens_to_ids_map is not None:
            pickle.dump(
                tokens_to_ids_map,
                open(os.path.join(output_file, "tokens_to_ids.pkl"), "wb"),
                -1,
            )
        if ids_to_tokens_map is not None:
            pickle.dump(
                ids_to_tokens_map,
                open(os.path.join(output_file, "ids_to_tokens.pkl"), "wb"),
                -1,
            )
        if subject_id_hadm_map is not None:
            pickle.dump(
                subject_id_hadm_map,
                open(os.path.join(output_file, "subject_id_hadm_map.pkl"), "wb"),
                -1,
            )

        return
    if processed_data:
        if source_sequences is not None:
            pickle.dump(
                source_sequences, open(output_file + "source_sequences.pkl", "wb"), -1
            )
        if target_sequences is not None:
            pickle.dump(
                target_sequences, open(output_file + "target_sequences.pkl", "wb"), -1
            )
        if old_to_new_ids_source is not None:
            pickle.dump(
                old_to_new_ids_source,
                open(output_file + "old_to_new_ids_source.pkl", "wb"),
                -1,
            )
        if old_to_new_ids_target is not None:
            pickle.dump(
                old_to_new_ids_target,
                open(output_file + "old_to_new_ids_target.pkl", "wb"),
                -1,
            )
        if source_tokens_to_ids is not None:
            pickle.dump(
                source_tokens_to_ids,
                open(output_file + "source_tokens_to_ids.pkl", "wb"),
                -1,
            )
        if target_tokens_to_ids is not None:
            pickle.dump(
                target_tokens_to_ids,
                open(output_file + "target_tokens_to_ids.pkl", "wb"),
                -1,
            )
        if hospital_ids_source is not None:
            pickle.dump(
                hospital_ids_source,
                open(output_file + "hospital_ids_source.pkl", "wb"),
                -1,
            )

        return
    else:
        if patients_visits_sequences is not None:
            pickle.dump(
                patients_visits_sequences,
                open(output_file + "patients_visits_sequences.pkl", "wb"),
                -1,
            )
        if tokens_to_ids_map is not None:
            pickle.dump(
                tokens_to_ids_map, open(output_file + "tokens_to_ids.pkl", "wb"), -1
            )
        if code_to_description_map is not None:
            pickle.dump(
                code_to_description_map,
                open(output_file + "code_to_description.pkl", "wb"),
                -1,
            )
