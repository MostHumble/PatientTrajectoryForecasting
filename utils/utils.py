import os 
import pickle
from typing import Dict, Tuple, List


def load_data(path: str = 'outputData/originalData/' , updated_ids_to_types : bool = False, train : bool = False) -> Tuple[List[List[List[int]]], Dict[str, int], Dict[str, int], Dict[int, str]]:
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
        # load the data again

        if train:

            source_target_sequences = pickle.load(open(os.path.join(path, 'data.source_target_sequences'), 'rb'))
            ids_to_types_map = pickle.load(open(os.path.join(path, 'data.ids_to_tokens_map'), 'rb'))
            tokens_to_ids_map = pickle.load(open(os.path.join(path, 'data.tokens_to_ids_map'), 'rb'))
            ids_to_tokens_map = pickle.load(open(os.path.join(path, 'data.ids_to_tokens_map'), 'rb'))

            if updated_ids_to_types is not None:
                updated_ids_to_types =  pickle.load(open(os.path.join(path, 'data.updated_ids_to_types'), 'rb'))
                reverse_up_ids_to_types =  pickle.load(open(os.path.join(path, 'data.reverse_up_ids_to_types'), 'rb'))
                return source_target_sequences, ids_to_types_map, tokens_to_ids_map, ids_to_tokens_map, updated_ids_to_types, reverse_up_ids_to_types
            
            return source_target_sequences, ids_to_types_map, tokens_to_ids_map, ids_to_tokens_map
        
        else:

            patients_visits_sequences = pickle.load(open(os.path.join(path, 'data.patients_visits_sequences'), 'rb'))
            tokens_to_ids_map = pickle.load(open(os.path.join(path, 'data.tokens_to_ids_map'), 'rb')) 
            code_types = pickle.load(open(os.path.join(path, 'data.code_types'), 'rb'))

            return patients_visits_sequences, tokens_to_ids_map, code_types
        
def get_paths(config: dict, strategy : str = None, predict_procedure : bool = False, predict_drugs : bool = False,
               train : bool = False) -> dict:
    """
    creates relevant paths according the given parameters

    Args:
        config (dict): contains the loaded paths.yaml 
        strategy (str): strategy of training to adopt 
        predict_procedure (str): whether to load the files that contain procedure in the target sequences
        predict_drugs (str): whether to load the files that contain drugs in the target sequences
        train (bool): whether to return only the training path

    Retruns:
        paths (dict): dictonnary that contrains all relevant paths
    """
    
    if train and strategy is None:
        strategy = strategy.upper()
        if strategy not in ('TF', 'SDP'):
             raise ValueError('Wrong strategy, must choose either TF, SDP')

    output_files =  config['output_files']
    
    paths = {}
    output_file = config['output_path']
    root = config['root_path']
    mimic_iv_path = os.path.join(root, config['mimic_iv_path'])
    ccs_file_path = os.path.join(root, config['ccs_path'])

    if train:
        if predict_procedure and predict_drugs :
            paths['train_data_path'] = os.path.join(output_file, strategy, output_files['all_outputs_file'])
        if predict_procedure and not(predict_drugs):
            paths['train_data_path'] = os.path.join(output_file, strategy, output_files['diagnosis_procedure_output_file'])
        else:
            print(train)
            paths['train_data_path'] = os.path.join(output_file, strategy, output_files['diagnosis_output_file'])
        return paths
        
    for mimic_file, path in config['mimic_files'].items():
        paths[mimic_file] = os.path.join(mimic_iv_path, path)

    for ccs_file, path in config['css_files'].items():
        paths[ccs_file] = os.path.join(ccs_file_path, path)      

    return paths
    

def store_files(source_target_sequences : List[Tuple[List[int], List[int]]] = None,
                ids_to_types_map : Dict[int, str] = None,
                tokens_to_ids_map : Dict[str, int] = None,
                ids_to_tokens_map : Dict[int, str] = None, 
                updated_ids_to_types : Dict[int, int] = None,
                patients_visits_sequences = None, types = None,
                code_to_description_map = None, train = False,
                output_file : str = 'outputData/originalData/'):
    """
    I can't remember what this function does :p 
    """
    if not os.path.exists(output_file):
        os.makedirs(output_file)
    print(f"dumping in {os.path.join(output_file)}")

    
    if train:
        if updated_ids_to_types is not None:
            pickle.dump(updated_ids_to_types, open(os.path.join(output_file, 'data.updated_ids_to_types'), 'wb'), -1)
            reverse_up_ids_to_types = {v: k for k, v in updated_ids_to_types.items()}
            pickle.dump(reverse_up_ids_to_types, open(os.path.join(output_file, 'data.reverse_up_ids_to_types'), 'wb'), -1)  

        pickle.dump(source_target_sequences, open(os.path.join(output_file, 'data.source_target_sequences'), 'wb'), -1)
        pickle.dump(ids_to_types_map, open(os.path.join(output_file, 'data.ids_to_types_map'), 'wb'), -1)
        pickle.dump(tokens_to_ids_map, open(os.path.join(output_file, 'data.tokens_to_ids_map'), 'wb'), -1)
        pickle.dump(ids_to_tokens_map, open(os.path.join(output_file, 'data.ids_to_tokens_map'), 'wb'), -1)
        return
    else:
        pickle.dump(patients_visits_sequences, open(output_file + 'data.patients_visits_sequences', 'wb'), -1)
        pickle.dump(tokens_to_ids_map, open(output_file + 'data.tokens_to_ids_map', 'wb'), -1)
        pickle.dump(code_to_description_map, open(output_file + 'data.code_to_description_map', 'wb'), -1)