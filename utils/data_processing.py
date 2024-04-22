import os
import pickle
import numpy as np
from collections import defaultdict
from typing import Dict, Optional, Tuple, List, Union

def list_tuples(x : List[List[int]], y : List[List[int]]) -> List[Tuple[List[int], List[int]]]:
    pairs = []
    for i, a in enumerate(zip(x,y)):
        pairs.append(a)
    return pairs

def prepare_for_tf(sequence : List[List[int]]) -> Tuple[List[List[int]]]:
    """
    Prepares the input sequence for trajectory forecasting training by creating pairs of input and output sequences.

    Args:
        sequence (List[List[int]]): The input sequence of integers.

    Returns:
        List[Tuple[List[List[int]], List[List[int]]]]: A list of pairs, where each pair consists of an input sequence and its corresponding output sequence.
    """
    X, y, pairs = list(), list(), list()
    for i in range(len(sequence)):
        # find the end of this pattern
        if i+1 >= len(sequence):
            break
        X.append(sequence[:i+1])
        y.append(sequence[i+1:])
    pairs = list_tuples(X, y)
    return pairs


def prepare_for_sdp(sequence: List[List[int]]):
    """
    Prepare the sequence for Sequential disease prediction modeling.

    Args:
        sequence (list): The input sequence.

    Returns:
        list: A list of pairs containing the input sequence and the corresponding target sequence.

    """
    
    X, y, pairs = list(), list(), list()
    for i in range(len(sequence)):
        if i + 1 >= len(sequence):
            break
        X.append(sequence[:i+1])
        y.append([sequence[i+1]])
        
    pairs = list_tuples(X, y)
    return pairs

def format_data(sequences : List[List[List[int]]],  strategy : Optional[str] = 'TF') -> List[Tuple[List[int], List[int]]]:
    """
    Formats the original sequences based on the specified data format.

    Args:
        sequences (List[List[List[int]]]): The original sequences to be formatted.
        strategy (str, optional): The data format to use. Can be either 'TF' for trajectory forecasting or 'SDP' for sequential disease prediction. Defaults to 'TF'.

    Returns:
        List[Tuple[List[List[int]]]]: The formatted pairs of input and output sequences.
    """

    source_target_sequences = []
    
    for i in range(len(sequences)):
        # Trajectory forecasting (TF): predict until the end of EOH
        if strategy == 'TF':
            source_target_sequences.extend(prepare_for_tf(sequences[i]))
        # Sequential disease prediction (SDP): predict until the next visit
        elif strategy == 'SDP':
            source_target_sequences.extend(prepare_for_sdp(sequences[i]))
        else:
            raise Exception('Wrong strategy, must choose either TF, SDP')
    return source_target_sequences

def reset_integer_output(source_target_sequences: List[List[int]]) -> Tuple[List[List[int]], Dict[int, int]]:
    """
    Resets the integer output codes in the given sequence of pairs.

    Args:
        source_target_sequences (List[List[int]]): A list of pairs where each pair contains an integer and a list of codes.

    Returns:
        Tuple[List[List[int]], Dict[int, int]]: A tuple containing the updated pairs and a dictionary mapping the old codes to new codes.
    """
    updated_source_target_sequences = []
    # keep same ids for special tokens
    old_to_new_map = defaultdict(lambda: len(old_to_new_map), {0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5})

    for pair in source_target_sequences:
        list_of_visits = []
        for target_visit in pair[1]:
            updated_target_visit = []
            for code in target_visit:
                updated_target_visit.append(old_to_new_map[code])
            list_of_visits.append(updated_target_visit)
        updated_source_target_sequences.append((pair[0], list_of_visits))
        
    return updated_source_target_sequences, dict(old_to_new_map)


def filter_codes(source_target_sequences : List[Tuple[List[List[int]], List[List[int]]]], ids_to_types_map: Dict[int, str],\
                  procedure : bool = False , drugs : bool = False, reset_target_map : bool = False)\
    -> Tuple[Tuple[List[List[int]]], Union[Dict[int, int]], None] :
    """
    Filters the codes of target sequences to remove indicated types, and flattens the both sequnces.

    Args:
    - source_target_sequences (list): List of pairs containing the input and output sequences.
    - ids_to_types_map (dict): Dictionary mapping codes to their corresponding types.
    - procedure (bool): Flag indicating whether to remove procedure codes in the output. Default is False.
    - drugs (bool): Flag indicating whether to remove drug codes in the output. Default is False.
    - reset_target_map (bool) : !Experimental! Flag indicating whether to remap after deleting the code.

    Returns:
    - updated_source_target_sequences (list): List of updated pairs containing the special tokens, and flattend
    - old_to_new_map (dict): a mapping of the old to new codes for the target sequences. 
    """

    updated_source_target_sequences = []

    if procedure and drugs:
        print("\nRemoving drug and procedure codes from target sequences")
        for pair in source_target_sequences:
            list_of_target_visits = []
            for target_visit in pair[1]:
                updated_target_visit = []
                for code in target_visit:
                    if not (ids_to_types_map[code] == 'P' or ids_to_types_map[code] == 'DR'):
                        updated_target_visit.append(code)
                list_of_target_visits.append(updated_target_visit)
            updated_source_target_sequences.append((pair[0], list_of_target_visits))
            

    if drugs and not(procedure):
        print("\nOnly removing drug codes from target sequences")
        for pair in source_target_sequences:
            list_of_target_visits = []
            for target_visit in pair[1]:
                updated_target_visit = []
                for code in target_visit:
                    if not (ids_to_types_map[code] == 'DR'):
                        updated_target_visit.append(code)
                list_of_target_visits.append(updated_target_visit)
            updated_source_target_sequences.append((pair[0], list_of_target_visits))

    if not(drugs) and procedure:
        print("\Only removing procedure codes from target sequences")
        for pair in source_target_sequences:
            list_of_target_visits = []
            for target_visit in pair[1]:
                updated_target_visit = []
                for code in target_visit:
                    if not (ids_to_types_map[code] == 'P'):
                        updated_target_visit.append(code)
                list_of_target_visits.append(updated_target_visit)
            updated_source_target_sequences.append((pair[0], list_of_target_visits))

    if not(procedure) and not(drugs):
        print("\nkeeping all codes")
        updated_source_target_sequences = source_target_sequences.copy()
    
    if reset_target_map:
        updated_source_target_sequences, old_to_new_map = reset_integer_output(updated_source_target_sequences)
        return updated_source_target_sequences, old_to_new_map

    return updated_source_target_sequences, None





def prepare_sequences(source_target_sequences: List[Tuple[List[int], List[int]]], tokens_to_ids_map: Dict[str, int], truncate : Optional[bool] = False,
                       pad: Optional[bool] = False, input_max_length: Optional[int] = None ,
                        target_max_length: Optional[int] = None ) -> Tuple[List[List[int]], List[List[int]]]:
    """
    Adds special tokens to the input and output sequences in the given list of pairs.

    Args:
        source_target_sequences (List[Tuple[List[int],List[int]]]): A list of pairs, where each pair contains an input source and target sequence.
        tokens_to_ids_map (Dict[str,int]): A dictionary containing special tokens.
        truncate (Optional[bool]): If True, truncates sequences to `input_max_length` or `output_max_length` if provided.
        pad (Optional[bool]): If True, pads sequences to `input_max_length` or `output_max_length` if provided.
        source_max_length (Optional[int]): Maximum length for source sequences.
        target_max_length (Optional[int]): Maximum length for target sequences.

    Returns:
        Tuple[List[List[int]], List[List[int]]]: Lists of formated source and target sequences.
    """
    
    updated_source_sequences = []
    updated_target_sequences = []

    for pair in source_target_sequences:
        input_sequences, output_sequences = pair
        input_sequence_spec, output_sequence_spec = [], []
        # Adding special tokens to input sequence
        for input_sequence in input_sequences:
            input_sequence =  [tokens_to_ids_map['BOV']] + input_sequence + [tokens_to_ids_map['EOV']]
            input_sequence_spec.extend(input_sequence)
        input_sequence_spec = [tokens_to_ids_map['BOH']] + input_sequence_spec + [tokens_to_ids_map['EOH']]                   
        
        # Adding special tokens to output sequence
        for output_sequence in output_sequences:
            output_sequence =  [tokens_to_ids_map['BOV']] + output_sequence + [tokens_to_ids_map['EOV']] 
            output_sequence_spec.extend(output_sequence)
        output_sequence_spec = [tokens_to_ids_map['BOS']] + output_sequence_spec + [tokens_to_ids_map['EOH']]
            
        # Truncate or pad input sequence
        if input_max_length is not None:
            if truncate:
                input_sequence_spec = input_sequence_spec[:input_max_length]
            if pad:
                input_sequence_spec += [tokens_to_ids_map['PAD']] * (input_max_length - len(input_sequence_spec))
        
        # Truncate or pad output sequence
        if target_max_length is not None:
            if truncate:
                output_sequence_spec = output_sequence_spec[:target_max_length]
            if pad:
                output_sequence_spec += [tokens_to_ids_map['PAD']] * (target_max_length - len(output_sequence_spec))
        
        updated_source_sequences.append(input_sequence_spec)
        updated_target_sequences.append(output_sequence_spec)
        
    return updated_source_sequences, updated_target_sequences

def train_test_val_split(source_sequences : List[List[int]], target_sequences : List[List[int]],
                test_size : float = 0.05, valid_size : float = 0.05, random_state = None)\
    -> Tuple[Tuple[List[int], List[int]], Tuple[List[int], List[int]], Tuple[List[int], List[int]]]:
    """
    Generates the train, test, validation splits for botht source sequences, and target sequences

    Args:
        source_sequences (list): list of source sequences.
        target_sequences (list): list of target sequences.
        test_size (flaot): the fraction of the test set
        valid_size (flaot): the fraction of the test set
        random_state (int): Seed for random number generation.

    Returns:
        Tuple[Tuple[List[int], List[int]], Tuple[List[int], List[int]], Tuple[List[int], List[int]]]:
          A tuple containing the train, test, and validation splits for both source and target sequences.
        
        
    """
    # Validate inputs
    if not 0 <= test_size <= 1 or not 0 <= valid_size <= 1:
        raise ValueError("test_size and valid_size must be in the range [0, 1].")
    
    if random_state is not None:
        np.random.seed(random_state)
    
    source_sequences = np.array(source_sequences)
    target_sequences = np.array(target_sequences)
    dataSize = len(source_sequences)
    idx = np.random.permutation(dataSize)
    nTest = int(np.ceil(test_size * dataSize))
    nValid = int(np.ceil(valid_size * dataSize))
    
    test_idx = idx[:nTest]
    valid_idx = idx[nTest:nTest+nValid]
    train_idx = idx[nTest+nValid:]

    train = {'source_sequences': source_sequences[train_idx], 'target_sequences' : target_sequences[train_idx]}
    test = {'source_sequences': source_sequences[test_idx],  'target_sequences' : target_sequences[test_idx]}
    valid = {'source_sequences':  source_sequences[valid_idx], 'target_sequences' : target_sequences[valid_idx]}
    
    return (train, test, valid)


def get_optimal_embedding_size(source_sequences : List[List[int]], target_sequences : List[List[int]]) -> Dict[str, Union[set, int]]:
    """
    Get the unique elements in the given sequences.

    Args:
        source_sequences (list): List of source sequences.
        target_sequences (list): List of target sequences.
    Returns:
        Dict[str, Union[set, int]]: A dictionary containing the unique elements in the source and target sequences, the number of unique elements in each sequence, and the maximum element in each sequence.
    """
    # Flatten the lists of lists
    source_flat = [item for sublist in source_sequences for item in sublist]
    target_flat = [item for sublist in target_sequences for item in sublist]
    
    # Create sets of unique elements
    unique_source = set(source_flat)
    unique_target = set(target_flat)

    len_unique_source = len(unique_source)
    len_unique_target = len(unique_target)
    max_unique_source = max(unique_source)
    max_unique_target = max(unique_target)
    
    data_properties = {
        'unique_source': unique_source,
        'unique_target': unique_target,
        'len_unique_source': len_unique_source,
        'len_unique_target': len_unique_target,
        'max_unique_source': max_unique_source,
        'max_unique_target': max_unique_target
    }



def get_embedding_size(max_unique : int, len_unique : int, unique_data : set, multiplier :int = 64) -> Tuple[int, Optional[Dict[int, int]]]:
    if max_unique - multiplier > len_unique:
        print('must reformat, too much memory would be lost in embedding')
        # create a new mapping of the codes to the new codes
        old_ids_to_new_mapping = {code: i for i, code in enumerate(unique_data)}
        return (len(old_ids_to_new_mapping) // multiplier + 1) * multiplier, old_ids_to_new_mapping
    else:
        # create the embedding size that is rounded to the nearest multiple of 64 of the max unique
        return (max_unique // multiplier + 1) * multiplier, None

def get_optimal_embedding_size(source_sequences : List[List[int]], target_sequences : List[List[int]], multiplier :int = 64) -> Dict[str, Union[set, int]]:
    """
    Get the unique elements in the given sequences and calculate optimal embedding size, creates corresponding mapping if needed.

    Args:
        source_sequences (list): List of source sequences.
        target_sequences (list): List of target sequences.
        multiplier (int): The multiplier to use for the embedding size.
    Returns:
        Dict[str,Union[int, int, list, list, dict, dict]]: A dictionary containing the optimal embedding size for the source and target sequences, the source and target sequences, and the mapping of the new codes to old codes.
    """
    # Flatten the lists of lists
    source_flat = [item for sublist in source_sequences for item in sublist]
    target_flat = [item for sublist in target_sequences for item in sublist]
    
    # Create sets of unique elements
    unique_source = set(source_flat)
    unique_target = set(target_flat)

    len_unique_source = len(unique_source)
    len_unique_target = len(unique_target)
    max_unique_source = max(unique_source)
    max_unique_target = max(unique_target)
    
    data_properties = {}

    embedding_size_source, mapping_source = get_embedding_size(max_unique_source, len_unique_source, unique_source, multiplier)
    embedding_size_target, mapping_target = get_embedding_size(max_unique_target, len_unique_target, unique_target, multiplier)

    if mapping_source is not None:
        source_sequences = [[mapping_source[code] for code in sequence] for sequence in source_sequences]
        mapping_source = {v: k for k, v in mapping_source.items()}
    if mapping_target is not None:
        target_sequences = [[mapping_target[code] for code in sequence] for sequence in target_sequences]
        mapping_target = {v: k for k, v in mapping_target.items()}

    data_properties['embedding_size_source'] = embedding_size_source
    data_properties['embedding_size_target'] = embedding_size_target
    data_properties['source_sequences'] = source_sequences
    data_properties['target_sequences'] = target_sequences
    data_properties['new_to_old_ids_source'] = mapping_source
    data_properties['new_to_old_ids_target'] = mapping_target
    
    return data_properties

def get_embedding_size(max_unique : int, len_unique : int, unique_data : set, multiplier :int) -> Tuple[int, Optional[Dict[int, int]]]:
    """
    Get the optimal embedding size for the given data.

    Args:
        max_unique (int): The maximum unique element in the data.
        len_unique (int): The number of unique elements in the data.
        unique_data (set): The set of unique elements in the data.
        multiplier (int): The multiplier to use for the embedding size.
    Returns:
        Tuple[int, Optional[Dict[int, int]]]: A tuple containing the optimal embedding size and a mapping of the old to new codes if the data needs to be reformatted.
    """
    if max_unique - multiplier > len_unique:
        print('must reformat, too much memory would be lost in embedding')
        # create a new mapping of the codes to the new codes
        old_ids_to_new_mapping = {code: i for i, code in enumerate(unique_data)}
        return (len(old_ids_to_new_mapping) // multiplier + 1) * multiplier, old_ids_to_new_mapping
    else:
        # create the embedding size that is rounded to the nearest multiple of 64 of the max unique
        return (max_unique // multiplier + 1) * multiplier, None