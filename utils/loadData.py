import os
import pickle
import numpy as np
from collections import defaultdict
from typing import Dict, Optional, Tuple, List, Union

def loadData(path: str) -> Tuple[List[List[List[int]]], Dict[str, int], Dict[str, int], Dict[int, str]]:
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
        seqs = pickle.load(open(os.path.join(path, 'data.seqs'), 'rb'))
        types = pickle.load(open(os.path.join(path, 'data.types'), 'rb'))
        codeType = pickle.load(open(os.path.join(path, 'data.codeType'), 'rb'))
        reverseTypes = {v: k for k, v in types.items()}
        return seqs, types, codeType, reverseTypes

def listTuples(x : List[List[int]], y : List[List[int]]) -> List[Tuple[List[int], List[int]]]:
    pairs = []
    for i, a in enumerate(zip(x,y)):
        pairs.append(a)
    return pairs

def PrepareForTF(sequence : List[List[int]]) -> Tuple[List[List[int]]]:
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
    pairs = listTuples(X, y)
    return pairs


def PrepareForSDP(sequence: List[List[int]]):
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
        
    pairs = listTuples(X, y)
    return pairs

def formatData(sequences : List[List[List[int]]],  strategy : Optional[str] = 'TF') -> List[Tuple[List[int], List[int]]]:
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
            source_target_sequences.extend(PrepareForTF(sequences[i]))
        # Sequential disease prediction (SDP): predict until the next visit
        elif strategy == 'SDP':
            source_target_sequences.extend(PrepareForSDP(sequences[i]))
        else:
            raise Exception('Wrong strategy, must choose either TF, SDP')
    return source_target_sequences

def resetIntegerOutput(source_target_sequences: List[List[int]]) -> Tuple[List[List[int]], Dict[int, int]]:
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


def filterCodes(source_target_sequences : List[Tuple[List[List[int]], List[List[int]]]], ids_to_types_map: Dict[int, str],
                 diagnosis: bool = False, procedure : bool = False , drugs : bool = False, reset_target_map : bool = False)\
    -> Tuple[Tuple[List[List[int]]], Union[Dict[int, int]], None] :
    """
    Filters the codes of target sequences to remove indicated types, and flattens the both sequnces.

    Args:
    - source_target_sequences (list): List of pairs containing the input and output sequences.
    - ids_to_types_map (dict): Dictionary mapping codes to their corresponding types.
    - diagnosis (bool): Flag indicating whether to include diagnosis codes in the output. Default is False.
    - procedure (bool): Flag indicating whether to include procedure codes in the output. Default is False.
    - drugs (bool): Flag indicating whether to include drug codes in the output. Default is False.
    - reset_target_map (bool) : !Experimental! Flag indicating whether to remap after deleting the code.

    Returns:
    - updated_source_target_sequences (list): List of updated pairs containing the special tokens, and flattend
    - old_to_new_map (dict): a mapping of the old to new codes for the target sequences. 
    """

    updated_source_target_sequences = []

    if procedure and drugs:
        print("\nRemoving drug and procedure codes from target sequences")
        for i, pair in enumerate(source_target_sequences):
            newOutput = []
            for code in pair[1]:
                if ids_to_types_map[code] == 'D' or ids_to_types_map[code] == 'T':
                    newOutput.append(code)

            if len(newOutput) >= 4:
                updated_source_target_sequences.append((pair[0], newOutput))

    if drugs and not(procedure):
        print("\nRemoving only drug codes from target sequences")
        for i, pair in enumerate(source_target_sequences):
            newOutput = []
            for code in pair[1]:
                if not (ids_to_types_map[code] == 'DR'):
                    newOutput.append(code)
            if len(newOutput) >= 4:
                updated_source_target_sequences.append((pair[0], newOutput))

    if not(diagnosis) and not(procedure) and not(drugs):
        print("\nkeeping all codes")
        updated_source_target_sequences = source_target_sequences.copy()
    
    if reset_target_map:
        updated_source_target_sequences, old_to_new_map = resetIntegerOutput(updated_source_target_sequences)
        return updated_source_target_sequences, old_to_new_map

    return updated_source_target_sequences, None


def storeFiles(pair : List[Tuple[List[int], List[int]]], outTypes : Dict[int, int], codeType : Dict[int, str], types : Dict[str, int], reverseTypes : Dict[int, str], outFile: str):
    """
    I can't remember what this function does :p 
    """
    if not os.path.exists(outFile):
        os.makedirs(outFile)
    print(f"dumping in {os.path.join(outFile)}")

    pickle.dump(pair, open(os.path.join(outFile, 'data.seqs'), 'wb'), -1)
    pickle.dump(outTypes, open(os.path.join(outFile, 'data.outTypes'), 'wb'), -1)
    pickle.dump(codeType, open(os.path.join(outFile, 'data.codeType'), 'wb'), -1)
    pickle.dump(types, open(os.path.join(outFile, 'data.types'), 'wb'), -1)
    pickle.dump(reverseTypes, open(os.path.join(outFile, 'data.reverseTypes'), 'wb'), -1)
    reverseOutTypes = {v: k for k, v in outTypes.items()}
    pickle.dump(reverseOutTypes, open(os.path.join(outFile, 'data.reverseOutTypes'), 'wb'), -1)


def prepareSequences(source_target_sequences: List[Tuple[List[int], List[int]]], tokens_to_ids_map: Dict[str, int], truncate : Optional[bool] = False, pad\
                      : Optional[bool] = False, input_max_length: Optional[int] = None , target_max_length: Optional[int] = None, ) -> Tuple[List[List[int]], List[List[int]]]:
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

def split_data(data, test_frac=0.05, valid_frac=0.05):
    sequences = np.array(data)
    dataSize = len(sequences)
    np.random.seed(0) 
    idx = np.random.permutation(dataSize)
    nTest = int(np.ceil(test_frac * dataSize))
    nValid = int(np.ceil(valid_frac * dataSize))
    #def len_argsort(seq):
     #   return sorted(range(len(seq)), key=lambda x: len(seq[x]))

    test_idx = idx[:nTest]
    valid_idx = idx[nTest:nTest+nValid]
    train_idx = idx[nTest+nValid:]

    train_x = sequences[train_idx]
   
    test_x = sequences[test_idx]
   
    valid_x = sequences[valid_idx]
    
    
    '''    train_x = [sorted(seq) for seq in train_x]
    train_y = [sorted(seq) for seq in train_y]
    valid_x = [sorted(seq) for seq in valid_x]
    valid_y = [sorted(seq) for seq in valid_y]
    test_x = [sorted(seq) for seq in test_x]
    test_y = [sorted(seq) for seq in test_y]
    '''
    train = (train_x)
    test = (test_x)
    valid = (valid_x)
    return (train, test, valid)