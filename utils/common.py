import os
import pickle
import torch 
from typing import Dict, Optional, Tuple, List, Union

def load_data(outFile: str) -> Tuple[List[List[List[int]]], Dict[str, int], Dict[str, int], Dict[int, str]]:
        """
        Load data from the specified file.

        Args:
        - outFile (str): The path to the file containing the data.

        Returns:
        - Tuple[List[List[List[int]]], Dict[str, ], Dict[str, int], Dict[int, str]]: A tuple containing the loaded data.
            - The first element is a list of sequences, where each sequence is a list of events, and each event is a list of integers.
            - The second element is a dictionary mapping event types to their corresponding codes.
            - The third element is a dictionary mapping event codes to their corresponding types.
            - The fourth element is a dictionary mapping event codes to their corresponding types (reversed mapping).
        """
        # load the data again
        seqs = pickle.load(open(os.path.join(outFile, 'data.seqs'), 'rb'))
        types = pickle.load(open(os.path.join(outFile, 'data.types'), 'rb'))
        codeType = pickle.load(open(os.path.join(outFile, 'data.codeType'), 'rb'))
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

def formatData(originalSeqs : List[List[List[int]]],  dataFormat : Optional[str] = 'TF') -> List[Tuple[List[int], List[int]]]:
    """
    Formats the original sequences based on the specified data format.

    Args:
        originalSeqs (List[List[List[int]]]): The original sequences to be formatted.
        dataFormat (str, optional): The data format to use. Can be either 'TF' for trajectory forecasting or 'SDP' for sequential disease prediction. Defaults to 'TF'.
        max_lenght (int, optional): The minimum sequence length. Pairs with a length greater than this value will be removed. Defaults to 400.

    Returns:
        List[Tuple[List[List[int]]]]: The formatted pairs of input and output sequences.
    """

    pairs = []
    
    for i in range(len(originalSeqs)):
        # Trajectory forecasting (TF): predict until the end of EOH
        if dataFormat == 'TF':
            pairs.extend(PrepareForTF(originalSeqs[i]))
        # Sequential disease prediction (SDP): predict until the next visit
        elif dataFormat == 'SDP':
            pairs.extend(PrepareForSDP(originalSeqs[i]))
        else:
            raise Exception('Wrong strategy, must choose either TF, SDP')
    return pairs

def updateOutput(newPairs : List[Tuple[List[List[int]], List[List[int]]]], codeType: Dict[int, str], diagnosis: bool = False, procedure : bool = False , drugs : bool = False)\
    -> List[Tuple[List[List[int]]]] :
    """
    Update the output sequences based on the specified criteria.

    Args:
    - newPairs (list): List of pairs containing the input and output sequences.
    - codeType (dict): Dictionary mapping codes to their corresponding types.
    - diagnosis (int): Flag indicating whether to include diagnosis codes in the output. Default is 0.
    - procedure (int): Flag indicating whether to include procedure codes in the output. Default is 0.
    - drugs (int): Flag indicating whether to include drug codes in the output. Default is 0.
    - all_ (int): Flag indicating whether to keep all codes in the output. Default is 0.

    Returns:
    - updSeqs (list): List of updated pairs containing the input and updated output sequences.
    """

    updSeqs = []

    if procedure and drugs:
        print("\n Removing drug and procedure codes from output for forecasting diagnosis code only")
        for i, pair in enumerate(newPairs):
            newOutput = []
            for code in pair[1]:
                if codeType[code] == 'D' or codeType[code] == 'T':
                    newOutput.append(code)

            if len(newOutput) >= 4:
                updSeqs.append((pair[0], newOutput))

    if drugs and not(procedure):
        print("\n Removing only drug codes from output for forecasting diagnosis and procedure code only")
        for i, pair in enumerate(newPairs):
            newOutput = []
            for code in pair[1]:
                if not (codeType[code] == 'DR'):
                    newOutput.append(code)
            if len(newOutput) >= 4:
                updSeqs.append((pair[0], newOutput))

    if not(diagnosis) and not(procedure) and not(drugs):
        print("\n keeping all codes")
        updSeqs = newPairs.copy()

    return updSeqs

def resetIntegerOutput(updSeqs: List[List[int]]) -> Tuple[List[List[int]], Dict[int, int]]:
    """
    Resets the integer output codes in the given sequence of pairs.

    Args:
        updSeqs (List[List[int]]): A list of pairs where each pair contains an integer and a list of codes.

    Returns:
        Tuple[List[List[int]], Dict[int, int]]: A tuple containing the updated pairs and a dictionary mapping the old codes to new codes.
    """
    updPair = []
    outTypes = {}
    # keep same ids for special tokens
    outTypes.update({0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5}) 
    for i, pair in enumerate(updSeqs):
        newVisit = []
        for code in pair[1]:
            if code in outTypes:
                newVisit.append(outTypes[code])
            else:
                outTypes[code] = len(outTypes)
                newVisit.append(outTypes[code])
        updPair.append((pair[0], newVisit))
    return updPair, outTypes


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