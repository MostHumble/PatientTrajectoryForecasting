from typing import Dict, List, Tuple


def test_build_data_wnotes(subjects_visits_sequences: List[List[List[int]]], subject_id_hadm_id_map_notes: Dict[int, List[int]]):
    print(f'len(subjects_visits_sequences): {len(subjects_visits_sequences)}, len(subject_id_hadm_id_map_notes): {len(subject_id_hadm_id_map_notes)}')
    assert len(subjects_visits_sequences) == len(subject_id_hadm_id_map_notes), "not the same name of subjects between mapping, and sequences"
    for admissions_map, adm_seqs in zip(subject_id_hadm_id_map_notes.values(), subjects_visits_sequences):
        assert len(admissions_map) == len(adm_seqs), 'not the same number of admissions in the mapping and sequences'

def test_format_data_wnotes(subject_id_hadm_id_map_notes: Dict[int, List[int]], source_target_sequences: List[List[List[int]]]):
    patient_visits = list(subject_id_hadm_id_map_notes.values())
    k = 0
    for i in range(len(source_target_sequences) - 1):
        if len(source_target_sequences[i+1][0]) == 1:
            assert len(source_target_sequences[i][0]) == len(patient_visits[k]) - 1,  "The alignement of the created sequences and hospital admissions, per subject isn't the same after data formating"
            k += 1
        else:
            pass 


def test_format_data_hospital_ids(notes_hadm_id: List[List[int]], source_target_sequences: Tuple[List[List[int]]]):
    assert len(notes_hadm_id) == len(source_target_sequences), "the lenght of the formated hospital ids isn't the same as that of source target sequences"

    for i in range(len(notes_hadm_id)):
        assert len(notes_hadm_id[i]) == len(source_target_sequences[i][0]), "the formated hospital ids do not follow the same or as that of source target sequences"