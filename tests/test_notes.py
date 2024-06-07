from typing import List, Dict

def test_build_data_wnotes(subjects_visits_sequences: List[List[List[int]]], subject_id_hadm_id_map_notes: Dict[int, List[int]]):
    print(f'len(subjects_visits_sequences): {len(subjects_visits_sequences)}, len(subject_id_hadm_id_map_notes): {len(subject_id_hadm_id_map_notes)}')
    assert len(subjects_visits_sequences) == len(subject_id_hadm_id_map_notes), "not the same name of subjects between mapping, and sequences"
    for admissions_map, adm_seqs in zip(subject_id_hadm_id_map_notes.values(), subjects_visits_sequences):
        assert len(admissions_map) == len(adm_seqs), 'not the same number of admissions in the mapping and sequences'