import pandas as pd
import sys
import spacy
import re
import time
import scispacy
from tqdm import tqdm
from utils.notes import sent_tokenize_rules, TextPreprocessor
from tqdm.notebook import tqdm #else tqdm doens't seem to work
import numpy as np
import dask.dataframe as dd
import os
import warnings
from pandarallel import pandarallel
from spacy.language import Language
import argparse
# Disable warnings
warnings.filterwarnings('ignore')



#setting sentence boundaries
@Language.component("sbd_component")
def sbd_component(doc):
    for i, token in enumerate(doc[:-2]):
        # define sentence start if period + titlecase token
        if token.text == '.' and doc[i+1].is_title:
            doc[i+1].sent_start = True
        if token.text == '-' and doc[i+1].text != '-':
            doc[i+1].sent_start = True
    return doc

#convert de-identification text into one token
def fix_deid_tokens(text, processed_text):
    deid_regex  = r"\[\*\*.{0,15}.*?\*\*\]" 
    if text:
        indexes = [m.span() for m in re.finditer(deid_regex,text,flags=re.IGNORECASE)]
    else:
        indexes = []
    for start,end in indexes:
        processed_text.merge(start_idx=start,end_idx=end)
    return processed_text
    

def process_section(section, note, processed_sections):
    # perform spacy processing on section
    processed_section = nlp(section['sections'])
    processed_section = fix_deid_tokens(section['sections'], processed_section)
    processed_sections.append(processed_section)

def process_note_helper(note):
    # split note into sections
    note_sections = sent_tokenize_rules(note)
    processed_sections = []
    section_frame = pd.DataFrame({'sections':note_sections})
    section_frame.apply(process_section, args=(note,processed_sections,), axis=1)
    return(processed_sections)

def process_text(sent, note):
    sent_text = sent['sents'].text
    if len(sent_text) > 0 and sent_text.strip() != '\n':
        if '\n' in sent_text:
            sent_text = sent_text.replace('\n', ' ')
        note['text'] += sent_text + '\n'  

def get_sentences(processed_section, note):
    # get sentences from spacy processing
    sent_frame = pd.DataFrame({'sents': list(processed_section['sections'].sents)})
    sent_frame.apply(process_text, args=(note,), axis=1)

def process_note(note):
    try:
        note_text = note['text'] 
        note['text'] = ''
        processed_sections = process_note_helper(note_text)
        ps = {'sections': processed_sections}
        ps = pd.DataFrame(ps)
        ps.apply(get_sentences, args=(note,), axis=1)
        return note 
    except Exception as e:
        pass
        #print ('error', e)
        
def parse_args():

    parser = argparse.ArgumentParser(description='Prepare notes for patient trajectory forecasting')
    parser.add_argument('--mimic_notes_file', type=str, default='/home/sifal.klioui/physionet/mimic-iv-notes/2.2.1/discharge_summaries.csv', help='Path to the MIMIC-IV notes file')
    parser.add_argument('--output_dir', type=str, default='/home/sifal.klioui/physionet/', help='Output directory to save the processed notes')
    parser.add_argument('--clean_text', action='store_true', help='Clean the text')
    parser.add_argument('--lower', action='store_true', help='Convert text to lowercase')
    parser.add_argument('--remove_special_characters_mullenbach', action='store_true', help='Remove special characters as done in Mullenbach et al.')
    parser.add_argument('--remove_special_characters', action='store_true', help='Remove special characters')
    parser.add_argument('--remove_digits', action='store_true', help='Remove digits')
    parser.add_argument('--remove_accents', action='store_true', help='Remove accents')
    parser.add_argument('--remove_brackets', action='store_true', help='Remove brackets')
    parser.add_argument('--convert_danish_characters', action='store_true', help='Convert Danish characters')
    parser.add_argument('--apply_replace', action='store_true', help='Apply replace')
    parser.add_argument('--remove_adm_details', action='store_true', help='Remove admission details')

    return parser.parse_args()

if __name__ == '__main__':

    args = parse_args()

    # Load the spaCy model
    nlp = spacy.load('en_core_sci_md', disable=['tagger','ner'])
    nlp.add_pipe('sbd_component', before='parser')  

    # Load the notes
    start = time.time()
    notes = pd.read_csv(args.mimic_notes_file, keep_default_na = False)
    
    print('Number of notes: %d' %len(notes.index))
    notes['ind'] = list(range(len(notes.index)))

    pandarallel.initialize(progress_bar=True)
    # usefull if have a lot cpu cores, otherwise look into RAPIDS cuDF (pandas on gpu! :O ) 
    formatted_notes = notes.parallel_apply(process_note, axis=1)

    text_preprocessor = TextPreprocessor(clean_text=args.clean_text,
                                         lower=args.lower,
                                         remove_special_characters_mullenbach=args.remove_special_characters_mullenbach,
                                         remove_special_characters=args.remove_special_characters,
                                         remove_digits=args.remove_digits,
                                         remove_accents=args.remove_accents,
                                         remove_brackets=args.remove_brackets,
                                         convert_danish_characters=args.convert_danish_characters,
                                         apply_replace=args.apply_replace,
                                         remove_adm_details=args.remove_adm_details)
    
    with open(os.path.join(args.output_dir,'notes.txt'),'w') as f:

        for text in formatted_notes['text']:
            try:
                text = text_preprocessor(text)
            except TypeError as e:
            # doing this because found some nan values in the notes
                if str(e) == "expected string or bytes-like object" and type(text) == float:
                    continue
                else:
                    raise         
                    
            if text != None and len(text) != 0 :
                f.write(text)
                f.write('\n')
    

    end = time.time()

    print('Time taken to process notes: %f seconds' %(end-start))
