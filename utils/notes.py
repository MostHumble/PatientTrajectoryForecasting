import pandas
import os
import re 
from transformers import (
    BertTokenizer )
import pandas as pd
from datasets import Dataset
import pickle

class TextPreprocessor:
    def __init__(
        self,
        clean_text: bool = True,
        lower: bool = True,
        remove_special_characters_mullenbach: bool = True,
        remove_special_characters: bool = False,
        remove_digits: bool = True,
        remove_accents: bool = False,
        remove_brackets: bool = False,
        convert_danish_characters: bool = False,
        apply_replace: bool = True,
        remove_adm_details: bool = True
    ) -> None:
        self.clean_text = clean_text
        self.lower = lower
        self.remove_special_characters_mullenbach = remove_special_characters_mullenbach
        self.remove_digits = remove_digits
        self.remove_accents = remove_accents
        self.remove_special_characters = remove_special_characters
        self.remove_brackets = remove_brackets
        self.convert_danish_characters = convert_danish_characters
        self.apply_replace = apply_replace
        self.remove_adm_details = remove_adm_details
        self.replace_list = [['dr\.',''] ,['DR\.','']
                ,['m\.d\.',''] ,['M\.D\.','']
                ,[' yo ', ' years old ']
                ,['p\.o', 'orally. ']
                ,['P\.O', 'orally. ']
                ,[' po ', ' orally ']
                ,[' PO ', ' orally ']
                ,['q\.d\.', 'once a day. ']
                ,['Q\.D\.', 'once a day. ']
                ,['qd', 'once a day. ']
                ,['QD', 'once a day. ']
                ,['I\.M\.', 'intramuscularly. ']
                ,['i\.m\.', 'intramuscularly. ']
                ,['b\.i\.d\.', 'twice a day. ']
                ,['B\.I\.D\.', 'twice a day. ']
                ,['bid', 'twice a day. ']
                ,['BID', 'twice a day. ']
                ,['Subq\.', 'subcutaneous. ']
                ,['SUBQ\.', 'subcutaneous. ']
                ,['t\.i\.d\.', 'three times a day. ']
                ,['tid', 'three times a day. ']
                ,['T\.I\.D\.', 'three times a day. ']
                ,['TID', 'three times a day. ']
                ,['q\.i\.d\.', 'four times a day. ']
                ,['Q\.I\.D\.', 'four times a day. ']
                ,['qid', 'four times a day. ']
                ,['QID', 'four times a day. ']
                ,['I\.V\.', 'intravenous. ']
                ,['i\.v\.', 'intravenous. ']
                ,['q\.h\.s\.', 'before bed. ']
                ,['Q\.H\.S\.', 'before bed. ']
                ,['qhs', 'before bed. ']
                ,['Qhs', 'before bed. ']
                ,['QHS', 'before bed. ']
                ,[' hr ', ' hours ']
                ,[' hrs ', ' hours ']
                ,['hr(s)', 'hours']
                ,['O\.D\.', 'in the right eye. ']
                ,['o\.d\.', 'in the right eye. ']
                ,[' OD ', ' in the right eye ']
                ,[' od ', ' in the right eye ']
                ,[' 5X ', ' a day five times a day ']
                ,[' 5x ', ' a day five times a day ']
                ,[' OS ', ' in the left eye ']
                ,[' os ', ' in the left eye ']
                ,['q\.4h', 'every four hours. ']
                ,['Q\.4H', 'every four hours. ']
                ,['q24h', 'every 24 hours. ']
                ,['Q24H', 'every 24 hours. ']
                ,['q4h', 'every four hours. ']
                ,['Q4H', 'every four hours. ']
                ,['O\.U\.', 'in both eyes. ']
                ,['o\.u\.', 'in both eyes. ']
                ,[' OU ', ' in both eyes. ']
                ,[' ou ', ' in both eyes. ']
                ,['q\.6h', 'every six hours. ']
                ,['Q\.6H', 'every six hours. ']
                ,['q6h', 'every six hours. ']
                ,['Q6H', 'every six hours. ']
                ,['q\.8h', 'every eight hours. ']
                ,['Q\.8H', 'every eight hours. ']
                ,['q8h', 'every eight hours. ']
                ,['Q8H', 'every eight hours. ']
                ,['q8hr', 'every eight hours. ']
                ,['Q8hr', 'every eight hours. ']
                ,['Q8HR', 'every eight hours. ']
                ,['q\.12h', 'every 12 hours. ']
                ,['Q\.12H', 'every 12 hours. ']
                ,['q12h', 'every 12 hours. ']
                ,['Q12H', 'every 12 hours. ']
                ,['q12hr', 'every 12 hours. ']
                ,['Q12HR', 'every 12 hours. ']
                ,['Q12hr', 'every 12 hours. ']
                ,['q\.o\.d\.', 'every other day. ']
                ,['Q\.O\.D\.', 'every other day. ']
                ,['qod', 'every other day. ']
                ,['QOD', 'every other day. ']
                ,['prn', 'as needed.']
                ,['PRN', 'as needed.']]

    def __call__(self, note: str) -> str:

        if self.clean_text:
            note =  self.clean(note)
        if self.apply_replace:
            note = self.apply_replace_list(note)
        if self.lower:
            note = note.lower()
        if self.convert_danish_characters:
            # use regex instead of str.replace to avoid replacing characters in the middle of words
            note = re.sub(r'å', 'aa', note)
            note = re.sub(r'æ', 'ae', note)
            note = re.sub(r'ø', 'oe', note)
        if self.remove_accents:
            note = re.sub(r'é|è|ê', "e", note)
            note = re.sub(r'á|à|â', "a", note)
            note = re.sub(r'ô|ó|ò', "o", note)
        if self.remove_brackets:
            note = re.sub("\[|\]", "", note)
        if self.remove_special_characters:
            note = re.sub("[^a-zA-Z0-9 ]", "", note)
            note = re.sub("\n|/|-", " ", note)
        if self.remove_special_characters_mullenbach:
            note = re.sub("[^A-Za-z0-9]+", " ", note)
        if self.remove_digits:
            note = note.replace("\d+", "", regex=True)
        if self.remove_adm_details:
            note = re.sub("admission date:|discharge date:|date of birth:|addendum:|--|__|==", "", note)                
        note = note.strip()
        return note
    
    def apply_replace_list(self, x : str) -> str:
        """
        Preprocess text to replace common medical abbreviations with their full form
        """
        processed_text = x
        for find, replace in self.replace_list:
            processed_text = re.sub(find,replace,processed_text)
        return processed_text
    
    def clean(self, text):
        text = re.sub(u'\xa0 ', ' ', text)
        text = re.sub('&gt;', '>', text)
        text = re.sub('&lt;', '<', text)
        text = re.sub('&amp;', ' and ', text)
        text = re.sub('&#x20;', ' ', text)
        text = re.sub('\u2022', '\t', text)
        text = re.sub('\x1d|\xa0|\x1c|\x14', ' ', text)
        text = re.sub('### invalid font number [0-9]+', ' ', text)
        text = re.sub('[ ]+', ' ', text)

        return text


def make_chunks(notes_path: str, text_preprocessor : TextPreprocessor, chunk_size : int = 10_000):
    notes_reader = pandas.read_csv(notes_path, keep_default_na = False ,chunksize=chunk_size)
    text_data = []
    # make sure the notes folder exisits, if not create it and set permission to user owner only
    if not os.path.exists('/scratch/sifal.klioui/notes/train'):
        os.makedirs('/scratch/sifal.klioui/notes/train', mode =0o700, exist_ok = True)
        os.makedirs('/scratch/sifal.klioui/notes/test', mode =0o700, exist_ok = True)
        
    for i, chunk in enumerate(notes_reader):
        text_data.extend(text_preprocessor(chunk)['text'].tolist())
        if i < 30:
            with open(f'/scratch/sifal.klioui/notes/train/chunck_{i}.txt', 'w', encoding='utf-8') as fp:
                fp.write('\n'.join(text_data))
        else:
            with open(f'/scratch/sifal.klioui/notes/test/chunck_{i}.txt', 'w', encoding='utf-8') as fp:
                fp.write('\n'.join(text_data))
        text_data = []


def _tokenize_function(examples, tokenizer):
        # Remove empty lines
        examples['text'] = [
            line for line in examples['text'] if len(line) > 0 and not line.isspace()
        ]
        return tokenizer(
            examples['text'],
            padding=False,
            truncation=True,
            max_length=128, # the sequence lenght with which the model was pretrained
            # We use this option because DataCollatorForLanguageModeling (see below) is more efficient when it
            # receives the `special_tokens_mask`.
            return_special_tokens_mask=True,
        )


def preprocess_logits_for_metrics(logits, labels):
        if isinstance(logits, tuple):
            # Depending on the model and config, logits may contain extra tensors,
            # like past_key_values, but logits always come first
            logits = logits[0]
        return logits.argmax(dim=-1)


def prepare_sequences(eval_preds):
            preds, labels = eval_preds
            # preds have the same shape as the labels, after the argmax(-1) has been calculated
            # by preprocess_logits_for_metrics
            labels = labels.reshape(-1)
            preds = preds.reshape(-1)
            mask = labels != -100
            labels = labels[mask]
            preds = preds[mask]
            return preds, labels


# NOTE: this code is taken directly from Emily Alsentze's clinicalBERT  github repository 
# https://github.com/EmilyAlsentzer/clinicalBERT/blob/master/lm_pretraining/heuristic_tokenize.py



def strip(s):   
    return s.strip()

def is_inline_title(text):
    m = re.search('^([a-zA-Z ]+:) ', text)
    if not m:
        return False
    return is_title(m.groups()[0])

stopwords = set('of', 'on', 'or')
def is_title(text):
    if not text.endswith(':'):
        return False
    text = text[:-1]

    # be a little loose here... can tighten if it causes errors
    text = re.sub('(\([^\)]*?\))', '', text)

    # Are all non-stopwords capitalized?
    for word in text.split():
        if word in stopwords:
            continue
        if not word[0].isupper():
            return False

    # I noticed this is a common issue (non-title aapears at beginning of line)
    return (text == 'Disp')

def sent_tokenize_rules(text):

    # long sections are OBVIOUSLY different sentences
    text = re.sub('---+', '\n\n-----\n\n', text)
    text = re.sub('___+', '\n\n_____\n\n', text)
    text = re.sub('\n\n+', '\n\n', text)

    segments = text.split('\n\n')

    ### Separate section headers ###
    new_segments = []

    # deal with this one edge case (multiple headers per line) up front
    m1 = re.match('(Admission Date:) (.*) (Discharge Date:) (.*)', segments[0])
    if m1:
        new_segments += list(map(strip,m1.groups()))
        segments = segments[1:]

    m2 = re.match('(Date of Birth:) (.*) (Sex:) (.*)'            , segments[0])
    if m2:
        new_segments += list(map(strip,m2.groups()))
        segments = segments[1:]

    for segment in segments:
        # find all section headers
        possible_headers  = re.findall('\n([A-Z][^\n:]+:)', '\n'+segment)
        #assert len(possible_headers) < 2, str(possible_headers)
        headers = []
        for h in possible_headers:
            #print 'cand=[%s]' % h
            if is_title(h.strip()):
                #print '\tYES=[%s]' % h
                headers.append(h.strip())

        # split text into new segments, delimiting on these headers
        for h in headers:
            h = h.strip()

            # split this segment into 3 smaller segments
            ind = segment.index(h)
            prefix = segment[:ind].strip()
            rest   = segment[ ind+len(h):].strip()

            # add the prefix (potentially empty)
            if len(prefix) > 0:
                new_segments.append(prefix.strip())

            # add the header
            new_segments.append(h)

            # remove the prefix from processing (very unlikely to be empty)
            segment = rest.strip()

        # add the final piece (aka what comes after all headers are processed)
        if len(segment) > 0:
            new_segments.append(segment.strip())

    # copy over the new list of segments (further segmented than original segments)
    segments = list(new_segments)
    new_segments = []


    ### Low-hanging fruit: "_____" is a delimiter
    for segment in segments:
        subsections = segment.split('\n_____\n')
        new_segments.append(subsections[0])
        for ss in subsections[1:]:
            new_segments.append('_____')
            new_segments.append(ss)

    segments = list(new_segments)
    new_segments = []


    ### Low-hanging fruit: "-----" is a delimiter
    for segment in segments:
        subsections = segment.split('\n-----\n')
        new_segments.append(subsections[0])
        for ss in subsections[1:]:
            new_segments.append('-----')
            new_segments.append(ss)

    segments = list(new_segments)
    new_segments = []

    '''
    for segment in segments:
        print '------------START------------'
        print segment
        print '-------------END-------------'
        print
    exit()
    '''

    ### Separate enumerated lists ###
    for segment in segments:
        if not re.search('\n\s*\d+\.', '\n'+segment): 
            new_segments.append(segment)
            continue

        '''
        print '------------START------------'
        print segment
        print '-------------END-------------'
        print
        '''

        # generalizes in case the list STARTS this section
        segment = '\n'+segment

        # determine whether this segment contains a bulleted list (assumes i,i+1,...,n)
        start = int(re.search('\n\s*(\d+)\.', '\n'+segment).groups()[0])
        n = start
        while re.search('\n\s*%d\.'%n,segment): # SHOULD CHANGE TO: while re.search('\n\s*%d\.'%n,segment): #(CHANGED . to \.)
            n += 1
        n -= 1

        # no bulleted list
        if n < 1:
            new_segments.append(segment)
            continue

        '''
        print '------------START------------'
        print segment
        print '-------------END-------------'

        print start,n
        print 
        '''

        # break each list into its own line
        # challenge: not clear how to tell when the list ends if more text happens next
        for i in range(start,n+1):
            matching_text = re.search('(\n\s*\d+\.)',segment).groups()[0] 
            prefix  = segment[:segment.index(matching_text) ].strip()
            segment = segment[ segment.index(matching_text):].strip()
            if len(prefix)>0:
                new_segments.append(prefix)
            
        if len(segment)>0:
            new_segments.append(segment)

    segments = list(new_segments)
    new_segments = []

    ### Remove lines with inline titles from larger segments (clearly nonprose)
    for segment in segments:
        '''
        With: __PHI_6__, MD __PHI_5__
        Building: De __PHI_45__ Building (__PHI_32__ Complex) __PHI_87__
        Campus: WEST
        '''

        lines = segment.split('\n')

        buf = []
        for i in range(len(lines)):
            if is_inline_title(lines[i]):
                if len(buf) > 0:
                    new_segments.append('\n'.join(buf))
                buf = []
            buf.append(lines[i])
        if len(buf) > 0:
            new_segments.append('\n'.join(buf))

    segments = list(new_segments)
    new_segments = []

    # Going to put one-liner answers with their sections 
    # (aka A A' B B' C D D' -->  AA' BB' C DD' )
    N = len(segments)
    for i in range(len(segments)):
        # avoid segfaults
        if i==0: 
            new_segments.append(segments[i])
            continue

        if  segments[i].count('\n') == 0  and \
               is_title(segments[i-1]) and \
           not is_title(segments[i  ]):
            if (i == N-1) or is_title(segments[i+1]):
                new_segments = new_segments[:-1]
                new_segments.append(segments[i-1] + ' ' + segments[i])
            else:
                new_segments.append(segments[i])
            # currently If the code sees a segment that doesn't have any new lines and the prior line is a title 
            # *but* it is not the last segment and the next segment is not a title then that segment is just dropped
            # so lists that have a title header will lose their first entry
        else:
            new_segments.append(segments[i])

    segments = list(new_segments)
    new_segments = []

    return segments



def tokenize_notes(notes_path, tokenizer):
    
    tokenizer = BertTokenizer.from_pretrained(tokenizer)
    
    notes = pd.read_csv(notes_path) 
    notes.drop_duplicates(subset = ['hadm_id'], inplace = True)

    def tokenize_function(example):
        encodings = tokenizer(example['text'], truncation=True, padding='max_length', max_length=512)
        return encodings


    raw_datasets = Dataset.from_pandas(notes)
    tokenized_datasets = raw_datasets.map(
                tokenize_function,
                batched=True,
                num_proc=42, # just to have the cache reused (used for cache signature)
                remove_columns=['text'],
                load_from_cache_file= True,
                desc="Running tokenizer on every text in dataset",
            )
    
    return tokenized_datasets

def tokenize_and_reindex_hospital_notes(tokenized_datasets, hospital_ids_source, notes_path, tokenizer = 'bert-base-uncased', save_path = None):
    
    tokenized_datasets = tokenize_notes(notes_path, tokenizer)
    
    hadm_id_to_index = {entry['hadm_id']: idx for idx, entry in enumerate(tokenized_datasets)}
    
    hospital_ids_source_reindexed = [[hadm_id_to_index[hadm_id] for hadm_id in hadm_id_list] for hadm_id_list in hospital_ids_source]
    
    tokenized_datasets = tokenized_datasets.remove_columns("hadm_id")

    if save_path:
        tokenized_datasets.save_to_disk(save_path)
        pickle.dump(hospital_ids_source_reindexed, open(f"{save_path}hospital_ids_source_reindexed.pkl", "wb"))
        
    
    return tokenized_datasets, hospital_ids_source_reindexed