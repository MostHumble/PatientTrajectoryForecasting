import pandas
from tqdm.auto import tqdm 
import os
import re 

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

    def __call__(self, df: pandas.DataFrame) -> pandas.DataFrame:

        if self.clean_text:
            df['text'] = df['text'].apply(lambda x: self.clean(x))
        if self.apply_replace:
            df['text'] = df['text'].apply(lambda x: self.apply_replace_list(x))
        if self.lower:
            df['text'] = df['text'].str.lower()
        if self.convert_danish_characters:
            df['text'] = df['text'].str.replace("å", "aa", regex=True)
            df['text'] = df['text'].str.replace("æ", "ae", regex=True)
            df['text'] = df['text'].str.replace("ø", "oe", regex=True)
        if self.remove_accents:
            df['text'] = df['text'].str.replace("é|è|ê", "e", regex=True)
            df['text'] = df['text'].str.replace("á|à|â", "a", regex=True)
            df['text'] = df['text'].str.replace("ô|ó|ò", "o", regex=True)
        if self.remove_brackets:
            df['text'] = df['text'].str.replace("\[[^]]*\]", "", regex=True)
        if self.remove_special_characters:
            df['text'] = df['text'].str.replace("\n|/|-", " ", regex=True)
            df['text'] = df['text'].str.replace(
                "[^a-zA-Z0-9 ]", "", regex=True
            )
        if self.remove_special_characters_mullenbach:
            df['text'] = df['text'].str.replace(
                "[^A-Za-z0-9]+", " ", regex=True
            )
        if self.remove_digits:
            df['text'] = df['text'].str.replace("(\s\d+)+\s", " ", regex=True)
        if self.remove_adm_details:
            df['text'] = df['text'].str.replace(
                "admission date:|discharge date:|date of birth:|addendum:|--|__|==", "", regex=True
            )
                
        df['text'] = df['text'].str.replace("\s+", " ", regex=True)
        df['text'] = df['text'].str.strip()
        return df
    
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