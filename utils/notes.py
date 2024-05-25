import pandas
from tqdm.auto import tqdm 
import os
import re 


class TextPreprocessor:
    def __init__(
        self,
        lower: bool = True,
        remove_special_characters_mullenbach: bool = True,
        remove_special_characters: bool = False,
        remove_digits: bool = True,
        remove_accents: bool = False,
        remove_brackets: bool = False,
        convert_danish_characters: bool = False,
    ) -> None:
        self.lower = lower
        self.remove_special_characters_mullenbach = remove_special_characters_mullenbach
        self.remove_digits = remove_digits
        self.remove_accents = remove_accents
        self.remove_special_characters = remove_special_characters
        self.remove_brackets = remove_brackets
        self.convert_danish_characters = convert_danish_characters

    def __call__(self, df: pandas.DataFrame) -> pandas.DataFrame:
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

        df['text'] = df['text'].str.replace("\s+", " ", regex=True)
        df['text'] = df['text'].str.strip()
        return df

def make_chunks(notes : pandas.DataFrame):
    text_data = []
    file_count = 0
    # make sure the notes folder exisits, if not create it and set permission to user owner only
    if not os.path.exists('/scratch/enroot-sifal.klioui/notes'):
        os.makedirs('/scratch/enroot-sifal.klioui/notes')
        os.system('chmod 700 /scratch/enroot-sifal.klioui/notes')
    for sample in tqdm(notes['text']):
        sample = sample.replace('\n', '')
        text_data.append(sample)
        if len(text_data) == 10_000:
            # once we hit the 10K mark, save to file
            with open(f'/scratch/enroot-sifal.klioui/notes/chunck_{file_count}.txt', 'w', encoding='utf-8') as fp:
                fp.write('\n'.join(text_data))
            text_data = []
            file_count += 1
    # after saving in 10K chunks, we will have ~2082 leftover samples, we save those now too
    with open(f'/scratch/enroot-sifal.klioui/notes/chunck_{file_count}.txt', 'w', encoding='utf-8') as fp:
        fp.write('\n'.join(text_data))


def combine_sent(x, combined_sent_max_len=92):
    combined_sent = ''
    combined_sent_list = []
    for i in range(len(x)):
        sent = x[i]
        if sent!='.':
            sent = re.sub('^\s+|\n|\r',' ',
            re.sub('\s\s|\t|\.|\,||admission date:|discharge date:|date of birth:|addendum:|--|__|==','',
                   sent.lower())).strip()
            sent_len = len(sent.split(' '))
            combined_sent_len = len(combined_sent.split(' '))
            
            if i == 0:
                combined_sent = sent
                if len(x) == 1:
                    combined_sent_list.append(combined_sent)
                
            else:
                # when len of sentence + combined sent < 92, combine the existin combined list with current sentence
                if sent_len + combined_sent_len <= combined_sent_max_len:
                    combined_sent = combined_sent + ' . ' + sent
                    if i == len(x) - 1:
                        combined_sent_list.append(combined_sent) 
                else:
                # when len is longer then append current combined sent into final list and reinitialize combined sent with current sent 
                    combined_sent_list.append(combined_sent)
                    combined_sent = sent
                
    return combined_sent_list


replace_LIST = [['dr\.',''] ,['DR\.','']
                ,['m\.d\.',''] ,['M\.D\.','']
                ,['yo', 'years old. ']
                ,['p\.o', 'orally. ']
                ,['P\.O', 'orally. ']
                ,['po', 'orally. ']
                ,['PO', 'orally. ']
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
                ,['OD', 'in the right eye. ']
                ,['od', 'in the right eye. ']
                ,['5X', 'a day five times a day. ']
                ,['5x', 'a day five times a day. ']
                ,['OS', 'in the left eye. ']
                ,['os', 'in the left eye. ']
                ,['q\.4h', 'every four hours. ']
                ,['Q\.4H', 'every four hours. ']
                ,['q24h', 'every 24 hours. ']
                ,['Q24H', 'every 24 hours. ']
                ,['q4h', 'every four hours. ']
                ,['Q4H', 'every four hours. ']
                ,['O\.U\.', 'in both eyes. ']
                ,['o\.u\.', 'in both eyes. ']
                ,['OU', 'in both eyes. ']
                ,['ou', 'in both eyes. ']
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
                ,['PRN', 'as needed.']
                ,['[0-9]+\.','']]


def preprocess_re_sub(x : str , replace_LIST : list) -> str:
    """
    Preprocess text to replace common medical abbreviations with their full form
    """
    processed_text = x
    for find,replace in replace_LIST:
        processed_text = re.sub(find,replace,processed_text)
    return processed_text

import calendar

email_map = 'mail'
org_map = ['hospital', 'company', 'university']
location_map = ['address', 'location' 'ward', 'state', 'country']
name_map = 'name'
number_map = ['number', 'telephone']
date_map = ['month', 'year', 'day'] + [calendar.month_name[i].lower() for i in range(1,13)]

def filter_brackets(text: str) -> str:

    unk_list = set(re.findall(r'\[\*.+?\*\]', text))
    unk_map = {}
    for elem in unk_list:
        find = False
        elem = elem.lower()
        if email_map in elem:
            unk_map[elem] = 'email'
            continue
        for w in org_map:
            if w in elem:
                unk_map[elem] = 'unknown {}'.format(w)
                find = True
                continue
        if find: continue
        for w in location_map:
            if w in elem:
                unk_map[elem] = 'unknown {}'.format(w)
                find = True
                continue
        if find: continue
        if name_map in elem:
            unk_map[elem] = 'unknown person'
            continue
        for w in date_map:
            if w in elem or re.search('[a-zA-Z]', elem) is None:
                unk_map[elem] = 'unknown date'
                find = True
                continue
        if find: continue
        unk_map[elem] = '[UNK]'
    for elem in unk_list:
        text = text.replace(elem, unk_map[elem])
    text = text.replace("unknown person unknown person", "unknown person")
    return text

from unidecode import unidecode

def clean_excel_text(text):

    if not isinstance(text, str):
        return text
    ## Clean
    clean = text
    clean = re.sub('[\t\r\n]', '   ', clean)
    clean = re.sub(u'\xa0 ', ' ', clean)
    clean = re.sub('&gt;', '>', clean)
    clean = re.sub('&lt;', '<', clean)
    clean = re.sub('&amp;', ' and ', clean)
    clean = re.sub('&#x20;', ' ', clean)
    clean = re.sub('\u2022', '\t', clean)
    clean = re.sub('\x1d|\xa0|\x1c|\x14', ' ', clean)
    clean = re.sub('### invalid font number [0-9]+', ' ', clean)
    clean = re.sub('[ ]+', ' ', clean)
    clean = unidecode(clean)
    return clean
