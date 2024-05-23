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