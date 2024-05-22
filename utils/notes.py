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

    def __call__(self, note: str) -> str:
        if self.lower:
            note = note.str.lower()
        if self.convert_danish_characters:
            note = note.str.replace("å", "aa", regex=True)
            note = note.str.replace("æ", "ae", regex=True)
            note = note.str.replace("ø", "oe", regex=True)
        if self.remove_accents:
            note = note.str.replace("é|è|ê", "e", regex=True)
            note = note.str.replace("á|à|â", "a", regex=True)
            note = note.str.replace("ô|ó|ò", "o", regex=True)
        if self.remove_brackets:
            note = note.str.replace("\[[^]]*\]", "", regex=True)
        if self.remove_special_characters:
            note = note.str.replace("\n|/|-", " ", regex=True)
            note = note.str.replace(
                "[^a-zA-Z0-9 ]", "", regex=True
            )
        if self.remove_special_characters_mullenbach:
            note = note.str.replace(
                "[^A-Za-z0-9]+", " ", regex=True
            )
        if self.remove_digits:
            note = note.str.replace("(\s\d+)+\s", " ", regex=True)

        note = note.str.replace("\s+", " ", regex=True)
        note = note.str.strip()
        return note
    