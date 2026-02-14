import re
def tokenise(text):
    text = str(text)
    # Surround structural characters with spaces so they become independent tokens
    for char in ['(', ')', '{', '}', '[', ']', ';', ',', '.', '!']:
        text = text.replace(char, f" {char} ")
    
    text = re.sub(r'\s+', ' ', text).strip()
    return text.split()
import re
PUNCT = r'[!"#$%&()*+,\-./:;<=>?@\[\]^_`{|}~]'

# TOKENISATION for Normal Textual data(# Not code #) 

# def tokenise(text):
#     text = str(text).lower() # covert into lower
#     text = re.sub(r'('+PUNCT+r')',r' \1',text) #introducing commas
#     # betweeen punc char, so that they dont get inc in words
#     text = re.sub(r'\s+', ' ', text).strip()
#     text = text.replace(',',"")
#     text = text.replace('!',"")
#     text = text.replace('.',"")
#     text = text.replace('"',"")
#     # itnroducing space between each commas.
#     return text.split()