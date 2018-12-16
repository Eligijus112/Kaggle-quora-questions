"""
Module for text preprocessing
"""

from nltk.corpus import stopwords
import pandas as pd
import string
import re
import inflection

class clean: 
    
    def to_lower(string_vec):
        """
        Makes every word in a string vector lowercase
        """
        cleaned_string = [string.lower() for string in string_vec]
        cleaned_string = pd.Series(cleaned_string)
        return cleaned_string
        
    def rm_stop_words(string_vec):
        """
        Removes common stop words from a string vector
        """
        cleaned_string = []
        stop_words = stopwords.words("english")   
        for text in string_vec:
            text = [word for word in text.split() if word not in stop_words]
            text = ' '.join(text)
            cleaned_string.append(text)
        return pd.Series(cleaned_string)    
    
    def rm_punctuations(string_vec):
        """
        Removes punctuations and other special characters from a string vector
        """
        cleaned_string = [re.sub(r"[^a-zA-Z0-9 ]","",s) for s in string_vec]
        return pd.Series(cleaned_string)
    
    def rm_digits(string_vec):
        """
        Removes digits from a string vector
        """
        regex = re.compile('[%s]' % re.escape(string.digits))
        cleaned_string = [regex.sub('', s) for s in string_vec]
        return pd.Series(cleaned_string)
    
    def clean_ws(string_vec):
        """
        Cleans whitespaces
        """
        cleaned_string = [s.strip() for s in string_vec]
        cleaned_string = [re.sub( '\s+', ' ', s) for s in string_vec]
        return pd.Series(cleaned_string)
    
    def singularize(string_vec):
        """
        Singularizes nouns in a string vector
        """
        cleaned_string = []
        for text in string_vec:
            text = [inflection.singularize(word) for word in text.split()]
            text = ' '.join(text)
            cleaned_string.append(text)
        return pd.Series(cleaned_string) 
        
        
        