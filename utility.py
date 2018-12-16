"""
Utility functions
"""

from nltk import word_tokenize
import pandas as pd

class utility:
    
    def term_freq(string_vec):
        """
        Calculates the token frequency in a string vector
        """
        
        ## Tokenizing the strings
        
        tokens = [word_tokenize(s) for s in string_vec]
        tokens_all = []
        for item in range(len(tokens)):
            tokens_all = tokens_all + tokens[item]
        
        ## Calculating the count of each unique token
        
        unique_tokens = list(set(tokens_all))
        count_dict = [{'term' : token, 
                       'count' : tokens_all.count(token)}
                        for token in unique_tokens]
        
        ## Making a pandas data frame from the dictionary
        
        count_frame = pd.DataFrame.from_dict(count_dict)
        count_frame = count_frame.sort_values('count', ascending=False)        
        
        return count_frame