# For Problem 3
#
# we will use regular expressions to replace certain types of named entity substrings with special tokens.
#
# Please implement the ner() function below, and feel free to use the re library. 

import re

# DO NOT modify any function definitions or return types, as we will use these to grade your work.
# However, feel free to add new functions to the file to avoid redundant code.


def cyber_ner(input_string):
    
    import calendar

    IP_ADDRESS_REGEX = r'\b[0-9]{0,3}\.[0-9]{0,3}\.[0-9]{0,3}\.[0-9]{0,3}\b'

    TIME_REGEX = r'((?:[0-1]?[0-9]|2[0-3])(?:[0-5][0-9])?[ ]*[APapHh][MmrR][Ss]*)|((?:[0-1]?[0-9]|2[0-3]):([0-5][0-9]):(?:[0-5][0-9])?[ ]*[APapHh][MmrR][Ss]*)|(\d{1,2}:\d{1,2}(?::[0-5][0-9])?(?:[ ]*[APapHh][MmrR][Ss]*)?)'

    full_months = [month for month in calendar.month_name if month]
    short_months = [d[:3] for d in full_months]
    months = '|'.join(short_months + full_months)
    months = months + '|([0-9])(?:[0-9])?'
    sep = r'([-/ ])*'
    day = r'[0-9](?:[0-9])?(?:[SsNnRrTt][TtDdHh])?'
    year = r'([0-9][0-9])\'*(?:[0-9][0-9])?'
    day_or_year = r'(([0-9](?:[0-9])?(?:[SsNnRrTt][TtDdHh])?)|(([0-9][0-9])(?:[0-9][0-9])?))'
    DATE = re.compile(rf'((?:{day}{sep})?)({months})((?:{sep}{day_or_year})?)((?:{sep}{year})?)((?:{sep}{months})?)')
    DATE_REGEX = DATE.pattern
    
    EMAIL_ADDRESS_REGEX = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b'
    
    WEB_ADDRESS_REGEX = r'(?:(http|https)\:\/\/)?(?:www\.)?[A-Za-z0-9.-]+(?:\.[A-Za-z]{2,})+(?:[/\w-]*)*|(?:(http|https)\:\/\/)?(?:www\.)?[A-Za-z0-9-]+(?:\[A-Za-z]{2,})'
    
    DOLLAR_AMOUNT_REGEX = r'\$\d{1,3}(?:,\d{3})*(?:\.\d{2})?'

    REGEX_DICT = {
        'IP_ADDRESS': IP_ADDRESS_REGEX,
        'DOLLAR_AMOUNT': DOLLAR_AMOUNT_REGEX,
        'TIME': TIME_REGEX,
        'DATE': DATE_REGEX,
        'EMAIL_ADDRESS': EMAIL_ADDRESS_REGEX,
        'WEB_ADDRESS': WEB_ADDRESS_REGEX,
    }

    for entity, regex in REGEX_DICT.items():
        input_string = re.sub(regex, f'{entity}', input_string)

    return input_string


    # = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = #

