import glob
import io
import os
import re
import argparse
import pandas as pd

import zipfile
from urllib.request import urlopen
from urllib.parse import urlparse
from datasets import Dataset, DatasetDict

def load_zipfile() -> pd.DataFrame:
    path = 'https://www.boj.or.jp/en/research/wps_rev/wps_2023/data/wp23e14.zip'

    # process the zipfile from the bank of japan
    remotezip = urlopen(path) # read the zip file as a string
    zipinmemory = io.BytesIO(remotezip.read()) # convert from a string to bytes
    zip = zipfile.ZipFile(zipinmemory) # pass the bytes to python's zipfile handler
    
    # read the excel spreadsheet
    xlsx_path = zip.namelist()[0] # get the name of the excel file
    with zip.open(xlsx_path) as f:
        data = pd.read_excel(f, sheet_name=1, skiprows=[0,1], date_format='%m/%d/%y')
    
    # pandas has trouble with some column names, so rename those
    data = data.rename(columns={'Unnamed: 0':'date', 'Unnamed: 1':'bank', 
                                'Unnamed: 2':'speech', 'Unnamed: 9':'id'})
    return data

def fetch_speeches(years):
    paths = ['https://bis.org/speeches/speeches_' + str(year) + '.zip' for year in years]
    speeches = pd.DataFrame()

    for path in paths:
        remotezip = urlopen(path) # read the zip file as a string
        zipinmemory = io.BytesIO(remotezip.read()) # convert from a string to bytes
        zip = zipfile.ZipFile(zipinmemory) # pass the bytes to python's zipfile handler
        
        # read the excel spreadsheet
        xlsx_path = zip.namelist()[0] # get the name of the excel file
        with zip.open(xlsx_path) as f:
            data = pd.read_csv(f)
        
        data['id'] = [('').join(os.path.basename(urlparse(url).path).split('.')[:-1]) for url in data['url']]  
        speeches = pd.concat([speeches, data])
    return speeches

class DataLoader:
    def __init__(self, years = [1997, 1998, 1999, 2000, 2001, 2002, 2003, 2004, 2005,
                                2006, 2007, 2008, 2009, 2010, 2011, 2012, 2013, 2014,
                                2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024]) -> None:
        self.search_index = fetch_speeches(years)

    def fetch_text(self, id: str, sentence_chunking: int = 1):
        speech = self.search_index[self.search_index.id == id]
        
        try:
            assert len(speech) == 1
        except AssertionError:
            print('search error:', id, 'found', len(speech), 'matching patterns')
            return '', '', '', '', '', '', '', ''

        url = speech['url'].iloc[0]
        title = speech['title'].iloc[0]
        description = speech['description'].iloc[0]
        author = speech['author'].iloc[0]
        date = speech['date'].iloc[0]
        text = speech['text'].iloc[0]

        # use regex to clean the text
        text = text.replace('\n', ' ')
        url_pattern = re.compile(r'https?://\S+|www\.\S+')
        bullet_pattern = re.compile(r'\s(\d+)\.\s')
        space_pattern = re.compile(r'\s+')
        text = url_pattern.sub('', text)
        text = bullet_pattern.sub('', text)
        text = space_pattern.sub(' ', text)

        speechdata_pd = self.build_dataframe(text, id, sentence_chunking)
        speechdata_hf = Dataset.from_pandas(speechdata_pd)

        return url, title, description, author, date, text, speechdata_pd, speechdata_hf
    
    def build_dataframe(self, text: str, id: str, sentence_chunking: int) -> pd.DataFrame:
        speechdata = pd.DataFrame()
        # read the text and write it to a dataframe
        sentences = text.split('.')
        sentence_chunked = ['.'.join(sentences[i*sentence_chunking:(i+1)*sentence_chunking]) for i in range((len(sentences) // sentence_chunking)+1)]
        id_col = [id]*len(sentence_chunked)

        speechdata['id'] = id_col
        speechdata['paragraph'] = sentence_chunked
        return speechdata

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('id', nargs='?', default=None, help='number of speeches')
    args = parser.parse_args()

    dat = DataLoader()
    url, title, description, author, date, text, speechdata_pd, speechdata_hf = dat.fetch_text(args.id, sentence_chunking=5)

    print(title)
    print(text)
    print(speechdata_pd)
