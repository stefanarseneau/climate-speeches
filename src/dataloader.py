import glob
from typing import Type
import io
import os
import argparse
import pandas as pd

from pdfminer.pdfparser import PDFParser
from pdfminer.pdfdocument import PDFDocument
from pdfminer.pdfinterp import PDFResourceManager, PDFPageInterpreter
from pdfminer.pdfpage import PDFPage
from pdfminer.converter import XMLConverter, HTMLConverter, TextConverter
from pdfminer.layout import LAParams
import unicodedata

import zipfile
from urllib.request import urlopen
from datasets import Dataset, DatasetDict

class CleanReader:
    def __init__(self, url: str) -> None:
        self.url = url

        # https://pypdf.readthedocs.io/en/stable/user/post-processing-in-text-extraction.html
        temp_text = self.read_pdf()
        self.text = self.string_format(temp_text)

    def read_pdf(self) -> str:
        remotepdf = urlopen(self.url)
        pdfmemory = io.BytesIO(remotepdf.read())
        
        rsrcmgr = PDFResourceManager()
        retstr = io.StringIO()
        codec = 'utf-8'
        laparams = LAParams()
        laparams.all_texts = False
        laparams.detect_vertical = False
        device = TextConverter(rsrcmgr, retstr, codec=codec, laparams=laparams)
        # Create a PDF interpreter object.
        interpreter = PDFPageInterpreter(rsrcmgr, device)
        for page in PDFPage.get_pages(pdfmemory):
            interpreter.process_page(page)
            text = retstr.getvalue()

        text = text.replace("\\n", "\n") # read in the pdf
        return text

    def string_format(self, text: str) -> str:
        # clean the pdf formatting artifacts
        text = text.split('\n\n')
        text = [line.replace('\n', ' ') for line in text]
        text = [unicodedata.normalize('NFKD', line) for line in text if line]
        return text


class DataLoader:
    def __init__(self, n: int, savefile: bool) -> None:
        self.data = self.load_zipfile()
        self.n = n if n is not None else len(self.data)
        self.savefile = savefile
        self.speechdata_pd = self.build_dataframe()
        self.speechdata_hf = Dataset.from_pandas(self.speechdata_pd)

    def load_zipfile(self) -> pd.DataFrame:
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

    
    def build_dataframe(self) -> pd.DataFrame:
        self.data['url'] = ['https://bis.org/review/{}.pdf'.format(self.data['id'][i]) 
                       for i in range(len(self.data))]
        speechdata = pd.DataFrame(columns=['id', 'paragraph'])
        for i in range(self.n):
            # read the text and write it to a dataframe
            text = CleanReader(self.data['url'][i]).text
            id_col = [self.data['id'][i]]*len(text)
            text_df = pd.DataFrame({'id': id_col, 'paragraph': text})
            speechdata = pd.concat([speechdata, text_df])
            # save the file
            if self.savefile:
                text_df.to_csv(f'data/{self.data['id'][i]}.csv')
        return speechdata

    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', nargs='?', default=None, help='number of speeches')
    parser.add_argument('--savefile', action='store_true')
    args = parser.parse_args()
    
    n = int(args.n)
    savefile = args.savefile

    a = DataLoader(n, savefile)
    a.speechdata_pd.to_csv('test.csv')
