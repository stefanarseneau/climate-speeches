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

import requests
import re
import pytesseract
from pdf2image import convert_from_bytes

import zipfile
from urllib.request import urlopen
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

class CleanReader:
    def __init__(self, url: str) -> None:
        self.url = url

        # https://pypdf.readthedocs.io/en/stable/user/post-processing-in-text-extraction.html
        temp_text = self.read_pdf()
        self.text = self.string_format(temp_text)

    def read_pdf(self) -> str:
        remotepdf = requests.get(self.url, stream=True, timeout=30).raw.read()
        pdfmemory = io.BytesIO(remotepdf)
        
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

        if '(cid:' in text:
            pages = convert_from_bytes(remotepdf)
            
            text = ""
            for pageNum,imgBlob in enumerate(pages):
                text += pytesseract.image_to_string(imgBlob,lang='eng')

        url_pattern = re.compile(r'https?://\S+|www\.\S+')
        bullet_pattern = re.compile(r'\s(\d+)\.\s')
        space_pattern = re.compile(r'\s+')
        text = url_pattern.sub('', text)
        text = bullet_pattern.sub('', text)
        text = space_pattern.sub(' ', text)
        return text

    def string_format(self, text: str) -> str:
        # clean the pdf formatting artifacts
        text = text.replace('\n', ' ')
        text = unicodedata.normalize('NFKD', text)       
        return text


class DataLoader:
    def __init__(self, id: str, sentence_chunking = 1) -> None:
        self.id = id
        self.url = 'https://bis.org/review/{}.pdf'.format(id)
        self.text = CleanReader(self.url).text

        self.speechdata_pd = self.build_dataframe(sentence_chunking)
        self.speechdata_hf = Dataset.from_pandas(self.speechdata_pd)
    
    def build_dataframe(self, sentence_chunking: int) -> pd.DataFrame:
        speechdata = pd.DataFrame()
        # read the text and write it to a dataframe
        sentences = self.text.split('.')
        sentence_chunked = ['.'.join(sentences[i*sentence_chunking:(i+1)*sentence_chunking]) for i in range((len(sentences) // sentence_chunking)+1)]
        id_col = [self.id]*len(sentence_chunked)

        speechdata['id'] = id_col
        speechdata['paragraph'] = sentence_chunked
        return speechdata

    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('id', nargs='?', default=None, help='number of speeches')
    args = parser.parse_args()

    dat = DataLoader(args.id, sentence_chunking=3)
    dat.speechdata_pd.to_csv('test.csv')
