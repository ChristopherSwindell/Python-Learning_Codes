##import PyPDF2 as p2
##import os
##entries = os.listdir('Alb_Real_Estate/')
##print(entries)
##object = open('Alb_Real_Estate/87001.pdf','rb')
##reader = PyPDF2.PdfFileReader(object)
##print(reader.numPages)
##page = reader.getPage(0)
##print(page.extractText())

##PDFfile = open('Alb_Real_Estate/87001.pdf','rb')
##pdfread = p2.PdfFileReader(PDFfile)

'''Extract Single Page'''
##x = pdfread.getPage(0)
##print(x.extractText())
##print(pdfread.getIsEncrypted())
##print(pdfread.getDocumentInfo())
##print(pdfread.getNumPages())

'''Tabula-Py'''
from tabula import read_pdf
from tabulate import tabulate
df = read_pdf('Alb_Real_Estate/87001.pdf')
print(pdf)
