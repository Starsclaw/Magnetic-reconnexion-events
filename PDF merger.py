import os, PyPDF2

#Ask user where the PDFs are
userpdflocation=input('Where are your files')
userpdflocation=userpdflocation.replace('\\','/')

#Sets the scripts working directory to the location of the PDFs
os.chdir(userpdflocation)

#Ask user for the name to save the file as
file_name=input('What should I call the file?')

#Get all the PDF filenames
Merged_file = []
for filename in os.listdir('.'):
    if filename.endswith('.pdf'):
        Merged_file.append(filename)

pdfWriter = PyPDF2.PdfFileWriter()

#loop through all PDFs
for filename in Merged_file:
#rb for read binary
    pdfFile = open(filename,'rb')
    pdfReader = PyPDF2.PdfFileReader(pdfFile)
#Opening each page of the PDF
    for pageNum in range(pdfReader.numPages):
        pageObj = pdfReader.getPage(pageNum)
        pdfWriter.addPage(pageObj)
#save PDF to file, wb for write binary
pdfOutput = open(file_name+'.pdf','wb')
#Outputting the PDF
pdfWriter.write(pdfOutput)
#Closing the PDF writer
pdfOutput.close()