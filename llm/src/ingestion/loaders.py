from langchain_community.document_loaders import PyPDFLoader

#load pdf
def load_pdf(file_path : str) :
    loader = PyPDFLoader(file_path= file_path)
    
    #return documents (since pdf => each page one doc)
    return loader.load()
