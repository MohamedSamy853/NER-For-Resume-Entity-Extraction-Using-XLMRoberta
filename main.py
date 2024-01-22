from transformers import pipeline 
import gradio as gr
from PyPDF2 import PdfReader
import fitz  
import typing as t
pipe = pipeline("ner" , "./model/saved_model", aggregation_strategy="simple"  )

def filter_entitiles(output:t.Dict) ->t.List[t.Dict[str , t.Dict[str , str]]]:
    return [ent for ent in output if ent['score'] > 0.39 ]

def ner(text : str):
    if len(text) > 1000:
    
        output = pipe(text, aggregation_strategy="simple")
        return {"text": text, "entities": filter_entitiles(output)}
    else :
        full_text = ""
        full_entities ={}
        start = 0
        for i in range(1000, len(text) , 1000):
            sub_text = text[start: i]
            output = pipe(sub_text , aggregation_strategy="simple")
            entities = filter_entitiles(output)
            full_entities|=entities
            full_text+=sub_text
            start = i
        return {"text":full_text , "entities": filter_entitiles(output)}
    
            
            



def read_pdf(file_path : t.BinaryIO):
    with open(file_path, "rb") as file:
        pdf_reader = PdfReader(file)
        text = ""
        for page_num in range(len(pdf_reader.pages)):
            text += pdf_reader.pages[page_num].extract_text()
    return text


def predict(file:t.BinaryIO):
    
    text = read_pdf(file)
    output = ner(text)
    return output

    


demo = gr.Interface(
    fn = predict , 
    inputs=gr.File() ,
    outputs= gr.HighlightedText()
)

if __name__ =='__main__':
    demo.launch(inbrowser=True, inline=True)


