import os
import vertexai.preview.generative_models as generative_models

from typing import Optional, List

from vertexai.generative_models import GenerativeModel, Part

from google.cloud import storage

from config import Config

os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = Config.GOOGLE_APPLICATION_CREDENTIALS

class Gemini():
    def __init__(self):
        self.model = 'gemini-1.5-flash-001'
        self.temperature = 0.2
        self.max_tkns = 8192
        self.top_p = 0.95

        self.llm = self.initialize_gemini()
    
    def initialize_gemini(self):
        safety_settings = {generative_models.HarmCategory.HARM_CATEGORY_UNSPECIFIED: generative_models.HarmBlockThreshold.BLOCK_NONE,
                           generative_models.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: generative_models.HarmBlockThreshold.BLOCK_NONE,
                           generative_models.HarmCategory.HARM_CATEGORY_HATE_SPEECH: generative_models.HarmBlockThreshold.BLOCK_NONE,
                           generative_models.HarmCategory.HARM_CATEGORY_HARASSMENT: generative_models.HarmBlockThreshold.BLOCK_NONE,
                           generative_models.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: generative_models.HarmBlockThreshold.BLOCK_NONE}
    
        agent_prompt = """You're a service agent in charge of generating drafts for multiple drug discovery reports in a pharma company.
        Use the attached documents and the user request to write the draft of a specific section based on the following rules:
        1. Talk in a formal and professional manner.
        2. Use ALL the documents when necessary. Do NOT ignore any of them.
        3. Use ONLY the information on the given documents to generate your answer.
        4. Give ONLY the draft as response. Do NOT interact with the user."""
        
        model = GenerativeModel(self.model,
                                generation_config = {"temperature": self.temperature,
                                                     "max_output_tokens": self.max_tkns,
                                                     "top_p": self.top_p},
                                system_instruction = agent_prompt,
                                safety_settings = safety_settings)
        print('LLM initialized')
        return model
    
    def create_docs(self, paths):
        """Formats the documents from the user's Cloud Storage folder so the LLM can read them
        Inputs:
            - paths: The list of documents in cloud storage uris"""
        
        docs = []
        
        bucket, remain = paths[0].replace("gs://", "").split("/", 1)
        blob_name = remain.rsplit("/", 1)[0]

        for path in paths:
            # Read from the file
            filename = path.rsplit("/", 1)[1]
            storage_client = storage.Client()
            bkt = storage_client.bucket(bucket)
            blob = bkt.blob("/".join([blob_name, filename]))
            
            with blob.open("rb") as f:
                data = f.read()
                
            # Create the VertexAI document
            docs.append(Part.from_data(mime_type = "application/pdf",
                                       data = data))

        print(f'{len(docs)} documents retrieved')
        return docs
    
    def generate_draft(self, query:str, paths:Optional[List]=[]):
        if not paths:
            msg = [query]
        else:
            docs = self.create_docs(paths)
            msg = docs + [query]


        chat = self.llm.start_chat(response_validation = False)

        draft = chat.send_message(msg)
        return draft.text