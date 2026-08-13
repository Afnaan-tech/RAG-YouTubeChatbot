import os
from dotenv import load_dotenv
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import ChatGoogleGenerativeAI,GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

url = input("Enter YouTube URL: ")
video_id = url.split("v=")[1].split("&")[0]  #it will extract the video id

   
try:
    youtube_api = YouTubeTranscriptApi()
    transcript_list = youtube_api.fetch(video_id, languages=["en"])     

    transcript = " ".join(chunk.text for chunk in transcript_list)

except TranscriptsDisabled:
    print("No captions available for this video.")

splitter=RecursiveCharacterTextSplitter(chunk_size=3000,chunk_overlap=200)
chunks=splitter.create_documents([transcript])

embedding_model = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
vector_store=FAISS.from_documents(chunks,embedding_model)

retriever=vector_store.as_retriever(search_type="similarity",search_kwargs={"k":4})#returns 4 most relevant chunks

prompt=PromptTemplate(template="YOU ARE A HELPFUL AI ASSISTANT ANSWER ONLY FROM THE PROVIDED TRANSCRIPT   {context} QUESTION:{question}",input_variables=['context','question'])
llm=ChatGoogleGenerativeAI(model="gemini-3.6-flash")

#all the page_content gets concanated into one str
def format_docs(retrieved_docs):
    context="\n\n".join(doc.page_content for doc in retrieved_docs)
    return context

parser=StrOutputParser()

chain1=RunnableParallel({
    "context":retriever | RunnableLambda(format_docs),
    "question":RunnablePassthrough()
})
chain2= prompt | llm | parser
final_chain=chain1 | chain2


question="summarize what is happening in the video"
result=final_chain.invoke(question)
print(result)