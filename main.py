from langchain.embeddings import OpenAIEmbeddings
from youtube_transcript_api import YouTubeTranscriptApi

from langchain.vectorstores import FAISS
from langchain_core.output_parsers import StrOutputParser
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
import os
import streamlit as st
import requests
from dotenv import load_dotenv


from urllib.parse import urlparse, parse_qs

def extract_video_id(url):
    parsed = urlparse(url)

    if parsed.netloc in ["youtu.be"]:
        return parsed.path.lstrip("/")

    elif parsed.netloc in ["www.youtube.com","youtube.com"]:
        if parsed.path == "/watch":
            return parse_qs(parsed.query).get("v", [None])[0]
        elif parsed.path.startswith("/embed/") or parsed.path.startswith("/shorts/"):
            return parsed.path.split("/")[2]
    return None

load_dotenv(".env")


openai_api_key = os.getenv("OPENAI_API_KEY", st.secrets.get("OPENAI_API_KEY"))
langchain_api_key = os.getenv("LANGCHAIN_API_KEY", st.secrets.get("LANGCHAIN_API_KEY"))

os.environ["OPENAI_API_KEY"] = openai_api_key
os.environ["LANGCHAIN_API_KEY"] = langchain_api_key


st.title("🎯 YouTube DeepSummarizer")


url=st.text_input("Enter the Youtube video link")

button=st.button("Summarize")

if button and url:
    with st.spinner("Creating Summary..."):
        try:
            short=""
            short_url = extract_video_id(url)
            if short_url=="":
                st.error("Could not extract video ID. Please check the YouTube URL.")
                st.stop()

            try:
                transcript_list = YouTubeTranscriptApi.list_transcripts(short_url)
            except Exception as e:
                st.error(f"Transcript list error: {e}")
                st.stop()

            code = ""
            transcript = ""
            for i in transcript_list:
                if i.is_generated:
                    code = {
                        "language": i.language,
                        "language_code": i.language_code
                    }
                    transcript = i.fetch()
                    break
            if code=="":
                st.error("No auto-generated transcript found.")
                st.stop()

            c = " ".join(entry.text for entry in transcript)

            
            splitter=RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200,
                length_function=len
            )
    
            chunks = splitter.split_text(c)

            # Vector store
            documents = [Document(page_content=chunk) for chunk in chunks]
            vectordb=FAISS.from_documents(documents, OpenAIEmbeddings())



            retriever=vectordb.as_retriever()



            llm=ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)

            prompt=PromptTemplate(
                input_variables=["context", "input"],
                                template="""
                                You are an expert summarizer and technical writer and a good translator if it is in foreign language convert all to english and suggest a title for the summary in capitals.

            Your goal is to write a **very detailed, long-form analysis** of the following transcript. Please perform the following tasks:

            1. **In-Depth Summary**: Write a comprehensive, multi-paragraph summary of the entire transcript. Cover all important ideas, events, and technical details. Expand on every concept mentioned. Use as much detail as possible, writing at least 8–10 paragraphs.

            2. **Detailed Highlights**: Create a bulleted list of highlights, -each accompanied by an emoji. Write at least 2 to 3 sentences for each highlight, explaining *why* it matters or how it connects to the broader theme.

            3. **Key Insights with Explanation**: Write 5 to 7 structured sections, each as a fully developed paragraph. Each insight should dive deep into a particular idea or lesson mentioned in the transcript. These sections should read like mini-essays and show thoughtful interpretation.

            Use formal tone, elaborate explanations, and do not simplify the language. Be thorough and exhaustive.

            Here is the transcript:
            {context}
                                
                                """
            )

            document_chain=create_stuff_documents_chain(llm=llm, prompt=prompt)


            retrieval_chain=create_retrieval_chain(retriever,document_chain)

            response = retrieval_chain.invoke({"input": "Summarize the text"})
            st.markdown("### 📝 Summary Output")
            st.write(response['answer'])

        except Exception as e:
            st.error(f"Error: {e}")
                            

        


    
