﻿# YT_SUMMARIZER

🎯 YouTube DeepSummarizer is a web application that provides detailed, long-form summaries of YouTube videos by extracting and analyzing their transcripts. This tool leverages LangChain, OpenAI's GPT models, and other advanced technologies to generate a comprehensive, insightful, and structured summary of video content.

Features
Automatic Video Transcript Extraction: Extracts the transcript of YouTube videos using the YouTube Transcript API.

Summarization: Uses OpenAI’s language models to generate detailed and structured summaries, including:

In-depth summary of the entire video

Highlighted key points with explanations

Insights and lessons learned from the video

Language Detection and Translation: Automatically detects the language of the transcript and translates the summary to English (optional feature).

Easy Integration with Streamlit: Provides a simple user interface for inputting video links and generating summaries.

Installation
Prerequisites
Python 3.6+

Git

OpenAI API Key - You’ll need an OpenAI API key for accessing GPT models.

Step-by-Step Setup
Clone the repository:

bash
Copy
Edit
git clone https://github.com/MetalCloth/YT_SUMMARIZER.git
cd YT_SUMMARIZER
Install the required Python packages:

bash
Copy
Edit
pip install -r requirements.txt
Create a .env file in the root directory and add your OpenAI API key:

ini
Copy
Edit
OPENAI_API_KEY=your_openai_api_key
LANGCHAIN_API_KEY=your_langchain_api_key
Run the Streamlit app:

bash
Copy
Edit
streamlit run app.py
Dependencies
streamlit - For building the web interface.

requests - For making HTTP requests.

youtube-transcript-api - For extracting video transcripts.

langchain - For chaining AI-based models and tools.

openai - For interacting with OpenAI's language models.

faiss - For vector search and retrieval of relevant chunks.

Usage
Open the application in your browser (it will run locally at http://localhost:8501).

Paste the YouTube video URL into the provided input field.

Click on the Summarize button to generate the summary.

The summary will be displayed below the input field, including:

A detailed, multi-paragraph summary of the transcript.

Key insights and highlights, each with explanations.

Optionally, the translated summary (if the transcript language is detected as non-English).

Example
Video URL: https://www.youtube.com/watch?v=1L509JK8p1I

Output:
Summary: A detailed, multi-paragraph analysis of the video’s content, covering all key points and insights.

Highlights: A bulleted list of major highlights from the video, each with an emoji and brief explanation.

Key Insights: 5-7 sections, each diving deep into a specific lesson or concept from the video.

Contributing
