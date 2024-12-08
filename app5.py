import streamlit as st
import re
import pdfplumber
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lsa import LsaSummarizer
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# Initialize Sumy summarizer and SentenceTransformer
summarizer = LsaSummarizer()
embedding_model = SentenceTransformer('paraphrase-mpnet-base-v2')

# Define functions
def extract_text_from_pdf(pdf_path):
    with pdfplumber.open(pdf_path) as pdf:
        full_text = ''.join([page.extract_text() + '\n' for page in pdf.pages if page.extract_text()])
    return full_text

def clean_text(text):
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'(Table\s*\d+.*?\.)(?=\s|$)', '', text, flags=re.IGNORECASE)
    text = re.sub(r'(Figure\s*\d+.*?\.)(?=\s|$)', '', text, flags=re.IGNORECASE)
    return text.strip()

def extract_sections(text):
    title_match = re.search(r'^[A-Z][A-Z\s]+$', text, re.MULTILINE)
    title = title_match.group(0) if title_match else "Title Not Found"
    
    abstract = re.search(r'(?s)(?<=\bABSTRACT\b)(.*?)(?=\bINTRODUCTION\b)', text, re.IGNORECASE)
    introduction = re.search(r'(?s)(?<=\bINTRODUCTION\b)(.*?)(?=\bLITERATURE SURVEY\b)', text, re.IGNORECASE)
    problem_statement = re.search(r'(?s)(?<=\bPROBLEM STATEMENT AND OBJECTIVES OF THE PROPOSED WORK\b)(.*?)(?=\bPROPOSED METHOD\b)', text, re.IGNORECASE)
    proposed_method = re.search(r'(?s)(?<=\bPROPOSED METHOD\b)(.*?)(?=\bRESULTS AND DISCUSSION\b)', text, re.IGNORECASE)
    conclusion = re.search(r'(?s)(?<=\bCONCLUSION AND FUTURE ENHANCEMENTS\b)(.*?)(?=\bREFERENCES\b)', text, re.IGNORECASE)

    return {
        "Title": title,
        "Abstract": clean_text(abstract.group(0)).strip() if abstract else "Abstract Not Found",
        "Introduction": clean_text(introduction.group(0)).strip() if introduction else "Introduction Not Found",
        "Problem Statement": clean_text(problem_statement.group(0)).strip() if problem_statement else "Problem Statement Not Found",
        "Proposed Method": clean_text(proposed_method.group(0)).strip() if proposed_method else "Proposed Method Not Found",
        "Conclusion": clean_text(conclusion.group(0)).strip() if conclusion else "Conclusion Not Found"
    }

def summarize_sections(sections):
    summaries = {}
    for section, content in sections.items():
        if section == "Title":
            summaries[section] = content
        else:
            # Create a PlaintextParser and summarize
            parser = PlaintextParser.from_string(content, Tokenizer("english"))
            summary = summarizer(parser.document, 2)  # Summarize into 2 sentences

            # Join sentences into a single string
            summarized_text = ' '.join(str(sentence) for sentence in summary)

            # Ensure the summary ends with a complete sentence
            sentences = summarized_text.split('. ')
            complete_summary = '. '.join(sentences[:2]) + '.' if len(sentences) > 1 else summarized_text
            
            # Optionally limit character count for brevity
            if len(complete_summary) > 300:  # Adjust the length as needed
                complete_summary = ' '.join(complete_summary.split()[:60]) + '...'  # Show a truncated version
            
            summaries[section] = complete_summary
    return summaries


def calculate_semantic_similarity(sections):
    cleaned_sections = {section: clean_text(content) for section, content in sections.items()}
    embeddings = embedding_model.encode(list(cleaned_sections.values()), show_progress_bar=True)
    
    similarity_matrix = cosine_similarity(embeddings)

    weighted_similarity_scores = {}
    section_names = list(sections.keys())

    
    adjusted_weights = {
        "Title": 1.5,
        "Abstract": 1.8,
        "Introduction": 1.2, 
        "Problem Statement": 1.0,  
        "Proposed Method": 2.0,
        "Conclusion": 0.8,  
    }

    for i in range(len(section_names)):
        for j in range(i + 1, len(section_names)):
            weight = adjusted_weights.get(section_names[i], 1.0) * adjusted_weights.get(section_names[j], 1.0)
            weighted_similarity_scores[(section_names[i], section_names[j])] = similarity_matrix[i, j] * weight

    return weighted_similarity_scores


def check_relevance(similarity_scores):
    scaled_scores = {pair: score * 5 for pair, score in similarity_scores.items()}
    average_similarity = np.mean(list(scaled_scores.values()))
    
    return "Yes, the sections are relevant." if average_similarity >= 4 else "No, the sections are not relevant."

# Streamlit Interface
st.title("Document Summarization and Semantic Similarity Checker")
uploaded_file = st.file_uploader("Upload a PDF document", type="pdf")

if uploaded_file:
    with st.spinner("Processing the PDF..."):
        try:
            # Step 1: Extract text from the PDF
            pdf_text = extract_text_from_pdf(uploaded_file)

            # Step 2: Extract sections
            sections = extract_sections(pdf_text)

            # Step 3: Summarize the sections
            summarized_sections = summarize_sections(sections)

            # Display Summarized Sections
            st.subheader("--- Summarized Sections ---")
            for section_name, summary in summarized_sections.items():
                st.write(f"*{section_name} Summary:*\n{summary}")
                st.write("-" * 40)

            # Step 4: Calculate semantic similarity
            similarity_scores = calculate_semantic_similarity(summarized_sections)

            # Step 5: Check relevance
            relevance = check_relevance(similarity_scores)

            # Display similarity scores and relevance
            st.subheader("Semantic Similarity Scores:")
            for pair, score in similarity_scores.items():
                st.write(f"{pair}: {score * 5:.2f}")

            st.subheader("Relevance Check:")
            st.write(relevance)

        except Exception as e:
            st.error(f"An error occurred while processing the PDF: {e}")
