# Document Lens

I built Document Lens to make it easier to work through research papers and other long PDFs. Instead of constantly searching through a document to find a detail, I can upload it, ask a question, and see an answer alongside the relevant passages and page numbers.

The app uses retrieval-augmented generation (RAG): it searches the document for relevant text before sending that context to Gemini to generate an answer.

![Document Lens application](screenshots/demo.png)

## Features

- Upload a text-based PDF and ask questions in a chat interface.
- Retrieve relevant sections using semantic similarity rather than keyword matching alone.
- Display the passages used for each answer, including their page numbers and similarity scores.
- Keep the extracted document index in memory for follow-up questions during the session.
- Respond when the supplied passages don't contain the requested information.

## Built with

- **Python** for document processing and retrieval
- **Streamlit** for the interface
- **pypdf** for PDF text extraction
- **Gemini Embedding** (`gemini-embedding-001`) for document and query embeddings
- **NumPy** for cosine-similarity search
- **Gemini 3.5 Flash-Lite** (`gemini-3.5-flash-lite`) for generating answers

## How it works

1. **Extract:** Read selectable text from the uploaded PDF and preserve its page numbers.
2. **Chunk:** Split each page into overlapping text chunks (140 words per chunk, with 25 words of overlap).
3. **Embed:** Create 768-dimensional embeddings for the chunks using Gemini's document-retrieval embedding mode.
4. **Retrieve:** Embed the question, compare it with stored vectors using cosine similarity, and select the three most relevant chunks.
5. **Answer:** Send the question and retrieved passages to Gemini with instructions to use only that context and include page references.

For now, embeddings are stored in memory rather than in a separate vector database. This keeps the implementation straightforward, although it means the document needs to be processed again after the session resets.

## Getting started

**Requirements:** Python 3.10+ and a Gemini API key from [Google AI Studio](https://aistudio.google.com/api-keys). Model access and free-tier limits depend on your Google AI Studio project. Google AI Pro is separate from Gemini API usage.

Clone the repository:

```bash
git clone https://github.com/razaraees10/document-lens-rag.git
cd document-lens-rag
```

Create and activate a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Install the dependencies:

```bash
python -m pip install -r requirements.txt
```

Copy the example environment file:

```bash
cp .env.example .env
```

Add your own Gemini API key to `.env`:

```dotenv
GEMINI_API_KEY=your_api_key_here
```

Start the app:

```bash
python -m streamlit run app.py
```

On Windows, use `.venv\Scripts\activate` to activate the environment and `copy .env.example .env` to create the environment file.

## Trying it out

The repository includes `sample_document.pdf` for a quick test. Upload it, select **Process PDF**, and try asking:

- When does the robotics club meet?
- What is the annual membership fee?
- What sensor is used for obstacle detection?
- Who founded the robotics club?

The last question is intentionally not answered in the sample document. It helps check whether the assistant acknowledges missing information rather than inventing it.

## Tests

Run the unit tests with:

```bash
python -m unittest discover -s tests -v
```

The tests use mock API responses and don't require a Gemini API key.

## Current limitations

- Only PDFs with extractable text are supported; scanned pages need OCR.
- The app processes one PDF at a time, with a 10 MB upload limit and a maximum of 80 chunks.
- Embeddings aren't saved between sessions.
- Answers can still contain mistakes, and retrieved passages aren't a guarantee of correctness.
- Gemini API requests are subject to model availability and project quotas. Avoid submitting sensitive documents unless you've reviewed the API's data-handling terms.

I'd like to extend this with saved document collections, support for multiple papers, and better handling of tables and scanned pages.
