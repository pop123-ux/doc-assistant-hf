# doc-assistant-hf
An AI-powered Document Intelligence tool built with Hugging Face Transformers and Gradio. Extracts precise answers and generates summaries from PDF, DOCX, and TXT files using specialized local models.

## Features
* **Multi-Format Extraction**: Reads and processes `.pdf`, `.docx`, and `.txt` files automatically.
* **Document QA**: Uses a `RoBERTa` model fine-tuned on SQuAD to extract exact, context-based answers.
* **Text Summarization**: Uses a `BART` model optimized for news and long-form documents to generate clean summaries.
* **Interactive UI**: Clean and simple web interface powered by Gradio with live text-pasting and file upload options.
* **Optimized Execution**: Direct model pipeline execution bypassing deprecated task constraints.

## Tech Stack
* **Framework**: Python 3.12, Gradio
* **AI/ML**: Hugging Face Transformers, PyTorch
* **Parsers**: PyPDF / PyMuPDF, python-docx
