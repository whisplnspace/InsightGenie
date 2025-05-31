# 🧠 InsightGenie: Your AI-Powered Data Analyst

![InsightGenie Banner](https://huggingface.co/datasets/huggingface/brand-assets/resolve/main/hf-logo-with-title.svg)

> Upload. Ask. Analyze. Visualize. – All in one place.

[![Live Demo](https://img.shields.io/badge/🚀%20Live%20Demo-Click%20Here-brightgreen?style=for-the-badge)](https://huggingface.co/spaces/whisplnspace/InsightGenie)

---

## 🖼️ Homepage Preview

![Image](https://github.com/user-attachments/assets/bf1d5251-2577-47d0-a6c2-c30e75ab53d9)

---

## 🔍 What is InsightGenie?

**InsightGenie** is an interactive, Streamlit-powered tool that acts as your personal AI data analyst. Simply upload your datasets or documents, ask natural language questions, and get actionable insights, detailed analysis, and beautiful visualizations—all powered by Google's Gemini LLM.

---

## ✨ Features

* 📁 **Multi-format file support**: CSV, Excel, PDF, DOCX, TXT, Images (OCR).
* 🤖 **LLM-Powered Q\&A**: Ask anything about your data, get AI-generated insights.
* 📊 **Smart Visualizations**: Bar, Line, Pie, and Scatter plots using Seaborn and Matplotlib.
* 📥 **Downloadable Outputs**: Save insights as text, data as CSV, or charts as PNG.
* 🕘 **Interactive History**: View or clear your Q\&A history in-app.
* 🎨 **Modern UI**: Sleek and responsive layout with sidebar control panel.

---

## 🚀 Live Demo

Explore the app here:
👉 [**InsightGenie on Hugging Face Spaces**](https://huggingface.co/spaces/whisplnspace/InsightGenie)

---

## 🧰 Technologies Used

* [Streamlit](https://streamlit.io/) – Frontend UI framework
* [Google Gemini](https://ai.google.dev/) – Generative AI for natural language data analysis
* [Pandas](https://pandas.pydata.org/) – Data processing
* [Seaborn & Matplotlib](https://seaborn.pydata.org/) – Visualization
* \[PyMuPDF, docx, pytesseract] – File/text extraction

---

## 🗂️ File Types Supported

* `.csv`, `.xlsx`, `.txt`, `.docx`, `.pdf`, `.png`, `.jpg`, `.jpeg`

---

## 🛠️ Setup Locally

### 1. Clone the repo

```bash
git clone https://github.com/whisplnspace/InsightGenie.git
cd InsightGenie
```

### 2. Install requirements

```bash
pip install -r requirements.txt
```

### 3. Add your Gemini API key

Create a `.env` file and add:

```
GEMINI_API_KEY=your_google_generativeai_api_key
```

### 4. Run Streamlit app

```bash
streamlit run app.py
```

---

## 🤝 Contributing

Contributions, issues, and suggestions are welcome! Feel free to open a PR or issue.

---


## 🙏 Acknowledgements

* [Google Generative AI](https://ai.google.dev/)
* [Streamlit](https://streamlit.io/)
* [Hugging Face Spaces](https://huggingface.co/spaces)
