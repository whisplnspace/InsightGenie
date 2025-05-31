import os
from dotenv import load_dotenv
import streamlit as st
import pandas as pd
import docx
import fitz  # PyMuPDF
import pytesseract
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns
import io
import base64
import google.generativeai as genai

# Load API key
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
genai.configure(api_key=GEMINI_API_KEY)
MODEL = "gemini-2.0-flash"

def extract_text(file):
    ext = os.path.splitext(file.name)[1].lower()
    if ext == ".txt":
        return file.read().decode("utf-8"), None
    elif ext == ".docx":
        doc = docx.Document(file)
        return "\n".join([p.text for p in doc.paragraphs]), None
    elif ext == ".csv":
        df = pd.read_csv(file)
        return df.to_csv(index=False), df
    elif ext == ".xlsx":
        df = pd.read_excel(file)
        return df.to_csv(index=False), df
    elif ext == ".pdf":
        doc = fitz.open(stream=file.read(), filetype="pdf")
        return "\n".join([page.get_text() for page in doc]), None
    elif ext in [".jpg", ".jpeg", ".png"]:
        image = Image.open(file)
        return pytesseract.image_to_string(image), None
    else:
        raise ValueError("Unsupported file type")

def query_llm(prompt):
    try:
        model = genai.GenerativeModel(MODEL)
        response = model.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        return f"Gemini Error: {e}"

def get_table_download_link(df):
    """Generates a link allowing the data in a given panda dataframe to be downloaded as csv"""
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()  # some strings <-> bytes conversions necessary here
    href = f'<a href="data:file/csv;base64,{b64}" download="data.csv">📥 Download data as CSV</a>'
    return href

def get_text_download_link(text, filename="answer.txt"):
    """Creates a link to download text as a file"""
    b64 = base64.b64encode(text.encode()).decode()
    return f'<a href="data:file/txt;base64,{b64}" download="{filename}">📥 Download answer as TXT</a>'

def get_image_download_link(fig, filename="plot.png"):
    """Creates a download link for matplotlib figure"""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    buf.seek(0)
    b64 = base64.b64encode(buf.read()).decode()
    href = f'<a href="data:file/png;base64,{b64}" download="{filename}">📥 Download plot as PNG</a>'
    return href

def main():
    st.set_page_config(page_title="InsightGenie: AI Data Analyst", page_icon="🧠", layout="wide")

    # Session state initialization for Q&A history
    if "history" not in st.session_state:
        st.session_state.history = []

    st.markdown(
        """
        <style>
        .title {
            font-size: 36px;
            font-weight: 900;
            color: #31708E;
            margin-bottom: 10px;
        }
        .subtitle {
            font-size: 18px;
            color: #555555;
            margin-bottom: 30px;
        }
        .stButton>button {
            background-color: #31708E;
            color: white;
            font-weight: bold;
            padding: 10px 20px;
        }
        .stButton>button:hover {
            background-color: #25506B;
            color: #f0f0f0;
        }
        .question-input {
            font-size: 16px;
            padding: 10px;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    st.markdown("<div class='title'>🧠 InsightGenie: Your AI Data Analyst</div>", unsafe_allow_html=True)
    st.markdown("<div class='subtitle'>Upload your data file, ask your questions, and get insights along with beautiful visualizations.</div>", unsafe_allow_html=True)

    with st.sidebar:
        st.header("Upload & Ask")
        uploaded_file = st.file_uploader("Upload your file", type=["csv", "xlsx", "txt", "docx", "pdf", "png", "jpg", "jpeg"])
        question = st.text_area("What would you like to know?", placeholder="Type your question here...", height=100)
        submit = st.button("🔍 Analyze")
        st.markdown("---")
        if st.session_state.history:
            st.subheader("💬 Question & Answer History")
            for i, qa in enumerate(reversed(st.session_state.history), 1):
                st.markdown(f"**Q{i}:** {qa['question']}")
                st.markdown(f"**A{i}:** {qa['answer']}")
                st.markdown("---")
            if st.button("🗑️ Clear History"):
                st.session_state.history = []
                st.experimental_rerun()

    if uploaded_file:
        try:
            raw_text, df = extract_text(uploaded_file)
            st.success(f"✅ Successfully processed **{uploaded_file.name}**")
            if df is not None:
                st.markdown("### Preview of the data")
                with st.expander("Show raw dataframe preview (first 10 rows)"):
                    st.dataframe(df.head(10))
                st.markdown(f"**Data shape:** {df.shape[0]} rows × {df.shape[1]} columns")
                st.markdown(get_table_download_link(df), unsafe_allow_html=True)
            else:
                with st.expander("Show extracted text preview"):
                    st.text(raw_text[:1000] + "..." if len(raw_text) > 1000 else raw_text)
        except Exception as e:
            st.error(f"File processing error: {e}")
            return
    else:
        raw_text, df = None, None

    answer = None
    if submit:
        if not uploaded_file:
            st.warning("⚠️ Please upload a file first.")
            return
        if not question or question.strip() == "":
            st.warning("⚠️ Please enter a question.")
            return
        with st.spinner("🤖 Generating insight from your data..."):
            try:
                prompt = f"You are a data analyst. Analyze the following data and answer the question:\n\nData:\n{raw_text}\n\nQuestion: {question}"
                answer = query_llm(prompt)
                st.markdown("## 💡 InsightGenie Answer")
                st.markdown(f"<div style='font-size:18px; line-height:1.5;'>{answer}</div>", unsafe_allow_html=True)
                st.markdown(get_text_download_link(answer), unsafe_allow_html=True)

                # Save Q&A to history
                st.session_state.history.append({"question": question, "answer": answer})
            except Exception as e:
                st.error(f"LLM error: {e}")

    if df is not None:
        st.markdown("---")
        st.subheader("📊 Visualize Your Data")

        with st.expander("Data Summary and Filter"):
            st.write(df.describe(include='all').T)

            st.markdown("### Filter Data (Optional)")
            filter_col = st.selectbox("Filter column", options=[None] + list(df.columns), index=0)
            filtered_df = df
            if filter_col:
                unique_vals = df[filter_col].dropna().unique()
                selected_vals = st.multiselect(f"Select values for {filter_col}", options=unique_vals)
                if selected_vals:
                    filtered_df = df[df[filter_col].isin(selected_vals)]
                st.markdown(f"Filtered data shape: {filtered_df.shape[0]} rows × {filtered_df.shape[1]} columns")

            st.dataframe(filtered_df.head(10))

        col1, col2, col3, col4 = st.columns([3, 3, 3, 2])
        with col1:
            x_col = st.selectbox("Select X-Axis", filtered_df.columns, index=0)
        with col2:
            y_col = st.selectbox("Select Y-Axis", filtered_df.columns, index=1 if len(filtered_df.columns) > 1 else 0)
        with col3:
            chart_type = st.selectbox("Chart Type", ["Bar", "Line", "Scatter", "Pie"])
        with col4:
            plot_size = st.slider("Plot Size", 5, 15, 8)

        if st.button("📈 Generate Plot"):
            try:
                sns.set(style="whitegrid")
                fig, ax = plt.subplots(figsize=(plot_size, plot_size * 0.6))
                if chart_type == "Bar":
                    sns.barplot(data=filtered_df, x=x_col, y=y_col, ax=ax, palette="viridis")
                elif chart_type == "Line":
                    sns.lineplot(data=filtered_df, x=x_col, y=y_col, ax=ax, marker="o", color="darkblue")
                elif chart_type == "Scatter":
                    sns.scatterplot(data=filtered_df, x=x_col, y=y_col, ax=ax, color="darkgreen", s=80)
                elif chart_type == "Pie":
                    ax.clear()
                    grouped = filtered_df.groupby(x_col)[y_col].sum()
                    ax.pie(grouped, labels=grouped.index, autopct="%1.1f%%", startangle=140, colors=sns.color_palette("viridis", len(grouped)))
                    ax.axis("equal")
                    ax.set_title(f"Pie Chart: {y_col} by {x_col}")
                    st.pyplot(fig)
                    st.markdown(get_image_download_link(fig), unsafe_allow_html=True)
                    return
                ax.set_title(f"{chart_type} Plot: {y_col} vs {x_col}", fontsize=16, weight='bold')
                ax.set_xlabel(x_col, fontsize=12)
                ax.set_ylabel(y_col, fontsize=12)
                st.pyplot(fig)
                st.markdown(get_image_download_link(fig), unsafe_allow_html=True)
            except Exception as e:
                st.error(f"Plotting error: {e}")

if __name__ == "__main__":
    main()
