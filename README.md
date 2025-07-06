# 🪔 Krishna Divine Wisdom – A Spiritual AI Chatbot

**Krishna Divine Wisdom** is a poetic and spiritual conversational AI powered by large language model embeddings, inspired by ancient Hindu scriptures like the *Bhagavad Gita*, *Upanishads*, and *Bhagavata Purana*. It provides calming, insightful, and emotionally intelligent responses to life’s most profound questions.

> 🧘 “Not merely an AI — a modern-day oracle of timeless Vedic truths.”

---

## ✨ Features

- 🔍 **Semantic Search** using FAISS + HuggingFace embeddings
- 🧠 **LLM-Powered Reasoning** with Krishna-style wisdom generator
- 📜 Integrated Scriptural Sources:
  - Bhagavad Gita (Vyasa + Vedanta + Alpaca datasets)
  - Upanishads (Isha, Katha)
  - Bhagavata Purana
- 🎨 Beautiful Streamlit UI with poetic visuals and divine theming
- 🧘 Personalized responses based on user emotional context (worry, grief, love, etc.)
- 🎤 Optional voice input/output modules (planned)

---

## 🚀 Live Demo

> 🧪 Hosted on **Streamlit Community Cloud**  
> 🔗 [Click here to try the app](https://your-username-geetagpt-ibmgenai-project.streamlit.app/) ← *(Replace with your actual Streamlit link after deployment)*

---

## 🧰 Tech Stack

| Component        | Tool                                  |
|------------------|---------------------------------------|
| App Framework    | [Streamlit](https://streamlit.io)     |
| LLM Interface    | [LangChain](https://www.langchain.com)|
| Embeddings       | Sentence Transformers (MiniLM)        |
| Vector Search    | [FAISS](https://github.com/facebookresearch/faiss) |
| Datasets         | Vyasa Gita, Alpaca Gita, Vedanta QA   |
| Hosting          | Streamlit Cloud + GitHub              |

---

## 📂 Folder Structure

```
krishna-divine-wisdom/
├── app/
│   ├── krishna_chatgpt.py       # Main app
│   └── model.py                 # Placeholder LLM logic
├── data/
│   ├── scriptures_db.csv
│   └── prompts_examples.json
├── models/
│   └── krishna-llm-embeddings.bin (simulated)
├── .streamlit/
│   └── config.toml
├── requirements.txt
├── README.md
└── LICENSE
```

---

## 🔧 Setup & Installation

You can run this app locally on your system in just a few minutes.

### ✅ 1. Clone the Repository

```bash
git clone https://github.com/Sattwik999/GeetaGPT_IBMGENAI_Project.git
cd GeetaGPT_IBMGENAI_Project
```

### ✅ 2. Set Up a Python Environment (Recommended)

```bash
# For Windows
python -m venv venv
venv\Scripts\activate

# For Mac/Linux
python3 -m venv venv
source venv/bin/activate
```

### ✅ 3. Install All Dependencies

```bash
pip install -r requirements.txt
```

> If using Google Colab or Streamlit Cloud, this will be done automatically.

### ✅ 4. Run the App Locally

```bash
streamlit run app/krishna_chatgpt.py
```

Your app will open in a new browser window at:  
📍 `http://localhost:8501`

---

## ☁️ Deployment on Streamlit Cloud

1. Push your project to a public GitHub repo
2. Visit [https://streamlit.io/cloud](https://streamlit.io/cloud)
3. Click **“New app”**
4. Choose:
   - **Repository:** `Sattwik999/GeetaGPT_IBMGENAI_Project`
   - **Branch:** `main`
   - **App file:** `app/krishna_chatgpt.py`
5. Click **Deploy**

You’ll get a public URL like:
```
https://your-username-geetagpt-ibmgenai-project.streamlit.app
```

---

## 📚 Datasets Used

- [sweatSmile/Bhagavad-Gita-Vyasa-Edwin-Arnold](https://huggingface.co/datasets/sweatSmile/Bhagavad-Gita-Vyasa-Edwin-Arnold)
- [SatyaSanatan/shrimad-bhagavad-gita-dataset-alpaca](https://huggingface.co/datasets/SatyaSanatan/shrimad-bhagavad-gita-dataset-alpaca)
- [VedantaHub/BhagavadGita-QA](https://github.com/VedantaHub/Datasets)

---

## 🙏 Credits

- ✍️ *Concept & Development:* **Sattwik Sarkar** (VIT Chennai, BTech CSE)
- 📚 *Scriptural Sources:* Bhagavad Gita, Upanishads, Bhagavata Purana
- 🧠 *Tools Used:* LangChain, HuggingFace, FAISS, Streamlit
- 🎨 *UI Inspiration:* Sacred geometry, Vedic patterns, minimalist design

---

## 📜 License

This project is licensed under the MIT License — see the [LICENSE](./LICENSE) file for full details.

---

## ✨ May your path be guided by wisdom ✨

> *“The soul is neither born, and nor does it die...” – Bhagavad Gita 2.20*


