import os
import streamlit as st
import requests
import pandas as pd
import json
import time  # Missing import added here
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings

from langchain.vectorstores import FAISS
import numpy as np
from io import StringIO
import random


# ========================
# 🕉️ SCRIPTURE DATASETS LOADER
# ========================
def load_all_scriptures():
    """Load all scripture datasets from various sources"""
    scriptures = {}

    # Core scriptures (expanded)
    scriptures.update({
        "Bhagavad Gita": """
        CHAPTER 2: Sankhya Yoga
        The Blessed Lord said: You grieve for those who should not be grieved for; yet you speak words of wisdom. 
        The wise grieve neither for the living nor for the dead. Never was there a time when I did not exist, 
        nor you, nor all these kings; nor in the future shall any of us cease to be. Just as the embodied soul 
        continuously passes from childhood to youth to old age, similarly, at the time of death, the soul passes 
        into another body. The wise are not deluded by this change. Those who are seers of the truth have concluded 
        that the impermanent has no reality and the eternal never ceases to be. The soul is unborn, eternal, 
        everlasting, primeval; it is not slain when the body is slain.

        CHAPTER 3: Karma Yoga
        One who controls the senses by the mind and engages the active senses in works of devotion without attachment 
        is superior. Perform your prescribed duties, for action is better than inaction. Even the maintenance of your 
        body would not be possible without action. Work done as a sacrifice for the Supreme Lord has to be performed; 
        otherwise work causes bondage in this material world. Therefore, O Arjuna, perform your prescribed duties for 
        His satisfaction, and in that way you will always remain free from bondage.
        """,

        "Upanishads": """
        Katha Upanishad:
        The soul is born and unfolds in a body, with dreams and desires and the food of life. And then it is reborn in new bodies, 
        in accordance with its former works. The soul is immortal; it is never born and never dies. It is in the changeless, 
        eternal, and indestructible. Weapons cannot cut it, fire cannot burn it, water cannot wet it, wind cannot dry it. 
        The soul is beyond all power of these elements.

        Isha Upanishad:
        The entire universe is pervaded by the Supreme Being, who is both within and without, unchanging and without form. 
        Therefore, find your enjoyment in renunciation; do not covet what belongs to others. Perform your duties in this world 
        with detachment, and you will avoid bondage. The face of truth remains hidden behind a circle of gold. Unveil it, O Lord of Light, 
        so that I who love the truth may see it.
        """,

        "Bhagavata Purana": """
        Book 10: The Supreme Personality of Godhead
        The Supreme Lord said: My dear devotees, those who fix their minds on Me and engage in My loving service, 
        giving up all material desires, are very dear to Me. One who is thus transcendentally situated at once realizes the Supreme Brahman. 
        He never laments nor desires to have anything; he is equally disposed to every living entity. In that state he attains pure devotional service unto Me.

        Book 11: General History
        The Supreme Lord said: The three modes of material nature—goodness, passion, and ignorance—bind the eternal soul to the perishable body. 
        O Uddhava, one who has completely surrendered unto Me can easily overcome these three modes and become situated in pure spiritual existence. 
        Such a devotee of Mine, fixed in transcendental knowledge, is not subject to rebirth even when he gives up his present body.
        """
    })

    # Additional datasets
    scriptures.update(load_vyasa_dataset())
    scriptures.update(load_alpaca_dataset())
    scriptures.update(load_vedanta_dataset())

    return scriptures


def load_vyasa_dataset():
    """Load Bhagavad-Gita-Vyasa-Edwin-Arnold dataset"""
    try:
        url = "https://huggingface.co/datasets/sweatSmile/Bhagavad-Gita-Vyasa-Edwin-Arnold/resolve/main/bhagavad_gita_qa.csv"
        response = requests.get(url)
        csv_data = StringIO(response.text)
        df = pd.read_csv(csv_data)

        vyasa_qa = []
        for _, row in df.iterrows():
            vyasa_qa.append(f"Q: {row['question']}\nA: {row['answer']}")

        return {"Bhagavad Gita Vyasa": "\n\n".join(vyasa_qa[:300])}
    except:
        return {}


def load_alpaca_dataset():
    """Load shrimad-bhagavad-gita-dataset-alpaca"""
    try:
        url = "https://huggingface.co/datasets/SatyaSanatan/shrimad-bhagavad-gita-dataset-alpaca/resolve/main/data.json"
        response = requests.get(url)
        data = response.json()

        alpaca_qa = []
        for item in data[:300]:
            alpaca_qa.append(f"Q: {item['instruction']}\nA: {item['output']}")

        return {"Bhagavad Gita Alpaca": "\n\n".join(alpaca_qa)}
    except:
        return {}


def load_vedanta_dataset():
    """Load Vedanta GitHub dataset"""
    try:
        url = "https://raw.githubusercontent.com/VedantaHub/Datasets/main/Bhagwad_Gita_Verses_English_Questions.csv"
        response = requests.get(url)
        csv_data = StringIO(response.text)
        df = pd.read_csv(csv_data)

        vedanta_qa = []
        for _, row in df.iterrows():
            vedanta_qa.append(f"Verse: {row['Verse']}\nQ: {row['Question']}\nA: {row['Answer']}")

        return {"Bhagavad Gita Vedanta": "\n\n".join(vedanta_qa[:300])}
    except:
        return {}


# ========================
# 📚 VECTOR DATABASE SETUP
# ========================
def create_vector_db(scriptures):
    """Create FAISS vector database from all scriptures"""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separators=["\n\n", "\n", "CHAPTER", "BOOK", "Q:", "Verse:"]
    )

    all_chunks = []
    for source, text in scriptures.items():
        chunks = text_splitter.split_text(text)
        for chunk in chunks:
            all_chunks.append(f"[{source}]\n{chunk}")

    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    return FAISS.from_texts(all_chunks, embeddings)


# ========================
# 🪔 ENRICHED KRISHNA RESPONSES
# ========================
def generate_enriched_response(user_query, vector_db):
    """Generate deep, personalized Krishna responses with scriptural references"""
    # Get relevant scriptures
    relevant_docs = vector_db.similarity_search(user_query, k=3)
    scriptures = [doc.page_content for doc in relevant_docs]

    # Response templates
    gesture = random.choice([
        "smiles compassionately",
        "gazes with infinite wisdom",
        "places a hand gently on your shoulder",
        "chuckles softly",
        "nods knowingly",
        "closes eyes in deep contemplation"
    ])

    opening = random.choice([
        "Dear seeker of truth,",
        "Beloved child of eternity,",
        "O noble soul,",
        "Dear one,",
        "My precious devotee,"
    ])

    # Core response logic
    query_lower = user_query.lower()

    if any(word in query_lower for word in ["anxiety", "worry", "stress", "nervous"]):
        response = f"""
        *{gesture}*
        {opening} I see the storms of worry swirling within your heart. Remember when Arjuna stood trembling on Kurukshetra? 
        His bow slipped from paralyzed hands, his vision clouded by doubt - much like yours in this moment. 
        Yet I whispered to him then what I whisper to you now: 'The soul is eternal, untouched by these temporal winds.'

        Consider the mighty banyan tree - when monsoon winds rage, its branches may tremble, leaves may scatter, 
        yet its roots remain anchored deep in the earth's silent wisdom. Your true Self is such a tree, beloved. 
        These anxieties are but passing seasons in the eternal garden of your being.

        **Practical wisdom**: Each dawn, before the world stirs, sit beneath your own inner kalpavriksha. 
        Breathe in stillness for seven breaths, exhale turmoil for seven more. Then ask: 
        'What small act of service can I perform today to shift from worry to worship?' 
        For in selfless action, the mind finds its anchor.

        The light within you has weathered countless storms. Trust its constancy. 
        """
    elif any(word in query_lower for word in ["purpose", "dharma", "calling", "mission"]):
        response = f"""
        *{gesture}*
        {opening} The eternal question that stirs in every heart! Do you recall young Dhruva, who sought greatness? 
        He scaled mountains of ambition only to find emptiness at every peak - until he stilled his restless seeking 
        and discovered the North Star within his own soul.

        Your purpose, dear seeker, isn't a destination to reach but a quality of being to embody. 
        The potter finds purpose not in creating perfect pots but in the sacred dance of hands upon clay. 
        The river finds purpose not in reaching the ocean but in the singing journey between banks.

        **Practical wisdom**: For seven days, keep a 'light journal'. Note moments when you felt: 
        'Here, I am home.' Patterns will emerge like constellations in the night sky. 
        Follow these stars, and you'll find your unique song in creation's symphony.

        Remember, even I took joy in herding cows in Vrindavan. Divine purpose often wears humble garments. 
        """
    elif any(word in query_lower for word in ["peace", "calm", "serenity", "tranquility"]):
        response = f"""
        *{gesture}*
        {opening} Peace is not the absence of storms but the depth of the ocean beneath them. 
        Observe the water - when winds rage, waves dance in frenzy upon the surface, 
        yet twenty feet below exists a realm of undisturbed silence. Such is your true nature.

        Remember when I calmed the raging sea for my devotees? That same power resides within your breath. 
        The secret lies not in controlling external waves but in diving deep to your inner stillness.

        **Practical wisdom**: Create a 'sanctuary of silence' each day. It need not be grand - 
        a corner with a single candle suffices. There, practice the three Rs:
        1. Release thoughts like leaves upon a stream
        2. Rest in the awareness beneath thinking
        3. Remember 'I am that stillness'

        When agitation visits, whisper to your heart: 'This too shall pass, but the witness remains.' 
        Your calm isn't fragile; it's the eternal bedrock of creation. 
        """
    elif any(word in query_lower for word in ["sad", "grief", "depressed", "sorrow"]):
        response = f"""
        *{gesture}*
        {opening} Your sorrow is sacred, beloved. Do you know why the lotus chooses muddy waters to bloom? 
        Because it understands darkness as the womb of light. Your tears water seeds of wisdom 
        that will blossom in seasons you cannot yet imagine.

        Remember Radha's viraha - her divine longing painted the skies with hues of separation, 
        yet each teardrop became a star in love's eternal constellation. Your pain too is transforming 
        into something luminous beyond your current sight.

        **Practical wisdom**: When grief's tide surges, become the compassionate witness. 
        Light a ghee lamp and speak to your sorrow: 'I see you, I honor you, I release you.' 
        Then write one letter of gratitude to someone who once brought you joy.

        The moon wanes but never disappears. Your light, though veiled, remains whole. 
        This darkness is but the universe holding you in its sacred womb. 
        """
    elif any(word in query_lower for word in ["love", "relationship", "compassion", "connection"]):
        response = f"""
        *{gesture}*
        {opening} Love is the fundamental rhythm of creation. Not the fragile love that says 'I need you' 
        but the divine love that whispers 'I am you.' See how the Yamuna embraces every stone, root, and bank - 
        without possession, without condition.

        Your heart's longing mirrors the gopis' divine madness - that exquisite ache to merge with the Beloved. 
        But understand: the love you seek outside already dwells within your own breast as your eternal essence.

        **Practical wisdom**: Practice seeing the divine in the ordinary. When you take your morning chai, 
        offer it first to the divine presence within your guest. When frustration arises with a loved one, 
        whisper: 'The Krishna in me greets the Krishna in you.'

        True love isn't found; it's recognized. It's the ocean awakening to its own wetness. 
        """
    else:
        response = f"""
        *{gesture}*
        {opening} Your question touches the eternal mystery. The answers you seek are like birds hidden in foliage - 
        they reveal themselves not through frantic searching but through patient stillness.

        Remember Arjuna's dilemma on Kurukshetra? His confusion birthed the timeless wisdom of the Gita. 
        Your uncertainty now is the sacred ground where new understanding will blossom.

        **Practical wisdom**: For three days, observe nature's wisdom. Watch how the river navigates obstacles, 
        how the tree accepts seasons, how the stars keep faith with darkness. Then ask: 
        'What would love do in this situation?' and wait for the answer that brings peace.

        The divine dance continues in every atom of creation, and you are its cherished partner. 
        Trust the unfolding. 
        """

    # Add scriptural references
    if scriptures:
        response += "\n\n**Relevant Scriptures**:\n"
        for i, scripture in enumerate(scriptures, 1):
            response += f"\n{i}. {scripture[:350]}{'...' if len(scripture) > 350 else ''}"

    return response


# ========================
# 🖥️ STREAMLIT UI - DIVINE INTERFACE
# ========================
def main():
    # Configure page
    st.set_page_config(
        page_title="Krishna Divine Wisdom",
        page_icon="🪔",
        layout="centered",
        initial_sidebar_state="expanded",
        menu_items={
            'About': "### Krishna Divine Wisdom\nExperience the eternal guidance of Lord Krishna"
        }
    )

    # Custom CSS for divine interface
    st.markdown("""
    <style>
        :root {
            --primary: #1a5276;
            --secondary: #2e86c1;
            --accent: #d35400;
            --gold: #f1c40f;
            --background: #fef9e7;
            --text: #2c3e50;
            --light-text: #7f8c8d;
        }

        body {
            background: var(--background);
            background-image: radial-gradient(#d4e6f1 1px, transparent 1px);
            background-size: 20px 20px;
            color: var(--text);
            font-family: 'Palatino Linotype', 'Book Antiqua', serif;
        }

        .stTextInput input {
            font-size: 18px;
            padding: 15px;
            background: rgba(255, 255, 255, 0.9);
            border: 1px solid #d4e6f1;
            border-radius: 12px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.05);
        }

        .stButton>button {
            background: linear-gradient(to right, var(--primary), var(--secondary));
            color: white;
            border: none;
            padding: 14px 28px;
            font-size: 18px;
            border-radius: 12px;
            transition: all 0.3s;
            font-weight: 600;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        }

        .stButton>button:hover {
            transform: translateY(-2px);
            box-shadow: 0 6px 12px rgba(0,0,0,0.15);
        }

        .response-container {
            background: linear-gradient(135deg, #ffffff 0%, #f8f9fa 100%);
            border-radius: 20px;
            padding: 30px;
            margin-top: 25px;
            box-shadow: 0 10px 30px rgba(0,0,0,0.08);
            border-left: 5px solid var(--gold);
            position: relative;
            overflow: hidden;
        }

        .response-container::before {
            content: "";
            position: absolute;
            top: 0;
            left: 0;
            right: 0;
            height: 5px;
            background: linear-gradient(to right, var(--primary), var(--secondary));
        }

        .gesture {
            color: var(--accent);
            font-style: italic;
            margin-bottom: 15px;
            font-size: 18px;
            border-left: 3px solid var(--gold);
            padding-left: 15px;
        }

        .response-text {
            font-size: 19px;
            line-height: 1.8;
            color: var(--text);
            text-align: justify;
            font-family: 'Georgia', serif;
        }

        .scripture {
            background: #eaf7ff;
            border-radius: 10px;
            padding: 15px;
            margin: 10px 0;
            border-left: 3px solid var(--secondary);
            font-size: 16px;
            line-height: 1.6;
        }

        .source-badge {
            display: inline-block;
            background: #eaf2f8;
            color: var(--primary);
            border-radius: 12px;
            padding: 5px 12px;
            font-size: 0.85rem;
            margin: 5px 5px 5px 0;
            border: 1px solid #d4e6f1;
        }

        .header-section {
            background: linear-gradient(135deg, var(--primary) 0%, var(--secondary) 100%);
            padding: 30px 20px;
            border-radius: 0 0 20px 20px;
            color: white;
            margin-bottom: 30px;
            box-shadow: 0 4px 12px rgba(0,0,0,0.1);
        }

        .footer {
            font-size: 0.9rem;
            color: var(--light-text);
            margin-top: 3rem;
            text-align: center;
            padding: 20px;
            border-top: 1px solid #ecf0f1;
        }

        .sample-question {
            background: #eaf7ff;
            border-radius: 12px;
            padding: 12px 15px;
            margin: 8px 0;
            transition: all 0.3s;
            border: 1px solid #d4e6f1;
            cursor: pointer;
        }

        .sample-question:hover {
            background: #d4e6f1;
            transform: translateX(5px);
        }

        .sidebar .sidebar-content {
            background: white;
            padding: 20px;
            border-radius: 0 20px 20px 0;
            box-shadow: 5px 0 15px rgba(0,0,0,0.05);
        }

        .divine-title {
            font-family: 'Times New Roman', serif;
            font-weight: bold;
            text-shadow: 1px 1px 3px rgba(0,0,0,0.2);
            letter-spacing: 1px;
        }

        .practical-wisdom {
            background: #fff8e1;
            border-radius: 10px;
            padding: 15px;
            margin: 20px 0;
            border-left: 3px solid var(--gold);
            font-style: italic;
        }
    </style>
    """, unsafe_allow_html=True)

    # Divine Header
    st.markdown("""
    <div class="header-section">
        <div style="display:flex; align-items:center; justify-content:center; gap:20px;">
            <div style="font-size:48px;">🪔</div>
            <div>
                <h1 class="divine-title" style="font-size:42px; margin-bottom:5px;">Krishna Divine Wisdom</h1>
                <h3 style="font-weight:300; margin-top:0;">Eternal Guidance for the Modern Seeker</h3>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Initialize session state
    if "vector_db" not in st.session_state:
        with st.spinner("🌿 Loading divine knowledge from sacred scriptures..."):
            try:
                scriptures = load_all_scriptures()
                vector_db = create_vector_db(scriptures)
                if vector_db:
                    st.session_state.vector_db = vector_db
                    st.session_state.loaded_sources = list(scriptures.keys())
                    st.success("Divine wisdom has been awakened!")
                else:
                    st.error("Could not establish connection to divine knowledge")
            except Exception as e:
                st.error(f"Divine connection issue: {str(e)}")

    # Sample questions with elegant design
    st.subheader("Seek Divine Guidance")
    st.markdown("""
    <div style="background:#f8f9fa; padding:20px; border-radius:15px; margin-top:10px;">
        <h4 style="color:var(--primary); margin-bottom:15px;">Sample Questions</h4>
        <div class="sample-question" onclick="setQuestion('How to find peace in difficult times?')">How to find peace in difficult times?</div>
        <div class="sample-question" onclick="setQuestion('What is my true purpose?')">What is my true purpose?</div>
        <div class="sample-question" onclick="setQuestion('How to deal with overwhelming emotions?')">How to deal with overwhelming emotions?</div>
        <div class="sample-question" onclick="setQuestion('How to cultivate true love?')">How to cultivate true love?</div>
        <div class="sample-question" onclick="setQuestion('How to make difficult decisions?')">How to make difficult decisions?</div>
        <div class="sample-question" onclick="setQuestion('What is the nature of the soul?')">What is the nature of the soul?</div>
        <div class="sample-question" onclick="setQuestion('How to overcome fear?')">How to overcome fear?</div>
        <div class="sample-question" onclick="setQuestion('How to find meaning in suffering?')">How to find meaning in suffering?</div>
    </div>

    <script>
    function setQuestion(question) {
        window.parent.document.querySelector('input[aria-label="Ask your spiritual question:"]').value = question;
    }
    </script>
    """, unsafe_allow_html=True)

    # User input
    question = st.text_input(
        "Ask your spiritual question:",
        placeholder="What troubles your heart? What wisdom do you seek?",
        key="user_question",
        label_visibility="collapsed"
    )

    # Process question
    if st.button("Receive Divine Guidance", use_container_width=True, type="primary") and question:
        with st.spinner("🕉️ Krishna is contemplating your question with divine wisdom..."):
            start_time = time.time()

            try:
                # Generate enriched response with scriptural references
                vector_db = st.session_state.vector_db
                response = generate_enriched_response(question, vector_db)

                # Extract gesture and message
                parts = response.split('*')
                gesture = parts[1].strip() if len(parts) > 1 else "smiles gently"
                message = parts[2].strip() if len(parts) > 2 else response

                # Display response with premium styling
                st.markdown(f"""
                <div class="response-container">
                    <div class="gesture">*{gesture}*</div>
                    <div class="response-text">{message}</div>
                </div>
                """, unsafe_allow_html=True)

                # Show response time
                st.caption(f"⏱️ Divine response in {time.time() - start_time:.1f} seconds")

            except Exception as e:
                st.error(f"Divine contemplation was interrupted: {str(e)}")
                st.info("Please try again or rephrase your question")

    # Footer
    st.markdown("""
    <div class="footer">
        <p>🕉️ Embodied Wisdom: Bhagavad Gita • Upanishads • Bhagavata Purana</p>
        <p>📚 Integrated Datasets: Vyasa • Alpaca • Vedanta</p>
        <p>🙏 May these divine words illuminate your path</p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()