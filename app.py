import streamlit as st
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

st.set_page_config(page_title="Find Similar Things", page_icon="🧒", layout="centered")

st.title("🧒 Find Similar Things!")
st.markdown("### Type a word and find similar things!")
st.divider()

word_list = [
    "cat", "dog", "lion", "tiger", "elephant", "bird", "fish",
    "apple", "orange", "banana", "grape", "mango", "strawberry",
    "car", "bus", "train", "plane", "boat", "bicycle",
    "red", "blue", "green", "yellow", "purple", "pink",
    "happy", "sad", "excited", "angry", "surprised",
    "run", "jump", "swim", "fly", "walk", "dance",
    "school", "teacher", "student", "book", "pencil",
    "house", "garden", "kitchen", "bedroom", "playground"
]

user_word = st.text_input("Enter a word:", placeholder="Example: cat, apple, car...")

col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    search_button = st.button("🔍 Find Similar Things!")

if search_button:
    if user_word.strip():
        with st.spinner("🔍 Searching..."):
            all_words = [user_word] + word_list
            vectorizer = TfidfVectorizer(analyzer='char_wb', ngram_range=(2,3))
            vectors = vectorizer.fit_transform(all_words)
            similarities = cosine_similarity(vectors[0:1], vectors[1:]).flatten()
            top_indices = similarities.argsort()[-5:][::-1]
            
            st.success("✅ Similar Things Found!")
            st.markdown(f"### Things similar to **'{user_word}'**:")
            
            for idx in top_indices:
                similarity_percent = int(similarities[idx] * 100)
                st.markdown(f"🎯 **{word_list[idx]}** - {similarity_percent}% similar")
    else:
        st.warning("⚠️ Please enter a word first!")

with st.sidebar:
    st.header("ℹ️ About")
    st.info("""
    **Find Similar Things App**
    
    Helps kids find similar words!
    
    **How to use:**
    1. Type any word
    2. Click the button
    3. See similar things!
    """)
    st.markdown("**Try:** cat | apple | car | happy")

st.divider()
st.caption("Made with ❤️ for Kids | © 2025")