import os
import tempfile
import streamlit as st
import google.generativeai as genai
from PIL import Image
import io
import fitz  # PyMuPDF
import re
import json
import requests
from urllib.parse import urlparse
import numpy as np
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Configure page
st.set_page_config(
    page_title="GameMaster AI - Board Game Companion",
    page_icon="🎲",
    layout="wide"
)

# Initialize session state variables
if "uploaded_manual" not in st.session_state:
    st.session_state.uploaded_manual = None
if "manual_text" not in st.session_state:
    st.session_state.manual_text = ""
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "current_game" not in st.session_state:
    st.session_state.current_game = None
if "manual_images" not in st.session_state:
    st.session_state.manual_images = []
if "page_count" not in st.session_state:
    st.session_state.page_count = 0

# Gemini API setup
def initialize_genai():
    api_key = ("AIzaSyBwgtrxT5tKuHymxOQ_im5IRFOoB7Qf3FA")
    
    genai.configure(api_key=api_key)
    
    # Configure model settings
    generation_config = {
        "temperature": 0.2,
        "max_output_tokens": 4096,
        "top_p": 0.95,
        "top_k": 64,
    }
    
    safety_settings = [
        {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
        {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
        {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
        {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
    ]
    
    # Initialize models
    text_model = genai.GenerativeModel(
        model_name="gemini-1.5-pro",
        generation_config=generation_config,
        safety_settings=safety_settings
    )
    
    vision_model = genai.GenerativeModel(
        model_name="gemini-1.5-pro-vision",
        generation_config=generation_config,
        safety_settings=safety_settings
    )
    
    return text_model, vision_model

# Document processing functions
def extract_text_from_pdf(pdf_file):
    """Extract text and images from uploaded PDF."""
    with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
        tmp_file.write(pdf_file.getvalue())
        tmp_file_path = tmp_file.name
    
    doc = fitz.open(tmp_file_path)
    st.session_state.page_count = doc.page_count
    
    text_content = ""
    images = []
    
    for page_num in range(doc.page_count):
        page = doc[page_num]
        
        # Extract text
        text = page.get_text()
        text_content += f"\n\n--- PAGE {page_num + 1} ---\n\n{text}"
        
        # Extract images
        image_list = page.get_images(full=True)
        for img_idx, img_info in enumerate(image_list):
            xref = img_info[0]
            base_image = doc.extract_image(xref)
            image_bytes = base_image["image"]
            
            # Convert to PIL Image
            try:
                img = Image.open(io.BytesIO(image_bytes))
                # Only store reasonably sized images (likely diagrams, not tiny icons)
                if img.width > 100 and img.height > 100:
                    images.append({
                        "image": img,
                        "page": page_num + 1,
                        "description": f"Image from page {page_num + 1}"
                    })
            except Exception as e:
                st.error(f"Error processing image: {e}")
    
    # Clean up
    doc.close()
    os.unlink(tmp_file_path)
    
    return text_content, images

def create_vector_store(text):
    """Create a vector store from the document text."""
    # Split text into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separators=["\n\n", "\n", " ", ""]
    )
    chunks = text_splitter.split_text(text)
    
    # Add page numbers and metadata to chunks
    docs = []
    for chunk in chunks:
        # Extract page number using regex
        page_match = re.search(r"--- PAGE (\d+) ---", chunk)
        page_num = int(page_match.group(1)) if page_match else 0
        
        # Create document with metadata
        docs.append({
            "content": chunk,
            "metadata": {"page": page_num}
        })
    
    # Create embeddings and vector store
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key="AIzaSyDJNmx7PKmb92aHcrwBK7L5IKHipNzjVck")
    
    # Create FAISS vector store
    vector_store = FAISS.from_texts(
        [doc["content"] for doc in docs],
        embeddings,
        metadatas=[doc["metadata"] for doc in docs]
    )
    
    return vector_store

def load_manual_from_url(url):
    """Download and process manual from URL."""
    try:
        response = requests.get(url)
        if response.status_code != 200:
            return None, f"Failed to download from URL. Status code: {response.status_code}"
            
        # Parse URL to get filename
        parsed_url = urlparse(url)
        filename = os.path.basename(parsed_url.path)
        
        # Save to temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(filename)[1]) as tmp_file:
            tmp_file.write(response.content)
            tmp_file_path = tmp_file.name
            
        # Process based on file type
        if filename.lower().endswith('.pdf'):
            with open(tmp_file_path, 'rb') as f:
                file_bytes = io.BytesIO(f.read())
                text, images = extract_text_from_pdf(file_bytes)
            
            # Clean up
            os.unlink(tmp_file_path)
            return text, images
        else:
            os.unlink(tmp_file_path)
            return None, "Only PDF files are supported for URL imports."
            
    except Exception as e:
        return None, f"Error processing URL: {str(e)}"

# RAG and query functions
def query_manual(query, vector_store, text_model):
    """Query the manual using RAG."""
    if not vector_store:
        return "Please upload a game manual first."
    
    # Step 1: Retrieve relevant chunks
    retriever = vector_store.as_retriever(search_kwargs={"k": 5})
    retrieved_docs = retriever.get_relevant_documents(query)
    
    # Format context for the model
    context = "\n\n".join([f"[Page {doc.metadata['page']}]: {doc.page_content}" for doc in retrieved_docs])
    
    # Step 2: Generate response with context
    prompt = f"""You are GameMaster AI, an expert board game assistant. 
Answer the following question about a board game based only on the provided manual excerpts.
If the answer cannot be determined from the excerpts, say so clearly.
Always cite the specific page numbers from the manual in your answer.

MANUAL EXCERPTS:
{context}

USER QUESTION:
{query}

Your answer:"""

    response = text_model.generate_content(prompt)
    return response.text

def analyze_game_image(image, question, vision_model):
    """Analyze a game image with a specific question."""
    prompt = f"""Analyze this board game image and answer the following question:
    
Question: {question}

Provide a detailed explanation based on what you can see in the image."""

    response = vision_model.generate_content([prompt, image])
    return response.text

def simulate_game_turn(game_name, player_action, game_state, text_model):
    """Simulate a game turn using function calling."""
    prompt = f"""You are GameMaster AI, an expert in board games, particularly {game_name}.
Given the current game state and the player's action, determine the outcome.
Think step by step through the rules and explain your reasoning.

CURRENT GAME STATE:
{json.dumps(game_state, indent=2)}

PLAYER ACTION:
{player_action}

Determine:
1. If the action is valid according to the rules
2. The updated game state after this action
3. Any consequences of this action (points scored, new cards drawn, etc.)
4. Advice for the player's next move

Format your response as a step-by-step walkthrough of the rules applied."""

    response = text_model.generate_content(prompt)
    return response.text

def create_game_summary(game_name, text_model, manual_text):
    """Generate a structured summary of the game."""
    prompt = f"""Create a comprehensive but concise summary of the board game "{game_name}" based on the manual text.
Structure your response with these sections:
- Game Overview (1-2 sentences)
- Players and Setup
- Goal/Winning Conditions
- Core Mechanics (3-5 bullet points)
- Turn Structure
- Key Components
- Estimated Play Time
- Complexity Level (Easy, Medium, Hard)

Base your summary ONLY on information found in the manual text:

{manual_text[:8000]}  # Taking first part of the manual for key info

Format the response as a well-structured markdown document."""

    response = text_model.generate_content(prompt)
    return response.text

# UI Components
def sidebar():
    """Create sidebar for manual upload and settings."""
    with st.sidebar:
        st.title("🎲 GameMaster AI")
        st.markdown("Your AI board game companion")
        st.divider()
        
        # Manual input options
        st.subheader("Game Manual")
        
        # Method selector
        input_method = st.radio("Choose input method:", ["Upload PDF", "Enter URL"])
        
        if input_method == "Upload PDF":
            uploaded_file = st.file_uploader("Upload game manual (PDF):", type=["pdf"])
            if uploaded_file and uploaded_file != st.session_state.uploaded_manual:
                st.session_state.uploaded_manual = uploaded_file
                with st.spinner("Processing manual..."):
                    st.session_state.manual_text, st.session_state.manual_images = extract_text_from_pdf(uploaded_file)
                    st.session_state.vector_store = create_vector_store(st.session_state.manual_text)
                    st.session_state.current_game = uploaded_file.name.split('.')[0]
                st.success(f"Processed {st.session_state.page_count} pages with {len(st.session_state.manual_images)} images")
        
        else:  # URL option
            url = st.text_input("Enter URL to game manual (PDF):")
            if url and st.button("Load Manual"):
                with st.spinner("Downloading and processing manual..."):
                    text, result = load_manual_from_url(url)
                    if text:
                        st.session_state.manual_text = text
                        st.session_state.manual_images = result
                        st.session_state.vector_store = create_vector_store(text)
                        # Extract game name from URL
                        parsed_url = urlparse(url)
                        filename = os.path.basename(parsed_url.path)
                        st.session_state.current_game = filename.split('.')[0]
                        st.success(f"Processed manual with {len(st.session_state.manual_images)} images")
                    else:
                        st.error(result)
        
        # Game info if manual is loaded
        if st.session_state.current_game:
            st.divider()
            st.subheader(f"Current Game: {st.session_state.current_game}")
            if st.button("Generate Game Summary"):
                with st.spinner("Creating game summary..."):
                    summary = create_game_summary(
                        st.session_state.current_game, 
                        text_model,
                        st.session_state.manual_text
                    )
                    st.session_state.chat_history.append({"role": "user", "content": "Generate a summary of this game"})
                    st.session_state.chat_history.append({"role": "assistant", "content": summary})
        
        st.divider()
        # Credits
        st.caption("Powered by Google Gemini API")
        
        # Clear session
        if st.button("Reset Session"):
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.experimental_rerun()

def main_content():
    """Create main content area."""
    if not st.session_state.current_game:
        st.title("🎲 Welcome to GameMaster AI")
        st.write("Upload a board game manual to get started!")
        
        with st.expander("What can GameMaster AI do?"):
            st.markdown("""
            - 📚 **Learn game rules** - Ask questions about any aspect of the game
            - 🖼️ **Understand game components** - Upload images for analysis
            - 🎮 **Practice gameplay** - Simulate turns and get feedback
            - 📊 **Get game summaries** - Quick overview of game mechanics
            - 🧠 **Rule interpretations** - Clarify ambiguous rules
            """)
        
        
    else:
        # Chat interface
        tabs = st.tabs(["Chat with GameMaster", "Manual Images", "Game Simulation"])
        
        # Chat Tab
        with tabs[0]:
            st.subheader(f"Chat about {st.session_state.current_game}")
            
            # Display chat history
            for message in st.session_state.chat_history:
                with st.chat_message(message["role"]):
                    st.markdown(message["content"])
            
            # Chat input
            user_input = st.chat_input("Ask about game rules, setup, strategy...")
            if user_input:
                # Display user message
                st.session_state.chat_history.append({"role": "user", "content": user_input})
                with st.chat_message("user"):
                    st.markdown(user_input)
                
                # Get AI response
                with st.chat_message("assistant"):
                    with st.spinner("Thinking..."):
                        response = query_manual(user_input, st.session_state.vector_store, text_model)
                        st.markdown(response)
                
                # Add to history
                st.session_state.chat_history.append({"role": "assistant", "content": response})
        
        # Manual Images Tab
        with tabs[1]:
            st.subheader("Game Manual Images")
            
            if not st.session_state.manual_images:
                st.info("No images were found in the manual. Try uploading a different manual or providing a URL to a manual with images.")
            else:
                # Display images in grid
                image_question = st.text_input("Ask a question about an image (select an image first):")
                
                # Create grid of images
                cols = st.columns(3)
                selected_image = None
                
                for i, img_data in enumerate(st.session_state.manual_images):
                    col_idx = i % 3
                    with cols[col_idx]:
                        st.image(img_data["image"], caption=f"Page {img_data['page']}", use_column_width=True)
                        if st.button(f"Select Image {i+1}", key=f"img_{i}"):
                            selected_image = img_data["image"]
                            st.session_state.selected_image_idx = i
                
                # Analysis section
                if selected_image and image_question:
                    st.divider()
                    st.subheader("Image Analysis")
                    
                    with st.spinner("Analyzing image..."):
                        analysis = analyze_game_image(selected_image, image_question, vision_model)
                    
                    st.markdown(analysis)
                    
                    # Add to chat history
                    img_desc = f"*Asked about image from page {st.session_state.manual_images[st.session_state.selected_image_idx]['page']}: {image_question}*"
                    st.session_state.chat_history.append({"role": "user", "content": img_desc})
                    st.session_state.chat_history.append({"role": "assistant", "content": analysis})
        
        # Game Simulation Tab
        with tabs[2]:
            st.subheader("Game Simulation")
            st.markdown("Practice gameplay and rule application")
            
            # Simplified game state management
            if "game_simulation_state" not in st.session_state:
                st.session_state.game_simulation_state = {
                    "players": [],
                    "current_turn": 0,
                    "game_phase": "setup",
                    "board_state": {},
                    "player_resources": {}
                }
            
            # Setup section
            with st.expander("Game Setup", expanded=st.session_state.game_simulation_state["game_phase"] == "setup"):
                num_players = st.number_input("Number of Players:", min_value=1, max_value=8, value=2)
                player_names = []
                
                for i in range(num_players):
                    player_names.append(st.text_input(f"Player {i+1} Name:", value=f"Player {i+1}"))
                
                if st.button("Initialize Game"):
                    # Reset state
                    st.session_state.game_simulation_state = {
                        "players": player_names,
                        "current_turn": 0,
                        "game_phase": "playing",
                        "board_state": {"initialized": True},
                        "player_resources": {name: {"points": 0} for name in player_names}
                    }
                    
                    # Generate setup instructions
                    setup_query = f"How do I set up {st.session_state.current_game} for {num_players} players? Give step-by-step instructions."
                    setup_instructions = query_manual(setup_query, st.session_state.vector_store, text_model)
                    
                    # Add to chat history
                    st.session_state.chat_history.append({"role": "user", "content": setup_query})
                    st.session_state.chat_history.append({"role": "assistant", "content": setup_instructions})
                    
                    st.success("Game initialized! Check the instructions below.")
                    st.markdown(setup_instructions)
            
            # Gameplay section
            if st.session_state.game_simulation_state["game_phase"] == "playing":
                st.subheader("Current Game State")
                
                # Display current player
                current_player = st.session_state.game_simulation_state["players"][
                    st.session_state.game_simulation_state["current_turn"] % len(st.session_state.game_simulation_state["players"])
                ]
                st.markdown(f"**Current Player:** {current_player}")
                
                # Display game state
                st.json(st.session_state.game_simulation_state["player_resources"])
                
                # Player action
                player_action = st.text_area("Describe your move or action:", 
                                           placeholder="Example: I want to play the red card and take 2 resource tokens")
                
                if st.button("Submit Action"):
                    with st.spinner("Processing action..."):
                        action_result = simulate_game_turn(
                            st.session_state.current_game,
                            player_action,
                            st.session_state.game_simulation_state,
                            text_model
                        )
                    
                    st.markdown("### Action Result")
                    st.markdown(action_result)
                    
                    # Update turn
                    st.session_state.game_simulation_state["current_turn"] += 1
                    
                    # Add to chat history
                    st.session_state.chat_history.append({"role": "user", "content": f"*Game simulation: {player_action}*"})
                    st.session_state.chat_history.append({"role": "assistant", "content": action_result})
                
                # Common actions based on game
                st.markdown("### Common Actions")
                common_actions_query = f"What are 5 common actions or moves in {st.session_state.current_game}? List them briefly."
                
                if "common_actions" not in st.session_state:
                    with st.spinner("Loading common actions..."):
                        common_actions = text_model.generate_content(common_actions_query).text
                        st.session_state.common_actions = common_actions
                
                st.markdown(st.session_state.common_actions)

# Main app
def main():
    global text_model, vision_model
    
    # Initialize Gemini models
    text_model, vision_model = initialize_genai()
    
    # Set up layout
    sidebar()
    main_content()

if __name__ == "__main__":
    main()
