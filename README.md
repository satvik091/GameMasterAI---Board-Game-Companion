# GameMaster AI - Board Game Companion

GameMaster AI is an intelligent assistant that helps users learn, explore, and master any board game using the power of Google's Gemini AI models and Streamlit.

## Features

- **Upload Board Game Manuals**: Process PDFs or provide URLs to game manuals
- **Rule Explanations**: Ask questions about game rules and get accurate answers
- **Image Understanding**: Analyze game components, board setups, and cards
- **Game Simulation**: Practice gameplay with virtual turns and feedback
- **Structured Summaries**: Get concise overviews of games and their mechanics
- **Vector Search**: All answers grounded in the actual game manual content

## Installation

1. Clone this repository:
   ```
   git clone https://github.com/yourusername/gamemasterai.git
   cd gamemasterai
   ```

2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Get a Gemini API key from [Google AI Studio](https://ai.google.dev/)

4. Create a `.streamlit/secrets.toml` file with your API key:
   ```
   GEMINI_API_KEY = "your-api-key-here"
   ```

5. Run the application:
   ```
   streamlit run app.py
   ```

## Usage

1. Upload a board game manual (PDF) or provide a URL to one
2. Ask questions about the rules, setup, or strategy
3. Upload photos of your game setup for analysis
4. Simulate turns and gameplay
5. Get structured summaries and explanations

## GenAI Capabilities Used

- **Document Understanding**: Parse and understand uploaded board game manuals
- **Retrieval-Augmented Generation (RAG)**: Retrieve and synthesize answers based on manual content
- **Few-Shot Prompting**: Answer rule-related questions with examples from prompt libraries
- **Image Understanding**: Interpret game setup diagrams or uploaded photos
- **Function Calling**: Simulate turns and apply rules dynamically via structured function calls
- **Agents**: The in-app "GameMaster Agent" walks users through gameplay
- **Structured Output**: Provide rule breakdowns and game summaries in structured formats
- **Embeddings + Vector Search**: Store chunked rulebooks for fast semantic retrieval
- **Grounding**: All answers cite specific rule sections or document pages
- **Long Context Window**: Enables understanding of large, multi-page manuals
- **Context Caching**: Caches prior rule interpretations for improved interactions

## Requirements

- Python 3.8+
- Internet connection (for API access)
- Gemini API key

## License

MIT License

## Contributing

Contributions welcome! Please feel free to submit a Pull Request.
