# YouTube-Video-Summarization-App

A YouTube Video Summarization app built using open-source LLMs and frameworks like Llama 2, Haystack, Whisper, and Streamlit. This app runs smoothly on a CPU, as the Llama 2 model is in GGUF format and is loaded through Llama.cpp.

## Description

- Utilizes Llama 2 for language processing.
- Uses Haystack for efficient search and retrieval.
- Employs Whisper for audio transcription.
- Built with Streamlit for an interactive web interface.
- Runs efficiently on CPU.

## Installation

1. **Clone the repository**:
    ```sh
    git clone https://github.com/5ud21/YouTube-Video-Summarization-App.git
    cd YouTube-Video-Summarization-App
    ```

2. **Create a virtual environment**:
    ```sh
    python -m venv venv
    ```

3. **Activate the virtual environment**:
    - On Windows:
        ```sh
        .\venv\Scripts\activate
        ```
    - On macOS/Linux:
        ```sh
        source venv/bin/activate
        ```

4. **Install the required dependencies**:
    ```sh
    pip install -r requirements.txt
    ```

## Usage

1. **Run the Streamlit app**:
    ```sh
    streamlit run app.py
    ```

2. **Open your web browser** and navigate to `http://localhost:8501` to access the app.

3. **Enter the YouTube video URL** and click on the "Summarize" button to get the summary of the video.

## Tools Used

- **Llama 2**: For language processing.
- **Haystack**: For search and retrieval.
- **Whisper**: For audio transcription.
- **Streamlit**: For building the web interface.
- **Llama.cpp**: For loading the Llama 2 model in GGUF format.

## License

This project is licensed under the MIT License. See the LICENSE file for details.

## Acknowledgements

- Thanks to the developers of Llama 2, Haystack, Whisper, and Streamlit for their amazing tools.
- Special thanks to the open-source community for their contributions and support.