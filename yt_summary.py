import streamlit as st
from pytube import YouTube
from haystack.nodes import PromptNode, PromptModel
from haystack.nodes.audio import WhisperTranscriber
from haystack.pipelines import Pipeline
from model_add import LlamaCPPInvocationLayer
import time

# Set the page configuration for the Streamlit app
st.set_page_config(
    layout="wide"
)

def download_video(url):
    """
    Downloads the audio stream of a YouTube video.

    Args:
        url (str): The URL of the YouTube video.

    Returns:
        str: The file path of the downloaded audio.
    """
    yt = YouTube(url)
    video = yt.streams.filter(abr='160kbps').last()
    return video.download()

def initialize_model(full_path):
    """
    Initializes the PromptModel with the specified model path.

    Args:
        full_path (str): The file path to the model.

    Returns:
        PromptModel: An instance of the PromptModel.
    """
    return PromptModel(
        model_name_or_path=full_path,
        invocation_layer_class=LlamaCPPInvocationLayer,
        use_gpu=False,
        max_length=512
    )

def initialize_prompt_node(model):
    """
    Initializes the PromptNode with the specified model.

    Args:
        model (PromptModel): The initialized PromptModel.

    Returns:
        PromptNode: An instance of the PromptNode.
    """
    summary_prompt = "deepset/summarization"
    return PromptNode(model_name_or_path=model, default_prompt_template=summary_prompt, use_gpu=False)

def transcribe_audio(file_path, prompt_node):
    """
    Transcribes the audio file and generates a summary using the PromptNode.

    Args:
        file_path (str): The file path of the audio file.
        prompt_node (PromptNode): The initialized PromptNode.

    Returns:
        dict: The output of the transcription and summarization pipeline.
    """
    whisper = WhisperTranscriber()
    pipeline = Pipeline()
    pipeline.add_node(component=whisper, name="whisper", inputs=["File"])
    pipeline.add_node(component=prompt_node, name="prompt", inputs=["whisper"])
    output = pipeline.run(file_paths=[file_path])
    return output

def main():
    """
    The main function that runs the Streamlit app.
    """
    # Set the title and background color
    st.title("YouTube Video Summarizer 🎥")
    st.markdown('<style>h1{color: orange; text-align: center;}</style>', unsafe_allow_html=True)
    st.subheader('Built with Llama 2 🦙, Haystack, Streamlit and ❤️')
    st.markdown('<style>h3{color: pink;  text-align: center;}</style>', unsafe_allow_html=True)

    # Expander for app details
    with st.expander("About the App"):
        st.write("An app which can generate an exhaustive summary of the input YouTube video.")
        st.write("Enter a YouTube URL in the input box and click 'Submit' to start.")

    # Input box for YouTube URL
    youtube_url = st.text_input("YouTube URL")

    # Submit button
    if st.button("Submit") and youtube_url:
        start_time = time.time()  # Start the timer
        # Download video
        file_path = download_video(youtube_url)

        # Initialize model
        full_path = "llama-2-7b-32k-instruct.Q4_K_S.gguf"
        model = initialize_model(full_path)
        prompt_node = initialize_prompt_node(model)
        # Transcribe audio
        output = transcribe_audio(file_path, prompt_node)

        end_time = time.time()  # End the timer
        elapsed_time = end_time - start_time

        # Display layout with 2 columns
        col1, col2 = st.columns([1,1])

        # Column 1: Video view
        with col1:
            st.video(youtube_url)

        # Column 2: Summary View
        with col2:
            st.header("A summary of the input YouTube Video")
            st.write(output)
            st.success(output["results"][0].split("\n\n[INST]")[0])
            st.write(f"Time taken: {elapsed_time:.2f} seconds")

if __name__ == "__main__":
    main()