from pytube import YouTube
from haystack.nodes import PromptNode, PromptModel
from haystack.nodes.audio import WhisperTranscriber
from haystack.pipelines import Pipeline
from model_add import LlamaCPPInvocationLayer

# Define the prompt template for summarization
summary_prompt = "deepset/summarization"

def youtube2audio(url: str):
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

# Initialize the WhisperTranscriber
whisper = WhisperTranscriber()

# Define the model path
full_path = "llama-2-7b-32k-instruct.Q4_K_S.gguf"

# Initialize the PromptModel
model = PromptModel(model_name_or_path=full_path, invocation_layer_class=LlamaCPPInvocationLayer, use_gpu=False, max_length=512)

# Print the model details
print(model)

# Initialize the PromptNode
prompt_node = PromptNode(model_name_or_path=model, default_prompt_template=summary_prompt, use_gpu=False)

# Print the prompt node details
print("###############################################")
print(prompt_node)
print("###############################################")

# Download the audio from the YouTube video
file_path = youtube2audio("https://www.youtube.com/watch?v=h5id4erwD4s")

# Initialize the Pipeline
pipeline = Pipeline()
pipeline.add_node(component=whisper, name="whisper", inputs=["File"])
pipeline.add_node(component=prompt_node, name="prompt", inputs=["whisper"])

# Run the pipeline
output = pipeline.run(file_paths=[file_path])

# Print the results
print(output["results"])
print(output["results"][0].split("\n\n[INST]")[0])