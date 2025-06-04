from huggingface_hub import hf_hub_download

# Download the model
model_path = hf_hub_download(
    repo_id="TheBloke/Llama-2-7b-Chat-GGUF",
    filename="llama-2-7b-chat.Q4_K_M.gguf",
    repo_type="model"
)

print(f"Model downloaded to: {model_path}") 