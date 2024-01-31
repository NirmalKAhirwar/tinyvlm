import torch
import re
import gradio as gr
from moondream import Moondream, detect_device
from threading import Thread
from transformers import TextIteratorStreamer, CodeGenTokenizerFast as Tokenizer

device, dtype = detect_device()
if device != torch.device("cpu"):
    print("Using device:", device)
    print("If you run into issues, pass the `--cpu` flag to this script.")
    print()

model_id = "vikhyatk/moondream1"
tokenizer = Tokenizer.from_pretrained(model_id)
moondream = Moondream.from_pretrained(model_id).to(device=device, dtype=dtype)
moondream.eval()


def answer_question(img, prompt):
   
    image_embeds = moondream.encode_image(img)
    streamer = TextIteratorStreamer(tokenizer, skip_special_tokens=True)
    thread = Thread(
        target=moondream.answer_question,
        kwargs={
            "image_embeds": image_embeds,
            "question": prompt,
            "tokenizer": tokenizer,
            "streamer": streamer,
        },
    )
    thread.start()

    buffer = ""
    for new_text in streamer:
        clean_text = re.sub("<$|END$", "", new_text)
        buffer += clean_text
        yield buffer.strip("<END")
        # print("buffer: ", buffer)


with gr.Blocks() as demo:
    gr.Markdown(
        """
        # 🌔 TinyVLM -- Chat with Images!
        ### A tiny vision language model. [GitHub](https://github.com/NirmalKAhirwar/tinyvlm.git)
        """
    )
    with gr.Row():
        prompt = gr.Textbox(label="Input Prompt", placeholder="Type here...", scale=4)
        submit = gr.Button("Submit")
        
    with gr.Row():
        img = gr.Image(type="pil", label="Upload an Image")
        output = gr.TextArea(label="Response", info="Please wait for a few seconds..")
        
        # audio_btn = gr.Audio(label="Listen", type="file", source="empty.mp3")
        
        
    submit.click(answer_question, [img, prompt], output)
    print("Outputs: ", answer_question)
    prompt.submit(answer_question, [img, prompt], output)
    print("Outputs: ", answer_question)

demo.queue().launch(debug=True, share=True)
