import os
import PIL.Image
import torch
from transformers import AutoModelForCausalLM
from janus.models import MultiModalityCausalLM, VLChatProcessor
from janus.CSD import generate # AR, JD, SJD, GSD, CSD

if __name__ == '__main__':

    # specify the path to the model
    model_path = "deepseek-ai/Janus-Pro-7B"
    vl_chat_processor: VLChatProcessor = VLChatProcessor.from_pretrained(model_path)
    tokenizer = vl_chat_processor.tokenizer

    vl_gpt: MultiModalityCausalLM = AutoModelForCausalLM.from_pretrained(
        model_path, trust_remote_code=True
    )
    vl_gpt = vl_gpt.to(torch.bfloat16).cuda().eval()

    conversation = [
        {
            "role": "<|User|>",
            "content": "A stunning princess from kabul in red, white traditional clothing, blue eyes, brown hair",
        },
        {"role": "<|Assistant|>", "content": ""},
    ]

    sft_format = vl_chat_processor.apply_sft_template_for_multi_turn_prompts(
        conversations=conversation,
        sft_format=vl_chat_processor.sft_format,
        system_prompt="",
    )
    prompt = sft_format + vl_chat_processor.image_start_tag

    visual_img, processed_tokens, elapsed_time = generate(
        vl_gpt,
        vl_chat_processor,
        prompt,
        0,  #  gpu_id
    )

    os.makedirs('generated_samples', exist_ok=True)
    for i in range(visual_img.shape[0]):
        save_path = os.path.join('generated_samples', "img_{}.jpg".format(i))
        PIL.Image.fromarray(visual_img[i]).save(save_path)

    print(processed_tokens, elapsed_time)