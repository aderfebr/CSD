import os
import time
import PIL.Image
import torch
import numpy as np
import math
from transformers import AutoModelForCausalLM
from janus.models import MultiModalityCausalLM, VLChatProcessor

class Speculative_Sampler:
    def sample(self, draft_tokens, target_tokens, draft_prob, target_prob):
        # draft_tokens: [B, L], target_tokens: [B, L], draft_prob: [B, L, V], target_prob: [B, L, V]
        
        target_prob = torch.clamp(target_prob, min=1e-10)
        draft_prob = torch.clamp(draft_prob, min=1e-10)

        B, L = draft_tokens.shape
        
        device = draft_tokens.device
        rs = torch.rand(target_prob.shape, device=device)
        resampled_target_tokens = target_tokens.clone()
        resampled_target_probs = target_prob.clone()

        first_misaligned_token_inds = []
        
        entropy = -torch.sum(target_prob*torch.log(target_prob),dim=-1)
        tvd = 0.5*torch.sum(torch.abs(target_prob-draft_prob),dim=-1)
        accept = (target_prob/draft_prob).clamp(max=1)

        for b in range(B):
            first_misaligned_token_index = L
            
            for i in range(L):
                cls_idx = draft_tokens[b, i]
                
                r = rs[b, i, cls_idx]

                ep = entropy[b,i]*0.5 if tvd[b,i]<0.65 else 0
                accept_prob = (accept[b,i,cls_idx] + ep).clamp(max=1)

                if r < accept_prob:
                    resampled_target_tokens[b, i] = cls_idx
                    resampled_target_probs[b, i, :] = draft_prob[b, i, :]
                else:
                    first_misaligned_token_index = i
                    
                    token_target_prob = target_prob[b, i]
                    token_draft_prob = draft_prob[b, i]
                    
                    pos_delta_logits = (token_target_prob - token_draft_prob).clamp(min=0).log()
                    probs = torch.softmax(pos_delta_logits, dim=-1)
                    
                    resampled_token = torch.multinomial(probs, num_samples=1).squeeze(-1)
                    
                    resampled_target_tokens[b, i] = resampled_token
                    resampled_target_probs[b, i, :] = probs
                    break
            
            first_misaligned_token_inds.append(first_misaligned_token_index)
        
        return first_misaligned_token_inds, resampled_target_tokens, resampled_target_probs

def shift(input_tokens, input_probs, parallel_size, vocab_size, offset, window_size):
    input_tokens = input_tokens[:,offset:]
    input_probs = input_probs[:,offset:]

    n = window_size-input_tokens.shape[1]
    if n>0:
        draft_probs = torch.ones((parallel_size, vocab_size), device='cuda') / vocab_size
        draft_tokens = torch.multinomial(draft_probs, num_samples=n)
        draft_probs = draft_probs.unsqueeze(1).repeat(1, n, 1)
        input_tokens = torch.cat([input_tokens,draft_tokens],dim=1)
        input_probs = torch.cat([input_probs,draft_probs],dim=1)
    elif n<0:
        input_tokens = input_tokens[:, :window_size]
        input_probs = input_probs[:, :window_size]

    return input_tokens, input_probs

@torch.inference_mode()
def generate(
    mmgpt: MultiModalityCausalLM,
    vl_chat_processor: VLChatProcessor,
    prompt: str,
    gpu_id: int,
    window_size: int = 16,
    temperature: float = 1,
    parallel_size: int = 1,
    cfg_weight: float = 5,
    image_token_num_per_image: int = 576,
    img_size: int = 384,
    patch_size: int = 16,
):
    device = f'cuda:{gpu_id}'
    t_start = time.time()

    input_ids = vl_chat_processor.tokenizer.encode(prompt)
    input_ids = torch.LongTensor(input_ids)

    tokens = torch.zeros((parallel_size*2, len(input_ids)), dtype=torch.int).cuda()
    for i in range(parallel_size*2):
        tokens[i, :] = input_ids
        if i % 2 != 0:
            tokens[i, 1:-1] = vl_chat_processor.pad_id

    vocab_size = 16384

    draft_probs = torch.ones((parallel_size*2, vocab_size), device=device) / vocab_size
    draft_tokens = torch.multinomial(draft_probs, num_samples=window_size-1)
    tokens = torch.cat([tokens, draft_tokens], dim=1)

    inputs_embeds = mmgpt.language_model.get_input_embeddings()(tokens)
    generated_tokens = torch.zeros((parallel_size, image_token_num_per_image), dtype=torch.int).cuda()

    skip_cnt = 0
    cnt_sum = 0

    sampler = Speculative_Sampler()

    for i in range(image_token_num_per_image):

        if skip_cnt > 0:
            skip_cnt -= 1
            continue

        outputs = mmgpt.language_model.model(inputs_embeds=inputs_embeds, use_cache=True, past_key_values=outputs.past_key_values if i != 0 else None)
        hidden_states = outputs.last_hidden_state
        
        logits = mmgpt.gen_head(hidden_states[:, -window_size:, :])
        logit_cond = logits[0::2, :]
        logit_uncond = logits[1::2, :]
        
        logits = logit_uncond + cfg_weight * (logit_cond-logit_uncond)
        next_probs = torch.softmax(logits / temperature, dim=-1)

        probs = next_probs.flatten(0, 1)
        next_tokens = torch.multinomial(probs, num_samples=1).squeeze(1)
        next_tokens = next_tokens.view(next_probs.shape[:-1])

        if i>0:
            first_misaligned_input_token_inds, next_tokens, next_probs = sampler.sample(
                draft_tokens = prev_tokens,
                target_tokens = next_tokens,
                draft_prob = prev_probs,
                target_prob = next_probs
            )
            skip_cnt = min(first_misaligned_input_token_inds)
            cnt_sum += skip_cnt

        skip_cnt = min(min(skip_cnt, image_token_num_per_image-i-1),window_size-1)
        generated_tokens[:, i:i+skip_cnt+1] = next_tokens[:, :skip_cnt+1]

        if -window_size+1+skip_cnt<0:
            new_past = []
            for layer_key, layer_value in outputs.past_key_values:
                truncated_key = layer_key[:, :, :-window_size+1+skip_cnt, :]
                truncated_value = layer_value[:, :, :-window_size+1+skip_cnt, :]
                new_past.append((truncated_key, truncated_value))
            outputs.past_key_values = tuple(new_past)
            del new_past

        prev_tokens, prev_probs = shift(next_tokens, next_probs, parallel_size, vocab_size, skip_cnt, window_size)
        inputs_embeds = mmgpt.prepare_gen_img_embeds(torch.cat([prev_tokens, prev_tokens], dim=0))
        prev_tokens, prev_probs = shift(prev_tokens, prev_probs, parallel_size, vocab_size, 1, window_size)

    processed_tokens = image_token_num_per_image-cnt_sum
    elapsed_time = time.time()-t_start

    dec = mmgpt.gen_vision_model.decode_code(generated_tokens.to(dtype=torch.int), shape=[parallel_size, 8, img_size//patch_size, img_size//patch_size])
    dec = dec.to(torch.float32).cpu().numpy().transpose(0, 2, 3, 1)

    dec = np.clip((dec + 1) / 2 * 255, 0, 255)

    visual_img = np.zeros((parallel_size, img_size, img_size, 3), dtype=np.uint8)
    visual_img[:, :, :] = dec

    return visual_img, processed_tokens, elapsed_time

if __name__ == '__main__':

    # specify the path to the model
    model_path = "/home/data/wmc/Janus-Pro-7B"
    vl_chat_processor: VLChatProcessor = VLChatProcessor.from_pretrained(model_path)
    tokenizer = vl_chat_processor.tokenizer

    vl_gpt: MultiModalityCausalLM = AutoModelForCausalLM.from_pretrained(
        model_path, trust_remote_code=True
    )
    vl_gpt = vl_gpt.to(torch.bfloat16).cuda().eval()

    conversation = [
        {
            "role": "<|User|>",
            "content": "a photo of four dogs",
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
    )

    os.makedirs('generated_samples', exist_ok=True)
    for i in range(visual_img.shape[0]):
        save_path = os.path.join('generated_samples', "img_{}.jpg".format(i))
        PIL.Image.fromarray(visual_img[i]).save(save_path)

    print(processed_tokens, elapsed_time)