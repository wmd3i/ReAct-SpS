import torch
import transformers
import time
import utils
from basellama import BaseLlama

class Llama(BaseLlama):
    def __init__(self, model_id='meta-llama/Llama-2-70b-chat-hf', temperature=1e-10):
        super().__init__()
        torch.manual_seed(1339)
        bnb_config = transformers.BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type='nf4',
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16
        )
        self.model = transformers.AutoModelForCausalLM.from_pretrained(
            model_id,
            quantization_config=bnb_config,
            device_map='balanced',
        )
        self.name = 'Llama-2-70b'
        self.model.eval()
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(model_id)
        self.temperature = temperature
    
    def __call__(self, prompt: str, max_tokens: int, stop = None):
        input_ids = self.tokenizer(prompt, return_tensors="pt", add_special_tokens=True).input_ids
        prompt_len = len(prompt)
        for _ in range(max_tokens):
            outputs = self.model(input_ids)
            next_token_logits = outputs.logits[:, -1, :]
            next_token_id = self.sample_fn(next_token_logits, self.temperature)
            input_ids = torch.cat([input_ids, next_token_id.unsqueeze(-1)], dim=-1)
            if input_ids.shape[1] >= 2048:
                data = self.tokenizer.decode(input_ids[0], skip_special_tokens=True)
                raise utils.MemoryLimitExceededError("Memory limit exceeded with token shape {}".format(input_ids.shape),
                                                    extra_data=data[prompt_len:-len(stop)])
            if stop and _ > 1:
                generated_text = self.tokenizer.decode(input_ids[0], skip_special_tokens=True)[prompt_len:]
                if stop in generated_text:
                    break
        output_str = self.tokenizer.decode(input_ids[0], skip_special_tokens=True)
        return output_str[prompt_len:-len(stop)]
        
    
    def sample_model(self, model,
                 input_ids,
                 nb_tokens,
                 temperature=0.5):
        for _ in range(nb_tokens):
            outputs = model(input_ids)
            next_token_logits = outputs.logits[:, -1, :]
            next_token_id = self.sample_fn(next_token_logits, temperature)
            input_ids = torch.cat([input_ids, next_token_id.unsqueeze(-1)], dim=-1)
        return input_ids
    
    def measure(self, texts, MAX_NEW_TOKENS=64, batch_size=1):
        input_ids = self.tokenizer(texts[0], return_tensors="pt").input_ids
        input_ids = torch.stack([input_ids[0]] * batch_size).to(self.model.device)
        generated_ids = self.sample_model(self.model, input_ids, MAX_NEW_TOKENS)

        start_time = time.time()
        nb_tokens = 0 
        for text in texts[1:]: # skip the 1st text since it's already generated for warmup
            print("Completing text:", text)
            intermediate_time = time.time()
            input_ids = self.tokenizer(text, return_tensors="pt").input_ids
            input_ids = torch.stack([input_ids[0]] * batch_size).to(self.model.device)
            generated_ids = self.sample_model(self.model, input_ids, MAX_NEW_TOKENS)
            nb_tokens += generated_ids.shape[1] - input_ids.shape[1]
            print("Completion: ", self.tokenizer.decode(
                generated_ids[0], skip_special_tokens=True))
            print("Time: {:.2f}s".format(time.time() - intermediate_time))
            print("========\n")
        print("nb_tokens", nb_tokens)
        ms_per_token = (time.time() - start_time)*1000 / nb_tokens
        return generated_ids, ms_per_token
    