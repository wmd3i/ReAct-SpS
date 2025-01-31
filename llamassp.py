import torch
import transformers
import time
import utils
from basellama import BaseLlama

class LlamaSSP(BaseLlama):
    def __init__(self, draft_id='meta-llama/Llama-2-7b-chat-hf', target_id='meta-llama/Llama-2-70b-chat-hf', temperature=1e-10):
        super().__init__()
        torch.manual_seed(1339)
        bnb_config = transformers.BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type='nf4',
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16
        )
        self.draft_model = transformers.AutoModelForCausalLM.from_pretrained(
            draft_id,
            quantization_config=bnb_config,
            device_map='balanced',
        )
        self.target_model = transformers.AutoModelForCausalLM.from_pretrained(
            target_id,
            quantization_config=bnb_config,
            device_map='balanced',
        )
        self.name = "SpS"
        self.draft_model.eval()
        self.target_model.eval()
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(draft_id)
        self.temperature = temperature
    
    
    def _draft_sample_k(self, model, input_ids, K):
        """sample K tokens from the draft model autoregressively
        draft_logits are a (B, K, V) tensor
        inputs_plus_k are a (B, T+K) tensor
        """
        inputs_plus_k = input_ids
        draft_logits = []
        for t in range(K):
            outputs = model(inputs_plus_k)
            next_token_logits = outputs.logits[:, -1, :]
            next_token_id = self.sample_fn(next_token_logits, self.temperature)
            inputs_plus_k = torch.cat(
                [inputs_plus_k, next_token_id.unsqueeze(1)],
                dim=1)
            draft_logits.append(next_token_logits)
        draft_logits = torch.stack(draft_logits, dim=1)
        return inputs_plus_k, draft_logits


    def _target_sample_from_distribution(self, target_distribution, draft_distribution):
        distribution = (target_distribution - draft_distribution)
        distribution = torch.max(distribution,
                                torch.zeros_like(distribution))
        distribution = distribution / distribution.sum(dim=-1, keepdim=True)
        return torch.multinomial(distribution, num_samples=1).squeeze(-1)


    def _ssp_iteration(self, target_model, draft_model, input_ids, K=4):
        _, T = input_ids.shape
        # sample K tokens from the draft model autoregressively
        # inputs_plus_k = (B, T+K); draft_logits = (B, K, V)
        inputs_plus_k, draft_logits = self._draft_sample_k(draft_model, input_ids, K) 
        # get the logits for the same tokens from the target model
        # target_logits = (B, K+1, V)
        target_logits = target_model(inputs_plus_k).logits[:, -K-1:, :]
        target_distribution = self.get_temperature_distribution(target_logits, self.temperature)
        draft_distribution = self.get_temperature_distribution(draft_logits, self.temperature)
        # Accept-reject token loop
        all_accepted = True
        for t in range(1, K+1):
            sampled_ratios = (
                target_distribution[:1, t-1, inputs_plus_k[0, T+t-1]]
                / draft_distribution[:1, t-1, inputs_plus_k[0, T+t-1]]
            )
            sampled_ratios = torch.min(sampled_ratios,
                                    torch.ones_like(sampled_ratios))
            rs = torch.rand_like(sampled_ratios)
            if (rs < sampled_ratios).any(): 
                input_ids = torch.cat(
                    [input_ids, inputs_plus_k[:, T + t-1].unsqueeze(1)],
                    dim=1)
            else:
                all_accepted = False
                next_token_id = self._target_sample_from_distribution(
                    target_distribution[:1, t-1, :],
                    draft_distribution[:1, t-1, :])
                input_ids = torch.cat(
                    [input_ids, next_token_id.unsqueeze(1)],
                    dim=1)
                break
        # if all tokens are accepted, sample a last one
        if all_accepted:
            next_token_id = self.sample_fn(target_logits[:1, -1, :], self.temperature)
            input_ids = torch.cat(
                [input_ids, next_token_id.unsqueeze(1)],
                dim=1)
            
        return input_ids


    def __call__(self, prompt, max_tokens, K=4, stop=None):
        input_ids = self.tokenizer(prompt, return_tensors="pt").input_ids
        B, T = input_ids.shape
        prompt_len = len(prompt)
        assert B == 1, "Batch size must be 1, implement the fixes for B > 1"
        while input_ids.shape[1] < T + max_tokens:
            if input_ids.shape[1] >= 2048:
                data = self.tokenizer.decode(input_ids[0], skip_special_tokens=True)
                raise utils.MemoryLimitExceededError("Memory limit exceeded with token shape {}".format(input_ids.shape),
                                                    extra_data=data[prompt_len:-len(stop)])
            input_ids = self._ssp_iteration(self.target_model, self.draft_model, input_ids, K)
            if stop:
                generated_text = self.tokenizer.decode(input_ids[0], skip_special_tokens=True)[prompt_len:]
                if stop in generated_text:
                    break
        output_str = self.tokenizer.decode(input_ids[0], skip_special_tokens=True)
        return output_str[prompt_len:-len(stop)]
    
    def ssp(self, target_model, draft_model, min_nb_tokens, input_ids, K=4):
        B, T = input_ids.shape
        assert B == 1, "Batch size must be 1, implement the fixes for B > 1"

        while input_ids.shape[1] < T + min_nb_tokens:
            input_ids = self._ssp_iteration(target_model, draft_model, input_ids, K)
        return input_ids


    def measure(self, texts, MAX_NEW_TOKENS=64, K=4, batch_size=1):
        nb_tokens = 0
        # Warmup
        input_ids = self.tokenizer(texts[0], return_tensors="pt").input_ids
        input_ids = torch.stack(
            [input_ids[0]] * batch_size).to(self.draft_model.device)
        generated_ids = self.ssp(self.target_model, self.draft_model, MAX_NEW_TOKENS,
                            input_ids, K=K)
        start_time = time.time()
        for text in texts[1:]:
            print("Completing text:", text)
            intermediate_time = time.time()
            input_ids = self.tokenizer(text, return_tensors="pt").input_ids
            input_ids = torch.stack(
                [input_ids[0]] * batch_size).to(self.draft_model.device)
            generated_ids = self.ssp(self.target_model, self.draft_model, MAX_NEW_TOKENS,
                                input_ids, K=K)
            nb_tokens += generated_ids.shape[1] - input_ids.shape[1]
            print("Completion: ", self.tokenizer.decode(
                generated_ids[0], skip_special_tokens=True))
            print("Time: {:.2f}s".format(time.time() - intermediate_time))
            print("========\n")
        ms_per_token = (time.time() - start_time)*1000 / nb_tokens
        return generated_ids, ms_per_token
    