import torch
from abc import ABC, abstractmethod

class BaseLlama(ABC):
    @abstractmethod
    def __init__(self):
        pass
    
    @abstractmethod
    def __call__(self):
        pass
    
    @abstractmethod
    def measure(self, texts, MAX_NEW_TOKENS=64, batch_size=1):
        pass
    
    def sample_fn(self, logits, temperature):
        probs = self.get_temperature_distribution(logits, temperature)
        return torch.multinomial(probs, num_samples=1).squeeze(-1)
        
    def get_temperature_distribution(self, logits, temperature):
        return torch.softmax(logits / temperature, dim=-1)
    
    def print_results(self, tokens_s, outputs):
        latency_sec_per_token = round(tokens_s / 1000, 3)
        throughput_tokens_per_sec = 1 / latency_sec_per_token
        print("Results for ", self.name)
        print(f"Throughput: {throughput_tokens_per_sec:.2f} tokens/sec")
        print(f"Latency: {latency_sec_per_token:.3f} sec/token")