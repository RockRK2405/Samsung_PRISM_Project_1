from __future__ import annotations

import math

try:
    import torch
    import transformers
    from transformers import AutoModelForCausalLM, AutoTokenizer
    _AVAILABLE = True
except ImportError:
    _AVAILABLE = False


def is_available() -> bool:
    """Check if torch and transformers are installed."""
    return _AVAILABLE


class PerplexityAnalyzer:
    _instance = None
    
    def __init__(self, model_name: str = "distilgpt2", device: str | None = None):
        """Load model. Auto-detect GPU."""
        if not is_available():
            raise RuntimeError("torch and transformers are required for PerplexityAnalyzer")
            
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
            
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForCausalLM.from_pretrained(model_name).to(self.device)
            self.model.eval()
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
        except Exception as e:
            raise RuntimeError(f"Failed to load model {model_name}: {e}")
    
    @classmethod
    def get_instance(cls) -> 'PerplexityAnalyzer':
        """Singleton accessor. Creates instance on first call."""
        if cls._instance is None:
            if not is_available():
                return None
            cls._instance = cls()
        return cls._instance
    
    def compute_perplexity(self, text: str, max_length: int = 512, stride: int = 256) -> float:
        """Compute perplexity of text. Returns float or nan for invalid input."""
        if not text or len(text.strip()) < 5:
            return float('nan')
            
        try:
            encodings = self.tokenizer(text, return_tensors="pt")
            input_ids = encodings.input_ids.to(self.device)

            if input_ids.size(1) < 2:
                return float('nan')

            seq_len = input_ids.size(1)
            nlls = []
            prev_end_loc = 0

            for begin_loc in range(0, seq_len, stride):
                end_loc = min(begin_loc + max_length, seq_len)
                trg_len = end_loc - prev_end_loc

                input_ids_chunk = input_ids[:, begin_loc:end_loc]
                target_ids = input_ids_chunk.clone()
                target_ids[:, :-trg_len] = -100

                with torch.no_grad():
                    outputs = self.model(input_ids_chunk, labels=target_ids)
                    neg_log_likelihood = outputs.loss * trg_len

                nlls.append(neg_log_likelihood)
                prev_end_loc = end_loc

                if end_loc == seq_len:
                    break

            ppl = torch.exp(torch.stack(nlls).sum() / end_loc)
            return float(ppl.cpu())
        except Exception:
            return float('nan')
    
    def compute_token_surprisals(self, text: str, max_length: int = 512) -> list[float]:
        """Get per-token surprisal values (negative log probs)."""
        if not text or len(text.strip()) < 5:
            return []
            
        try:
            # Tokenize but truncate for simple surprisal analysis
            encodings = self.tokenizer(text, return_tensors="pt", max_length=max_length, truncation=True)
            input_ids = encodings.input_ids.to(self.device)
            
            if input_ids.size(1) < 2:
                return []
                
            with torch.no_grad():
                outputs = self.model(input_ids, labels=input_ids)
                logits = outputs.logits
                
            # Compute token-level cross entropy
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = input_ids[..., 1:].contiguous()
            loss_fct = torch.nn.CrossEntropyLoss(reduction="none")
            
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
            surprisals = loss.cpu().tolist()
            
            return surprisals
        except Exception:
            return []
    
    def compute_surprisal_stats(self, text: str) -> dict:
        """Returns mean, variance, skewness, kurtosis, max, entropy."""
        surprisals = self.compute_token_surprisals(text)
        
        if not surprisals or len(surprisals) < 2:
            return {
                'mean': 0.0, 'variance': 0.0, 'skewness': 0.0, 
                'kurtosis': 0.0, 'max_surprisal': 0.0, 'entropy': 0.0
            }
            
        import numpy as np
        from scipy.stats import skew, kurtosis
        
        arr = np.array(surprisals)
        
        # Calculate discrete entropy (approximation from probabilities)
        probs = np.exp(-arr)
        # Normalize to form a valid distribution for entropy calculation
        if np.sum(probs) > 0:
            probs_norm = probs / np.sum(probs)
            entropy = -np.sum(probs_norm * np.log(probs_norm + 1e-10))
        else:
            entropy = 0.0
            
        return {
            'mean': float(np.mean(arr)),
            'variance': float(np.var(arr)),
            'skewness': float(skew(arr)),
            'kurtosis': float(kurtosis(arr)),
            'max_surprisal': float(np.max(arr)),
            'entropy': float(entropy)
        }
    
    def score(self, text: str) -> float:
        """0-1 AI probability. Low perplexity -> high AI probability."""
        ppl = self.compute_perplexity(text)
        if math.isnan(ppl) or ppl <= 0:
            return 0.5
            
        try:
            # Convert perplexity to an AI probability score via sigmoid
            log_ppl = math.log(ppl)
            # Center at log(ppl) = 3.5 (~33 perplexity), scale by 0.8
            # Low perplexity (e.g. 15, log=2.7) -> higher probability
            # The original requirement was 1 / (1 + exp((log(ppl) - 3.5) / 0.8))
            # Wait, low perplexity means AI.
            # If ppl=10, log=2.3. (2.3-3.5)/0.8 = -1.5. 1/(1+exp(-1.5)) = 0.82 (High AI prob)
            # If ppl=100, log=4.6. (4.6-3.5)/0.8 = +1.375. 1/(1+exp(1.375)) = 0.20 (Low AI prob)
            score = 1.0 / (1.0 + math.exp((log_ppl - 3.5) / 0.8))
            return score
        except Exception:
            return 0.5
    
    def analyze(self, text: str) -> dict:
        """Full analysis combining all metrics."""
        ppl = self.compute_perplexity(text)
        if math.isnan(ppl):
            return {
                'available': True,
                'perplexity': None,
                'ai_score': 0.5,
                'surprisal_stats': {
                    'mean': 0.0, 'variance': 0.0, 'skewness': 0.0, 
                    'kurtosis': 0.0, 'max_surprisal': 0.0, 'entropy': 0.0
                }
            }
            
        score = self.score(text)
        stats = self.compute_surprisal_stats(text)
        
        return {
            'available': True,
            'perplexity': ppl,
            'ai_score': score,
            'surprisal_stats': stats
        }


def analyze_text_perplexity(text: str) -> dict:
    """Convenience function. Returns analysis dict or safe fallback."""
    if not is_available():
        return {
            'available': False,
            'perplexity': None,
            'ai_score': 0.5,
            'surprisal_stats': {}
        }
        
    try:
        analyzer = PerplexityAnalyzer.get_instance()
        if analyzer is None:
            return {
                'available': False,
                'perplexity': None,
                'ai_score': 0.5,
                'surprisal_stats': {}
            }
        return analyzer.analyze(text)
    except Exception:
        return {
            'available': False,
            'perplexity': None,
            'ai_score': 0.5,
            'surprisal_stats': {}
        }
