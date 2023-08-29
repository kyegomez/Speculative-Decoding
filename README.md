[![Multi-Modality](agorabanner.png)](https://discord.gg/qUtxnK2NMf)

# Speculative-Decoding
My own implementation of "Fast Inference from Transformers via Speculative Decoding"

## Code
```python

import torch
import torch.nn.functional as F

class SpeculativeDecoder:
    def __init__(self, Mp, Mq, gamma):
        """
        Initialize the SpeculativeDecoder.
        
        Parameters:
        - Mp (nn.Module): The target model.
        - Mq (nn.Module): The more efficient approximation model.
        - gamma (int): The number of completions to generate.
        """
        self.Mp = Mp
        self.Mq = Mq
        self.gamma = gamma
        
    def speculative_sampling(self, p, q):
        """
        Perform speculative sampling from distribution p(x) using distribution q(x).
        
        Parameters:
        - p (torch.Tensor): The target distribution p(x).
        - q (torch.Tensor): The approximation distribution q(x).
        
        Returns:
        - x (int): The sampled token from p(x).
        """
        try:
            # Sample x from q(x)
            x = torch.multinomial(q, 1).item()
            
            # Keep x if q(x) <= p(x), otherwise reject and resample from adjusted distribution
            if q[x] > p[x]:
                p_0 = F.normalize(torch.clamp(p - q, min=0), p=1, dim=0)
                x = torch.multinomial(p_0, 1).item()
                
            return x
        except Exception as e:
            print("An error occurred in speculative_sampling: ", str(e))
    
    def speculative_decoding_step(self, prefix):
        """
        Perform one step of speculative decoding.
        
        Parameters:
        - prefix (torch.Tensor): The conditioning prefix.
        
        Returns:
        - new_prefix (torch.Tensor): The updated prefix.
        """
        try:
            # Sample γ guesses from Mq autoregressively
            prefix = prefix.unsqueeze(0)  # Add batch dimension
            guesses = []
            for i in range(self.gamma):
                with torch.no_grad():
                    q = self.Mq(prefix)
                guess = torch.multinomial(q[0], 1)
                guesses.append(guess)
                prefix = torch.cat((prefix, guess.unsqueeze(0)), dim=1)
            
            # Run Mp in parallel
            p_values = []
            for i in range(self.gamma + 1):
                with torch.no_grad():
                    p = self.Mp(prefix[:, :len(prefix[0]) - i])
                p_values.append(p[0])
            
            # Determine the number of accepted guesses
            n = self.gamma
            for i in range(self.gamma):
                ri = torch.rand(1).item()
                if ri > p_values[i][guesses[i].item()] / q[0][guesses[i].item()]:
                    n = i - 1
                    break
            
            # Adjust the distribution from Mp if needed
            p_0 = p_values[n + 1]
            if n < self.gamma:
                q_n = q[0][guesses[n].item()]
                p_0 = F.normalize(torch.clamp(p_0 - q_n, min=0), p=1, dim=0)
            
            # Return one token from Mp, and n tokens from Mq
            t = torch.multinomial(p_0, 1)
            new_prefix = torch.cat((prefix[0, :len(prefix[0]) - self.gamma + n], t))
            
            return new_prefix
        except Exception as e:
            print("An error occurred in speculative_decoding_step: ", str(e))

# Example Usage
# Make sure to define your models Mp and Mq, and provide the prefix and gamma value
# Mp = some_target_model
# Mq = some_approximation_model
# prefix = torch.tensor([some_initial_tokens])
# gamma = some_integer_value

# decoder = SpeculativeDecoder(Mp, Mq, gamma)
# new_prefix = decoder.speculative_decoding_step(prefix)


```

## Citations

```bibtex
@inproceedings{Leviathan2022FastIF,
    title   = {Fast Inference from Transformers via Speculative Decoding},
    author  = {Yaniv Leviathan and Matan Kalman and Y. Matias},
    booktitle = {International Conference on Machine Learning},
    year    = {2022},
    url     = {https://api.semanticscholar.org/CorpusID:254096365}
}
```

```bibtex
@inproceedings{sun2023spectr,
    title     = {SpecTr: Fast Speculative Decoding via Optimal Transport},
    author    = {Ziteng Sun and Ananda Theertha Suresh and Jae Hun Ro and Ahmad Beirami and Himanshu Jain and Felix Yu and Michael Riley and Sanjiv Kumar},
    booktitle = {Workshop on Efficient Systems for Foundation Models @ ICML2023},
    year      = {2023},
    url       = {https://openreview.net/forum?id=d0mGsaheuT}
}
```

```bibtex
@article{Chen2023AcceleratingLL,
    title     = {Accelerating Large Language Model Decoding with Speculative Sampling},
    author    = {Charlie Chen and Sebastian Borgeaud and Geoffrey Irving and Jean-Baptiste Lespiau and L. Sifre and John M. Jumper},
    journal   = {ArXiv},
    year      = {2023},
    volume    = {abs/2302.01318},
    url       = {https://api.semanticscholar.org/CorpusID:256503945}
}
```
