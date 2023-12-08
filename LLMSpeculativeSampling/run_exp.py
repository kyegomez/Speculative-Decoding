from main import generate 
# from dist_inf import generate
import pandas as pd


model_configs = [
    # ("/p/realai/amir/Speculative-Decoding/LLMSpeculativeSampling/models1/Baichuan-7B", "/p/realai/amir/Speculative-Decoding/LLMSpeculativeSampling/models1/Baichuan-7B"), 
    # ("/p/realai/amir/Speculative-Decoding/LLMSpeculativeSampling/models1/Baichuan-7B", "/p/realai/amir/Speculative-Decoding/LLMSpeculativeSampling/models1/Baichuan2-13B-Base"), 
    # ("/p/realai/amir/Speculative-Decoding/LLMSpeculativeSampling/models1/Baichuan2-13B-Base", "/p/realai/amir/Speculative-Decoding/LLMSpeculativeSampling/models1/Baichuan2-13B-Base"), 
    # ("/p/realai/amir/Speculative-Decoding/LLMSpeculativeSampling/models1/Orca-2-7b", "/p/realai/amir/Speculative-Decoding/LLMSpeculativeSampling/models1/Orca-2-7b"),
    ("/p/realai/amir/Speculative-Decoding/LLMSpeculativeSampling/models1/Orca-2-7b", "/p/realai/amir/Speculative-Decoding/LLMSpeculativeSampling/models1/Orca-2-13b"),
    # ("/p/realai/amir/Speculative-Decoding/LLMSpeculativeSampling/models1/Orca-2-13b", "/p/realai/amir/Speculative-Decoding/LLMSpeculativeSampling/models1/Orca-2-13b"),
    # ("/p/realai/amir/Speculative-Decoding/LLMSpeculativeSampling/models1/bloom-560m", "/p/realai/amir/Speculative-Decoding/LLMSpeculativeSampling/models1/bloom-560m"),
    # ("/p/realai/amir/Speculative-Decoding/LLMSpeculativeSampling/models1/bloom-560m", "/p/realai/amir/Speculative-Decoding/LLMSpeculativeSampling/models1/bloomz-7b1"),
    # ("/p/realai/amir/Speculative-Decoding/LLMSpeculativeSampling/models1/bloomz-7b1", "/p/realai/amir/Speculative-Decoding/LLMSpeculativeSampling/models1/bloomz-7b1"),
]

input_texts = [
    "What are the best vegan restaurants in New York City?"
    "How can I improve my time management skills?"
    "Explain the theory of relativity in simple terms.",
    "Suggest some exercises for beginners in yoga.",
    "What are the top five science fiction books of the last decade?"]

gamma_vals = [i for i in range(4, 8)]
gpu_num = 0
df = pd.DataFrame(columns=['approx_model_name', 'target_model_name', 'gamma', 'alpha', 'approx_tokens_p_sec', 'target_tokens_p_second', 'sp_tokens_p_second', 'input_str_index'])


for i in range(len(model_configs)):
    for txt_idx in range(len(input_texts)):
        for j in gamma_vals:

            try: 
                alpha, sp_tok_p_sec, approx_tok_p_sec, target_tok_p_sec = generate(
                    input_text=input_texts[txt_idx], approx_model_name=model_configs[i][0],
                    target_model_name=model_configs[i][1], random_seed=0, num_tokens=50,
                    gamma=j,
                    use_benchmark=True,
                    gpu_num=gpu_num)
            except RuntimeError as e:
                # handle the error
                print(e)

                gpu_num+= 1
                gpu_num %= 7

                alpha, sp_tok_p_sec, approx_tok_p_sec, target_tok_p_sec = generate(
                    input_text=input_texts[txt_idx], approx_model_name=model_configs[i][0],
                    target_model_name=model_configs[i][1], random_seed=0, num_tokens=50,
                    gamma=j,
                    use_benchmark=True,
                    gpu_num=gpu_num)

            
            new_row = pd.DataFrame({
                'approx_model_name': [model_configs[i][0]], 
                'target_model_name': [model_configs[i][1]], 
                'gamma': [j],
                'alpha': [alpha],
                'approx_tokens_p_sec': [approx_tok_p_sec],
                'target_tokens_p_second': [target_tok_p_sec],
                'sp_tokens_p_second': [sp_tok_p_sec],
                'input_str_index': [txt_idx]
            }, index=[0])


            df = pd.concat([df, new_row], ignore_index=True)
            df.to_csv(f'exp_orca_singleGPU.csv')