# pip install transformers==4.49.0 lm_eval==0.4.8 accelerate==0.34.2
# pip install antlr4-python3-runtime==4.11 math_verify sympy hf_xet

module load anaconda3
conda activate ~/scratch/envs/llada

# Set the environment variables first before running the command.
export HF_ALLOW_CODE_EVAL=1
export HF_DATASETS_TRUST_REMOTE_CODE=true

# conditional likelihood estimation benchmarks
accelerate launch eval_llada.py --tasks gpqa_main_n_shot --num_fewshot 5 --limit 32 --model llada_dist --batch_size 8 --model_args model_path='/home/hice1/jmoutahir3/scratch/LLada-Reasoning/llada_local_1.5',cfg=0.5,is_check_greedy=False,mc_num=128

accelerate launch eval_llada.py --tasks truthfulqa_mc2 --num_fewshot 0 --limit 32 --model llada_dist --batch_size 8 --model_args model_path='/home/hice1/jmoutahir3/scratch/LLada-Reasoning/llada_local_1.5',cfg=2.0,is_check_greedy=False,mc_num=128

accelerate launch eval_llada.py --tasks arc_challenge --num_fewshot 0 --limit 32 --model llada_dist --batch_size 8 --model_args model_path='/home/hice1/jmoutahir3/scratch/LLada-Reasoning/llada_local_1.5',cfg=0.5,is_check_greedy=False,mc_num=128

accelerate launch eval_llada.py --tasks hellaswag --num_fewshot 0 --limit 32 --model llada_dist --batch_size 8 --model_args model_path='/home/hice1/jmoutahir3/scratch/LLada-Reasoning/llada_local_1.5',cfg=0.5,is_check_greedy=False,mc_num=128

accelerate launch eval_llada.py --tasks winogrande --num_fewshot 5 --limit 32 --model llada_dist --batch_size 8 --model_args model_path='/home/hice1/jmoutahir3/scratch/LLada-Reasoning/llada_local_1.5',cfg=0.0,is_check_greedy=False,mc_num=128

accelerate launch eval_llada.py --tasks piqa --num_fewshot 0 --limit 32 --model llada_dist --batch_size 8 --model_args model_path='/home/hice1/jmoutahir3/scratch/LLada-Reasoning/llada_local_1.5',cfg=0.5,is_check_greedy=False,mc_num=128

accelerate launch eval_llada.py --tasks mmlu --num_fewshot 5 --limit 32 --model llada_dist --batch_size 1 --model_args model_path='/home/hice1/jmoutahir3/scratch/LLada-Reasoning/llada_local_1.5',cfg=0.0,is_check_greedy=False,mc_num=1

accelerate launch eval_llada.py --tasks cmmlu --num_fewshot 5 --limit 32 --model llada_dist --batch_size 1 --model_args model_path='/home/hice1/jmoutahir3/scratch/LLada-Reasoning/llada_local_1.5',cfg=0.0,is_check_greedy=False,mc_num=1

accelerate launch eval_llada.py --tasks ceval-valid --num_fewshot 5 --limit 32 --model llada_dist --batch_size 1 --model_args model_path='/home/hice1/jmoutahir3/scratch/LLada-Reasoning/llada_local_1.5',cfg=0.0,is_check_greedy=False,mc_num=1


# conditional generation benchmarks
accelerate launch eval_llada.py --tasks bbh --limit 32 --model llada_dist --model_args model_path='/home/hice1/jmoutahir3/scratch/LLada-Reasoning/llada_local_1.5',gen_length=1024,steps=1024,block_length=1024

accelerate launch eval_llada.py --tasks gsm8k --limit 32 --model llada_dist --model_args model_path='/home/hice1/jmoutahir3/scratch/LLada-Reasoning/llada_local_1.5',gen_length=1024,steps=1024,block_length=1024

accelerate launch eval_llada.py --tasks minerva_math --limit 32 --model llada_dist --model_args model_path='/home/hice1/jmoutahir3/scratch/LLada-Reasoning/llada_local_1.5',gen_length=1024,steps=1024,block_length=1024

accelerate launch eval_llada.py --tasks humaneval --limit 32 --model llada_dist --confirm_run_unsafe_code --model_args model_path='/home/hice1/jmoutahir3/scratch/LLada-Reasoning/llada_local_1.5',gen_length=1024,steps=1024,block_length=1024

accelerate launch eval_llada.py --tasks mbpp --limit 32 --model llada_dist --confirm_run_unsafe_code --model_args model_path='/home/hice1/jmoutahir3/scratch/LLada-Reasoning/llada_local_1.5',gen_length=1024,steps=1024,block_length=1024