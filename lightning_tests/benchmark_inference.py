from litgpt import LLM
from pprint import pprint
from litgpt.api import benchmark_dict_to_markdown_table

llm = LLM.load("checkpoints/google/gemma-2-2b/")

print('---------')

text, bench_d = llm.benchmark(prompt="První věc, kterou jsem udělal je vzal", 
                              top_k=1, stream=True, num_iterations=10)

for key in bench_d: # warmup token takes longer
    bench_d[key] = bench_d[key][1:]

print(text)

print(benchmark_dict_to_markdown_table(bench_d))
