from datasets import load_dataset, load_from_disk
from langchain.chains.prompt_selector import is_chat_model
from langchain_core.output_parsers.string import StrOutputParser
import evaluate
import json
import sys
import os
import numpy as np
import time
import copy
from datetime import datetime
from .prompts import PROMPT_SELECTOR

local_dir = os.path.dirname(os.path.abspath(__file__))
home_dir = os.path.dirname(os.path.dirname(local_dir))
sys.path.append(home_dir + "/tools/morphology")
from lemmatization import MorphoDiTa
from word_roots import DeriNet


class Evaluator:
    def __init__(self, local=False):
        print("\nInitializing SQAD evaluator")
        if local:
            self.load_local()
        else:
            self.load_hf()
        
        print("Loading morphological tools")
        self.lemmatizer = MorphoDiTa()
        if not self.lemmatizer.lemmatizer:
            self.lemmatizer = False
        self.root_lexicon = None
        if not self.lemmatizer:
            print("Failed to load lemmatizer model, morphological analysis will be skipped")
        else:
            try:
                self.root_lexicon = DeriNet()
            except:
                print("Failed to load root lexicon, root extraction will be skipped")

    def load_hf(self):
        print("Loading dataset from Hugging Face")
        self.dataset = load_dataset("fewshot-goes-multilingual/cs_squad-3.0", split="test")
        #self.dataset.save_to_disk(local_dir + "/data/test")

    def load_local(self):
        print("Loading dataset locally")
        self.dataset = load_from_disk(local_dir + "/data/test")
    
    def morpho_analyze(self, answer):
        tokens = [tok.rstrip(",.?!") for tok in answer.lower().rstrip('\r\n').split()]
        lemmas = ""
        roots = ""
        for token in tokens:
            lemma = self.lemmatizer.lemmatize(token)
            lemmas += lemma + " "      
            if self.root_lexicon is not None:
                roots += self.root_lexicon.get_root(lemma) + " "
        return lemmas, roots
    
    def run_eval(self, llm, result_file, stop_idx=np.inf):
        info = f'\nCommencing SQAD evaluation at {datetime.now().strftime("%H:%M:%S, %d/%m/%Y")}'
        print(info)
        with open (result_file, "a") as rf:
            rf.write("\n\n*** SQAD ***" + info + "\n")

        prompt = PROMPT_SELECTOR.get_prompt(llm)
        str_parser = StrOutputParser()

        references = []
        predictions = []
        ref_lemmas = []
        pred_lemmas = []
        ref_roots = []
        pred_roots = []
        count = 0
        cum_time = 0.

        for i, example in enumerate(self.dataset):
            if i+1 > stop_idx:
                break
            print(f"\rExample {i+1} / {len(self.dataset)}", end="")
            context = example["context"]
            question = example["question"]

            try:
                start_time = time.time()
                if is_chat_model(llm):
                    result = llm.invoke(prompt.format_prompt(context=context, question=question).to_messages())
                else:
                    result = llm.invoke(prompt.format_prompt(context=context, question=question).text)
                result = str_parser.invoke(result)
                end_time = time.time()
            except Exception as e:
                print(f"\nExample skipped due to an LLM Error: {e}")
                continue
            
            id = example["item_id"]
            ref = {
                "answers": {"text": example["answers"], "answer_start": [d["start"] for d in example["occurrences"]]},
                "id": id
            }
            ans = {
                "prediction_text": result,
                "id": id
            }
            references.append(ref)
            predictions.append(ans)

            if self.lemmatizer:
                ans_lemmas = []
                ans_roots = []
                for ans_text in example["answers"]:
                    lems, roots = self.morpho_analyze(ans_text)
                    ans_lemmas.append(lems)
                    ans_roots.append(roots)

                ref_lem_dict = copy.deepcopy(ref)
                ref_lem_dict["answers"]["text"] = ans_lemmas
                ref_root_dict = copy.deepcopy(ref)
                ref_root_dict["answers"]["text"] = ans_roots

                lems, roots = self.morpho_analyze(result)
                pred_lem_dict = copy.deepcopy(ans)
                pred_lem_dict["prediction_text"] = lems
                pred_root_dict = copy.deepcopy(ans)
                pred_root_dict["prediction_text"] = roots

                ref_lemmas.append(ref_lem_dict)
                pred_lemmas.append(pred_lem_dict)
                if self.root_lexicon is not None:
                    ref_roots.append(ref_root_dict)
                    pred_roots.append(pred_root_dict)
            count += 1
            cum_time += end_time - start_time
            
        print("\nComputing metrics")

        lines = "\nResults:\n"
        if count > 0:
            metric = evaluate.load("squad")
            res = metric.compute(predictions=predictions, references=references)
            lines += f"Exact Match: {res['exact_match']:.2f}\n"
            lines += f"BoW F1: {res['f1']:.2f}\n"

            if self.lemmatizer:
                res_lemma = metric.compute(predictions=pred_lemmas, references=ref_lemmas)
                lines += f"Lemmas Exact Match: {res_lemma['exact_match']:.2f}\n"
                lines += f"Lemmas BoW F1: {res_lemma['f1']:.2f}\n"

                if self.root_lexicon is not None:
                    res_root = metric.compute(predictions=pred_roots, references=ref_roots)
                    lines += f"Roots Exact Match: {res_root['exact_match']:.2f}\n"
                    lines += f"Roots BoW F1: {res_root['f1']:.2f}\n"
            
            lines += f"Average inference time: {cum_time/count:.2f}s\n"

        lines += f"Total valid examples used: {count}\n"

        with open(result_file, "a") as rf:
            rf.write(lines)
        print(lines)
