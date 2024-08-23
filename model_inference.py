from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from datasets import load_dataset
import torch
import os
from utils import *

seed = 228

model_name = "google/gemma-2-2b-it"

model = AutoModelForCausalLM.from_pretrained(model_name,
                                             device_map="auto",
                                             torch_dtype=torch.float16  # Ensure model loads in float16
                                             )

# model = activate_adapter(model, "training/lora/cp/instruction_gemma2-2b-it/checkpoint-2000")
# model = activate_adapter(model, "training/lora/cp/instruction_gemma2-2b-it/checkpoint-5000")

model = activate_adapter(model, "training/unsloth/checkpoint-2000")
# model = activate_adapter(model, "training/unsloth/checkpoint-13000")

# model.config.use_cache = False

tokenizer = AutoTokenizer.from_pretrained(model_name)

peft_model = model
peft_model.eval()

# input_prompt = "Ahoj, prosím, napiš kratký odstavec o tom, jak se cítí kočky. Jsou šťastné nebo spíše smutné?"
# input_prompt = "Jaké město je větší: New York nebo Praha nebo Yekaterinburg? Kolik lidí tam žije?"
# input_prompt = "Jaké slovo chybí? 'Moje kočka je velmi ________ se mnou.' Napiš jen pisméno odpovědí a-d. \na). šťastná\nb). sťastný\nc). štastné\nd). šťastnou"
input_prompt = "0.002 = 1000\n1 = x?"
# input_prompt = """Zde je příklad otázky a odpovědi z testu:
# Otázka:
# Vysokotlaké systémy zabraňují vzduchu stoupat do chladnějších oblastí atmosféry, kde může kondenzovat voda. Jaký bude nejpravděpodobnější výsledek, pokud se systém vysokého tlaku udrží v oblasti po dlouhou dobu?
# Možnosti:
# A) mlha
# B) Déšť
# C) sucho
# D) Tornádo
# Odpověď:
# C

# Teď odpovězte na následující otázku. Napíšte pouze písmeno odpovědi.
# Otázka:
# Která oblast by byla nejlepší pro výzkum, aby se našly způsoby, jak snížit problémy životního prostředí způsobené lidmi?
# Možnosti:
# A) Přeměna slunečního světla na elektřinu
# B) Hledání nových zásob uhlí
# C) Nalezení ložisek, která obsahují ropu
# D) Přeměna lesů na zemědělskou půdu
# Odpověď:"""
# input_prompt = """Kontext:
# Karibská krize (též Kubánská krize) byla mezinárodní politická krize. Hrozilo, že přeroste v jaderný konflikt. Vypukla v roce 1962 v důsledku rozmístění sovětských raket středního doletu na Kubě, kterým SSSR odpověděl na umístění amerických raket v Turecku. V reakci na to vyhlásily Spojené státy americké blokádu Kuby, která měla zabránit dopravení dalších raket na toto území....
# Otázka:
# Ve kterém roce vypukla Karibská krize?	
# Odpověď:
# 1962

# Kontext:
# Ekvádor, oficiálně Ekvádorská republika (španělsky Ecuador nebo República del Ecuador), je stát v Jižní Americe, který leží na rovníku a zároveň je jeho pobřeží omýváno vodami Tichého oceánu. Jeho sousedy jsou Kolumbie na severu a Peru na jihovýchodě. Součástí Ekvádoru jsou rovněž Galapágy. Španělské slovo Ecuador znamená "rovník". Vlajka Ekvádoru se skládá ze tří vodorovných pruhů. Vrchní pruh se rozkládá na vrchní polovině vlajky a má barvu žlutou. O spodní polovinu vlajky se dělí barva modrá a červená . Žlutá a červená barva symbolizují vlajku Španělskou, protože Španělsko tuto oblast kolonizovalo. Modrá má symbolizovat oceán kterým je Ekvádor a Španělsko odděleno. Do modré části a kouskem do žluté části pak ještě zasahuje...	
# Otázka:
# Jsou Galapágy součástí Ekvádoru?	
# Odpoveď:
# ano

# Kontext:
# ... Pohoří Jura (především Švábská a Franská Alba) pochází z doby jury. Z trias v Německu, z kterého pochází většina pískovců, v pohoří Jury ale převažuje vápenec. V třetihorách následovala eroze, zarovnání terénu a vytvoření nížin. Aktivní vulkanismus není v Německu pozorován, vulkanické horniny ale dosvědčují dřívější vulkanickou činnost. Nachází se především v pohoří Eifel. V pohoří Eifel jsou prameny s výskytem kysličníku uhličitého, gejzíry atd. Německo se nachází na eurasijské kontinentální desce, přesto se vyskytují slabá zemětřesení, především v Porúří. === Půdy === Na území Německa převládají drnopodzolové půdy a lesní hnědozemě. Drnopodzolové půdy se nejčastěji vyskytují na písčitých a štěrkopísčitých ledovcových usazeninách v Severoněmecké nížině. Lesní hnědozemě se vyskytují nejčastěji v Středoněmecké vysočině. == Příroda == === Flóra === Německo leží v mírném klimatickém pásmu. Značnou část jeho...	
# Otázka:
# V jakem klimatickém pásmu leží Německo?	
# Odpověď:"""


# input_tokens = tokenizer(input_prompt, return_tensors="pt")["input_ids"].to("cuda")  # Ensure input tokens are also in float16
# # input_tokens = torch.tensor(input_tokens).to("cuda").long()
# print(input_tokens)

messages = [
    {"role": "user", "content": input_prompt},
]

input_ids = tokenizer.apply_chat_template(messages,
                                          return_tensors="pt", 
                                          return_dict=True,
                                          add_generation_prompt=True).to("cuda")

outputs = model.generate(**input_ids, max_new_tokens=200)
print(tokenizer.decode(outputs[0]))

# with torch.no_grad():  # Disabling gradient computation
#     generation_output = peft_model.generate(
#         input_ids=input_ids["input_ids"],
#         max_new_tokens=100,
#         do_sample=True,
#         top_k=10,
#         top_p=0.9,
#         temperature=1e-9,
#     )

# op = tokenizer.decode(generation_output[0], skip_special_tokens=False)
# print(f"\033[94m>>>>\033[0m{op}\033[94m<<<<\033[0m")
