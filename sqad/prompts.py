# flake8: noqa
from langchain.chains.prompt_selector import ConditionalPromptSelector, is_chat_model
from langchain.prompts import PromptTemplate
from langchain.prompts.chat import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate,
)


task = """Pro zadaný kontext a s ním souvisejícím otázku vygeneruj správnou odpověď. Odpovídej minimálním počtem slov extrahovaných z kontextu bez dalšího komentáře, případně odpověz pouze "ano" nebo "ne".

"""

few_shot = """Zde je 5 ukázkových příkladů:

Kontext:
Karibská krize (též Kubánská krize) byla mezinárodní politická krize. Hrozilo, že přeroste v jaderný konflikt. Vypukla v roce 1962 v důsledku rozmístění sovětských raket středního doletu na Kubě, kterým SSSR odpověděl na umístění amerických raket v Turecku. V reakci na to vyhlásily Spojené státy americké blokádu Kuby, která měla zabránit dopravení dalších raket na toto území....
Otázka:
Ve kterém roce vypukla Karibská krize?	
Odpověď:
1962

Kontext:
Ekvádor, oficiálně Ekvádorská republika (španělsky Ecuador nebo República del Ecuador), je stát v Jižní Americe, který leží na rovníku a zároveň je jeho pobřeží omýváno vodami Tichého oceánu. Jeho sousedy jsou Kolumbie na severu a Peru na jihovýchodě. Součástí Ekvádoru jsou rovněž Galapágy. Španělské slovo Ecuador znamená "rovník". Vlajka Ekvádoru se skládá ze tří vodorovných pruhů. Vrchní pruh se rozkládá na vrchní polovině vlajky a má barvu žlutou. O spodní polovinu vlajky se dělí barva modrá a červená . Žlutá a červená barva symbolizují vlajku Španělskou, protože Španělsko tuto oblast kolonizovalo. Modrá má symbolizovat oceán kterým je Ekvádor a Španělsko odděleno. Do modré části a kouskem do žluté části pak ještě zasahuje...	
Otázka:
Jsou Galapágy součástí Ekvádoru?	
Odpoveď:
ano

Kontext:
... Pohoří Jura (především Švábská a Franská Alba) pochází z doby jury. Z trias v Německu, z kterého pochází většina pískovců, v pohoří Jury ale převažuje vápenec. V třetihorách následovala eroze, zarovnání terénu a vytvoření nížin. Aktivní vulkanismus není v Německu pozorován, vulkanické horniny ale dosvědčují dřívější vulkanickou činnost. Nachází se především v pohoří Eifel. V pohoří Eifel jsou prameny s výskytem kysličníku uhličitého, gejzíry atd. Německo se nachází na eurasijské kontinentální desce, přesto se vyskytují slabá zemětřesení, především v Porúří. === Půdy === Na území Německa převládají drnopodzolové půdy a lesní hnědozemě. Drnopodzolové půdy se nejčastěji vyskytují na písčitých a štěrkopísčitých ledovcových usazeninách v Severoněmecké nížině. Lesní hnědozemě se vyskytují nejčastěji v Středoněmecké vysočině. == Příroda == === Flóra === Německo leží v mírném klimatickém pásmu. Značnou část jeho...	
Otázka:
V jakem klimatickém pásmu leží Německo?	
Odpověď:
v mírném

Kontext:
Fonologie (fonémika) je lingvistická věda, která podobně jako fonetika zkoumá zvukovou stránku přirozeného jazyka. Na rozdíl od fonetiky ji však zajímají pouze zvukové rozdíly, které mají v daném jazyce nějakou funkci (schopnost rozlišovat význam). Fonologie je nauka o funkci hlásek, zatímco fonetika je nauka o tvorbě hlásek ve zvukovém ústrojí, jejich šíření a vnímání. == Foném == Hlavním zájmem zkoumání ve fonologii je foném – nejmenší zvuková jednotka, která odlišuje slova mezi sebou. Výslovnostní varianty fonému jsou alofony: V češtině se např. nerozlišují fonémy /n/ a /ŋ/ ("n" ve slově náš proti "n" ve slově banka), protože rozdíl mezi nimi nemůže měnit význam. Vyslovované [n] a [ŋ] jsou proto považovány za alofony jediného fonému /n/. Naproti tomu v angličtině tento rozdíl hraje roli, viz např. thin [θ] (tenký) vs. thing [θ] (věc). Alofonní výslovnost je především zájmem fonetiky, která sleduje, jak se to které slovo správně vyslovuje,...
Otázka:
Jak se nazývá nejmenší zvuková jednotka?
Odpověď:
foném

Kontext:
... "jmenovat", ale je současně i volbou jeho osobního patrona, průvodce a životního vzoru tohoto jména. Proto, když církev slaví památku daného svatého, slaví jej i ti, kdo nesou jeho jméno, neboť je to den jejich patrona. Bývá zvykem dávat při křtu i jmen víc. V některých katolických zemích se přibírá ještě další jméno i při biřmování, např. v Česku. V Itálii naopak tento zvyk neznají. Při oslavách je zvykem nositelům jména popřát a případně je i obdarovat. Vzhledem k množství světců je mnoho rodných jmen příslušných k více dnům a naopak, k jednotlivým dnům připadá řada svatých. Ve Švédsku publikuje oficiální seznam jmenin Královská švédská akademie věd. Český občanský kalendář původně sice vycházel z církevního kalendáře, ale v průběhu času doznal mnoha změn...
Otázka:
Kdo ve Švédsku publikuje oficiální seznam jmenin?
Odpověď:
Královská švédská akademie věd

"""

request = """Nyní vygeneruj odpověď pro následující zadání:
Kontext:
{context}
Otázka:
{question}
Odpověď:
"""

input_variables = ["context", "question"]
input_types = {"context": "str", "question": "str"}


prompt_template = task + few_shot + request
PROMPT = PromptTemplate(
    template=prompt_template, input_variables=input_variables, input_types=input_types
)

system_template = task + few_shot
msg_template = request
messages = [
    SystemMessagePromptTemplate.from_template(system_template),
    HumanMessagePromptTemplate.from_template(msg_template),
]
CHAT_PROMPT = ChatPromptTemplate.from_messages(messages)

PROMPT_SELECTOR = ConditionalPromptSelector(
    default_prompt=PROMPT, conditionals=[(is_chat_model, CHAT_PROMPT)]
)

