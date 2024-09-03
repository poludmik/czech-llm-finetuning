import os
from ufal.morphodita import Morpho, TaggedLemmas

local_dir = os.path.dirname(os.path.abspath(__file__))

class MorphoDiTa(object):
    def __init__(self, path=local_dir+"/czech-morfflex/czech-morfflex2.0-220710.dict"):
        self.lemmatizer = Morpho.load(path)

    def lemmatize(self, word):
        lem_store = TaggedLemmas()
        res = self.lemmatizer.analyze(word, self.lemmatizer.GUESSER, lem_store)
        return lem_store[0].lemma
    
    def tag(self, word):
        lem_store = TaggedLemmas()
        res = self.lemmatizer.analyze(word, self.lemmatizer.GUESSER, lem_store)
        return lem_store[0].tag