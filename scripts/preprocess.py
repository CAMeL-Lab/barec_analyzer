import argparse
import re
from camel_tools.utils.charmap import CharMapper
from camel_tools.utils.transliterate import Transliterator
from camel_tools.tokenizers.word import simple_word_tokenize
from camel_tools.disambig.bert import BERTUnfactoredDisambiguator
from camel_tools.morphology.database import MorphologyDB
from camel_tools.morphology.analyzer import Analyzer
from camel_tools.utils.dediac import dediac_ar

parser = argparse.ArgumentParser(description="Preprocess raw text to different input variants.")
parser.add_argument('--input', type=str, required=True, help='path to input text file containing raw text data')
parser.add_argument('--input_var', type=str, required=True, help='Input variant that raw text will be processed into (e.g., Word, D3Tok)')
parser.add_argument('--output', type=str, required=True, help='path to output file to save processed text data')
parser.add_argument('--db', type=str, required=False, help='path to morphological database to use for processing')
args = parser.parse_args()

input_file = args.input
input_variant = args.input_var
output_file = args.output
db_path = args.db

if db_path is None:
    print("No database path provided. Using default morphology database.")

print('Processing input variant:', input_variant)

def clean(text):
    arclean = CharMapper.mapper_from_json("arclean_map.json")
    arclean_translit = Transliterator(arclean)
    out = arclean_translit.transliterate(text)
    out = re.sub(r'(?<=\B)ى(?=\B)', 'ي', out)
    return out

with open(input_file, 'r', encoding='utf-8') as f:
    sentences = f.read().split("\n")
clean_sentences = [clean(str(line)) for line in sentences]
simple_tokenized_sentences = [simple_word_tokenize(line, split_digits=True) for line in clean_sentences]
word_sentences = [" ".join(line) for line in simple_tokenized_sentences]

if input_variant == "Word":
    with open(output_file, 'w', encoding='utf-8') as f:
        for line in word_sentences:
            f.write(line + "\n")
else:
    if db_path is None:
        bert = BERTUnfactoredDisambiguator.pretrained(model_name='msa', pretrained_cache=False, top=1)
    else:
        db = MorphologyDB(db_path,"a")
        analyzer = Analyzer(db, cache_size=100000, backoff='ADD_PROP')
        bert = BERTUnfactoredDisambiguator.pretrained(model_name='msa', pretrained_cache=False, top=1)
        bert._analyzer = analyzer
    disambiguated_sentences = bert.disambiguate_sentences(simple_tokenized_sentences)
    lex_sentences = []
    d3tok_sentences = []
    d3lex_sentences = []
    for i in range(len(disambiguated_sentences)):
        lex_sentence = [dediac_ar(disambig.analyses[0][1]['lex']) for disambig in disambiguated_sentences[i]]
        d3tok_sentence = [dediac_ar(disambig.analyses[0][1]['d3tok']).replace("_+", " +").replace("+_", "+ ") for disambig in disambiguated_sentences[i]]
        d3lex_sentence = []
        for j in range(len(d3tok_sentence)):
            d3tok_word = d3tok_sentence[j].split(" ")
            lex_word = lex_sentence[j]
            for segment in d3tok_word:
                if "+" not in segment or segment == "+":
                    d3lex_word = d3tok_sentence[j].replace(segment, lex_word)
            d3lex_sentence.append(d3lex_word)
        d3tok_sentence = " ".join(d3tok_sentence)
        lex_sentence = " ".join(lex_sentence)
        d3lex_sentence = " ".join(d3lex_sentence)
        d3tok_sentences.append(d3tok_sentence)
        lex_sentences.append(lex_sentence)
        d3lex_sentences.append(d3lex_sentence)
    if input_variant == "D3Tok":
        with open(output_file, 'w', encoding='utf-8') as f:
            for line in d3tok_sentences:
                f.write(line + "\n")
    elif input_variant == "Lex":
        with open(output_file, 'w', encoding='utf-8') as f:
            for line in lex_sentences:
                f.write(line + "\n")
    elif input_variant == "D3Lex":
        with open(output_file, 'w', encoding='utf-8') as f:
            for line in d3lex_sentences:
                f.write(line + "\n")
    else:
        raise ValueError(f"Unknown input variant: {input_variant}. Supported variants are: Word, D3Tok, Lex, D3Lex.")