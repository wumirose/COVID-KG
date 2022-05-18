import re
import requests
import arxiv
import pandas as pd
import hashlib
import pdf2image
import pytesseract
from tqdm import tqdm

from bs4 import BeautifulSoup

from itertools import combinations
from transformers import AutoTokenizer
from zero_shot_re import RelTaggerModel, RelationExtractor


def arxiv_search(searchquery, max_results):
    '''Read ARXIV ARTICLES
    using arxiv API'''
    search = arxiv.Search(
      query = searchquery,
      max_results = max_results,
      sort_by = arxiv.SortCriterion.SubmittedDate
    )
    all_article_ls = []
    all_articles = []
    i = 0
    for result in search.results():
        year = result.published
        title = result.title
        authors = result.authors
        abstract = result.summary
        all_article_ls.append({'title': title, \
        'author(s)': authors, 'synopsis': abstract, 'year': year})

        url = result.pdf_url
        pdf = requests.get(url,  timeout = 30)
        doc = pdf2image.convert_from_bytes(pdf.content)
        # Get the article text
        article = []

        for i, data in enumerate(doc):
            txt = pytesseract.image_to_string(data).encode("utf-8")
            article.append(txt.decode("utf-8"))
        i+=1
        print('Article', i, 'Total Pages', i)
        article_txt = " ".join(article)
        all_articles.append(article_txt)

    return  all_articles, all_article_ls


def read_url(url):
    '''Read URL CONTENTS'''
    if url.endswith('pdf'):
        pdf = requests.get(url,  timeout = 30)
        doc = pdf2image.convert_from_bytes(pdf.content)

        # Get the article text
        article = []
        for i, data in enumerate(doc):
            txt = pytesseract.image_to_string(data).encode("utf-8")
            article.append(txt.decode("utf-8"))
        print('Total Pages:', i)
        article_txt = " ".join(article)
    else:
        res = requests.get(url)
        doc = res.text
        soup = BeautifulSoup(doc, 'html5lib')
        for script in soup(["script", "style", 'aside']):
            script.extract()
        blog_txt = " ".join(re.split(r'[\n\t]+', soup.get_text()))
        article_txt = nlp(blog_txt)
        print('Entity lenghth:', len(article_txt.ents))
    return article_txt


def clean_article(article_txt):
    '''remove all rows atrating with unwanted seqs, subheadings etc
       returns text
    '''
    if article_txt.find("Introduction") > 0:
        text = article_txt.split("Introduction")[1]
        if text.find("Conflict") > 0:
            text = text.split("Conflict of Interest")[0]
    cleantxt = " ".join([row.rstrip() for row in text.split("\n") if len(row) > 1 \
                      # and not row.endswith("al.")\
                      # and not row.startswith("al.")\
                      and not row.startswith("Arch")\
                      and not row.startswith("Volume")\
                      and not row.startswith(" Khan")\
                      and not row.startswith("12")\
                      and not row.startswith("(https")\
                      and not row.startswith("Table")\
                      and not row.startswith("Figure") ])

    text = re.sub(r'- ', '-', cleantxt)#improper hiving
    text = re.sub(r' -', '-', text)#improper hiving
    text = re.sub(r'\[.*?]', '', text)#[2]citation
    text = re.sub(r'http\S+','', text)#https
    text = re.sub(r'www\S+','', text)#www
    text = re.sub(r' +',' ', text)#extra space
    # text = re.sub(r'i.e.',' ', text)#i.e.
    # text = re.sub(r'ie.,',' ', text)#ie.,
    text = re.sub(r'13',' ', text)#ie.,
    text = re.sub(r'b-coronavirus', 'SARS-CoV-2', text)
    text = re.sub(r'beta-coronavirus', 'SARS-CoV-2', text)
    text = re.sub(r'beta-SARS-COV-2', 'SARS-CoV-2', text)
    text = re.sub(r'SARS-COV-2', 'SARS-CoV-2', text)
    text = re.sub(r'-2-2', '-2', text)
    text = re.sub(r'2019-nCoV', 'COVID-19', text)
    text = re.sub(r'“B-Sitosterol:', 'Beta-sitosterol', text)
    text = re.sub(r'\[-sitosterol', 'Beta-sitosterol', text)
    text = re.sub(r'B-sitosterol', 'Beta-sitosterol', text)
    text = re.sub(r'^-sitosterol', 'Beta-sitosterol', text)
    text = re.sub(r'B-sitostero', 'Beta-sitosterol', text)
    return text

def joining(text):
    '''join incomplete lines together
    return list sentences = ['sent1', 'sent2', ...]
    '''
    all_text = []
    all_texts = []
    start = ' '
    for i, txt in enumerate(text):
        if re.search("^[a-z]", txt) is None:
            txts = txt
        else:
            txts = start + ' ' + txt
        start = txts
        all_text.append(txts)

    for i, word in enumerate(all_text):
        if word.endswith('i.e.'):
            sen = word+ ' ' + all_text[i+1]
            all_texts.append(sen)
        else:
            all_texts.append(word)
    return all_texts

def stop_remove(STOP_WORDS, sentences_):
    stopwords = list(STOP_WORDS)
    subwords = ['Introduction', 'Conclusion', 'Abstract', 'Results',  'Discussion', 'Table', 'Figure', 'et.', 'et', 'al.', 'al', 'i.e.', 'ie.,', 'Title']
    for i in subwords:
        if i not in stopwords:
            stopwords.append(i)
    sentences = []
    i = 0
    for sentence in sentences_:
        wordlist = []
        words = sentence.split() 
        if len(words) > 1: 
            for word in words:
                if word not in stopwords:
                    wordlist.append(word)
        sentences.append(' '.join(wordlist))   
    return sentences

def query_raw(text, url="http://bern2.korea.ac.kr/plain"):
    """BERN Biomedical entity linking API"""
    return requests.post(url, json = {'text': text}, verify = False).json()

def parse_entity(entity_list):
    """Diambiguate the parsed entities
    Thanks to Tomas https://colab.research.google.com/github/tomasonjo/blogs/blob/master/bionlp/bioNLP2graph.ipynb
    the notebook was soo insightful"""
    virus_species = ['SARS-CoV', 'SARS-CoV-2', 'Severe acute respiratory syndrome coronavirus']
    disease = ['COVID', 'SARS', 'Coronavirus']
    glyco = ['SARS-CoV-2 MP', 'SARS-CoV-2 Sp Glycoprotein', 'S proteins', \
        'S protein SARS-CoV-2 SARS-CoV-2 sp glycoprotein', 'SARS-CoV-2 sp glycoprotein', \
        'SARS-CoV-2 SARS-CoV-2 sp glycoprotein','COVID-19 (SARS-CoV-2) sp glycoprotein',\
        'S protein SARS-CoV-2', 'Coronavirus sp glycoprotein S1']
    parsed_entities = []
    # rows_ent, row_ent_name = [], []
    for entities in entity_list:
        e = []
        # If there are biomedical entities in the sentence
        if not entities.get('annotations'): 
            #get only the text
            parsed_entities.append({'text': entities['text'], 'timestamp':entities['timestamp'], 'text_sha256': hashlib.sha256(entities['text'].encode('utf-8')).hexdigest()})
        else:
            spanns = []
            span_names = []
            i = 0
            for entity in entities['annotations']:
                if entity['mention'] != 'Severe' and len(entity['mention']) > 2 and entity['mention'] != 'pig':
                    if (entity['mention'] in disease and entity['obj'] != 'disease'):
                        #Entityt type: Correction
                        entity_id = ''.join(entity['id']) + '*'
                        entity_type = 'disease'
                        entity_proba = 1 
                        if entity['mention'] == 'COVID':
                            entity_name = 'COVID-19' 
                            spanns.append((entity['span']['begin'], entity['span']['end'])) 
                            span_names.append(entity_name)
                        else:
                            entity_name = entity['mention']   
                    elif (entity['mention'] in virus_species and entity['obj'] != 'species'):
                        entity_id = ''.join(entity['id']) + '*'
                        entity_type = 'species'
                        entity_proba = 1 
                        if entity['mention'] == 'Severe acute respiratory syndrome coronavirus':
                            entity_name = 'SARS-CoV-2'
                            spanns.append((entity['span']['begin'], entity['span']['end'])) 
                            span_names.append(entity_name) 
                        entity_name = entity['mention']
                    elif (entity['mention'] == 'RBD' and entity['obj'] != 'gene'):
                        entity_id = ''.join(entity['id']) + '*'
                        entity_type = 'gene'
                        entity_proba = 1 
                        entity_name = entity['mention']
                    else:
                        #Name Disambiguation
                        entity_id =  entity['id']
                        entity_type = entity['obj']
                        entity_proba = entity['prob'] 
                        entity_name = None
                        if entity['mention'] in glyco and entity['mention']!= 'SARS-CoV-2 Spike Glycoprotein':
                            entity_name = 'SARS-CoV-2 Spike Glycoprotein' 
                            spanns.append((entity['span']['begin'], entity['span']['end'])) 
                            span_names.append(entity_name)
                        elif entity['mention'] == 'angiotensin converting enzyme-2':
                            entity['mention'] = 'ACE-2'
                            entity_name =  entity['mention']
                            spanns.append((entity['span']['begin'], entity['span']['end'])) 
                            span_names.append(entity_name)
                        elif entity['mention'] == 'Muntingia':
                            entity_name = 'Muntingia Calabura'
                            spanns.append((entity['span']['begin'], entity['span']['end'])) 
                            span_names.append(entity_name) 
                        elif entity['mention'] == 'N, E, M':
                            entity_name = 'NEM Protein'
                            spanns.append((entity['span']['begin'], entity['span']['end'])) 
                            span_names.append(entity_name)      
                        else:
                            entity_name = entity['mention']       
                    e.append({'entity_id': entity_id, 'entity_type': entity_type, 'entity': entity_name, 'entity_proba': entity_proba})#, 'entity_span': entity_span})
            # print('#', i, entities['text'])
            # i+=1
            for i, s in enumerate(spanns):
                if len(s)>=1:
                    txts = entities['text'][s[0]:s[1]]
                    # print(i, 'old:',txts)
                    n_text = span_names[i]
                    # print(i, 'new:',n_text)
                    entities['text'] = entities['text'].replace(txts, n_text)
                    # print(i, 'replaced:', entities['text'])
            parsed_entities.append({'entities':e, 'text':entities['text'], 'timestamp':entities['timestamp'], 'text_sha256': hashlib.sha256(entities['text'].encode('utf-8')).hexdigest()})         
    return parsed_entities

def make_df(parsed_entities):
    """Turn the parsed entity to a pandas frame
    from BERN2 Json format"""
    df = pd.json_normalize(parsed_entities)
    d = {k: [] for k in list(df['entities'][0][0].keys())}
    d['text'],  d['timestamp'],  d['text_sha256'] = [], [], []
    for rowid, entities in enumerate(df['entities']):
        if type(entities) == list:
            text = df['text'][rowid]
            timestamp = df['timestamp'][rowid]	
            text_sha256 = df['text_sha256'][rowid]
            for entity in entities: 
                for k in list(entity.keys()):
                    d[k].append(entity[k])
                d['text'].append(text)
                d['timestamp'].append(timestamp)
                d['text_sha256'].append(text_sha256)
    dataset = pd.DataFrame(d)
    return dataset


def predict_relations(parsed_entities, prior_rel):
    """ Use the prior Rule-based Matching relations 
    to predict relationships between entitites
    """   
    model = RelTaggerModel.from_pretrained("fractalego/fewrel-zero-shot")
    tokenizer = AutoTokenizer.from_pretrained("bert-large-uncased-whole-word-masking-finetuned-squad"
)
    extractor = RelationExtractor(model, tokenizer, prior_rel)

    # Candidate sentence where there is more than a single entity present
    candidates = [s for s in parsed_entities if (s.get('entities'))]
    predicted_rels = []
    for c in tqdm(candidates):
      combs = combinations([{'name':x['entity'], 'id':x['entity_id'], 'type':x['entity_type']} for x in c['entities']], 2)
      for pair in list(combs):
        try:
          ranked_rels = extractor.rank(text=c['text'].replace(",", " "), head=pair[0]['name'], tail=pair[1]['name'])
          # Define threshold for the most probable relation
          if ranked_rels[0][1] > 0.70:
            # print(combination)
            head = pair[0]['id']
            head_name = pair[0]['name']
            head_type = pair[0]['type']
            tail = pair[1]['id']
            tail_name = pair[1]['name']
            tail_type = pair[1]['type']
            rel = ranked_rels[0][0]
            if head != tail:
              predicted_rels.append({'head': head, 'head_name': head_name, 'head_type': head_type, 'tail': tail, 'tail_name': tail_name, 'tail_type': tail_type, 'con': rel, 'source': c['text_sha256']})
        except:
          pass
    return predicted_rels
