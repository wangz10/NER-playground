'''Utils for loading datasets.
'''
import re
import nltk

def parse_NCBI_disease_corpus(fn):
    '''
    returns data: [(doc_texts, {'entities': [(ent_start, ent_end, ent_type), ...]}), ...]
    '''
    data = []
    c = 0 # error counter
    with open(fn, 'r') as f:
        fulltext = f.read()
        for pmid_doc in fulltext.split('\n\n'):

            lines = pmid_doc.strip().split('\n')
            texts = re.sub(r'[0-9]+\|t\|', '', lines[0]) + '\n' + re.sub(r'[0-9]+\|a\|', '', lines[1])
            entities = []
            for line in lines[2:]:
                sl = line.split('\t')
                start, end = int(sl[1]), int(sl[2])
                entity, entity_type = sl[3], sl[4]
                if texts[start:end] != entity:
                    c += 1
                    # print (c, texts[start:end], entity)
                else:
                    entities.append( (start, end, entity_type) )
                    
            data.append((texts, {'entities': entities}))
    return data


def convert_to_annot_tokens(data):
    '''
    Convert the data into lists of tokens with POS-tags and entity_type:
    [
        [(token, pos_tag, ent_type), ...], # 1st document
        [(token, pos_tag, ent_type), ...], # 2nd document
        ...
    ]
    Multi-token entities will be dropped.
    '''
    data_annot = [None] * len(data)
    for i in range(len(data)):
        text = data[i][0]
        entities = data[i][1]['entities']

        pos_tags = nltk.pos_tag(nltk.word_tokenize(text))

        d_entities = {} # entity: entity_type
        for ent in entities:
            ent_name = text[ent[0]:ent[1]]
            d_entities[ent_name] = ent[2]

        doc_annot = []
        for token, pos in pos_tags:
            if token in d_entities:
                ent_type = d_entities[token]
            else:
                ent_type = 'IR'
            doc_annot.append( (token, pos, ent_type) )

        data_annot[i] = doc_annot    
    return data_annot

