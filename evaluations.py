import nereval
from nereval import Entity

def evaluate_f1(data_true, data_pred):
    '''
    data_true: [[(texts, {'entities': [(start, end, label_type), ...]}), ...]]
    data_pred: [[(text, label_type, start), ...]]
    '''
    n_docs = len(data_true)
    y_true = [None] * n_docs
    y_pred = [None] * n_docs
    
    for i in range(n_docs):
        y_true[i] = [Entity(data_train[i][0][ent[0]:ent[1]], ent[2], ent[0]) for ent in data_true[i][1]['entities']]
        y_pred[i] = [Entity(ent[0], ent[1], ent[2]) for ent in data_pred[i]]
    
    return nereval.evaluate(y_true, y_pred)


def evaluate_f1_for_spacy_model(data_true, nlp):
    '''
    data_true: [[(texts, {'entities': [(start, end, label_type), ...]}), ...]]
    nlp: a spaCy NER model
    '''
    n_docs = len(data_true)
    data_pred = [None] * n_docs
    
    for i in range(n_docs):
        pred_ents = nlp(data_true[i][0]).ents
        data_pred[i] = [(ent.text, ent.label_, ent.start_char) for ent in pred_ents]
    return evaluate_f1(data_true, data_pred)
