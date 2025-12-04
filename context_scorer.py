import os
from nltk.tokenize import sent_tokenize
from nltk.stem.snowball import SnowballStemmer
import torch.nn.functional as F
import torch
from itertools import product

from sentence_transformers import SentenceTransformer, util
from transformers import pipeline, AutoTokenizer, AutoModelForMaskedLM, AutoConfig
import numpy as np
from transformers import AutoModelForSequenceClassification
import pickle

def load_data(article_dir, relation_dir):
    #Get article and relation files
    article_names = sorted(os.listdir(article_dir))
    relation_files = sorted(os.listdir(relation_dir))
    articles = []
    article_relations = []
    stemmed_articles = []
    stemmer = SnowballStemmer("english")
    article_stemmed_relations = []
    for i in range(len(article_names)):
        article_name = article_names[i]
        sentence_to_inds = {}
        with open(article_dir+article_name, "r") as f:
            passage = f.read()
            sentences = sent_tokenize(passage)
            stemmed_passage = []
            for sentence in sentences:
                stemmed = []
                for word in sentence.split(" "):
                    stemmed.append(stemmer.stem(word))
                stemmed_passage.append(" ".join(stemmed))
            articles.append((article_name, sentences))
            stemmed_articles.append(stemmed_passage)
        relations = []
        relation_fname = relation_files[i]
        
        stemmed_relations = []
        with open(relation_dir+relation_fname) as f:
            lines = f.readlines()
            for line in lines:
                if len(line.strip()) > 0:
                    stemmed_relation= []
                    relation = []
                    j = 0
                    for words in line.strip().split(" | "):
                        if j == 1:
                            lower_words = words.lower()
                        else:
                            lower_words = words
                        temp = []
                        relation.append(words)
                        for word in lower_words.split():
                            temp.append(stemmer.stem(word))
                        stemmed_relation.append(" ".join(temp))
                        j+=1
                    stemmed_relations.append(stemmed_relation)
                    relations.append(relation)
        article_relations.append(relations)
        article_stemmed_relations.append(stemmed_relations)
    return articles, article_relations, stemmed_articles, article_stemmed_relations


def bert_masked_prediction(tokenizer, model, text, device, top_k_per_mask=5):
    
    input_tokens = tokenizer(text, return_tensors="pt").to(device)
    mask_positions = (input_tokens["input_ids"] == tokenizer.mask_token_id).nonzero(as_tuple=True)[1]
    
    with torch.no_grad():
        logits = model(**input_tokens).logits
    log_probs = F.log_softmax(logits, dim=-1)
    mask_log_probs = log_probs[0, mask_positions, :]
    topk = [torch.topk(mask_log_probs[i], top_k_per_mask) for i in range(len(mask_positions))]
    sequences = []
    for combo in product(*[range(top_k_per_mask) for _ in range(len(mask_positions))]):
        # combo is like (i1, i2, i3, ...) for N masks
        score = sum(topk[m].values[combo[m]].item() for m in range(len(combo)))
        token_ids = [topk[m].indices[combo[m]].item() for m in range(len(combo))]
        repeated = False
        for k in range(len(token_ids)-1):
            if token_ids[k] == token_ids[k+1]:
                repeated = True
        if not repeated:
            sequences.append((score, token_ids))

    # Sort by probability computed from joint score
    
    # attach probabilities to sequences
    sequences.sort(key=lambda x: x[0], reverse=True)
    sequences = sequences[:5]
    scores = torch.tensor([s for s, _ in sequences])
    probs = torch.softmax(scores, dim=0)
    sequences = [(p, token_ids) for (s, token_ids), p in zip(sequences, probs)]
    results = []
    str_mask_ind = text.find("<mask>")
    for score, token_ids in sequences:
        masks = " ".join(["<mask>"]*len(token_ids))
        decoded_tokens = [tokenizer.decode([tid]).strip() for tid in token_ids]
        new_relation = " ".join(decoded_tokens)
        results.append([score.item(), new_relation, text[:str_mask_ind]+new_relation+text[str_mask_ind+len(masks):]])
    return results


def get_bert_probs(model_name, articles, article_relations, stemmed_articles, article_stemmed_relations):
    """
    Input:
        model_name: name of BERT model to use for MLM (currently only works for roberta-large)
        articles: List of string, each string is the entire article
        article_relations: List of List of List
            Inner list contains all the relations for the given article
            Innermost list is [cause,relation,effect]
        stemmed_articles: List of string, each string is a stemmer applied to each word in the article
        article_stemmed_relations: List of List of List
            Structurally same as article_relations but a stemmer is applied to 
            each word in the cause, relation, and effect strings
         
    Use SentenceBERT to find the original sentence the relation was extracted from.
    Then, use a BERT/MLM model to do masked language prediction by masking out the relation part
    of the <cause,relation,effect> triple in the original sentence.
    If the extracted relation does not verbatim exist in the original text, check if the word
    stems exist in the original sentence. If they do, then mask out the corresponding original words.
    If the extracted relation does not verbatim exist and the stemmed words also do not verbatim exist,
    save the semantically most similar text to be the context/premise and relation as the hypothesis, will
    use an MNLI model to check if context entails the relation.

    Returns: 
        article_probs: List of List of Dict
            Inner list represents an article and each element (Dict) maps (cause,relation,effect) to the match score
            as described above
        missing_triples: List of List of List
            Inner list represents an article and each element (List) consists of:
                index 0: the 5 most relevant/semantically similar sentences from SentenceBERT for the given (cause,relation,effect) triple
                index 1: the cause,relation,effect triple as a phrase/sentence as 1 string
                index 2: (cause,relation,effect)
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sentence_model = SentenceTransformer('msmarco-MiniLM-L6-cos-v5').to(device)
    sentence_model.eval()
    bert_tokenizer = AutoTokenizer.from_pretrained(model_name)
    bert_model = AutoModelForMaskedLM.from_pretrained(model_name).to(device)
    bert_model.eval()
    articles_probs = []
    articles_missing_triples = []
    for i in range(len(articles)):
        article_name = articles[i][0]
        sentences = articles[i][1]
        stemmed_sentences = stemmed_articles[i]
        stemmed_relations = article_stemmed_relations[i]
        relations = article_relations[i]
        relations_as_str = []
        for relation in relations:
            relations_as_str.append(" ".join(relation).strip())
        #Find the semantically most similar sentences for each relation in the article
        article_mat = sentence_model.encode_document(sentences, convert_to_tensor=True, device=device, normalize_embeddings=True)
        relations_as_str_mat = sentence_model.encode_query(relations_as_str, convert_to_tensor=True, device=device, normalize_embeddings=True)
        best_fit_sents = util.semantic_search(relations_as_str_mat, article_mat, top_k=5)
        missing_triples = []
        relation_probs = {}
        bert_scores = []
        for j in range(len(best_fit_sents)):
            cause,relation,effect = relations[j]
            ind_found = 0
            for k in range(len(best_fit_sents[j])):
                match = best_fit_sents[j][k]
                matched_sent = sentences[int(match['corpus_id'])]
                if relation in matched_sent:
                    ind_found = int(match['corpus_id'])
                    break
            matched_sent = sentences[ind_found]
            #Find words in the sentence
            relation_words= relation.split(" ")
            start_ind = matched_sent.find(relation)
            sent_enc = sentence_model.encode(matched_sent)
            stemmed_relation = " ".join(stemmed_relations[j])
            stemmed_start_ind = stemmed_sentences[ind_found].find(stemmed_relation)
            if start_ind != -1 or stemmed_start_ind != -1:
                masks = " ".join(["<mask>"]*len(relation_words))
                if start_ind != -1:
                    masked_sent = matched_sent[:start_ind] + masks + matched_sent[start_ind+len(relation):]
                else:
                    #Find the word stems and original words, mask out the words corresponding to the relation
                    matched_sent_as_list = matched_sent.split(" ")
                    stemmed_sent_as_list = stemmed_sentences[ind_found].split(" ")
                    stemmed_relation_as_list = stemmed_relation.split(" ")
                    relation_walker = 0
                    ent_start_ind = 0
                    for k in range(len(stemmed_sent_as_list)):
                        if stemmed_sent_as_list[k] == stemmed_relation_as_list[relation_walker]:
                            if relation_walker == 0:
                                ent_start_end = k
                            relation_walker+=1
                        elif relation_walker > 0:
                            relation_walker = 0
                        if relation_walker == len(stemmed_relation_as_list):
                            break
                    relation_start_ind = ent_start_ind+len(stemmed_relations[j][0].split(" "))
                    for k in range(len(masks)):
                        matched_sent_as_list[k+relation_start_ind] ="<mask>"
                    masked_sent = " ".join(matched_sent_as_list)
                predictions = bert_masked_prediction(bert_tokenizer, bert_model, masked_sent, device, top_k_per_mask=3)
                
                best_match_ind = 0
                match_score = 0
                best_match = predictions[0]
                for k in range(len(predictions)):
                    score, predicted, new_sent = predictions[k]
                    cur_match_score = sentence_model.similarity(sent_enc, sentence_model.encode(new_sent))
                    if cur_match_score > 0.85:
                        match_score+=cur_match_score.item()*score
                        
                    
                relation_probs[(cause,relation, effect)] = match_score
                bert_scores.append(match_score)                
            else:
                
                premises = []
                for match in best_fit_sents[j]:
                    premises.append(sentences[int(match["corpus_id"])])
                missing_triples.append([" ".join(premises), relations_as_str[j], (cause,relation,effect)])
                
        articles_missing_triples.append(missing_triples)
        articles_probs.append((article_name, relation_probs))
        # if len(bert_scores) > 0:
        #     print("BERT scores:", np.percentile(np.array(bert_scores), [0,10,20,30,40,50,60,70,80,90,100]))
        # else:
        #     print("NO BERT SCORES?")
    return articles_probs, articles_missing_triples


def get_mnli_probs(model_name, article_dir, premise_hypos_per_article, output_dir):
    """
    Input:
        model_name: name of the MNLI model to use
        article_dir: path to directory containing articles
        premise_hypos_per_article: 2nd return value of get_bert_probs
        output_dir: output directory (only used to write a file for debugging)
    Use an MNLI model to predict if the extracted relation is entailed by the surrounding context

    Returns:
        all_relation_to_scores: List of List of Dict
            Inner list represents an article and each element (Dict) maps (cause,relation,effect) to the match score
            as entailment probability as given by the MNLI model
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = AutoConfig.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name).to(device)
    model.eval()
    
    #Make actual ouptut dir based on model_name
    act_output_dir =output_dir+"/"+model_name 
    os.makedirs(act_output_dir, exist_ok=True)
    
    #Get article and relation files
    article_names = sorted(os.listdir(article_dir))
    #Find the index corresponding to entailment
    entail_ind = 0
    if "ENTAILMENT" in config.label2id:
        entail_ind = config.label2id["ENTAILMENT"]
    elif "entailment" in config.label2id:
        entail_ind = config.label2id['entailment']
    all_relation_to_scores = []
    all_scores = []
    for i in range(len(premise_hypos_per_article)):
        article_name = article_names[i]
        probabilities = []
        relation_and_label = []
        premise_and_hypos = premise_hypos_per_article[i]
        article_relation_to_scores = {}
        for premise, hypothesis, hypothesis_parts in premise_and_hypos:
            inputs = tokenizer(premise, hypothesis, return_tensors="pt", truncation=True).to(device)
            with torch.no_grad():
                outputs = model(**inputs)
                probs = F.softmax(outputs.logits, dim=-1).squeeze().cpu().numpy()
            
            probabilities.append(probs[entail_ind])
            relation_and_label.append([hypothesis, config.id2label[np.argmax(probs)], str(probs.max())])
            article_relation_to_scores[hypothesis_parts] =  probs[entail_ind]
        all_relation_to_scores.append((article_name, article_relation_to_scores))
        entailment_scores = np.array(probabilities)
        with open(act_output_dir+"/"+article_name, "a+") as f:
            for line in relation_and_label:
                f.write(" ".join(line)+"\n")
        
        all_scores = all_scores+probabilities
        # print("Stats for article", article_name)
        # print("Min entailment_score",np.min(entailment_scores))
        # print("Max entailment_score",np.max(entailment_scores))
        # print("Median entailment_score", np.median(entailment_scores))
        # print("Percentiles",np.percentile(entailment_scores, [0,10,20,30,40,50,60,70,80,90,100]))
    # all_entailment_scores = np.array(all_scores)
    # print("All stats")
    # print("Min entailment_score",np.min(all_entailment_scores))
    # print("Max entailment_score",np.max(all_entailment_scores))
    # print("Median entailment_score", np.median(all_entailment_scores))
    # print("Percentiles",np.percentile(all_entailment_scores, [0,10,20,30,40,50,60,70,80,90,100]))
    return all_relation_to_scores

def get_context_probs(articles_dir, relations_dir, output_dir):
    articles, article_relations, stemmed_articles, article_stemmed_relations = load_data(articles_dir, relations_dir)
    bert_probs, premise_hypos_per_article = get_bert_probs("FacebookAI/roberta-large", articles, article_relations, stemmed_articles, article_stemmed_relations)
    with open(output_dir+"bert_scores.pkl", "wb") as f:
        pickle.dump(bert_probs, f)
    with open(output_dir+"missing_triples.pkl", "wb") as f:
        pickle.dump(premise_hypos_per_article, f)
    mnli_probs = get_mnli_probs("microsoft/deberta-large-mnli", articles_dir, premise_hypos_per_article, output_dir)
    relation_probs_per_article = {}
    for i in range(len(bert_probs)):
        if bert_probs[i][0] != mnli_probs[i][0]:
            print(bert_probs[i][0], mnli_probs[i][0])
            print("article name and index mismatch!")
        relation_probs_per_article[bert_probs[i][0]] =  bert_probs[i][1] | mnli_probs[i][1]
    with open("context_scores.pkl", "wb") as f:
        pickle.dump(relation_probs_per_article,f)
    return relation_probs_per_article

if __name__ == "__main__":
    articles_dir = "bitcoin_docs/"
    relations_dir = "relations/"
    output_dir = "outputs/"
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--bert", action="store_true")
    parser.add_argument('--mnli', action="store_true")
    parser.add_argument('--all', action="store_true")
    args = parser.parse_args()
    # get_context_probs(articles_dir, relations_dir, output_dir)
    articles, article_relations, stemmed_articles, article_stemmed_relations = load_data(articles_dir, relations_dir)
    if args.all:
        os.makedirs(output_dir, exist_ok=True)
        get_context_probs(articles_dir, relations_dir, output_dir)
    elif args.bert:
        bert_probs, premise_hypos_per_article = get_bert_probs("FacebookAI/roberta-large", articles, article_relations, stemmed_articles, article_stemmed_relations)
        with open(output_dir+"bert_scores.pkl", "wb") as f:
            pickle.dump(bert_probs, f)
        with open(output_dir+"missing_triples.pkl", "wb") as f:
            pickle.dump(premise_hypos_per_article, f)
    else:
        with open('bert_outputs/bert_scores.pkl', 'rb') as f:
            bert_probs = pickle.load(f)
        with open('bert_outputs/missing_triples.pkl', 'rb') as f:
            premise_hypos_per_article = pickle.load(f)
       # bert_probs, premise_hypos_per_article = get_bert_probs("FacebookAI/roberta-large", articles, article_relations, stemmed_articles, article_stemmed_relations)
        mnli_probs = get_mnli_probs("microsoft/deberta-large-mnli", articles_dir, premise_hypos_per_article, output_dir)
        relation_probs_per_article = []
        for i in range(len(bert_probs)):
            if bert_probs[0] != mnli_probs[0]:
                print("article name and index mismatch!")
            relation_probs_per_article.append((bert_probs[0], bert_probs[i][1] | mnli_probs[i][1]))
        with open(output_dir+"context_scores.pkl", "wb") as f:
            pickle.dump(relation_probs_per_article, f)