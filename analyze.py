import argparse, json, re, csv
from pathlib import Path
from collections import Counter
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer, ENGLISH_STOP_WORDS
from sklearn.decomposition import LatentDirichletAllocation, NMF

SEED=42
STOP = sorted(set(ENGLISH_STOP_WORDS)-{'not','no','never'})

def clean(s):
    s = re.sub(r"<[^>]+>", " ", str(s))
    s = re.sub(r"&[a-zA-Z]+;", " ", s)
    s = re.sub(r"https?://\S+|www\.\S+", " ", s)
    s = s.lower()
    s = re.sub(r"[^a-z\s']", " ", s)
    s = re.sub(r"\bbr\b", " ", s)
    return re.sub(r"\s+", " ", s).strip()
def main():
    p=argparse.ArgumentParser(); p.add_argument('--input',default='All_Beauty.jsonl');p.add_argument('--output',default='results');p.add_argument('--sample',type=int,default=10000); a=p.parse_args()
    out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
    counts=Counter(); eligible=[]; seen=set(); stars=Counter()
    with open(a.input,encoding='utf-8') as f:
        for line in f:
            r=json.loads(line);counts['total']+=1
            rating=r.get('rating'); stars[str(rating)]+=1
            raw=r.get('text') or ''
            if not raw.strip():counts['empty_text']+=1
            if rating not in (1,1.0,2,2.0):continue
            counts['low_rated']+=1
            if not raw.strip():continue
            if len(raw.split())<10:counts['low_rated_short']+=1;continue
            counts['eligible']+=1
            normalized=re.sub(r'\s+',' ',raw).strip().casefold()
            if normalized in seen:counts['duplicate_eligible_text']+=1;continue
            seen.add(normalized)
            eligible.append({'rating':rating,'text':raw,'parent_asin':r.get('parent_asin',''),'timestamp':r.get('timestamp')})
    counts['unique_eligible']=len(eligible)
    rng=np.random.default_rng(SEED);idx=rng.choice(len(eligible),size=min(a.sample,len(eligible)),replace=False)
    sample=[eligible[i] for i in idx]
    df=pd.DataFrame(sample);df['clean_text']=df['text'].map(clean)
    df.to_csv(out/'sample_metadata.csv',index=False)
    texts=df.clean_text.tolist()
    cv=CountVectorizer(stop_words=STOP,max_df=.90,min_df=5,max_features=5000,ngram_range=(1,2))
    tf=TfidfVectorizer(stop_words=STOP,max_df=.90,min_df=5,max_features=5000,ngram_range=(1,2))
    X=cv.fit_transform(texts);Y=tf.fit_transform(texts)
    lda=LatentDirichletAllocation(n_components=8,random_state=SEED,max_iter=10,learning_method='batch',n_jobs=-1)
    nmf=NMF(n_components=8,random_state=SEED,init='nndsvda',max_iter=250)
    L=lda.fit_transform(X);N=nmf.fit_transform(Y)
    rows=[]
    for name,model,vec,weights in [('LDA',lda,cv,L),('NMF',nmf,tf,N)]:
        terms=np.array(vec.get_feature_names_out());labels=weights.argmax(axis=1)
        for i,comp in enumerate(model.components_):
            words=terms[np.argsort(comp)[-12:][::-1]].tolist()
            ids=np.where(labels==i)[0]
            top=ids[np.argsort(weights[ids,i])[-3:][::-1]] if len(ids) else []
            rows.append({'model':name,'topic':i+1,'count':len(ids),'percent':round(100*len(ids)/len(df),2),'top_terms':', '.join(words),'example_reviews':' | '.join(df.iloc[j]['text'][:230].replace('\n',' ') for j in top)})
    pd.DataFrame(rows).to_csv(out/'topics.csv',index=False)
    result={'counts':dict(counts),'ratings':dict(stars),'sample_size':len(df),'random_seed':SEED,'vocabulary_count':len(cv.vocabulary_),'vocabulary_tfidf':len(tf.vocabulary_),'count_matrix_shape':X.shape,'tfidf_matrix_shape':Y.shape,'lda_perplexity':float(lda.perplexity(X)),'nmf_reconstruction_error':float(nmf.reconstruction_err_),'note':'Model-specific metrics are not directly comparable; manually inspect topics and representative documents.'}
    (out/'summary.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result,indent=2))
if __name__=='__main__':main()
