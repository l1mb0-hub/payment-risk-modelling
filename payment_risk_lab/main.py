"""Payment fraud detection with chronological holdout, economic threshold and shift checks."""
import argparse,json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.special import expit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score,average_precision_score,brier_score_loss,precision_recall_curve

FEATURES=['amount_log','cross_border','merchant_risk','device_age','velocity','late_night']

def simulate(n=40000,seed=42):
    if n<2000:raise ValueError('n>=2000')
    r=np.random.default_rng(seed)
    day=r.integers(0,120,n);amount=np.clip(r.lognormal(3.7,.95,n),1,2000)
    cross=r.binomial(1,.22,n);merchant=r.beta(2,6,n)
    age=r.exponential(120,n);velocity=r.poisson(1.7+.6*cross,n)
    night=r.binomial(1,.18,n)
    logit=-5.5+.47*np.log1p(amount)+1.1*cross+2.0*merchant-.007*age+.25*velocity+.35*night+.8*(day>=80)
    label=r.binomial(1,expit(logit))
    return pd.DataFrame({'transaction_id':np.arange(n),'day':day,'amount_log':np.log1p(amount),
                         'cross_border':cross,'merchant_risk':merchant,'device_age':age,'velocity':velocity,
                         'late_night':night,'is_fraud':label})

def cost(y,p,threshold,miss_cost=150,alert_cost=4):
    """Idealized model: every alert costs €4 and blocks fraud perfectly."""
    a=np.asarray(p)>=threshold;y=np.asarray(y)
    return float(np.mean(alert_cost*a+miss_cost*((y==1)&(~a))))

def psi(reference,current):
    edges=np.unique(np.quantile(reference,np.linspace(0,1,11)))
    if len(edges)<3:return 0.
    edges[0]=-np.inf;edges[-1]=np.inf
    a=np.maximum(np.histogram(reference,bins=edges)[0]/len(reference),1e-6)
    b=np.maximum(np.histogram(current,bins=edges)[0]/len(current),1e-6)
    return float(np.sum((b-a)*np.log(b/a)))

def run(output,n=40000,seed=42):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    d=simulate(n,seed);d.to_csv(output/'synthetic_payments.csv',index=False)
    train=d[d.day<70];val=d[(d.day>=70)&(d.day<90)];test=d[d.day>=90]
    model=make_pipeline(StandardScaler(),LogisticRegression(max_iter=500))
    model.fit(train[FEATURES],train.is_fraud)
    pv=model.predict_proba(val[FEATURES])[:,1]

    feasible=[t for t in np.linspace(.01,.99,99) if float((pv>=t).mean()) <= .10]
    threshold=float(min(feasible,key=lambda t:cost(val.is_fraud,pv,t)))

    pt=model.predict_proba(test[FEATURES])[:,1]
    result={'synthetic':True,'n':n,'split':{'train':len(train),'validation':len(val),'test':len(test)},
            'threshold_selected_on_validation':threshold,'maximum_validation_alert_rate':0.10,
            'test':{'fraud_rate':float(test.is_fraud.mean()),'roc_auc':float(roc_auc_score(test.is_fraud,pt)),
                    'average_precision':float(average_precision_score(test.is_fraud,pt)),
                    'brier':float(brier_score_loss(test.is_fraud,pt)),
                    'cost_per_transaction_eur':cost(test.is_fraud,pt,threshold),
                    'always_approve_cost_eur':float(150*test.is_fraud.mean()),
                    'alert_rate':float((pt>=threshold).mean())},
            'drift':{'fraud_rate_train':float(train.is_fraud.mean()),
                     'fraud_rate_test':float(test.is_fraud.mean()),
                     'merchant_risk_psi':psi(train.merchant_risk.to_numpy(),test.merchant_risk.to_numpy())}}
    (output/'results.json').write_text(json.dumps(result,indent=2))
    precision,recall,_=precision_recall_curve(test.is_fraud,pt)
    fig,ax=plt.subplots(figsize=(7,4));ax.plot(recall,precision,label='Logistic regression')
    ax.axhline(test.is_fraud.mean(),ls='--',label='Random baseline')
    ax.set(xlabel='Recall',ylabel='Precision',title='Future time-window performance',xlim=(0,1),ylim=(0,1))
    ax.legend();fig.tight_layout();fig.savefig(output/'precision_recall.png',dpi=160);plt.close(fig)
    print(json.dumps(result,indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',default='payment_risk_lab/outputs');p.add_argument('--seed',type=int,default=42)
    p.add_argument('--n',type=int,default=40000);a=p.parse_args();run(a.output,a.n,a.seed)
