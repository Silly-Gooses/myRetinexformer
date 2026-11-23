"""Exact seven-start notebook optimization and deterministic Train-only CV."""
import numpy as np
import pandas as pd
from scipy.optimize import minimize, minimize_scalar
from .config import SPLIT_SEED, CV_FOLDS, LAMBDA_GRID, REGULARIZATION_CENTER
from .io_utils import locked_json

def rgb_score(df,a):
    ar=df.rgb_a.to_numpy(float); br=df.rgb_b.to_numpy(float); cr=df.rgb_c.to_numpy(float)
    mse=np.maximum(ar*a*a+2*br*a+cr,1e-14)
    return float(np.mean(-10*np.log10(mse)))

def ycc_score(df,ay,ac):
    a=df.Ayy.to_numpy(float);b=df.Acc.to_numpy(float);c=df.Ayc.to_numpy(float)
    d=df.By.to_numpy(float);e=df.Bc.to_numpy(float);f=df.C.to_numpy(float)
    mse=np.maximum(a*ay*ay+b*ac*ac+2*c*ay*ac+2*d*ay+2*e*ac+f,1e-14)
    return float(np.mean(-10*np.log10(mse)))

def fit_rgb(df):
    result=minimize_scalar(lambda a:-rgb_score(df,float(a)),bounds=(0,1),method='bounded',options={'xatol':1e-7})
    choices=[0.,1.,float(result.x)]
    return float(max(choices,key=lambda a:rgb_score(df,a)))

def fit_ycc(tables,lam=0.,center=0.5):
    # tables: one dataset for local, or multiple datasets for macro-balanced joint fit
    assert len(tables)>=1
    def objective(x):
        ay,ac=x
        avg=float(np.mean([ycc_score(table,ay,ac) for table in tables]))
        return -avg + lam*((ay-center)**2+(ac-center)**2)
    starts=[(0,0),(1,1),(0,1),(1,0),(.5,.5),(.2,.8),(.8,.2)]
    candidates=starts.copy()
    for st in starts:
        result=minimize(objective,np.array(st,dtype=float),bounds=[(0,1),(0,1)],method='L-BFGS-B',
                        options={'maxiter':220,'ftol':1e-12})
        candidates.append(tuple(np.clip(result.x,0.,1.)))
    return tuple(map(float,min(candidates,key=objective)))

def fold_labels(ds,df):
    rng=np.random.default_rng(SPLIT_SEED+sum(map(ord,ds)))
    ordering=rng.permutation(len(df))
    ids=np.empty(len(df),dtype=np.int32)
    ids[ordering]=np.arange(len(df))%CV_FOLDS
    return ids

def chosen_lambda(cv,scope,ds=None):
    v=cv[cv.scope==scope]
    if ds is not None:v=v[v.dataset==ds]
    candidate=v.groupby('lam').holdout_psnr.mean()
    best=float(candidate.max())
    # Within 0.005 dB, choose stronger regularization (conservative, train-only tie-break).
    return float(max([float(lam) for lam,val in candidate.items() if val>=best-0.005]))

def fit_tables(tables, output):
    """Fit every ablation from Train coefficient tables and freeze the result."""
    output.mkdir(parents=True, exist_ok=True)
    if not tables or any(len(t) < CV_FOLDS for t in tables.values()):
        raise ValueError("Each dataset requires at least three Train rows")
    FOLDS={ds:fold_labels(ds,tables[ds]) for ds in list(tables)}
    CV_ROWS=[]
    for ds in list(tables):
        df=tables[ds];folds=FOLDS[ds]
        for lam in LAMBDA_GRID:
            for fold in range(CV_FOLDS):
                tr=df.iloc[np.flatnonzero(folds!=fold)]
                va=df.iloc[np.flatnonzero(folds==fold)]
                ay,ac=fit_ycc([tr],lam,REGULARIZATION_CENTER)
                CV_ROWS.append(dict(scope='dataset',dataset=ds,lam=lam,fold=fold,n_train=len(tr),n_holdout=len(va),
                                    alpha_y=ay,alpha_c=ac,holdout_psnr=ycc_score(va,ay,ac)))
    # Shared-model cross-validation: hold out corresponding folds from EACH dataset.
    for lam in LAMBDA_GRID:
        for fold in range(CV_FOLDS):
            tr=[tables[ds].iloc[np.flatnonzero(FOLDS[ds]!=fold)] for ds in list(tables)]
            ay,ac=fit_ycc(tr,lam,REGULARIZATION_CENTER)
            for ds in list(tables):
                va=tables[ds].iloc[np.flatnonzero(FOLDS[ds]==fold)]
                CV_ROWS.append(dict(scope='joint',dataset=ds,lam=lam,fold=fold,n_train=sum(len(x) for x in tr),
                                    n_holdout=len(va),alpha_y=ay,alpha_c=ac,holdout_psnr=ycc_score(va,ay,ac)))
    cv=pd.DataFrame(CV_ROWS)
    cv.to_csv(output/'train_cross_validation_folds.csv',index=False)

    # Lambda selected on mean of per-fold holdout PSNR. For joint equal-weight datasets via same folds/count.
    cv_summary=cv.groupby(['scope','dataset','lam'],as_index=False).holdout_psnr.mean()
    cv_summary.to_csv(output/'train_cross_validation_summary.csv',index=False)

    LOCAL_PARAMS={}
    for ds,df in tables.items():
        l=chosen_lambda(cv,'dataset',ds)
        ay,ac=fit_ycc([df],0.,REGULARIZATION_CENTER)
        ry,rc=fit_ycc([df],l,REGULARIZATION_CENTER)
        LOCAL_PARAMS[ds]=dict(alpha_rgb=fit_rgb(df),alpha_y=ay,alpha_c=ac,
                              alpha_y_regularized=ry,alpha_c_regularized=rc,lambda_regularized=l,
                              n_fit=len(df))
    lam_joint=chosen_lambda(cv,'joint')
    joint_y,joint_c=fit_ycc(list(tables.values()),0.,REGULARIZATION_CENTER)
    joint_reg_y,joint_reg_c=fit_ycc(list(tables.values()),lam_joint,REGULARIZATION_CENTER)
    JOINT_PARAMS=dict(alpha_y=joint_y,alpha_c=joint_c,alpha_y_regularized=joint_reg_y,
                      alpha_c_regularized=joint_reg_c,lambda_regularized=lam_joint)
    params={'experiment':'regjoint_v2','local':LOCAL_PARAMS,'joint':JOINT_PARAMS,'lambda_grid':LAMBDA_GRID,
            'selection_rule':'3fold Train-only CV; neutral prior center=.5; mean fold PSNR; stronger lambda on ties within .005dB'}
    locked_json(output / "selected_weights_train_only.json", params)
    return params
