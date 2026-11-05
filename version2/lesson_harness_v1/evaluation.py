"""Matched endpoints: human accuracy and synthetic agreement are never pooled."""
import numpy as np

from . import LEVELS,QUALITIES
from .training import balanced_weights


def supported_test_rows(train,test,observations):
    fitted={observations[i].label.quality for i in train}
    supported=[i for i in test if observations[i].label.quality in fitted]
    excluded=[{'observation_index':i,'quality':observations[i].label.quality,
               'reason':'no_training_labels_for_head'} for i in test if observations[i].label.quality not in fitted]
    return supported,excluded


def validate_probabilities(value,rows):
    p=np.asarray(value,float)
    if p.shape!=(rows,5) or not np.isfinite(p).all() or (p<0).any() or (p>1).any() or not np.allclose(p.sum(1),1,atol=1e-6,rtol=0):
        raise ValueError('matched finite ordinal distributions required')
    return p


def cluster_interval(delta,weights,people,lessons,replicates=2000):
    """Descriptive crossed resampling; not an independent-window confidence claim."""
    people=np.asarray(people);lessons=np.asarray(lessons)
    unique_people=np.unique(people);unique_lessons=np.unique(lessons)
    if len(unique_people)<2 or len(unique_lessons)<2:
        return {'interval':None,'reason':'fewer_than_two_people_or_lessons','replicates':0}
    rng=np.random.default_rng(1729);bootstrap=[]
    for _ in range(replicates):
        sampled_people=rng.choice(unique_people,len(unique_people),replace=True)
        sampled_lessons=rng.choice(unique_lessons,len(unique_lessons),replace=True)
        w=weights*np.array([(sampled_people==p).sum()*(sampled_lessons==l).sum() for p,l in zip(people,lessons)])
        if w.sum()>0:bootstrap.append(float(w@delta/w.sum()))
    return {'interval':np.quantile(bootstrap,[.025,.975]).tolist(),'reason':None,'replicates':len(bootstrap)}


def endpoints(observations,windows,indices,forecasts,*,replicates=2000):
    ids=np.asarray(indices,int)
    if ids.ndim!=1 or len(set(ids.tolist()))!=len(ids) or (ids<0).any() or (ids>=len(observations)).any():
        raise ValueError('unique supported observation indices required')
    probabilities={arm:validate_probabilities(p,len(ids)) for arm,p in forecasts.items()}
    required={'mean','content_ridge','content_network','content_tribe_network','full_eeg_informed','training_eeg_reassigned'}
    if set(probabilities)!=required:raise ValueError('all six prespecified arms required')
    result={}
    for quality in QUALITIES:
        result[quality]={}
        for origin in ['human','synthetic']:
            ix=[j for j,i in enumerate(ids) if observations[i].label.quality==quality and observations[i].label.origin==origin]
            total=sum(o.label.quality==quality and o.label.origin==origin for o in observations)
            interpretation='human_label_accuracy' if origin=='human' else 'synthetic_teacher_agreement_not_human_accuracy'
            if not ix:
                result[quality][origin]={'status':'not_evaluated','total_labels':total,'evaluated_labels':0,
                    'reason':'no_labels' if not total else 'no_training_supported_heldout_predictions','interpretation':interpretation,
                    'scores':None,'comparisons':None};continue
            obs=[observations[int(ids[j])] for j in ix];weights=balanced_weights(obs)
            y=np.array([o.label.distribution for o in obs]);scores={};errors={}
            for arm,p in probabilities.items():
                values=p[ix];errors[arm]=abs(values@LEVELS-y@LEVELS)
                scores[arm]={'MAE_0_10':float(weights@errors[arm]),
                    'ordinal_distribution_Brier':float(weights@((values-y)**2).sum(1))}
            comparisons={}
            people=[o.label.person for o in obs];lessons=[windows[o.window_indices[0]]['lesson'] for o in obs]
            for name,candidate,comparator in [
                ('additional_EEG_supervision','full_eeg_informed','content_tribe_network'),
                ('training_mean','full_eeg_informed','mean'),
                ('EEG_reassignment_control','full_eeg_informed','training_eeg_reassigned')]:
                baseline=scores[comparator]['MAE_0_10'];reduction=baseline-scores[candidate]['MAE_0_10']
                comparisons[name]={'candidate':candidate,'comparator':comparator,'MAE_reduction':reduction,
                    'relative_reduction_percent':100*reduction/baseline if baseline>0 else None,
                    'descriptive_crossed_cluster_95_percent':cluster_interval(errors[comparator]-errors[candidate],weights,people,lessons,replicates)}
            result[quality][origin]={'status':'evaluated','total_labels':total,'evaluated_labels':len(obs),
                'excluded_labels':total-len(obs),'participants':len(set(people)),'lessons':len(set(lessons)),
                'interpretation':interpretation,'scores':scores,'comparisons':comparisons}
    return result
