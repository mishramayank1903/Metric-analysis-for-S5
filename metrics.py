import torch
import itertools
import numpy as np


#################### CA‑SDR, error‑wise ####################

def _casdr_error_wise(est_dict, ref_dict, mixture, metricfunc):

    tp_lb = set(est_dict.keys()) & set(ref_dict.keys()) - {"silence"}
    fp_lb = set(est_dict.keys()) - tp_lb - {"silence"}
    fn_lb = set(ref_dict.keys()) - tp_lb - {"silence"}

    result = torch.tensor([], dtype=mixture.dtype, device=mixture.device)
    if tp_lb:
        tp_est_wf = torch.stack([est_dict[lb] for lb in tp_lb])  # [len(tp), T]
        tp_ref_wf = torch.stack([ref_dict[lb] for lb in tp_lb])  # [len(tp), T]
        repeated_mixture = mixture.unsqueeze(0).repeat(len(tp_lb), 1)
        score_est = metricfunc(tp_est_wf, tp_ref_wf)
        score_mix = metricfunc(repeated_mixture, tp_ref_wf)
        improvement = score_est - score_mix
        if improvement.ndim == 0:
            improvement = improvement.unsqueeze(0)
        result = torch.cat((result, score_est), dim=0)

    pens =torch.zeros(len(fp_lb) + len(fn_lb))
    result = torch.cat((result, pens), dim=0)    
    return result.mean().item(), tp_lb, fn_lb, fp_lb


#################### CA‑SDR, source-wise ####################

def _casdr_source_wise(est_dict, ref_dict, mixture, metricfunc):
    est_lb = list(est_dict.keys())
    ref_lb = list(ref_dict.keys())

    tp_lb = set(est_lb) & set(ref_lb) - {"silence"}
    fp_lb = set(est_lb) - tp_lb - {"silence"}
    fn_lb = set(ref_lb) - tp_lb - {"silence"}

    result = torch.tensor([], dtype=mixture.dtype, device=mixture.device)
    for lb in tp_lb:
        score_est = metricfunc(est_dict[lb], ref_dict[lb])
        score_mix = metricfunc(mixture, ref_dict[lb])
        improvement = score_est - score_mix
        if improvement.ndim == 0:
            improvement = improvement.unsqueeze(0)
        result = torch.cat((result, improvement), dim=0)

    pens = torch.zeros(len(fp_lb) + len(fn_lb), dtype=mixture.dtype,
                       device=mixture.device)
    result = torch.cat((result, pens), dim=0)
    mean_result = torch.sum(result) / len(ref_dict)
    return mean_result, tp_lb, fn_lb, fp_lb


#################### CASA‑SDR (error-wise, source‑wise, tp only) ####################

def _pad_dicts(est_dict, ref_dict):
    est_dict = dict(est_dict)
    ref_dict = dict(ref_dict)
    sample_shape = next(iter(est_dict.values())).shape if est_dict else torch.Size([1])

    while len(est_dict) < len(ref_dict):
        est_dict[f"missing_label_{len(est_dict)}"] = torch.zeros(sample_shape)
    while len(ref_dict) < len(est_dict):
        ref_dict[f"missing_label_{len(ref_dict)}"] = torch.zeros(sample_shape)

    assert len(est_dict) == len(ref_dict)
    return est_dict, ref_dict


def _best_perm(est_dict, ref_dict, metricfunc):
    est_dict, ref_dict = _pad_dicts(est_dict, ref_dict)
    est_items = list(est_dict.items())
    ref_values = list(ref_dict.values())

    best_metric_value = -np.inf
    best_perm = None
    for perm in itertools.permutations(est_items):
        perm_vals = [v for (_, v) in perm]
        metric_value = torch.tensor(0.0)
        for i in range(len(perm_vals)):
            metric_value = metric_value + metricfunc(perm_vals[i], ref_values[i])
        if metric_value.item() > best_metric_value:
            best_metric_value = metric_value.item()
            best_perm = perm
    return best_perm, list(ref_dict.keys()), list(ref_dict.values())


def _casasdr_error_wise(est_dict, ref_dict, mixture, metricfunc):
    best_perm, ref_keys, _ = _best_perm(est_dict, ref_dict, metricfunc)
    tp_lb, fp_lb, fn_lb = [], [], []
    result = []

    perm_keys = [k for (k, _) in best_perm]

    mismatches = set()
    for i, (k1, k2) in enumerate(zip(perm_keys, ref_keys)):
        if k1 == k2:
            improvement = metricfunc(best_perm[i][1], ref_dict[k2]) - \
                          metricfunc(mixture, ref_dict[k2])
            tp_lb.append(k1)
            result.append(improvement)
        elif k1.startswith("missing_label"):
            fn_lb.append(k2)
            result.append(torch.tensor(0.0))
        elif k2.startswith("missing_label"):
            fp_lb.append(k1)
        else:
            fp_lb.append(k1)
            fn_lb.append(k2)
            mismatches.add((k2, k1))
            result.append(torch.tensor(0.0))
    swap = 0
    for (a,b) in mismatches:
        if (b,a) in mismatches and a != b:
            swap += 1
    swap = swap // 2 # each swap is counted twice in the above loop
    if swap > 0:
        print("est keys:", est_dict.keys())
        print("ref keys:", ref_dict.keys())
        print("mismatches:", mismatches)


    agg_factor = len(tp_lb)+len(fp_lb)+len(fn_lb)
    
    metric_mean_value = torch.sum(torch.stack(result)) / agg_factor
    return metric_mean_value, tp_lb, fn_lb, fp_lb, swap


def _casasdr_source_wise(est_dict, ref_dict, mixture, metricfunc):
    best_perm, ref_keys, _ = _best_perm(est_dict, ref_dict, metricfunc)
    tp_lb, fp_lb, fn_lb = [], [], []
    result = []

    perm_keys = [k for (k, _) in best_perm]

    mismatches = set()
    for i, (k1, k2) in enumerate(zip(perm_keys, ref_keys)):
        if k1 == k2:
            improvement = metricfunc(best_perm[i][1], ref_dict[k2]) 
            tp_lb.append(k1)
            result.append(improvement)
        elif k1.startswith("missing_label"):
            fn_lb.append(k2)
            result.append(torch.tensor(0.0))
        elif k2.startswith("missing_label"):
            fp_lb.append(k1)
        else:
            fp_lb.append(k1)
            fn_lb.append(k2)
            mismatches.add((k2, k1))
            result.append(torch.tensor(0.0))
    metric_mean_value = torch.sum(torch.stack(result)) / len(ref_dict)
    return metric_mean_value, tp_lb, fn_lb, fp_lb


def _casasdr_tp(est_dict, ref_dict, mixture, metricfunc):
    best_perm, ref_keys, _ = _best_perm(est_dict, ref_dict, metricfunc)
    tp_lb, fp_lb, fn_lb = [], [], []
    result = []

    perm_keys = [k for (k, _) in best_perm]

    mismatches = set()
    for i, (k1, k2) in enumerate(zip(perm_keys, ref_keys)):
        if k1 == k2:
            improvement = metricfunc(best_perm[i][1], ref_dict[k2]) 
            # - \
            #               metricfunc(mixture, ref_dict[k2])
            tp_lb.append(k1)
            result.append(improvement)
        elif k1.startswith("missing_label"):
            fn_lb.append(k2)
        elif k2.startswith("missing_label"):
            fp_lb.append(k1)
        else:
            fp_lb.append(k1)
            fn_lb.append(k2)
    
    metric_mean_value = torch.sum(torch.stack(result)) / len(tp_lb) if len(tp_lb) > 0 else torch.tensor(0.0)
    return metric_mean_value, tp_lb, fn_lb, fp_lb



#################### classical SDRi (permutation‑invariant) ####################

def _classical_sdr(est_dict, ref_dict, mixture, metricfunc):
    est_dict, ref_dict = _pad_dicts(est_dict, ref_dict)
    est_items = list(est_dict.items())
    ref_values = list(ref_dict.values())

    best_metric_value = -np.inf
    for perm in itertools.permutations(est_items):
        perm_vals = [v for (_, v) in perm]
        est_stack = torch.stack(perm_vals)
        ref_stack = torch.stack(ref_values)
        mix_stack = mixture.unsqueeze(0).repeat(len(ref_values), 1)
        improvement = metricfunc(est_stack, ref_stack) 
        if improvement.item() > best_metric_value:
            best_metric_value = improvement.item()
    return best_metric_value

#################### Original CA‑SDR on true positives ####################

def _casdr_tp(est_dict, ref_dict, mixture, metricfunc):

    tp_lb = set(est_dict.keys()) & set(ref_dict.keys()) - {"silence"}
    fp_lb = set(est_dict.keys()) - tp_lb - {"silence"}
    fn_lb = set(ref_dict.keys()) - tp_lb - {"silence"}

    result = torch.tensor([], dtype=mixture.dtype, device=mixture.device)
    if tp_lb:
        tp_est_wf = torch.stack([est_dict[lb] for lb in tp_lb])  # [len(tp), T]
        tp_ref_wf = torch.stack([ref_dict[lb] for lb in tp_lb])  # [len(tp), T]
        repeated_mixture = mixture.unsqueeze(0).repeat(len(tp_lb), 1)
        score_est = metricfunc(tp_est_wf, tp_ref_wf)
        score_mix = metricfunc(repeated_mixture, tp_ref_wf)
        improvement = score_est - score_mix
        if improvement.ndim == 0:
            improvement = improvement.unsqueeze(0)
        result = torch.cat((result, improvement), dim=0)    
    else:
        result = torch.tensor(0.0, dtype=mixture.dtype, device=mixture.device)

    return result.mean().item(), tp_lb, fn_lb, fp_lb


#################### Unified metric ####################

def unified_metric(
    metric_type,
    mode,
    metricfunc,
    mixture,
    est_dict=None,
    ref_dict=None
):
    """
    metric_type: 'casdr', 'casasdr', or 'classical'
    mode: 'error_wise' or 'source_wise' or 'tp_only' (ignored for 'classical')
    Returns:
        For casdr/casasdr:
          (metric_value, n_tp, n_fn, n_fp)
        For classical:
          scalar metric_value
    """

    if metric_type == "classical":
        if est_dict is None or ref_dict is None:
            raise ValueError("classical mode expects est_dict/ref_dict.")
        return _classical_sdr(est_dict, ref_dict, mixture, metricfunc)

    if metric_type == "casdr":
        if est_dict is None or ref_dict is None:
            raise ValueError("casdr expects est_dict/ref_dict.")
        if mode == "error_wise":
            return _casdr_error_wise(est_dict, ref_dict, mixture, metricfunc)
        elif mode == "source_wise":
            return _casdr_source_wise(est_dict, ref_dict, mixture, metricfunc)
        elif mode == "tp_only":
            return _casdr_tp(est_dict, ref_dict, mixture, metricfunc)
        else:
            raise ValueError(f"Unknown mode: {mode}")

    if metric_type == "casasdr":
        if est_dict is None or ref_dict is None:
            raise ValueError("casasdr expects est_dict/ref_dict.")
        if mode == "error_wise":
            return _casasdr_error_wise(est_dict, ref_dict, mixture, metricfunc)
        elif mode == "source_wise":
            return _casasdr_source_wise(est_dict, ref_dict, mixture, metricfunc)
        elif mode == "tp_only":
            return _casasdr_tp(est_dict, ref_dict, mixture, metricfunc)
        else:
            raise ValueError(f"Unknown mode: {mode}")

    raise ValueError(f"Unknown metric_type: {metric_type}")
