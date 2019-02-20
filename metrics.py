import numpy as np


def pointwise(pds, gts):
  return np.mean(np.argmax(pds, axis=1) == np.argmax(gts, axis=1))


def mae(pds, gts):
  return np.mean(np.abs(pds[:, 1] - gts[:, 1]))


def pairwise(pds, gts, factors):
  # sort by factor
  pd_by_factor = {}
  for pd, gt, factor in zip(pds, gts, factors):
    factor_pd = pd_by_factor.setdefault(factor, {'pos': [], 'neg': []})
    if np.argmax(gt) == 1:
      factor_pd['pos'].append(pd[1])
    else:
      factor_pd['neg'].append(pd[1])

  count = 0
  for factor, pd_probs in pd_by_factor.items():
    for pos_prob, neg_prob in zip(pd_probs['pos'], pd_probs['neg']):
      if pos_prob > neg_prob:
        count += 2
      elif pos_prob == neg_prob: # break-even
        count += 1

  return count / len(gts)
