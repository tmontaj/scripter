def best_and_worest_per_batch(x, y, model, npick, loss, pre_model=None, post_model=None):
  if pre_model:
    x = pre_model(x)
  
  logits = model(x)
  loss_  = loss(y, logits)

  if post_model:
    logits = post_model(logits)
  print("loss",loss_.shape)
  print("logits",logits.shape)
  print("y",y.shape)
  print("x",x.shape)
  loss_axis = np.argsort(loss_)

  all = zip(loss_[loss_axis], x[loss_axis], logits[loss_axis], y[loss_axis])

  n = npick if npick <= x.shape[0] else x.shape[0] 
  
  best   = loss_axis[:n]
  worest = np.flip(loss_axis[-n:])

  best   = (loss_[best], x[best], logits[best], y[best])
  worest = (loss_[worest], x[worest], logits[worest], y[worest])

  return best, worest

def mutual_best_and_worest(best_overall, worest_overall, best, worest, n):
  best_mutual = []
  worest_mutual = []

  for i in range(n):
    if best_overall[i][0] > best[i][0]:
      best_mutual.append(best_overall[i])
    else:
      best_mutual.append(best[i])
    
    if worest_overall[i][0] < worest[i][0]:
      worest_mutual.append(worest_overall[i])
    else:
      worest_mutual.append(worest[-i])

  return (best_mutual, worest_mutual)

def best_and_worest_per_repleca(dataset, model, npick, global_npick, loss, pre_model=None, post_model=None):
  
  best_overall=None
  worest_overall=None

  for x,y in dataset:
    n = global_npick if global_npick <= x.shape[0] else x.shape[0]
    best, worest = best_and_worest_per_batch(x, y, model, npick, loss, pre_model, post_model)

    # best_overall, worest_overall = mutual_best_and_worest(best_overall, 
    #                                                       worest_overall, best, 
    #                                                       worest, n) if best_overall else (best, worest)

    if best_overall:
      best_overall, worest_overall = mutual_best_and_worest(best_overall, 
                                                          worest_overall, best, 
                                                          worest, n)
    else:
      best_overall, worest_overall = (best, worest)

  return best_overall, worest_overall