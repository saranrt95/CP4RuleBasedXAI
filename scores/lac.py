
def LAC_score(model, X, y):
    pr = model.predict_proba(X)
    #print(pr)
    if y == 0:
        score = 1 - pr[:, 0]
    elif y == 1:
        score = 1 - pr[:, 1]
    return score 