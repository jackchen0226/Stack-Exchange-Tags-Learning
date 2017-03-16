def partialf1(tp, fp, fn, dtp, dfp, dfn):
    p = tp/(tp+fp)
    r = tp/(tp+fn)
    dr = (dtp/(tp+fn))-((1/((tp+fn)**2))*(dtp+dfn)*tp)
    dp = (dtp/(tp+fp))-((1/((tp+fp)**2))*(dtp+dfp)*tp)

    dpf = (2 * (p**2))/((p+r)**2) * dr
    drf = (2 * (r**2))/((p+r)**2) * dp
    return dpf+drf

partialf1(2000.0, 345.0, 1200.0, 0.0, 0.0, 0.0)