import csv

def f1_score(tp, fp, fn):
    p = (tp*1.) / (tp+fp)
    r = (tp*1.) / (tp+fn)
    f1 = (2*p*r)/(p+r)
    return f1

if __name__ == '__main__':
	
	f1_score()