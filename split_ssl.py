import argparse
import random

if __name__ == '__main__' :
    parser = argparse.ArgumentParser()
    parser.add_argument('-file', type = str, required = True)
    args = parser.parse_args()
    ds = {}
    with open(args.file) as f :
        for line in f :
            vals = line.strip().split('\t')
            if not vals[1] in ds :
                ds[vals[1]] = []
                
            ds[vals[1]].append(vals[0]) 
    
    ftrain = open(args.file+'_train.txt', 'w')
    ftest  = open(args.file+'_test.txt', 'w')
            
    for cl in ds :        
        random.shuffle(ds[cl])
        lst = ds[cl]                
        for item in lst[:1000]:            
            ftrain.write("%s\n" % item)
        for item in lst[-100:]:            
            ftest.write("%s\n" % item)
    ftrain.close()
    ftest.close()
    print('Done')