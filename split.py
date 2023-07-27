import argparse
import random

if __name__ == '__main__' :
    parser = argparse.ArgumentParser()
    parser.add_argument('-file', type = str, required = True)
    parser.add_argument('-n_cat', type = int, required = True)
    args = parser.parse_args()
    n_cat = args.n_cat
    ds = {}
    with open(args.file) as f :
        for line in f :
            vals = line.strip().split('\t')
            if not vals[1] in ds :
                ds[vals[1]] = []
                
            ds[vals[1]].append(vals[0]) 
    
    
    ftrain = open(args.file+'_train.txt', 'w')
    ftest_k  = open(args.file+'_test_known.txt', 'w')
    ftest_u  = open(args.file+'_test_unknown.txt', 'w')
    
    keys = ds.keys()
    random.shuffle(keys)
    keys = keys[:n_cat]
    keys_u = keys[-n_cat:]        
    for cl in keys :        
        random.shuffle(ds[cl])
        lst = ds[cl]                
        for item in lst[:1000]:            
            ftrain.write('{}\n'.format(item))
        for item in lst[-1000:]:            
            ftest_k.write('{}\n'.format(item))
    for cl in keys_u:        
        random.shuffle(ds[cl])
        lst = ds[cl]                        
        for item in lst[-1000:]:            
            ftest_u.write('{}\n'.format(item))
    ftrain.close()
    ftest_k.close()
    ftest_u.close()
    print('Done')