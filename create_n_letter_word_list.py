import sys
import os

input_file = sys.argv[1]

if os.path.isfile(input_file):
    with open(input_file, 'r', encoding="utf8", errors='ignore') as f0:
        for line in f0.readlines():
            line = line.rstrip()
            wdlen = len(line)
            if wdlen > 3 and wdlen < 12:
                with open(input_file[:2]+'_phon_'+'_len'+str(wdlen)+'.txt', 'a') as f1:
                    f1.write(line+'\n')
