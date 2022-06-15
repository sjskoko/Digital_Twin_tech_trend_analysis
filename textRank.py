import numpy as np
import gensim
from urllib.request import urlretrieve, urlopen
import gzip
import zipfile


urlretrieve("http://nlp.stanford.edu/data/glove.6B.zip", filename="glove.6B.zip")
zf = zipfile.ZipFile('glove.6B.zip')
zf.extractall() 
zf.close()