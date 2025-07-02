#!/usr/bin/python
import os
from urllib.request import urlretrieve
import gzip
import pandas as pd
import numpy as np
import anndata as ad



outdir = '../Data/'


# Download expression data

if not os.path.exists(outdir):
    os.makedirs(outdir)

url="https://www.ncbi.nlm.nih.gov/geo/download/?acc=GSE96058&format=file&file=GSE96058%5Fgene%5Fexpression%5F3273%5Fsamples%5Fand%5F136%5Freplicates%5Ftransformed%2Ecsv%2Egz"
expr_file= os.path.join(outdir, 'gene_expression.csv.gz')

urlretrieve(url, expr_file)

# Download metadata
urls = [
    "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE96nnn/GSE96058/matrix/GSE96058-GPL11154_series_matrix.txt.gz",
    "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE96nnn/GSE96058/matrix/GSE96058-GPL18573_series_matrix.txt.gz"
]

meta_files = []
for url in urls:
    filename = os.path.join(outdir, os.path.basename(url))
    urlretrieve(url, filename)
    meta_files.append(filename)

# gunzip the downloaded files

for file in os.listdir(outdir):
    if file.endswith('.gz'):
        with gzip.open(os.path.join(outdir, file), 'rb') as f_in:
            with open(os.path.join(outdir, file[:-3]), 'wb') as f_out:
                f_out.write(f_in.read())


# remove the .gz files
for file in os.listdir(outdir):
    if file.endswith('.gz'):
        os.remove(os.path.join(outdir, file))

# Load the expression data
df = pd.read_csv(expr_file, index_col=0)


# Average expression data across replicates
replicates = [c for c in df.columns if c.endswith('repl')]
for repl in replicates:
    og = repl[:-4]
    df[og] = df[[og,repl]].mean(axis=1)
    del df[repl]


# get metadata

def get_meta(path):
    with open(path) as f:
        lines=f.readlines()    
    meta_data = [line for line in lines if line.startswith("!Sample")]
    meta_dict = {}
    for line in meta_data:
        if line.startswith("!Sample_title"):
            samples=line.replace("!Sample_title","").strip().split('\t')
            samples=[c.strip('"') for c in samples]
            meta_dict['Sample'] =samples
        elif line.startswith("!Sample_characteristics_ch1"):
            parts = line.strip().split('\t')
            values = [v.strip('"') for v in parts[1:]]  
            key = values[0].split(':')[0].strip()
            data = [v.split(':')[1].strip() for v in values]
            meta_dict[key]=data
    metadata = pd.DataFrame(meta_dict)
    return metadata

metadata = pd.concat([get_meta(p) for p in meta_files],axis=0)

# Select common samples

metadata=metadata[metadata['Sample'].isin(df.columns)]
df = df[metadata['Sample']]
metadata=metadata.set_index('Sample')

# Create AnnData object



X = df.T
adata = ad.AnnData(X=X, obs=metadata)

adata.write_h5ad(os.path.join(outdir, 'dataset.h5ad'))

