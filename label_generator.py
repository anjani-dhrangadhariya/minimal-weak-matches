# imports - general
from os import listdir
from os.path import isfile, join
import ast
import pandas as pd
from labeling_functions import OntologyLabelingFunctionX, loadStopWords, spansToLabels
import numpy as np
import os, random
from read_data import get_source_df, get_target_df


# imports - specific
import torch


################################################################################
# Initialize and set seed
################################################################################
def seed_everything( seed ):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


seed = 0
seed_everything(seed)
print('The random seed is set to: ', seed)

def get_picos(fpath):

    picos_dict = { 'intervention':'I' , 'intervention_syn':'I', 'participant':'P', 'outcome':'O', 'studytype':'O' }

    file_path = str( fpath )

    file_name = file_path.split('/')[-1]
    file_name = file_name.replace('.txt', '')


    try:
        if file_name in picos_dict:
            picos = picos_dict[file_name]
    except BaseException as error:
        raise 'An exception occurred: {}'.format(error)

    return picos

# get source term files from the input_sources directory
indir_sources = '/home/anjani/minimal-distant-matches/input_sources'
ds_sources = [join(indir_sources, f) for f in listdir(indir_sources) if isfile(join(indir_sources, f))]

# select a source file to use and load file contents into a dataframe
sources_df = get_source_df(ds_sources[0])

# get target corpus files from the input_targets directory
indir_targets = '/home/anjani/minimal-distant-matches/input_targets'
ds_targets = [join(indir_targets, f) for f in listdir(indir_targets) if isfile(join(indir_targets, f))]

# read a target file to use and load file contents into a dataframe
targets_df, targets_flatten_df = get_target_df( ds_targets[0], write_to_file = False )

# get picos label to label the target tokens (picos will be labeled as +1 labels)
picos = get_picos( ds_sources[0] )
print( 'The PICOS label is: ', picos )

# get stopwords (stopwords will be labeled as -1 labels)
sw_lf = loadStopWords()

# match sources to the targets and retrieve matching spans between source terms and target corpora
ds_matches = OntologyLabelingFunctionX( targets_df['text'], targets_df['tokens'], targets_df['offsets'], source_terms = sources_df['text'].tolist(), picos=picos, fuzzy_match=False, stopwords_general = sw_lf )

# convert the matching spans to labels
df_ds_labels = spansToLabels(matches=ds_matches, df_data=targets_df, picos=picos)
assert len( df_ds_labels ) == len( targets_df['tokens'] ) == len( targets_df['offsets'] )
targets_df['labels'] = df_ds_labels

# write the label annotations to a file in "output" directory
outdir = '/home/anjani/minimal-distant-matches/output_files'
# targets_df.to_csv(f'{outdir}/output.json', sep='\t')