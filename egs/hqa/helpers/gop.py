"""
Helper functions to compute GOP (Goodness of Pronunciation).
Basic concept (for each utterance):
1. Align data using lexicon and a model (forced alignment)
Output: a vector of length M, where M is number of frames.
Values are phonemes according to the alignment. I.e. some simulation of "real" values for a frame.
2. For the same utterance predict phonemes using the model,
but without the lexicon.
Output: MxN matrix, where M is number of frames, N is the number of phonemes,
values are probability for the corresponding phoneme to be at the frame with respect to the model.
3. If the model is good enough then for each frame alignment value must be the most probable one.
"""

import os
import glob

import numpy as np
import pandas as pd

from kaldi_io import phone_symb2int, pdf2phone, read_ali_from_stdout, read_feats_from_stdout
from utils import softmax

# Set environment
# load_dotenv('./.env', verbose=True, override=True)

def compute_gmm_probs(model_dir: str, feats_rspec: str, ali_rspec: str = None) -> pd.DataFrame:
    """Compute GOP probs for a GMM based model

    Computes probs for GOP using a GMM model. The algorithm is (for each utterance):
    1. Estimate 'real' phonemes for each frame by alignment (by `ali-to-phones --per-frame=true <model> <ali_rspec> ark,t:-`):
        Output: Vector of length M, M - number of frames
    2. Compute log-likelihoods of pdf_ids for each frame (by `gmm-compute-likes {model} {feats_rspec} ark,t:-`)
        Output: Matrix MxN, M - number of frames, N - number of pdf_ids
    3. Convert log-likelihoods to probabilities by softmax transformation: probs[i,:] = softmax(likes[i,:])
        Output: Matrix MxN, M - number of frames, N - number of pdf_ids.
    4. Map pdf_id to phonemes by `show-transitions {model_path}/phones.txt {model_path}/final.mdl {model_path}/final.occs | grep Transition-state`
    5. Sum pdf_id-probabilities by phoneme: probs_new[i,j] = sum(probs[i, <all pdf_ids of phoneme j>])
        Output: Matrix MxK, K - number of phonemes
    6. For each frame get: 
        max(probs_new[i,:]), argmax(probs_new[i,:]), probs_new[i,alignment_phoneme[i]], alignment_phoneme[i]
        This is one row for output DataFrame

    Parameters
    ----------
    model_dir : str
        A dir to a gmm model. Supposed to be standard kaldi model. For example, 'exp/mono_mfcc'
    feats_rspec: str
        A rspec for feats to provide for gmm-compute-likes. For example,
        feats_rspec = f'"ark,s,cs:apply-cmvn --utt2spk=ark:{data}/utt2spk scp:{data}/cmvn.scp scp:{data}/feats.scp ark:- | add-deltas ark:- ark:- |"'
    ali_rspec: str,None
        Standard kaldi rspec for an alignment file. For example, 'ark:"gunzip -c exp/mono_mfcc/ali.1.gz|"
        By default all ali.*.gz files from model_dir are taken.

    Returns
    -------
    probs: pd.DataFrame
        A pandas DataFrame of following structure:
        - utterance - name of utterance
        - max_prob - maximum phoneme-probability for the corresponding frame
        - max_phone - a phoneme having maximum probability for the frame (i.e. argmax for probability)
        - ali_prob - probability of a phoneme expected by alignment
        - ali_phone - a phoneme expected by alignment
    """
    model = os.path.join(model_dir, "final.mdl")
    phones_file = os.path.join(model_dir, "phones.txt")

    # Get mappings between phoneme int code and symbol
    symb2int = phone_symb2int(phones_file)
    int2symb = {y: x for x, y in symb2int.items()}

    # Get mapping for pdf id -> phoneme symbol
    pdf2symb = pdf2phone(model_dir)
    pdf2int = {k: symb2int[v] for k, v in pdf2symb.items()}

    # Get alignments
    if ali_rspec in None:
        # By default get alignments from all ali.*.gz files in the model_dir
        ali_files = sorted(glob.glob(os.path.join(model_dir, 'ali.*.gz')))
        alis = {}
        for ali_file in ali_files:
            alis.update(
                read_ali_from_stdout(model, f'ark:"gunzip -c {ali_file}|"')
                )
    else:
        alis = read_ali_from_stdout(model, ali_rspec)

    # Compute gmm probs for each pdf id
    gmm_likes_command = f"gmm-compute-likes {model} {feats_rspec} ark,t:-"
    gmm_likes = read_feats_from_stdout(gmm_likes_command)

    # Calculate GOP for each utterance
    probs_summary = {}
    for utterance, likes in gmm_likes.items():
        try:
            ali = alis[utterance]
        except KeyError:
            print(f"Could not find alignment for {utterance}")
            continue

        # Convert likes to probabilities by softmax transformation
        prob = softmax(likes)

        # Switch from pdf_id to phoneme: sum probs by phoneme
        prob = pd.DataFrame(data=prob, columns=pdf2int.values())
        prob = prob.groupby(by=df.columns, axis=1).sum()

        # Get probs summary:
        #   max probability phoneme (i.e. predicted phoneme)
        #  VS.
        #   alignment phoneme (i.e. real phoneme)
        probs_summary[utterance] = pd.DataFrame(
            {
                "max_prob": prob.max(axis=1),
                "max_phone": prob.idxmax(axis=1),
                "ali_prob": prob.lookup(np.arange(prob.shape[0]), ali),
                "ali_phone": ali,
            }
        )
    del gmm_likes

    # Convert dict of DataFrames to one DataFrame
    return pd.concat(probs_summary.values(), keys=probs_summary.keys())
