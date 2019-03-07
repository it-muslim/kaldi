"""
These are helper function for kaldi io operations. Working with alignments,
features and phonemes.
NOTE: It is assumed that all necessary commands exist (i.e are added to PATH).
"""

from typing import Dict
import subprocess
import numpy as np


def pdf2phone(model_path: str) -> Dict[int, str]:
    """Map pdf_id(s) to actual phones

    Uses kaldi's show-transition command, which gives output like:
        Transition-state 134: phone = ws hmm-state = 0 pdf = 43
         Transition-id = 283 p = 0.916493 count of pdf = 4323 [self-loop]
         Transition-id = 284 p = 0.0835068 count of pdf = 4323 [0 -> 1]
        Transition-state 135: phone = ws hmm-state = 1 pdf = 124
         Transition-id = 285 p = 0.896085 count of pdf = 3474 [self-loop]
         Transition-id = 286 p = 0.103915 count of pdf = 3474 [1 -> 2]
        Transition-state 136: phone = ws hmm-state = 2 pdf = 109
         Transition-id = 287 p = 0.876454 count of pdf = 2922 [self-loop]
         Transition-id = 288 p = 0.123546 count of pdf = 2922 [2 -> 3]
    All lines starting with "Transition-state" are considered as containing
    mapping for pdf_id and phone symbol.

    Parameters
    ----------
    model_path : str
        A path to a folder, containing files phones.txt, final.mdl
        and final.occs. For example, 'exp/mono_mfcc'

    Returns
    -------
    mapping : dict
        A dict of pairs {pdf_id: phone_symb}
    """
    mapping = {}
    command = f"show-transitions {model_path}/phones.txt {model_path}/final.mdl {model_path}/final.occs | grep Transition-state"
    proc = subprocess.Popen(
        command, stdout=subprocess.PIPE, shell=True, executable="/bin/bash"
    )
    while True:
        line = proc.stdout.readline()
        if line == b"":
            break
        else:
            vals = line.decode("utf-8").strip().split()
            assert len(vals) == 11
            mapping[int(vals[-1])] = vals[4]
    return mapping


def phone_symb2int(phones_file: str) -> Dict[str, int]:
    """Map phoneme symbols to integer codes

    Parameters
    ----------
    phones_file : str
        A path to a phones.txt file. For example, 'exp/mono_mfcc/phones.txt'

    Returns
    -------
    mapping : dict
        A dict of pairs {phone_symb: phone_int_code}
    """
    with open(phones_file) as file:
        code2phone = [x.rstrip("\n").split() for x in file.readlines()]
    return {phone: int(code) for phone, code in code2phone}


def phone_int2symb(phones_file: str) -> Dict[int, str]:
    """Is opposite to phone_symb2int"""
    symb2int = phone_symb2int(phones_file)
    return {y: x for x, y in symb2int.items()}


def read_feats_from_stdout(feats_command: str) -> Dict[str, np.array]:
    """Import feats(or feats-like) data as a numpy array

    As feats are generated "on-fly" in kaldi, there is no a feats file
    (except most simple cases like raw mfcc, plp or fbank).  So, that is why
    we take feats as a command rather that a file path. Can be applied to
    other commands (like gmm-compute-likes) generating an output in same
    format as feats, i.e:
    utterance_id_1  [
      70.31843 -2.872698 -0.06561285 22.71824 -15.57525 ...
      78.39457 -1.907646 -1.593253 23.57921 -14.74229 ...
      ...
      57.27236 -16.17824 -15.33368 -5.945696 0.04276848 ... -0.5812851 ]
    utterance_id_2  [
      64.00951 -8.952017 4.134113 33.16264 11.09073 ...
      ...

    Parameters
    ----------
    feats_command : str
        A command generating feats and printing them to stdout by 'ark,t:-'.
        For example,
        'copy-feats scp:data/test/feats.scp ark,t:-'
        'gmm-compute-likes exp/mono_mfcc/final.mdl "ark,s,cs:apply-cmvn --utt2spk=ark:train/utt2spk scp:train/cmvn.scp scp:train/feats.scp ark:- | add-deltas ark:- ark:- |" ark,t:-'

    Returns
    -------
    feats : numpy.array
        A dict of pairs {utterance: feats}
    """
    feats = {}
    # current_row = 0
    current_utter = None
    proc = subprocess.Popen(
        feats_command, stdout=subprocess.PIPE, shell=True, executable="/bin/bash"
    )
    line = proc.stdout.readline()
    while line != b"":
        try:
            line = line.decode("utf-8")
            if line.startswith("  "):
                # If feature values
                assert current_utter is not None
                feats[current_utter].append([float(x) for x in line.strip(" ]\n").split(" ")])
                # current_row += 1
            else:
                # If a new utterance
                current_utter = line.split(" ")[0]
                feats[current_utter] = []
                # print(f'CURRENT ROW: {current_row}')
                # print(f'CURRENT UTTER: {current_utter}')
        except:
            print('"' + line + '"')
            print(line.strip(" ]").split(" "))
            print(current_utter)
            raise
        line = proc.stdout.readline()
    feats = {k: np.array(v) for k, v in feats.items()}
    return feats


def read_ali_from_stdout(model: str, ali_rspec: str) -> Dict[str, np.array]:
    """Import alignments as a numpy array

    The function allows to get per-frame phoneme predictions from alignment
    files.  It has different syntax (input arguments) than read_feats...
    because usually there are many alignment files in one folder of a model.
    So, rspec for ali must be specified.

    Parameters
    ----------
    model : str
        A path to model for alignment. For example, 'exp/mono_mfcc/final.mdl'
    ali_rspec: str
        Standard kaldi rspec for an alignment file.
        For example, 'ark:"gunzip -c exp/mono_mfcc/ali.1.gz|"

    Returns
    -------
    ali : dict
        A dict of pairs {utterance: }
    """
    ali = {}
    command = (
        f"ali-to-phones --per-frame=true {model} {ali_rspec} ark,t:-"
    )
    proc = subprocess.Popen(
        command, stdout=subprocess.PIPE, shell=True, executable="/bin/bash"
    )
    line = proc.stdout.readline()
    while line != b"":
        values = line.decode("utf-8").strip(" \n").split(" ")
        ali[values[0]] = np.array([int(x) for x in values[1:]], dtype=np.int32)
        line = proc.stdout.readline()
    return ali
