#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Runs the probing with noise experiments

Copyright © 2023 Filip Klubička. Technological University Dublin, ADAPT Centre.
This work is licensed under a Creative Commons Attribution 4.0 International License
"""
from datetime import datetime
import argparse
import pickle
from src.NoiseProbe import NoiseProbe

def parse_arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-p", "--path", type=str, default="data/bigram_shift_small.pkl", help="Specify path to your data file, i.e. a pickle file containing a dictionary with candidate words/sentences, their embeddings and class labels.")
    parser.add_argument("-c", "--classes", type=str, default='binary', help="Type of classification task performed. Options: binary, multiclass. Default: binary")
    parser.add_argument("-e", "--embedding", type=str, default='bert', help="Specifies which type of encoder architecture generated the 'fixed' embeddings you want to load from the dataset file and use for training. Specifying bert_train will train new bert embeddings on new sentences, rather than load pretrained embeddings from the dictionary. Options: glove, bert, bert_train. Default: bert")
    parser.add_argument("-n", "--noise", type=str, default='vanilla', help="Type of noise injections/embedding modifications to be performed before training. See README for details. Options: rvec, vanilla, abn, abd, abnd, d1h, d2h. Default: vanilla")
    parser.add_argument("-b", "--baseline", default=False, action="store_true", help="Instead of training on the input vectors, evaluate a baseline model that makes random label predictions over the given dataset.")
    parser.add_argument("-r", "--runs", type=int, default=50, help="Number of times the model will be trained. Default: 50")
    args = parser.parse_args()
    return args

if __name__ == "__main__":

    # Parameters
    args = parse_arguments()

    # Load dataset
    data_file = pickle.load(open(args.path, "rb"), encoding='latin1')

    # Run full trainind and evaluation the given number of times
    print("accuracy, precision, recall, f1, auc_pred, auc_probs")
    for index in range(0,args.runs):
        now = datetime.now()
        current_time = now.strftime("%H:%M:%S")
        print("Time:", current_time, "*** BIGRAM SHIFT ***", args.noise, args.embedding, "*** Round:", index, "/", args.runs)
        probe = NoiseProbe(data_file, args.embedding, args.noise)
        probe.evaluate(args.classes, args.baseline)
    print("BIGRAM SHIFT: Finished", args.runs, args.noise, args.embedding, "evaluation runs.")