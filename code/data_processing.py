#!/usr/bin/env python
# -*- coding: UTF-8 -*-
import pandas as pd
import numpy as np
from rdkit import Chem
import random
import warnings
warnings.filterwarnings("ignore")

def basesDict():
    """
    Obtain SMILES strings of A, T, G, C molecule.
    """
    baseDict = {}
    with open('data/basesSMILES/ATGC.dict', 'r') as f:
        for line in f:
            line = line.strip().split(',')
            baseDict[line[0]] = line[1]

    return baseDict

def atom_feature(atom):
    """
    Features of atoms in base molecule.
    """
    return np.array(one_of_k_encoding_unk(atom.GetSymbol(),
                                          ['C', 'N', 'O']) +                      # # whether there is an atom in the molecule
                    one_of_k_encoding(atom.GetDegree(), [1, 2, 3]) +              # # the degree of the atom in the molecule
                    one_of_k_encoding(atom.GetTotalNumHs(), [0, 1, 2, 3]) +       # # the total number of hydrogens on the atom
                    one_of_k_encoding(atom.GetImplicitValence(), [0, 1, 2, 3]) +  # # the implicit valence
                    [atom.GetIsAromatic()])                                       # # feature dimensions are 15

def one_of_k_encoding(x, allowable_set):

    if x not in allowable_set:
        raise Exception("input {0} not in allowable set{1}:".format(x, allowable_set))

    return list(map(lambda s: x == s, allowable_set))

def one_of_k_encoding_unk(x, allowable_set):

    if x not in allowable_set:
        x = allowable_set[-1]

    return list(map(lambda s: x == s, allowable_set))


#......................................................................................................#

def norm_Adj(adjacency):
    """
    Obtain the normalized adjacency matrix.
    """
    I = np.array(np.eye(adjacency.shape[0]))                                # # identity matrix
    adj_hat = adjacency + I

    # D^(-1/2) * (A + I) * D^(-1/2)
    D_hat = np.diag(np.power(np.array(adj_hat.sum(1)), -0.5).flatten(), 0)  # # degree matrix
    adj_Norm = adj_hat.dot(D_hat).transpose().dot(D_hat)                    # # normalized adjacency matrix

    return adj_Norm

def norm_fea(features):
    """
    Obtain the normalized node feature matrix.
    """
    norm_fea = features / features.sum(1).reshape(-1, 1)                    # # normalized node feature matrix

    return norm_fea

def convert_to_graph(seq):
    """
    Obtain the molecular graph features of one sequence.
    """
    baseDict = basesDict()
    maxNumAtoms = 13

    # Molecules of bases from one sequence
    graphFeaturesOneSeq = []
    seqSMILES = [baseDict[b] for b in seq]
    for baseSMILES in seqSMILES:
        DNA_mol = Chem.MolFromSmiles(baseSMILES)##example：<rdkit.Chem.rdchem.Mol object at 0x0000019A512272B0>

        # Adjacency matrix
        AdjTmp = Chem.GetAdjacencyMatrix(DNA_mol)

        AdjNorm = norm_Adj(AdjTmp)

        # Node feature matrix (features of node (atom))（13，15）
        if AdjNorm.shape[0] <= maxNumAtoms:

            # Preprocessing of feature
            graphFeature = np.zeros((maxNumAtoms, 15))
            nodeFeatureTmp = []
            for atom in DNA_mol.GetAtoms():#example atom_feature: [1 0 0 0 1 0 0 1 0 0 0 1 0 0 1]
                nodeFeatureTmp.append(atom_feature(atom))

            nodeFeatureNorm = norm_fea(np.asarray(nodeFeatureTmp))

            # Molecular graph feature for one base
            graphFeature[0:len(nodeFeatureTmp), 0:15] = np.dot(AdjNorm.T, nodeFeatureNorm)


            # Append the molecualr graph features for bases in order
            graphFeaturesOneSeq.append(graphFeature)

    # Molecular graph features for one sequence
    graphFeaturesOneSeq = np.asarray(graphFeaturesOneSeq, dtype=np.float32)

    return graphFeaturesOneSeq  # #

#......................................................................................................#

def dataProcessing(path):
    """
    Read the data, and then encode the DNA molecular graph feature.
    """
    data = pd.read_csv(path)
    X = []
    for line in data['data']:
        seq = line.strip('\n')
        graphFeatures = convert_to_graph(seq)
        X.append(graphFeatures)
    X = np.asarray(X)                            # # (bs, 41, 13 ,15)
    y = np.array(data['label'], dtype=np.int32)

    return X, y

def prepareData(PositiveCSV, NegativeCSV):
    """
    :param PositiveCSV: the positive samples of input file with comma-separated values.
    :param NegativeCSV: the negative samples of input file with comma-separated values.
    :return           : DNA molecular graph features of positive and negative samples and their corresponding labels.
    """
    Positive_X, Positive_y = dataProcessing(PositiveCSV)
    Negitive_X, Negitive_y = dataProcessing(NegativeCSV)

    return Positive_X, Positive_y, Negitive_X, Negitive_y

def chunkIt(seq, num):
    """
    Divide the data based on k-folds.
    """
    avg = len(seq) / float(num)
    out = []
    last = 0.0
    while last < len(seq):
        out.append(seq[int(last):int(last + avg)])
        last += avg

    return out

def shuffleData(X, y):
    """
    :param X: data
    :param y: labels
    :return : data and labels after shuffle
    """
    index = [i for i in range(len(X))]
    random.seed(0)
    random.shuffle(index)
    X = X[index]
    y = y[index]

    return X, y










