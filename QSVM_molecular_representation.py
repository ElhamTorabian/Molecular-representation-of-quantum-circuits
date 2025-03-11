import csv
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
from itertools import combinations_with_replacement
import random

import seaborn as sns
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KernelDensity
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn import preprocessing
from sklearn.metrics import log_loss


import rdkit
from rdkit import Chem
from rdkit import DataStructs
from rdkit.Chem.Pharm2D import Gobbi_Pharm2D
from rdkit.Chem import AllChem
from rdkit.Chem import Draw


from qiskit import ClassicalRegister, QuantumCircuit, QuantumRegister
from qiskit.circuit.library import ZZFeatureMap, ZFeatureMap,PauliFeatureMap,NLocal
from qiskit.circuit import Parameter, ParameterVector
from qiskit_machine_learning.kernels import FidelityStatevectorKernel



def Good_Bad_split(Descriptors,Ave_accuracy):
    """
    Function to split the dataset into "Good" and "Bad" quantum circuits based on the average accuracy.
    
    Parameters:
    - Descriptors: list/array, list of quantum circuits.
    - Ave_accuracy: list/array, list of average accuracy values.

    Returns:
    - Good_design: list, list of "Good" quantum circuits.
    - Bad_design: list, list of "Bad" quantum circuits.
    - Desc: list, list of all quantum circuits.
    - labels: list, list of labels (1 for "Good" and 0 for "Bad").
    - Good_acc: list, list of average accuracy values for "Good" quantum circuits.
    - Bad_acc: list, list of average accuracy values for "Bad" quantum circuits.

    """
    Good_design=[]
    Good_acc=[]
    Bad_design=[]
    Bad_acc=[]
    Desc=[]
    labels=[]
    # Hidden threshold
    top = 0.63
    bot = 0.6
    # Perovskite threshold
    # top = 0.64
    # bot = 0.56
    print("Top threshold is: ",top)
    print("Bottom threshold is: ",bot)

    for i in range(len(Ave_accuracy)):
        if Ave_accuracy[i]>top:
            Good_design.append(Descriptors[i])
            Good_acc.append(Ave_accuracy[i])
            Desc.append(Descriptors[i])
            labels.append(1)
        elif Ave_accuracy[i]<bot:
            Bad_design.append(Descriptors[i])
            Bad_acc.append(Ave_accuracy[i])
            Desc.append(Descriptors[i])
            labels.append(0)
    print("Total good length:   ", len(Good_design))
    print("Total bad length:   ", len(Bad_design))
    return(Good_design,Bad_design,Desc,labels, Good_acc, Bad_acc)    

def Cleaning(Desc):
    Cleaned=[]
    for row in Desc:
        new = row.replace(',','')
        new = new.replace('U1','u')
        new = new.replace('U1 ','u')
        new = new.replace('X','x')
        new = new.replace('xn','n')
        new = new.replace('xm','m')
        new = new.replace(" ","")
        new = new.replace("''","")
        new = new.replace("'","")
        new = new.replace(' ','')
        new = new.replace("[","")
        new = new.replace("]","")
        Cleaned.append(new)
    return(Cleaned)
    

def circ_to_mol(seq,m):
    Descriptor=np.reshape(list(seq),(4,m))
    # Descriptor=np.reshape(list(seq),(5,m))
    # Descriptor=np.reshape(list(seq),(3,m))
    subs = []
    for row in Descriptor:
        a=''
        p_num = 0
        atom_num = 0
        ent_num = 0
        for i in range(len(row)):
            
            if row[i]=='u':
                a+='C'
                p_num+=1
                atom_num+=1
            elif row[i]=='x':
                a+='N'
                # atom_num+=1
                # a += 'C1CC1'
                atom_num += 3
                ent_num+=1
            elif row[i]=='n':
                a+='O'
                # atom_num+=1
                # a += 'C2CCC2'
                atom_num += 4
                ent_num+=1
            elif row[i]=='m':
                a+='S'
                # atom_num+=1
                # a += 'C3CCCC3'
                atom_num += 5
                ent_num+=1
            elif row[i]=='p':
                a+='P'
                # atom_num+=1
                # a += 'C3CCCC3'
                atom_num += 5
                ent_num+=1
        subs.append(a)
    #For n_feature = 4    
    SML=subs[0]+'C('+subs[1]+')('+subs[2]+')'+subs[3]
    
    #For n_feature = 5 
    # SML=subs[0]+'C('+subs[1]+')'+'('+subs[2]+')'+'C('+subs[3]+')'+subs[4]

    SML=SML.replace('()','')
    return(SML,p_num,atom_num,ent_num)





def SMILE_convert(Cleaned):
    """
    Function to convert quantum circuits to SMILES representation.
    
    Parameters:
    - Cleaned: list, list of quantum circuits descriptors.

    Returns:
    - SMILES: list, list of SMILES representation of the quantum circuits.
    - param_num: list, list of number of parametrized gates.
    - atoms_num: list, list of number of atoms.
    - Entanglement_num: list, list of number of entanglement gates.
    - Layers_num: list, list of number of layers.

    """
    SMILES = []
    param_num = []
    atoms_num = []
    Entanglement_num = []
    Layers_num = []
    for row in Cleaned:
        SML,p_n,atom_num,ent_num = circ_to_mol(row,int(len(row)/4))
        SMILES.append(SML)
        param_num.append(p_n)
        atoms_num.append(atom_num)
        Entanglement_num.append(ent_num)
        Layers_num.append(int(len(row)/4))
    return(SMILES,param_num,atoms_num,Entanglement_num, Layers_num)
    

def Finger_Convert(S):
    """
    Function to convert SMILES to molecular fingerprints.

    Parameters:
    - S: list, list of SMILES representation of molecules.

    Returns:
    - MOLES: list, list of molecules in RDKit format.
    - FPS: list, list of molecular fingerprints
    
    """

    MOLES=[]
    for row in S:
        MOLES.append(Chem.MolFromSmiles(row))
    FPS = [Chem.RDKFingerprint(x) for x in MOLES]
    return(MOLES,FPS)

def Bitstring_convert(FP):
    """
    Function to convert molecular fingerprints to bitstrings.

    Parameters:
    - FP: list, list of molecular fingerprints.

    Returns:
    - FParr: list, list of bitstrings.
    
    """

    FParr_T=[np.array(x) for x in FP]
    bit_T=["".join(x.astype(str)) for x in FParr_T]
    fp2_T=[DataStructs.cDataStructs.CreateFromBitString(x) for x in bit_T]
    FP_T=[list(x.GetOnBits()) for x in fp2_T]

    Lens=[len(x) for x in FP_T]
    L_arr=np.array(Lens)
    MAX=np.max(L_arr)
    print(MAX)
    Final_FP=[np.append(x,np.zeros((1, 591-len(x)))) for x in FP_T]
    return(Final_FP)


def Dataset_prep(x_train,x_test,label_train,label_test):
    """
    Function to prepare the dataset for training.

    Parameters:
    - x_train: list, list of training data.
    - x_test: list, list of testing data.
    - label_train: list, list of training labels.
    - label_test: list, list of testing labels.

    Returns:
    - x: np.array, input data.
    - labs: np.array, labels.
    - features: list, list of feature names.
    
    """

    Xtrain=np.array(x_train)
    Xtotal=np.vstack((x_train, x_test))
    Labels= np.hstack((label_train,label_test))
    print('Finger prints shape is:',Xtotal.shape)
    print('Labels shape is:',Labels.shape)
    labs = np.reshape(Labels,(len(Labels),1))
    print(labs.shape)
    final_data = np.concatenate([Xtotal,labs],axis=1)

    dataset = pd.DataFrame(final_data)
    dataset=dataset.rename(columns=lambda x: 'f'+str(x))
    MAX=591
    s='f'+str(MAX)
    dataset=dataset.rename(columns = {s:'Labels'})
    print(dataset)
    features = np.array(list(dataset.columns))
    f=features[:(MAX-1)]
    x = dataset.loc[:, f].values
    x = StandardScaler().fit_transform(x) # normalizing the features
    print(x.shape)
    return(x,labs,features)

def smi2cm(smi, dimensions = 2, Hs = True, return_xyz = False):
    """
    Function to convert SMILES to Coulomb matrix.

    Parameters:
    - smi: str, SMILES representation of the molecule.
    - dimensions: int, number of dimensions (2 or 3).
    - Hs: bool, whether to include Hydrogens.
    - return_xyz: bool, whether to return the XYZ matrix.

    Returns:
    - cij: np.array, Coulomb matrix.
    - xyzmatrix: np.array, XYZ matrix.
    
    """

    mol = Chem.MolFromSmiles(smi)
    if Hs:
        mol = Chem.AddHs(mol)
    else:
        mol = Chem.RemoveHs(mol)
    if int(dimensions) == 2:
        AllChem.Compute2DCoords(mol)
    elif int(dimensions) == 3:
        try:
            # Code that might raise ValueError
            AllChem.EmbedMolecule(mol)
            AllChem.UFFOptimizeMolecule(mol)
        except ValueError as e:
            if str(e) == "Bad Conformer Id":
                print("Bad Conformer Id")
                return None  # Skip to the next molecule   
    else:
        print("Invalid input for parameter: dimensions\nPlease only use either 2 or 3")
    xyz = Chem.MolToXYZBlock(mol)
    xyzmatrix = xyz_parse(xyz)
    if return_xyz:
        return xyzmatrix
    else:
        return gen_coulombmatrix(xyzmatrix)

def xyz_parse(xyz):
    """
    Function to parse XYZ coordinates.

    Parameters:
    - xyz: str, XYZ coordinates of the molecule.

    Returns:
    - xyzmatrix: np.array, XYZ matrix
    
    """

    nAtoms = int(xyz.split("\n")[0])
    xyzmatrix = np.ndarray((nAtoms, 4), dtype="float")
    ANs = {'H': 1, 'He': 2, 'Li': 3, 'Be': 4, 'B': 5, 'C': 6, 'N': 7, 'O': 8, 'F': 9, 'Ne': 10, 'Na': 11, 'Mg': 12, 'Al': 13, 'Si': 14, 'P': 15, 'S': 16, 'Cl': 17, 'Ar': 18, 'K': 19, 'Ca': 20, 'Sc': 21, 'Ti': 22, 'V': 23, 'Cr': 24, 'Mn': 25, 'Fe': 26, 'Co': 27, 'Ni': 28, 'Cu': 29, 'Zn': 30, 'Ga': 31, 'Ge': 32, 'As': 33, 'Se': 34, 'Br': 35, 'Kr': 36, 'Rb': 37, 'Sr': 38, 'Y': 39, 'Zr': 40, 'Nb': 41, 'Mo': 42, 'Tc': 43, 'Ru': 44, 'Rh': 45, 'Pd': 46, 'Ag': 47, 'Cd': 48, 'In': 49, 'Sn': 50, 'Sb': 51, 'Te': 52, 'I': 53, 'Xe': 54, 'Cs': 55, 'Ba': 56, 'La': 57, 'Ce': 58, 'Pr': 59, 'Nd': 60, 'Pm': 61, 'Sm': 62, 'Eu': 63, 'Gd': 64, 'Tb': 65, 'Dy': 66, 'Ho': 67, 'Er': 68, 'Tm': 69, 'Yb': 70, 'Lu': 71, 'Hf': 72, 'Ta': 73, 'W': 74, 'Re': 75, 'Os': 76, 'Ir': 77, 'Pt': 78, 'Au': 79, 'Hg': 80, 'Tl': 81, 'Pb': 82, 'Bi': 83, 'Po': 84, 'At': 85, 'Rn': 86, 'Fr': 87, 'Ra': 88, 'Ac': 89, 'Th': 90, 'Pa': 91, 'U': 92, 'Np': 93, 'Pu': 94, 'Am': 95, 'Cm': 96, 'Bk': 97, 'Cf': 98, 'Es': 99, 'Fm': 100, 'Md': 101, 'No': 102, 'Lr': 103, 'Rf': 104, 'Db': 105, 'Sg': 106, 'Bh': 107, 'Hs': 108, 'Mt': 109}
    i = 0
    for line in xyz.split("\n"):
        line = line.split()
        if len(line) == 4:
            xyzmatrix[i][0] = float(ANs[line[0]])
            xyzmatrix[i][1] = float(line[1])
            xyzmatrix[i][2] = float(line[2])
            xyzmatrix[i][3] = float(line[3])
            i+=1
    return xyzmatrix

def gen_coulombmatrix(xyzmatrix):

    """
    Function to generate Coulomb matrix.
    From: Rupp, M.; Tkatchenko, A.; Müller, K. R.; Von Lilienfeld, O. A. Fast and Accurate Modeling of Molecular Atomization Energies with Machine Learning. Phys. Rev. Lett. 2012, 108 (5), 1–5. https://doi.org/10.1103/PhysRevLett.108.058301.
    
    Parameters:
    - xyzmatrix: np.array, XYZ matrix of the molecule.

    Returns:
    - cij: np.array, Coulomb matrix.

    """
    nAtoms = int(xyzmatrix.shape[0])

    cij = np.zeros((nAtoms, nAtoms))

    for i in range(nAtoms):
        for j in range(nAtoms):
            if i == j:
                cij[i][j] = 0.5 * xyzmatrix[i][0] ** 2.4  # Diagonal term described by Potential energy of isolated atom
            else:
                dist = np.linalg.norm(np.array(xyzmatrix[i][1:]) - np.array(xyzmatrix[j][1:]))

                cij[i][j] = xyzmatrix[i][0] * xyzmatrix[j][0] / dist  # Pair-wise repulsion
            #cij[i][j] = float(cij[i][j])
    return cij
    
def eig_columb(S):
    """
    Function to calculate the eigenvalues of the Coulomb matrix.

    Parameters:
    - S: list, list of SMILES representation of molecules.

    Returns:
    - Eigs: list, list of eigenvalues of the Coulomb matrix
    
    """

    CoulombMats = []
    for row in S:
        Col_i = smi2cm(row, Hs = False)
        if Col_i is not None:
            CoulombMats.append(Col_i)

    Eigs = [np.linalg.eig(x)[0] for x in CoulombMats]
    
    return(Eigs)

def Gershgorin_descriptors(S):
    CoulombMats = []
    MinMax_radii = []
    for row in S:
        # print(row)
        Col_i = smi2cm(row, Hs = False)
        if Col_i is not None:
            CoulombMats.append(Col_i)
            R = np.zeros(len(Col_i[0])) # disk radii
            for i in range(len(Col_i[0])):
                R[i] = sum(abs(Col_i[i,:])) - abs(Col_i[i,i])
            MinMax_radii.append([np.min(R),np.max(R)])
        else:
            pass
    return(CoulombMats,MinMax_radii)


def physical_descriptors(eigs):
    """
    Function to calculate Gershgorin circles based descriptors from the eigenvalues of the Coulomb matrix.

    Parameters:
    - eigs: list, list of eigenvalues of the Coulomb matrix.

    Returns:
    - Desc_3D: list, list of Gershgorin circles based descriptors.
    
    """

    Desc_3D = []
    for row in eigs:
        x1 = np.min(row)
        x2 = np.mean(row)
        x3 = np.std(row)
        Desc_3D.append([x1,x2,x3])
    return(Desc_3D)
    

def random_mol(N,pool, num_L, num_features):
    """
    Function to generate random molecular structures based on the given pool of atoms.

    Parameters:
    - N: int, number of random molecules to generate.
    - pool: list, list of atoms to choose from.
    - num_L: int, number of atoms in each branch.
    - num_features: int, number of branches in each molecule.

    Returns:
    - branches: list, list of random branches.
    - rand_moles: list, list of random molecules in RDKit format
    - GG_descriptor: list, list of Gershgorin circles based descriptors
    
    """
    
    rand_smiles =[]
    random_branches = list(combinations_with_replacement(pool, r=num_L))
    rand_moles = []
    GG_descriptor = []
    branches = []

    for i in range(N):
        combs = random.sample(random_branches, num_features)
        branches.append(combs)
        subs = [''.join(comb) for comb in combs]
        # For n_features=4
        # slm = (subs[0]+'C('+subs[1]+')('+subs[2]+')'+subs[3]).replace('()','')
        # For n_features=5
        slm = (subs[0]+'C('+subs[1]+')'+'('+subs[2]+')'+'C('+subs[3]+')'+subs[4]).replace('()','')
        rand_smiles.append(slm)
        rand_moles.append(Chem.MolFromSmiles(slm))
        GG_descriptor.append(Gershgorin_descriptors([slm])[0])
    return(branches,rand_moles,GG_descriptor)

def calculate_bic(n, LL, num_params):
    """
    Calculate Bayesian Information Criteria (BIC) for the proposed model.

    Parameters:
    - n: int, number of samples.
    - LL: float, log-likelihood value.
    - num_params: int, number of parameters in the model.

    Returns:
    - bic: float, Bayesian Information Criteria (BIC) value.

    """

    bic = (-2) * LL + num_params * np.log(n)
    return bic


def evaluation(y_pred, Y_real, P_pred, param_num):
    """
    Calculate statistical criteria for the proposed model.

    Parameters:
    - y_pred: list, predicted labels.
    - Y_real: list, real labels.
    - P_pred: list, predicted probabilities.
    - param_num: int, number of parameters in the model.

    Returns:
    - acc0: float, accuracy for the first class.
    - acc1: float, accuracy for the second class.
    - Low_acc: float, lower accuracy between the two classes.
    - BIC: float, Bayesian Information Criteria (BIC) value.

    """

    count0 = 0
    count1 = 0
    t = 0
    r = 0

    Yr = Y_real.iloc[:, 0].tolist()

    for j in range(len(y_pred)):
        if Yr[j] == -1:
            t += 1
            if y_pred[j] == Yr[j]:
                count0 += 1
        else:
            r += 1
            if y_pred[j] == Yr[j]:
                count1 += 1

    acc0 = count0 / t
    acc1 = count1 / r
    Low_acc = min(acc0, acc1)

    LogLoss = log_loss(Yr, P_pred)
    BIC = calculate_bic(100, LogLoss, param_num)

    return acc0, acc1, Low_acc, BIC

def P_counter(comb):
        """
        Count the number of parameters in the circuit (here, the number of Z-rotation gates).

        Parameters:
        - comb: list, list of quantum circuits.

        Returns:
        - count: int, number of parameters in the circuit.
        """
        count = 0
        for row in comb:
            if row == 'U1':
                count += 1
        return count

def Test_random_mols(random_moles):
    """
    Test the performance of the quantum kernel on random molecular structures.

    Parameters:
    - random_moles: list, list of random molecular structures.

    Returns:
    - RESULT: pd.DataFrame, dataframe containing the results of the test.
    """

    X_train = pd.read_csv('./Datasets/mnist_5D/mnist_X_train.csv').reset_index(drop=True)
    Y_train = pd.read_csv('./Datasets/mnist_5D/mnist_y_train.csv').reset_index(drop=True)
    X_test = pd.read_csv('./Datasets/mnist_5D/mnist_X_test.csv').reset_index(drop=True)
    Y_test = pd.read_csv('./Datasets/mnist_5D/mnist_y_test.csv').reset_index(drop=True)

    scaler = preprocessing.MinMaxScaler(feature_range=(0, 2*np.pi))
    normalized_Xtrain = scaler.fit_transform(X_train.to_numpy())
    normalized_Xtest = scaler.transform(X_test.to_numpy())
    # normalized_Xval = scaler.transform(X_val.to_numpy())

    result = {'Descs': ['combinations'], 'Acc_average': 0}
    RESULT = pd.DataFrame(result, index=[0])

    for subs in random_moles:
        # print(subs)
        Qubits = len(subs[0])  # Number of qubits/features
        PARAM = ParameterVector('x', Qubits)
        qreg_q = QuantumRegister(Qubits, 'q')
        circuit = QuantumCircuit(qreg_q)
        for j in range(len(subs)):
            mole_convert(circuit, qreg_q, PARAM,subs[j],j)
        
        matrix_Zfeaturemap = [['H', 'U1']]
        for i in range(Qubits - 1):
            matrix_Zfeaturemap.append(['H', 'U1'])

        feature_map = circ_convert(circuit,qreg_q,PARAM,matrix_Zfeaturemap, 2)

        Circuit_added = circuit

        featuremap = feature_map & Circuit_added

        print(featuremap)

        # quantum_instance = QuantumInstance(backend, shots=1, seed_simulator=seed, seed_transpiler=seed)
        # Kernel = QuantumKernel(feature_map=featuremap, quantum_instance=quantum_instance, enforce_psd=False)
        Kernel = FidelityStatevectorKernel(feature_map=featuremap, shots=None)
        svc = SVC(kernel=Kernel.evaluate, probability=True, random_state=1234) #Fix random seed to make bic deterministic 
        svc.fit(normalized_Xtrain, np.array(Y_train).ravel())

        y_predicted = svc.predict(normalized_Xtest)
        # y_pred_plot = svc.predict(normalized_Xval)
        P_t = svc.predict_proba(normalized_Xtest)
        # P_v = svc.predict_proba(normalized_Xval)

        acc0_t, acc1_t, Low_t, BIC_t = evaluation(y_predicted, Y_test, P_t,1)
        # acc0_v, acc1_v, Low_v, BIC_v = evaluation(y_pred_plot, Y_val, P_v,1)

        result = {'Descs': [subs], 'Acc_average': (acc0_t+acc1_t)/2}
        print(result)
        RES = pd.DataFrame(result)
        RESULT = pd.concat([RESULT, RES], ignore_index=True)
        
    return RESULT  


def circ_convert(circuit,qreg_q,PARAM,seq, m):
    """
    Function to convert quantum circuits to Qiskit circuits.

    Parameters:
    - circuit: QuantumCircuit, Qiskit quantum circuit.
    - qreg_q: QuantumRegister, quantum register.
    - PARAM: list, list of parameters.
    - seq: list, list of quantum circuits descriptors.
    - m: int, number of layers.

    Returns:
    - circuit: QuantumCircuit, updated Qiskit quantum circuit.
    
    """

    Qubits = 4  # Number of qubits/features

    # Reshape the sequence list into a matrix representation
    Descriptor = np.reshape(list(seq), (Qubits, m), order='F')

    for i in range(len(Descriptor[0])):
        for j in range(len(Descriptor)):
            if Descriptor[j][i] == 'U1':
                circuit.p(PARAM[j], qreg_q[j])
            elif Descriptor[j][i] == 'H':
                circuit.h(qreg_q[j])
            elif Descriptor[j][i] == 'X':
                circuit.cx(qreg_q[j - 1], qreg_q[j])
            elif Descriptor[j][i] == 'Xn':
                circuit.cx(qreg_q[j - 2], qreg_q[j])
            elif Descriptor[j][i] == 'Xm':
                circuit.cx(qreg_q[j - 3], qreg_q[j])
            elif Descriptor[j][i] == 'Xp':
                circuit.cx(qreg_q[j - 4], qreg_q[j])

    return circuit


def mole_convert(circuit, qubits, var, seq, j):
    """
    Function to convert molecular structures to Qiskit circuits.

    Parameters:
    - circuit: QuantumCircuit, Qiskit quantum circuit.
    - qubits: list, list of qubits.
    - var: list, list of parameters.
    - seq: str, molecular structure.
    - j: int, number of layers.

    Returns:
    - circuit: QuantumCircuit, updated Qiskit quantum circuit.
    
    """
    n = 0  # Initialize parameter index counter
    for i in range(len(seq)):
        if seq[i] == 'C':
            circuit.p(1*var[j], qubits[j])
            n += 1
        elif seq[i] == 'N':
            circuit.cx(qubits[j - 1], qubits[j])
        elif seq[i] == 'O':
            circuit.cx(qubits[j - 2], qubits[j])
        elif seq[i] == 'S':
            circuit.cx(qubits[j - 3], qubits[j])
        elif seq[i] == 'P':
            circuit.cx(qubits[j - 3], qubits[j])



def Plot_figures(x_train, Dark_Good, Good_acc=None, Bad_acc=None, param_number=None, atoms_num=None, Entanglement_num=None, Layers_num=None, 
                 labels_train=None, Total_FINGER=None, figsize=(8, 8), plot_type='scatter', hline_y_top=None, hline_y_bottom=None, hline_xmin=0, hline_xmax=1, 
                 text_x=None, text_y=None, text_s=None, fontsize=16):
    """
    Function to plot various figures related to Gershgorin circles, PCA, and QC design analysis.
    
    Parameters:
    - x_train: np.array, input training data (first two columns assumed to represent descriptors).
    - Dark_Good: list/array, list indicating "Good QC design" indices.
    - Good_acc: list/array, accuracy for good QC designs (optional).
    - Bad_acc: list/array, accuracy for bad QC designs (optional).
    - param_number: list/array, number of parametrized gates/atoms (optional).
    - atoms_num: list/array, number of gates/atoms (optional).
    - Entanglement_num: list/array, number of CNOT gates/entanglement (optional).
    - Layers_num: list/array, number of layers (optional).
    - labels_train: list/array, labels for the training data (optional, used in PCA plots).
    - Total_FINGER: list/array, dataset used for PCA analysis (optional).
    - figsize: tuple, size of the figure (default: (8, 8)).
    - plot_type: str, type of plot to generate or title, with options like 'scatter', 'accuracy', 'pca_analysis', etc.
    - hline_y: float or list, y-coordinate(s) to draw horizontal line(s) (optional).
    - hline_xmin: float, minimum x-coordinate for horizontal line(s) (optional).
    - hline_xmax: float, maximum x-coordinate for horizontal line(s) (optional).
    - text_x: float, x-coordinate for text annotation (optional).
    - text_y: float, y-coordinate for text annotation (optional).
    - text_s: str, string for the text annotation (optional).
    - fontsize: int, font size for labels, ticks, and legends (default: 16).
    
    All plots have fixed fontsize of 14 for labels, ticks, and legends.
    """
    plt.figure(figsize=figsize)
    
    if plot_type == 'pca_analysis':
        # PCA analysis plot
        
        if Total_FINGER is None or labels_train is None:
            print("Total_FINGER or labels_train is missing. Cannot perform PCA analysis.")
            return
        
        # Step 1: PCA Variance Plot
        x_train_pca = Total_FINGER
        pca_finger = PCA(0.99)
        principalComponents = pca_finger.fit_transform(np.array(x_train_pca))
        variance = pca_finger.explained_variance_ratio_
        var = np.cumsum(np.round(variance, decimals=3))
        
        plt.figure(figsize=(8, 6))
        plt.ylabel('% Variance Explained', fontsize=fontsize)
        plt.xlabel('# of Features', fontsize=fontsize)
        plt.title('PCA Analysis', fontsize=fontsize)
        plt.plot(var)
        plt.show()
        
        print("Number of components: ", pca_finger.n_components_)

        # Step 2: PCA Scatter Plot (2D)
        principal_Df = pd.DataFrame(data=principalComponents[:, [0, 1]], columns=['principal component 1', 'principal component 2'])
        
        plt.figure(figsize=(10, 10))
        plt.xticks(fontsize=fontsize)
        plt.yticks(fontsize=fontsize)
        plt.xlabel('First Principal Component', fontsize=fontsize)
        plt.ylabel('Second Principal Component', fontsize=fontsize)
        
        targets = [1, 0]
        colors = ['r', 'g']
        markers = ['o','^']

        for target, color, marker in zip(targets, colors, markers):
            indicesToKeep = labels_train == target
            plt.scatter(principal_Df.loc[indicesToKeep, 'principal component 1'], 
                        principal_Df.loc[indicesToKeep, 'principal component 2'], c=color, s=20, marker=marker)
        
        # plt.legend(['Good', 'Bad'], prop={'size': fontsize})
        # plt.savefig('PCA_Hidden.pdf')
        plt.show()

        # Coefficients for the first two principal components
        idxs1 = np.array(list(zip(*sorted([(val, i) for i, val in enumerate(np.abs(pca_finger.components_[0]))], reverse=True))))
        DF_PC1 = pd.DataFrame(data=idxs1.T, columns=['Coefficient', 'Index'])
        print("Top 15 features influencing PC1:\n", DF_PC1[:15])

        idxs2 = np.array(list(zip(*sorted([(val, i) for i, val in enumerate(np.abs(pca_finger.components_[1]))], reverse=True))))
        DF_PC2 = pd.DataFrame(data=idxs2.T, columns=['Coefficient', 'Index'])
        print("Top 15 features influencing PC2:\n", DF_PC2[:15])
    
    if plot_type == 'scatter':
        # Scatter plot of Gershgorin circles
        plt.scatter(np.array(x_train)[:len(Dark_Good), 0], np.array(x_train)[:len(Dark_Good), 1], color='r', s=12, label='Performant QC design', marker='o')
        plt.scatter(np.array(x_train)[len(Dark_Good):, 0], np.array(x_train)[len(Dark_Good):, 1], color='g', s=12, label='Underperforming QC design', marker='^')
        plt.xlabel(r"$R_{\rm min}$", fontsize=fontsize)
        plt.ylabel(r"$R_{\rm max}$", fontsize=fontsize)
        plt.legend(fontsize=fontsize)
        
    elif plot_type == 'accuracy_vs_radius':
        # Scatter plot of accuracy vs. radius
        plt.scatter(np.array(x_train)[:len(Dark_Good), 0], Good_acc, color='r', s=12, label='Performant QC design', marker='o')
        plt.scatter(np.array(x_train)[len(Dark_Good):, 0], Bad_acc, color='g', s=12, label='Underperforming QC design', marker='^')
        
        if hline_y_top is not None:
            plt.hlines(y=hline_y_bottom, xmin=hline_xmin, xmax=hline_xmax, linestyles='--', color='g')
            plt.hlines(y=hline_y_top, xmin=hline_xmin, xmax=hline_xmax, linestyles='--', color='r')
        
        if text_x is not None and text_y is not None and text_s is not None:
            plt.text(x=text_x, y=text_y, s=text_s, fontsize=fontsize)
            
        plt.xlabel(r"$R_{\rm min}$", fontsize=fontsize)
        plt.ylabel('Average classification accuracy', fontsize=fontsize)
        plt.legend(fontsize=fontsize)
        plt.xlim((hline_xmin, hline_xmax))
    
    elif plot_type == 'accuracy_vs_index':
        # Scatter plot of accuracy vs. radius
        plt.scatter(range(100), Good_acc, color='r', s=12, label='Performant QC design', marker='o')
        plt.scatter(range(100), Bad_acc, color='g', s=12, label='Underperforming QC design', marker='^')
        
        if hline_y_top is not None:
            plt.hlines(y=hline_y_bottom, xmin=hline_xmin, xmax=hline_xmax, linestyles='--', color='g')
            plt.hlines(y=hline_y_top, xmin=hline_xmin, xmax=hline_xmax, linestyles='--', color='r')
        
        if text_x is not None and text_y is not None and text_s is not None:
            plt.text(x=text_x, y=text_y, s=text_s, fontsize=fontsize)
            
        plt.xlabel('Random molecule (quantum circuit) index', fontsize=fontsize)
        plt.ylabel('Average classification accuracy', fontsize=fontsize)
        plt.legend(fontsize=fontsize)
        plt.xlim((hline_xmin, hline_xmax))

    elif plot_type == 'kde':
        # KDE plot of Gershgorin circles
        sns.kdeplot(x=np.array(x_train)[:len(Dark_Good), 0], y=np.array(x_train)[:len(Dark_Good), 1], shade=True, cmap='Reds', bw_adjust=0.5)
        sns.kdeplot(x=np.array(x_train)[len(Dark_Good):, 0], y=np.array(x_train)[len(Dark_Good):, 1], shade=True, cmap='Greens', bw_adjust=0.5)
        plt.xlabel(r"$R_{\rm min}$", fontsize=fontsize)
        plt.ylabel(r"$R_{\rm max}$", fontsize=fontsize)
        plt.title('Probability density', fontsize=fontsize)
        plt.legend(['Performant QC', 'Underperforming QC'], fontsize=fontsize)
    
    elif plot_type == 'kde_prob_good' or plot_type == 'kde_prob_bad':
        # KDE probability calculation and contour plot for "Good" and "Bad"
        x = np.linspace(np.array(x_train)[:, 0].min(), np.array(x_train)[:, 0].max(), 100)
        y = np.linspace(np.array(x_train)[:, 1].min(), np.array(x_train)[:, 1].max(), 100)
        xx, yy = np.meshgrid(x, y)
        grid_points = np.vstack([xx.ravel(), yy.ravel()]).T
        
        # KDE fitting
        kde_good = KernelDensity(bandwidth=1.0, kernel='gaussian')
        kde_good.fit(np.array(x_train)[:len(Dark_Good), :2])
        kde_bad = KernelDensity(bandwidth=1.0, kernel='gaussian')
        kde_bad.fit(np.array(x_train)[len(Dark_Good):, :2])
        
        log_density_good = kde_good.score_samples(grid_points)
        log_density_bad = kde_bad.score_samples(grid_points)
        
        density_good = np.exp(log_density_good).reshape(xx.shape)
        density_bad = np.exp(log_density_bad).reshape(xx.shape)
        
        prob_good = density_good / (density_good + density_bad)
        prob_bad = density_bad / (density_good + density_bad)
        
        if  plot_type == 'kde_prob_good':
            # Plot probability for "Good"
            plt.contourf(xx, yy, prob_good, cmap='Reds', alpha=0.6, levels=20)
            plt.colorbar(label='Probability density', fontsize=fontsize)
        else:
            # Plot probability for "Bad"
            plt.contourf(xx, yy, prob_bad, cmap='Greens', alpha=0.6, levels=20)
            plt.colorbar(label='Probability density', fontsize=fontsize)
        
        plt.xlabel(r"$R_{\rm min}$", fontsize=fontsize)
        plt.ylabel(r"$R_{\rm max}$", fontsize=fontsize)
    
    elif plot_type == 'line_plot':
        # Line plots for probability densities
        x = np.linspace(np.array(x_train)[:, 0].min(), np.array(x_train)[:, 0].max(), 100)
        y = np.linspace(np.array(x_train)[:, 1].min(), np.array(x_train)[:, 1].max(), 100)
        xx, yy = np.meshgrid(x, y)
        grid_points = np.vstack([xx.ravel(), yy.ravel()]).T

        # KDE fitting
        kde_good = KernelDensity(bandwidth=1.0, kernel='gaussian')
        kde_good.fit(np.array(x_train)[:len(Dark_Good), :2])
        kde_bad = KernelDensity(bandwidth=1.0, kernel='gaussian')
        kde_bad.fit(np.array(x_train)[len(Dark_Good):, :2])
        
        log_density_good = kde_good.score_samples(grid_points)
        log_density_bad = kde_bad.score_samples(grid_points)
        
        density_good = np.exp(log_density_good).reshape(xx.shape)
        density_bad = np.exp(log_density_bad).reshape(xx.shape)
        
        prob_good = density_good / (density_good + density_bad)
        prob_bad = density_bad / (density_good + density_bad)

        prob_data_good = pd.DataFrame({'Descriptor1': xx.ravel(), 'Descriptor2': yy.ravel(), 'Probability': prob_good.ravel()})
        prob_data_bad = pd.DataFrame({'Descriptor1': xx.ravel(), 'Descriptor2': yy.ravel(), 'Probability': prob_bad.ravel()})
        
        sns.lineplot(data=prob_data_good, x='Descriptor1', y='Probability', label='Perfomant QC design', color='red')
        sns.lineplot(data=prob_data_bad, x='Descriptor1', y='Probability', label='Underperforming QC design', color='green', linestyle='--')
        plt.xlabel(r"$R_{\rm min}$", fontsize=fontsize)

        # sns.lineplot(data=prob_data_good, x='Descriptor2', y='Probability', label='Good QC design', color='red')
        # sns.lineplot(data=prob_data_bad, x='Descriptor2', y='Probability', label='Bad QC design', color='green')
        # plt.xlabel(r"$R_{\rm max}$", fontsize=fontsize)
            
        plt.ylabel('Probability density', fontsize=fontsize)
        plt.legend(fontsize=fontsize)
        # plt.xlim((70,320))
    
    elif plot_type == 'param_vs_radii':
        # Scatter plot of param_number vs radii
        plt.scatter(np.array(x_train)[:len(Dark_Good), 0], np.array(param_number)[:len(Dark_Good)], color='r', s=12, label='Performant QC design', marker='o')
        plt.scatter(np.array(x_train)[len(Dark_Good):, 0], np.array(param_number)[len(Dark_Good):], color='g', s=12, label='Underperforming QC design', marker='^')
        plt.xlabel(r"$R_{\rm min}$", fontsize=fontsize)
        plt.ylabel('Number of parametrized gates/Number of carbon atoms', fontsize=fontsize)
        plt.legend(fontsize=fontsize)

    elif plot_type == 'layers_vs_radii':
        # print(Layers_num)
        # Scatter plot of Layers_num vs radii
        plt.scatter(np.array(x_train)[:len(Dark_Good), 0], np.array(Layers_num)[:len(Dark_Good)], color='r', s=12, label='Performant QC design', marker='o')
        plt.scatter(np.array(x_train)[len(Dark_Good):, 0], np.array(Layers_num)[len(Dark_Good):], color='g', s=12, label='Underperforming QC design', marker='^')
        plt.xlabel(r"$R_{\rm min}$", fontsize=fontsize)
        plt.ylabel('Number of layers', fontsize=fontsize)
        plt.legend(fontsize=fontsize)

    # Customize ticks for all plots
    plt.xticks(fontsize=fontsize)
    plt.yticks(fontsize=fontsize)
    # plt.tight_layout()
    plt.show()




A=[]

# with open('./QCs_vs_accuracies/QCs_from_mol_4DHidden_100random.csv', newline='') as csvfile1:
# with open('./QCs_vs_accuracies/QCs_from_mol_4DPerovskite_100random.csv', newline='') as csvfile1:
with open('./QCs_vs_accuracies/QCs_4DHidden10000.csv', newline='') as csvfile1:
# with open('./QCs_vs_accuracies/QCs_4DPerovskite10000.csv', newline='') as csvfile1:
    desc1 = csv.reader(csvfile1, delimiter='|', quotechar='|')
    for row in desc1:
        A.append(row[0])
    
Descriptors=np.array(A)       
print("Total dataset length:  ", len(Descriptors))

C=[]       
# with open('./QCs_vs_accuracies/Acc_4DPerovskite_100random.csv', newline='') as csvfile6:
# with open('./QCs_vs_accuracies/Acc_4DHidden_100random.csv', newline='') as csvfile6:
with open('./QCs_vs_accuracies/Acc_4DHidden10000.csv', newline='') as csvfile6:
# with open('./QCs_vs_accuracies/Acc_4DPerovskite10000.csv', newline='') as csvfile6:
    desc6 = csv.reader(csvfile6, delimiter='|', quotechar='|')
    for row in desc6:
        C.append(float(row[0])) 

Ave_accuracy=np.array(C)  

# 8L Hidden dataset
# file_path = './QCs_vs_accuracies/Hidden_8L_desc_acc.csv'  # Replace with your actual file path
# df = pd.read_csv(file_path)

# # Select the 'QC_descriptor' and 'Energy' columns
# Descriptors = df['QC']
# Ave_accuracy = df['Acc']

Good_design,Bad_design,Desc,labels,Good_acc, Bad_acc=Good_Bad_split(Descriptors,Ave_accuracy)

Dark_Good = Good_design
Dark_Bad = Bad_design
print(len(Dark_Good))
print(len(Dark_Bad))
Total_descriptors=np.concatenate((Dark_Good,Dark_Bad))
print("Samples length:   ", len(Total_descriptors))
print(Total_descriptors[0])
Cleaned=Cleaning(Total_descriptors)
print(Cleaned[0])

#The part you can use to generate SMILES from QCs
Total_SMILE,param_number,atoms_num,Entanglement_num,Layers_num = SMILE_convert(Cleaned)
Moles,Total_FINGER=Finger_Convert(Total_SMILE)


#3D descriptors based on eigenvalues of Coulomb matrices
# Eigs = eig_columb(Total_SMILE)
# x_train = physical_descriptors(Eigs)

#The part you can use to generate Coulomb matrices from SMILES
CoulombMats, x_train = Gershgorin_descriptors(Total_SMILE)

l_g=np.ones((1,len(Dark_Good)))
l_b=np.zeros((1,len(Dark_Bad)))
labels_train= np.array(np.concatenate((l_g[0][:len(Dark_Good)],l_b[0][:len(Dark_Bad)])))

# Main graphs for 4D datasets
Plot_figures(x_train, Dark_Good, Good_acc, Bad_acc, figsize=(8, 8), plot_type='scatter')
Plot_figures(x_train, Dark_Good, Good_acc, Bad_acc, figsize=(12, 6), plot_type='line_plot')
Plot_figures(x_train, Dark_Good, Good_acc, Bad_acc, labels_train=labels_train, Total_FINGER=Total_FINGER, plot_type='pca_analysis')
# Plot_figures(x_train, Dark_Good, Good_acc, Bad_acc, param_number, figsize=(8, 8), plot_type='param_vs_radii')
# Plot_figures(x_train, Dark_Good, Good_acc, Bad_acc, Layers_num = Layers_num, figsize=(8, 8), plot_type='layers_vs_radii')

# Hidden dataset- 100 random molecules
# Plot_figures(x_train, Dark_Good, Good_acc, Bad_acc, figsize=(8, 8), plot_type='accuracy_vs_radius', hline_y_top=0.63, hline_y_bottom=0.6, hline_xmin=200, hline_xmax=320, text_x=250, text_y=0.615, text_s=r'$\pm 10\% $ margin')
# Perovskite dataset- 100 random molecules
# Plot_figures(x_train, Dark_Good, Good_acc, Bad_acc, figsize=(8, 8), plot_type='accuracy_vs_radius', hline_y_top=0.64, hline_y_bottom=0.56, hline_xmin=300, hline_xmax=360, text_x=325, text_y=0.615, text_s=r'$\pm 10\% $ margin')

# Graphs for random generated 5D datsets graphs
# data = pd.read_csv('./QCs_vs_accuracies/Hidden_5D_acc.csv')
# # data = pd.read_csv('./QCs_vs_accuracies/Mnist_5D_acc.csv')

# # # Extract the columns for the scatter plot
# rand_good = data['High']
# rand_bad = data['Low']

# 5D Mnist
# Plot_figures(x_train, Dark_Good, rand_good, rand_bad, figsize=(8, 8), plot_type='accuracy_vs_index', hline_y_top=0.91, hline_y_bottom=0.84, hline_xmin=0, hline_xmax=100)

# 5D Hidden
# Plot_figures(x_train, Dark_Good, rand_good, rand_bad, figsize=(8, 8), plot_type='accuracy_vs_index', hline_y_top=0.73, hline_y_bottom=0.68, hline_xmin=0, hline_xmax=100)
