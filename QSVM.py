#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 26 17:51:15 2024

This script implements the compositional algorithm to find the optimal quantum kernel for Quantum Support Vector Machine (QSVM)

Author: Elham Torabian
"""

# Import necessary libraries
import numpy as np
import pandas as pd
from itertools import combinations_with_replacement
from scipy.optimize import minimize
from scipy.stats import norm
# from datetime import datetime
import random

from qiskit_aer import Aer
from qiskit import QuantumCircuit, QuantumRegister
from qiskit_algorithms.utils import algorithm_globals
# from qiskit import IBMQ
from qiskit.circuit import ParameterVector
from qiskit_machine_learning.kernels import FidelityStatevectorKernel

from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.datasets import fetch_openml
from sklearn.utils import check_random_state
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import log_loss
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF

# from bayes_opt import UtilityFunction

from skopt import Optimizer
from skopt import gp_minimize
# from skopt.plots import plot_convergence, plot_gaussian_process
from bayes_opt import BayesianOptimization
import os

# task_id = int(os.getenv("SLURM_ARRAY_TASK_ID"))

print("Current path:  ", os.getcwd())

class QuantumSVMOptimizer:
    def __init__(self):
        # Initialize your variables here
        self.n_feature = 4  # Number of features
        self.L = 8  # Number of layers

        self.feature_dim = self.n_feature
        self.PARAM = ParameterVector('x', self.n_feature)
        self.pp = [1 for i in range(self.feature_dim*self.L)]

        self.Classical_kernel='rbf'

        # Load and preprocess data
        self.load_and_preprocess_data()
        
        # Prepare combinations of gates for the first layer
        Gates = ['U1', 'X', 'Xn', 'Xm', '0']
        self.All_one_l = list(combinations_with_replacement(Gates, r=self.n_feature*self.L))

        print(f"All {self.L} layer QCs are:   ", len(self.All_one_l))

    def load_and_preprocess_data(self):
        # Load your dataset here
        self.X_train = pd.read_csv('/scratch/st-rkrems-1/elhamtr/QC/QCs_descriptors/hidden_manifold_model_4D_1000/hidden_manifold_model_X_train.csv').reset_index(drop=True)
        self.Y_train = pd.read_csv('/scratch/st-rkrems-1/elhamtr/QC/QCs_descriptors/hidden_manifold_model_4D_1000/hidden_manifold_model_y_train.csv').reset_index(drop=True)
        self.X_test = pd.read_csv('/scratch/st-rkrems-1/elhamtr/QC/QCs_descriptors/hidden_manifold_model_4D_1000/hidden_manifold_model_X_test.csv').reset_index(drop=True)
        self.Y_test = pd.read_csv('/scratch/st-rkrems-1/elhamtr/QC/QCs_descriptors/hidden_manifold_model_4D_1000/hidden_manifold_model_y_test.csv').reset_index(drop=True)

        # Normalize data points
        scaler = preprocessing.MinMaxScaler(feature_range=(0, 2*np.pi))
        self.normalized_Xtrain = scaler.fit_transform(self.X_train.to_numpy())
        self.normalized_Xtest = scaler.transform(self.X_test.to_numpy())
        # self.normalized_Xval = scaler.transform(self.X_val.to_numpy())

    def circ_convert(self, seq, m, param):
        """
        Construct the quantum circuit based on the given sequence of gates and parameters.

        Args:
            seq (list): Sequence of gates (number of features x m).
            m (int): Number of layers.
            param (list): Parameters values of the Z-rotation gates.

        Returns:
            QuantumCircuit: The constructed quantum circuit.
        """
        n = 0  # Initialize parameter index counter
        Qubits = self.n_feature  # Number of qubits/features

        # Reshape the sequence list into a matrix representation
        Descriptor = np.reshape(list(seq), (Qubits, m), order='F')
        
        qreg_q = QuantumRegister(Qubits, 'q')
        circuit = QuantumCircuit(qreg_q)
        for i in range(len(Descriptor[0])):
            for j in range(len(Descriptor)):
                if Descriptor[j][i] == 'U1':
                    circuit.p(param[n] * self.PARAM[j], qreg_q[j])
                    n += 1
                elif Descriptor[j][i] == 'H':
                    circuit.h(qreg_q[j])
                elif Descriptor[j][i] == 'X':
                    circuit.cx(qreg_q[j - 1], qreg_q[j])
                elif Descriptor[j][i] == 'Xn':
                    circuit.cx(qreg_q[j - 2], qreg_q[j])
                elif Descriptor[j][i] == 'Xm':
                    circuit.cx(qreg_q[j - 3], qreg_q[j])

        return circuit

    def calculate_bic(self, n, LL, num_params):
        """
        Calculate Bayesian Information Criteria (BIC) for the proposed model.

        Args:
            n (int): Number of training points.
            num_params (int): Number of parameters in the model.
            LL (float): Log likelihood.

        Returns:
            BIC value of the proposed model (float).
        """
        bic = (-2) * LL + num_params * np.log(n)
        return bic


    def evaluation(self, y_pred, Y_real, P_pred, param_num):
        """
        Calculate statistical criteria for the proposed model.

        Args:
            y_pred (list/numpy array/dataframe): Predicted labels.
            Y_real (list/numpy array/dataframe): Real labels in the dataset.
            P_pred (list/numpy array/dataframe): Predicted probability of correct label.

        Returns:
            acc0 (float): True Positive Rate (TPR)
            acc1 (float): True Negative Rate (TNR)
            Low_acc (float): Min(TPR,TNR)
            BIC (float): Bayesian Information Criteria.
        """
        count0 = 0
        count1 = 0
        t = 0
        r = 0

        Yr = Y_real.iloc[:, 0].tolist()

        for j in range(len(y_pred)):
            if Yr[j] == 1:
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
        BIC = self.calculate_bic(len(self.normalized_Xtrain), LogLoss, param_num)

        return acc0, acc1, Low_acc, BIC


    def get_featuremap(self, comb, Param, m):
        """
        Args:
            comb (list): Given combination of gates.
            Param (list/numpy array): Parameters values of the Z-rotation gates.
            m (int): Number of layers.

        Returns:
            featuremep
        """
        par_map = [1, 1, 1, 1, 1, 1]
        matrix_Zfeaturemap = [['H', 'U1']]
        for i in range(self.n_feature - 1):
            matrix_Zfeaturemap.append(['H', 'U1'])

        feature_map = self.circ_convert(matrix_Zfeaturemap, 2, par_map)

        Circuit_added = self.circ_convert(comb, m, Param)

        featuremap = feature_map & Circuit_added

        return featuremap


    def classification(self, comb, Param, m):
        """
        QSVM algorithm for a given layout of gates (comb) with corresponding parameters.

        Args:
            comb (list): Given combination of gates.
            Param (list/numpy array): Parameters values of the Z-rotation gates.
            m (int): Number of layers.

        Returns:
            RESULT (dataframe): Statistical criteria for the test and validation subsets.
            BIC_v (float): The BIC corresponding to the quantum model
        """
        par_map = [1, 1, 1, 1, 1, 1]
        matrix_Zfeaturemap = [['H', 'U1']]
        for i in range(self.n_feature - 1):
            matrix_Zfeaturemap.append(['H', 'U1'])

        feature_map = self.circ_convert(matrix_Zfeaturemap, 2, par_map)

        Circuit_added = self.circ_convert(comb, m, Param)

        featuremap = feature_map & Circuit_added

        # quantum_instance = QuantumInstance(backend, shots=1, seed_simulator=seed, seed_transpiler=seed)
        # Kernel = QuantumKernel(feature_map=featuremap, quantum_instance=quantum_instance, enforce_psd=False)
        Kernel = FidelityStatevectorKernel(feature_map=featuremap, shots=None)
        svc = SVC(kernel=Kernel.evaluate, probability=True, random_state=1234) #Fix random seed to make bic deterministic 
        svc.fit(self.normalized_Xtrain, np.array(self.Y_train).ravel())

        y_predicted = svc.predict(self.normalized_Xtest)
        # y_pred_plot = svc.predict(self.normalized_Xval)
        P_t = svc.predict_proba(self.normalized_Xtest)
        # P_v = svc.predict_proba(self.normalized_Xval)

        acc0_t, acc1_t, Low_t, BIC_t = self.evaluation(y_predicted, self.Y_test, P_t,len(Param))
        # acc0_v, acc1_v, Low_v, BIC_v = self.evaluation(y_pred_plot, self.Y_val, P_v)

        result = {'Descs': [comb], 'Acc0': acc0_t, 'Acc1': acc1_t, 'Acc_ave': (acc0_t+acc1_t)/2, 'BIC_t': BIC_t}
        RESULT = pd.DataFrame(result)


        return RESULT, BIC_t


    def optimization(self, N_iter, Combs, l):
        """
        Optimize the parameters of the model (here, parameters of Z-rotation gates).

        Args:
            N_iter (int): Number of optimization iterations.
            Combs (list): A list of combination of gates to be optimized.
            l (int): Number of layers.

        Returns:
            Optimized_res (dataframe): Statistical criteria for the test and validation subsets for the optimized quantum circuits.
            Parameters (list): Optimized parameters.
        """        
        Optimized_res = pd.DataFrame()
        Parameters = []
        for i in range(len(Combs)):
            self.comb = Combs[i]
            Opt_params, MinBIC = self.BO_pack(N_iter, self.comb, l)
            ress, bic = self.classification(self.comb, Opt_params, l)

            Optimized_res = pd.concat([Optimized_res, ress], ignore_index=True, axis=0)
            Parameters.append(Opt_params)
        return Optimized_res, Parameters


    def objective_function(self, params):
        """
        Objective function for the optimization process (minimize BIC).

        Args:
            params (list/numpy array): Given parameters value of the model.

        Returns:
            BIC (float): Bayesian Information Criteria.
        """
        
        Result_total, bic = self.classification(self.comb, params, int(len(self.comb) / 4)) # 4 is number of features (qubits)
        return bic

    def P_counter(self, comb):
        """
        Count the number of parameters in the circuit (here, the number of Z-rotation gates).

        Args:
            comb (list): Given combination of gates.

        Returns:
            count (int): Number of parameters in the circuit.
        """
        count = 0
        for row in self.comb:
            if row == 'U1':
                count += 1
        return count


    def BO_pack(self, N, comb, m):
        """
        Perform Bayesian Optimization to optimize the parameters of the model using a Gaussian Process as a surrogate model.

        Args:
            N (int): Number of optimization iterations.
            comb (list): Combination of gates for which parameters are to be optimized.
            m (int): Number of layers.

        Returns:
            res.x (list/numpy array): Optimized parameters.
            res.fun (float): Optimized BIC.
        """
        pnum = self.P_counter(self.comb)
        pbounds = []
        for i in range(pnum):
            pbounds.append((0, 2 * 3.14))

        # Perform Bayesian Optimization using Gaussian Process
        res = gp_minimize(self.objective_function, pbounds, random_state=43, acq_func="LCB", n_calls=N, n_random_starts=5,
                        noise=0.1 ** 2)
        # plot = plot_convergence(res, yscale='log')
#         FIG = plot.get_figure()
#         if not hasattr(self, "irep"):
#             self.irep = 1
#         FIG.savefig('Convergence_1param'+str(self.L)+'L_rep'+str(self.irep)+'.png')
        print(f"BO_pack: params {res.x}, value {res.fun}")
        return res.x, res.fun



    def cleaning(self, Desc):
        """
        Clean the dataset that should be read from an Excel file.

        Args:
            Desc (list of lists): Quantum circuit descriptors.

        Returns:
            Cleaned (list of lists): Cleaned QC descriptors.
        """
        Cleaned = []
        for row in Desc:
            desc = []
            for i in range(len(row)):
                if row[i:i + 2] == 'U1':
                    desc.append('U1')
                elif row[i:i + 2] == "X'":
                    desc.append('X')
                elif row[i:i + 2] == 'Xn':
                    desc.append('Xn')
                elif row[i:i + 2] == 'Xm':
                    desc.append('Xm')
                elif row[i:i + 1] == '0':
                    desc.append('0')
                else:
                    pass

            if desc == []:
                pass
            else:
                Cleaned.append(desc)
        return Cleaned



    def Random_QC_QSVM(self):
        """
        Greedy search method to increase the complexity of the QCs and span the space of all combinations of gates.

        Args:
            M (int): Number of circuits chosen from M QCs to be optimized.
            K (int): Number of circuits chosen to be used to append to a new layers.
            L (int): Number of layers.

        Returns:
            RESULT (dataframe): Statistical result of the QCs with L layers.
        """
        self.pp[0:self.feature_dim*self.L] = [1 for i in range(self.feature_dim*self.L)]
        result = {'Descs': ['comb'], 'Acc0': 0, 'Acc1': 0, 'Acc_ave': 0, 'BIC_t': 0}
        RESULT = pd.DataFrame(result, index = [0])
        combs = random.sample(self.All_one_l, 10000)
#         combs = self.All_one_l[:10000]
        print("Len(selected_combs) is:   ", len(combs))
        # print("The comb is   :", list(combs[1]))
        for comb in list(combs):
        # ind = [10*(task_id-1),10*(task_id)]
        # for i in range(ind[0],ind[1]):
            Result_f, bic = self.classification(list(combs[i]), self.pp, self.L)
            RESULT = pd.concat([RESULT,Result_f], ignore_index=True)
        
        RESULT.sort_values(by='Acc_ave', inplace=True, ascending=False)
        # path = '/scratch/st-rkrems-1/elhamtr/QC/QCs_descriptors/Array_QSVM_Hidden_1000/QSVM_Hidden4D_' + str(self.L) + 'L_task'+str(task_id)+'.csv'
        # RESULT.to_csv(path)

        print("The best ", str(self.L), " layer circuit before optimization is:\n", RESULT.head(5))

        return RESULT
    
    def run(self):
        # Call your main processing steps here...
        self.Random_QC_QSVM()
        self.optimize_final(l = 5)
        # self.classical()

        # Add more logic as needed...

    def classical(self):
        '''
        Classical SVM applied to a given dataset
        '''
        classical_k=self.Classical_kernel
        rbf_svc = SVC(kernel=classical_k,probability=True)
        # print(f"shape is {Y_train.shape}")
        rbf_svc.fit(self.normalized_Xtrain, self.Y_train.to_numpy().ravel())
        Ypred_rbf=rbf_svc.predict(self.normalized_Xtest)
        # Ypred_rbf_val=rbf_svc.predict(self.normalized_Xval)
        # P_vrbf=rbf_svc.predict_proba(self.normalized_Xval)
        P_trbf=rbf_svc.predict_proba(self.normalized_Xtest)
        # acc0_vrbf,acc1_vrbf,acc_low_vrbf,BIC_vrbf=self.evaluation(Ypred_rbf_val,Y_val,P_vrbf)
        # print('The lowest classical (based on RBF) validation accuracy:   ', acc_low_vrbf)
        acc0_trbf,acc1_trbf,acc_low_trbf,BIC_trbf=self.evaluation(Ypred_rbf,self.Y_test,P_trbf,1)
        print('The average classical (based on RBF) test accuracy:   ',(acc0_trbf+acc1_trbf)/2)
        print('The BIC corresponding to classical (based on RBF) model:   ',BIC_trbf)

    def optimize_final(self, l, maxiter=20, irep=1):
        """
        L (int): Number of layers.

        Returns:
            RESULT (dataframe): Statistical result of the QCs with L layers.
        """
        self.irep = irep
        path = '/scratch/st-rkrems-1/elhamtr/QC/QCs_descriptors/Array_QSVM_Hidden_1000/QSVM_Greedy_' + str(l) + 'l_Hidden1000.csv'
        data = pd.read_csv(path)
        Data = data.to_numpy()
        Dess = []
        for row in Data:
            Dess.append(row[2])
		# sorteddict = self.cleaning(Dess[0:K])
        sorteddict = self.cleaning(Dess)
        To_opt = sorteddict
        print("To_opt are:  ", To_opt)
#                 To_opt = sorteddict
        out = []
        if(len(To_opt)>0):
            print("Optimizing quantum circuits parameters for layer", str(l), "...")
			# Number of iterations cannot be less than 20
            Opt_Result, Opt_parameters = self.optimization(20, np.array(To_opt), l)
            print("The best ", str(l), " layer circuit after optimization is:\n", Opt_Result.loc[0, :])
            out = Opt_Result['Descs'].tolist()
        path = '/scratch/st-rkrems-1/elhamtr/QC/QCs_descriptors/Array_QSVM_Hidden_1000/Opt_QSVM_random_' + str(l) + 'L_Hidden.csv'
        Opt_Result.to_csv(path)

        return out


# This main block initializes the class and calls the main method.
if __name__ == "__main__":
    qsvm_optimizer = QuantumSVMOptimizer()
    qsvm_optimizer.run()
