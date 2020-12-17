#Importing Dependencies
import sys
import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import rbf_kernel
from scipy import linalg as sLA
from sklearn import svm
from sklearn import discriminant_analysis
from sklearn.cluster import DBSCAN
import main

class Fault_Detection(object):
    # Variables
    clf_convergent = False
    normal_data = []
    data = []
    data_sample = 0
    data_stream = []
    reference_size = 0
    buffer_size = 0
    kernel = ""
    gamma = 0
    degree = 0
    eiglength = 0
    wcss_threshold = 0
    accuracy_loop = 0
    accuracy = []
    accuracy_total = 0
    label_buffer = []
    df_acc = []
    df_label = []
    classifier = "Support Vector Machines"

    #get control parameters
    def getReference(self, reference_size):
        self.reference_size = reference_size
        print(self.reference_size)
    def getBuffer(self, buffer_size):
        self.buffer_size = buffer_size
        print(self.buffer_size)
    def getKernel(self, kernel):
        self.kernel = kernel
        print(self.kernel)
    def getGamma(self, gamma):
        self.gamma = gamma
        print(self.gamma)
    def getDegree(self, degree):
        self.degree = degree
        print(self.degree)
    def getEig(self, eig):
        self.eiglength = eig
        print(self.eiglength)
    def getWcssThres(self, wcss_threshold):
        self.wcss_threshold = wcss_threshold
        print(self.wcss_threshold)
    def getClassifier(self, classifier):
        self.classifier = classifier
        print(self.classifier)  
        
    #Fuctions
    ##Robust Scaler to scale features down
    def robust_scaler_fit(self, normal_reference):
        self.rb = RobustScaler()
        self.normal_reference = self.rb.fit_transform(self.normal_reference.T)
        self.normal_reference = self.normal_reference.T
        return self.normal_reference, self.rb

    def robust_scaler_transform(self, data, rb):
        self.data = self.rb.transform(self.data.T)
        self.data = self.data.T
        return self.data
        
    ## KFDA
    def KFDA_normal(self, normal_data):
        K = rbf_kernel(self.normal_data, gamma = self.gamma)
        K = pd.DataFrame.from_records(K)
        #### Centering Kernel Matrix
        m = np.shape(self.normal_data)[0]
        seper_m = np.ones((m,m))/m
        Kc = K - seper_m.dot(K) - K.dot(seper_m) + seper_m.dot(K).dot(seper_m)
        #### KPCA
        eigenValues, eigenVectors = np.linalg.eig(Kc)
        eigenValues = np.real(eigenValues)
        eigenVectors = np.real(eigenVectors)
        idx = eigenValues.argsort()[::-1] #Outputs idx for sorting eigenValues and ordering by descending value
        eigenValues = eigenValues[idx] #sorting eigenValues by idx
        eigenVectors = np.real(eigenVectors[:,idx]) #sorting eigenVectors by idx
        self.KPCA_Matrix = []
        for i in range(len(eigenValues)):
            if (eigenValues[i] < 1e-15):
                end_index = i
                break
            scaled_eigenvector = eigenVectors[:,i]/(np.sqrt(eigenValues[i]))
            self.KPCA_Matrix.append(scaled_eigenvector)
        self.KPCA_Matrix = np.array(self.KPCA_Matrix)
        self.KPCA_Matrix = self.KPCA_Matrix.T
        #self.KPCA_Matrix = eigenVectors[:,:self.eiglength] / np.sqrt(eigenValues[:self.eiglength]) #PCA Transformation Matrix
        #### Transforming Kernel Matrix into KPCA-space
        self.normal_reference = self.KPCA_Matrix.T.dot(K)
        self.normal_reference = pd.DataFrame.from_records(self.normal_reference)
        ### FDA
        L = pd.Series([1]*m)
        classes = np.unique(L)
        k = len(classes)
        M = self.normal_reference.mean(axis = 1)
        try:
            st = np.diagflat(eigenValues[:end_index]) #Total variance
        except NameError:
            st = np.diagflat(eigenValues[:]) #Total variance
        #### Looping to find sb
        sb=0 #initial value
        yt = self.normal_reference.T
        yt = pd.DataFrame.from_records(yt)
        for j in range(k):
            a = 0
            a = np.where(L == classes[j])[0]
            Kj = pd.DataFrame()
            an = len(a)
            for ak in range(an):
                ai = a[ak]
                k2add = self.normal_reference.iloc[:,ai]
                Kj[ak] = k2add
            nj = Kj.shape[1]
            Kj = Kj.to_numpy()
            mj = Kj.mean(axis = 1)
            sb = sb + nj*np.outer((mj-M),(mj-M)) #Variance between class

        sb = sb/m #Scaling sb
        st_i = np.linalg.inv(st)
        eigenValues1, eigenVectors1 = np.linalg.eig(st_i.dot(sb))
        eigenValues1 = np.real(eigenValues1)
        eigenVectors1 = np.real(eigenVectors1)
        idx1 = eigenValues1.argsort()[::-1] #Outputs idx for sorting eigenValues and ordering by descending value
        eigenValues1 = eigenValues1[idx1] #sorting eigenValues by idx
        eigenVectors1 = np.real(eigenVectors1[:,idx1]) #sorting eigenVectors by idx
        self.KFDA_Matrix = eigenVectors1[:,:2] / np.sqrt(eigenValues1[:2]) #KFDA Transformation Matrix
        #### Transforming KFDA Matrix into KFDA matrix
        self.normal_reference = self.KFDA_Matrix.T.dot(self.normal_reference)
        self.normal_reference = pd.DataFrame.from_records(self.normal_reference)
        return self.normal_reference, self.KPCA_Matrix, self.KFDA_Matrix

    ## Test if wcss difference is above threshold
    def wcss_test(self, X, wcss_threshold):
        #X_wcss = self.rs_wcss.transform(X)
        X_wcss = X
        wcss = []
        for i in range(1, 4):
            kmeans = KMeans(n_clusters = i, init = 'k-means++', random_state = 0)
            kmeans.fit(X_wcss)
            wcss.append(kmeans.inertia_)
        self.wcssDiff = (wcss[0]-wcss[1])/(wcss[1]-wcss[2])
        print('wcss difference: ', self.wcssDiff, "wcss diff: ", self.wcssDiff)
        self.wcss_result = self.wcssDiff > self.wcss_threshold
        return self.wcss_result

    ## Clustering for label handling
    def k_means_clustering(self, X):
        kmeans = KMeans(n_clusters = 2, init = 'k-means++', random_state = 42)
        self.L_KKN = kmeans.fit_predict(self.X)
        return self.L_KKN

    def db_clustering(self):
        clustering = DBSCAN(eps=0.01, min_samples=1).fit(self.X)
        self.L_KKN = clustering.labels_
        unique_classes = np.unique(self.L_KKN)
        #print(unique_classes)
        big_count = 0
        for db_class in unique_classes:
            x = len(self.L_KKN[self.L_KKN == db_class])
            if big_count < x:
                big_count = x
                big_class = db_class
        
        for i in range(len(self.L_KKN)):
            if self.L_KKN[i] == big_class:
                self.L_KKN[i] = 1
            else:
                self.L_KKN[i] = 0
        return self.L_KKN

    ### Training SVM 
    def SVM_Train(self, X, L_KKN):
        self.clf= svm.SVC(class_weight= {0: .8, 1: 1})
        self.clf = self.clf.fit(self.X, self.L_KKN)
        return self.clf

    ## Training LDA
    def LDA_Train(self, X, L_KKN):
        self.clf= discriminant_analysis.LinearDiscriminantAnalysis()
        self.clf = self.clf.fit(self.X, self.L_KKN)
        return self.clf

    c = ['red', 'blue']
    def KFDA_data(self, data, normal_reference, KPCA_Matrix, KFDA_Matrix):
        self.data = rbf_kernel(self.normal_data, self.data, gamma = self.gamma)
        self.data = pd.DataFrame.from_records(self.data)
        self.data_KPCA = KPCA_Matrix.T.dot(self.data)
        self.data_KPCA = pd.DataFrame.from_records(self.data_KPCA)
        self.data_KFDA = KFDA_Matrix.T.dot(self.data_KPCA)
        self.data = pd.DataFrame.from_records(self.data_KFDA)
        return self.data

    def clfPredict(self, X, clf):
        self.L_pred = self.clf.predict(self.data.T)
        return self.L_pred

    def Label_accuracy(self):
        self.accuracy_loop = self.accuracy_loop + 1
        for labelpred, datalabel in zip(self.L_pred, self.data_label):
            datalabel = int(datalabel.decode("utf-8"))
            self.accuracy.append(labelpred == datalabel)
        self.accuracy = sum(self.accuracy)/self.buffer_size
        self.accuracy_total = (self.accuracy_total*(self.accuracy_loop-1) + self.accuracy)/self.accuracy_loop
        print(self.accuracy_total*100, self.accuracy, self.accuracy_loop)
        print("__________________________")
        df_entry = [self.accuracy_total*100, self.accuracy, self.accuracy_loop]
        self.df_acc.append(df_entry)
        self.accuracy = []

    def label_handler(self):
        if self.label_buffer == []:
            for x in self.L_pred:
                self.label_buffer.append(x)
        elif self.label_buffer != []:
            self.df_label.append([self.label_buffer[-1]])
            bslice = self.label_buffer[:-1]
            self.label_buffer.clear()
            lslice = self.L_pred[1:]
            for bidx, lidx in zip(bslice, lslice):
                if bidx != lidx:
                    if lidx == 1:
                        bidx = lidx
                self.label_buffer.append(bidx)
            self.label_buffer = [self.L_pred[0], *self.label_buffer]
            

    def clfConvergent(self):
        if (self.L_pred.tolist() == [1]*self.buffer_size):
            self.clf_convergent = True
            print("clf_convergent = ", self.clf_convergent)

    def dispatcher(self, data_stream_from_mqtt):
        self.data_stream = data_stream_from_mqtt
        self.data_label = self.data_stream.iloc[:,-1]
        self.data_stream = self.data_stream.iloc[:,:-1]
        self.data_sample = self.data_sample + 1
        print('Samples uploaded: ', self.data_sample)
        if self.data_sample == 3000:
            sys.exit()
        elif len(self.data_stream) == self.reference_size:
            if len(self.normal_data) == 0:
                print('reference size reached')
                self.normal_data = self.data_stream
                #self.normal_reference = self.normal_data
                self.normal_reference, self.KPCA_Matrix, self.KFDA_Matrix = self.KFDA_normal(self.normal_data)
                self.normal_reference, self.rb = self.robust_scaler_fit(self.normal_reference)
                #self.rs_wcss = RobustScaler()
                #self.rs_wcss.fit(self.normal_reference.T)
                self.all_negatives = np.zeros(self.buffer_size)
                return len(self.data_stream), self.normal_reference
            elif len(self.normal_reference) != 0:
                self.data = self.data_stream[-self.buffer_size:]
                self.data_label = self.data_label[-self.buffer_size:]
                self.data = self.KFDA_data(self.data, self.normal_reference, self.KPCA_Matrix, self.KFDA_Matrix)
                self.data = self.robust_scaler_transform(self.data, self.rb)
                #Joining normal_reference and data
                self.X = np.concatenate((self.normal_reference, self.data), axis=1)
                self.X = self.X.T
                self.wcss_result = self.wcss_test(self.X, self.wcss_threshold)
                if(self.clf_convergent == False): #Kalo garis pemisah SVM masih berubah ubah
                    if(self.wcss_result == True): #Kalo elbow udah jelas nunjukin jumlah cluster = 2
                        self.L_KKN = self.k_means_clustering(self.X)
                        #self.L_KKN = self.db_clustering()
                        print(self.L_KKN)
                        if(sum(self.L_KKN[:self.reference_size]) > .6*(self.reference_size)):
                            for i in range(len(self.L_KKN)):
                                if self.L_KKN[i] == 0:
                                    self.L_KKN[i] = 1
                                elif self.L_KKN[i] == 1:
                                    self.L_KKN[i] = 0
                        self.L_KKN[:self.reference_size] = 0*self.reference_size  #Forcing labels of normal_reference to 0
                        #print("punya clustering")
                        if self.classifier == "Support Vector Machines":
                            #print("Sekarang pakai SVM")
                            try:
                                #print("its not that weird")
                                self.clf = self.SVM_Train(self.X, self.L_KKN)
                                self.L_pred = self.clfPredict(self.data.T, self.clf)
                            except ValueError:
                                #print("yo its weird")
                                self.L_pred = self.all_negatives
                        elif self.classifier == "Fisher Discriminant Analysis":
                            #print("Sekarang pakai FDA")
                            try:
                                #print("its not that weird")
                                self.clf = self.LDA_Train(self.X, self.L_KKN)
                                self.L_pred = self.clfPredict(self.data.T, self.clf)
                            except ValueError:
                                #print("yo its weird")
                                self.L_pred = 0*self.buffer_size
                        self.Label_accuracy()
                        print(self.L_pred)
                        return len(self.data_stream), self.X, self.wcssDiff, self.L_pred
                    else:
                        print('wcss result: ', self.wcss_result)
                        self.L_pred = self.all_negatives
                        self.Label_accuracy()
                        #print("punya classifier")
                        print(self.L_pred)
                        return len(self.data_stream), self.X, self.wcssDiff
                elif(self.clf_convergent == True): #Kalo garis pemisah SVM udah stabil
                    if self.classifier == "Support Vector Machines":
                        self.L_pred = self.clfPredict(self.X, self.clf)
                    elif self.classifier == "Fisher Discriminant Analysis":
                        self.L_pred = self.clfPredict(self.X, self.clf)
                    self.Label_accuracy()
                    #self.label_handler()
                    return len(self.data_stream), self.X, self.L_pred.tolist()
        elif len(self.data_stream) != self.reference_size:
            print('Current data size: ', len(self.data_stream))
            return len(self.data_stream)
    pass

faultDetection = Fault_Detection()