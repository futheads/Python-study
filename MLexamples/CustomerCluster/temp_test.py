import pandas as pd
import numpy as np
import pickle
import random
from sklearn import cluster
import scipy.cluster.hierarchy as sch
from matplotlib import pyplot as plt
from numpy import mean
from functools import reduce

def oneHotEncoding(df, old_field):
    distinct_vals = list(set(reduce(lambda x,y: x+y, df[old_field])))
    cnt = len(distinct_vals)
    new_fields = [old_field+'_'+str(i) for i in range(cnt)]
    for i in range(cnt):
        df[new_fields[i]] = 0
        df[new_fields[i]] = df[old_field].map(lambda x: int(distinct_vals[i] in x))
    del df[old_field]
    return 1


def normalization(df,var,method='min-max'):
    '''
    :param df,var: the dataframe and column that need to be normalized. Please make sure it is not a constant, and also not empty
    :param method: normalization method
    :return: the normalized result or -1 for error input
    '''
    x = df[var]
    new_field = var + "_norm"
    if method == 'min-max':
        x_min = min(x)
        x_max = max(x)
        d = x_max - x_min
        df[new_field] = [(i-x_min)*1.0/d for i in x]
        del df[var]
        return 1
    elif method == 'zero-score':
        mu = np.mean(x)
        std = np.std(x)
        df[new_field] = [(i - mu) * 1.0 / std for i in x]
        del df[var]
        return 1
    else:
        print("Please specify the normalization method: min-max or zero-score")
        return -1

def makeupMissing(x,replaceVal):
    if np.isnan(x):
        return replaceVal
    else:
        return x

def minkovDist(x,y,p=2):
    '''
    :param x,y: the numpy arrays of two samples
    :param p: the degree of distance
    :return: the distance, or -1 if p is less than 1
    '''
    if p>=1:
        return (sum((x-y)**p)*1.0)**(1.0/p)
    else:
        print("p must be larger than or equal to 0")
        return -1

def KmeansAlgo(dataset, k):
    '''
    :param dataset: the dataset of clustering
    :param k: the number of clusters
    :return: the group label of each sample, the centroids and the cost
    '''
    N = dataset.shape[0]
    label = [0]*N
    # randomly select k samples as the initial centroids
    centroidsIndex = random.sample(range(N),k)
    centroids = [dataset[i,] for i in centroidsIndex]
    centroidsChanged = True
    while(centroidsChanged):
        centroidsChanged = False
        #Calculate the Minkovski distance between each point and the centriods
        #Assign the point to the group with min distance to the centroid of the group
        for i in range(N):
            dist_to_cent = [minkovDist(dataset[i,].getA()[0], centroid.getA()[0]) for centroid in centroids]
            label[i] = dist_to_cent.index(min(dist_to_cent))
        #Update the centroids using geometric centroid in the group
        for j in range(k):
            position = [p for p in range(N) if label[p] == j]
            clusterGroup = dataset[position]
            newCents = np.mean(clusterGroup, axis=0)
            #Judge whether the centrod is updated
            if minkovDist(newCents.getA()[0], centroids[j].getA()[0]) > 0.00001:
                centroidsChanged = True
                centroids[j] = newCents
            else:
                centroidsChanged = False
    #calculate the cost function
    cost = 0
    for i in range(N):
        centroid = centroids[label[i]]
        dist_to_cent = minkovDist(dataset[i,].getA()[0], centroid.getA()[0])
        cost += dist_to_cent**2
    cost = cost/N
    return {'group':label, 'centroids':centroids, 'cost':cost}



### Step 1: Data preprocessing ###
loan_table = pd.read_csv('input/loan_details.csv', header = 0, encoding='gb2312')
cust_table = pd.read_csv('input/customer_table.csv', header = 0, encoding='gb2312')
#cust_table is not necessary.


cust_id = cust_table['CUST_ID'].drop_duplicates().to_frame(name='CUST_ID')
cust_id.columns = ['CUST_ID']
loan_table_cust = cust_id.merge(loan_table, on='CUST_ID',how='inner')

id_freq = loan_table_cust.groupby(['CUST_ID'])['CUST_ID'].count()
id_freq_2 = id_freq.to_dict()
id_freq_3 = [k for k, v in id_freq_2.items() if v > 1]
id_freq_4 = pd.DataFrame({'CUST_ID':id_freq_3})     # 贷款信息多于一条的客户Id

'''
Some of the duplicates of ID are due to information re-submmission. For these applicants,
the 'CUST_ID','Loan_Type','Loan_Term','Start_Date','End_Date','Loan_Amt','Undisbursed_Amt','Business_Type_Code'
are the same
'''

#多个贷款产品的情况
dup_records = id_freq_4.merge(loan_table, on='CUST_ID',how='inner')[['CUST_ID','Loan_Type','Loan_Term','Start_Date','End_Date',
                                                                     'Loan_Amt','Undisbursed_Amt','Business_Type_Code']]
#First, we remove the dupliated records in dup_records
dup_records2 = dup_records.drop_duplicates()
id_dup = dup_records2.groupby(['CUST_ID'])['CUST_ID'].count().to_dict()

id_dup_1 = [k for k,v in id_dup.items() if v == 1]
id_dup_1_df = pd.DataFrame({'CUST_ID':id_dup_1})
drop_dup_1 = pd.merge(id_dup_1_df,loan_table, on= 'CUST_ID',how='left')
drop_dup_1b = drop_dup_1.groupby('CUST_ID').last()

id_all = list(id_freq.index)
id_non_dup = [i for i in id_all if i not in set(drop_dup_1b.index)]
id_non_dup_df = pd.DataFrame({'CUST_ID':id_non_dup})
id_non_dup_df_2 = pd.merge(id_non_dup_df,loan_table, on= 'CUST_ID',how='left')


id_loans = pd.concat([id_non_dup_df_2,drop_dup_1b])

### Second, we make up the missing value in the dataframe
### make up the missing value in Interest_Payment
temp = id_loans.apply(lambda x: int(makeupMissing(x.Interest_Payment,9)), axis=1)
id_loans['Interest_Payment'] = temp

### make up the missing value in Credit_Level
temp = id_loans.apply(lambda x: int(makeupMissing(x.Credit_Level,0)), axis=1)
id_loans['Credit_Level'] = temp

'''
initially, we convert the data from loan level to customer level. For the customers with mulitple loans,
we use the list to capture the details
'''
all_vars = list(id_loans.columns)
all_vars.remove('CUST_ID')
for var in all_vars:
    id_loans[var] = id_loans[var].apply(lambda x: [x])

id_loans_group = id_loans.groupby('CUST_ID').sum()
#file_id_loans_group.close()

#file_id_loans_group = open('/Users/Downloads/数据/银行客群聚类/processed data/id_loans_group.pkl','r')
#id_loans_group = pickle.load(file_id_loans_group)
#file_id_loans_group.close()

#### Derive the various features using loan detail table ####
# Also we normalize the variables if necessary
# No. of loan types
var1 = id_loans_group.apply(lambda x: len(set(x.Loan_Type)),axis=1)
var1 = var1.to_frame(name='No_Loan_Types')

#count of loans
var2 = id_loans_group.apply(lambda x: len(x.Loan_Type),axis=1)
var2 = var2.to_frame(name='No_Loan')

#max of loan terms
var3a = id_loans_group.apply(lambda x: max(x.Loan_Term),axis=1)
var3a = var3a.to_frame(name='Max_Loan_Terms')
normalization(var3a, 'Max_Loan_Terms')

#min of loan terms
var3b = id_loans_group.apply(lambda x: min(x.Loan_Term),axis=1)
var3b = var3b.to_frame(name='Min_Loan_Terms')
normalization(var3b, 'Min_Loan_Terms')

#mean of loan terms
var3c = id_loans_group.apply(lambda x: mean(x.Loan_Term),axis=1)
var3c = var3c.to_frame(name='Mean_Loan_Terms')
normalization(var3c, 'Mean_Loan_Terms')

#total loan amount
#var4 = id_loans.groupby('CUST_ID').Loan_Amt.sum()
var4a = id_loans_group.apply(lambda x: sum(x.Loan_Amt),axis=1)
var4a = var4a.to_frame(name='Total_Loan_Amt')

#mean loan amount
var4b = id_loans_group.apply(lambda x: mean(x.Loan_Amt),axis=1)
var4b = var4b.to_frame(name='Mean_Loan_Amt')

#total Undisbursed_Amt
#var5 = id_loans.groupby('CUST_ID').Undisbursed_Amt.sum()
var5a = id_loans_group.apply(lambda x: sum(x.Undisbursed_Amt),axis=1)
var5a = var5a.to_frame(name='Total_Undisbursed_Amt')

#mean Undisbursed_Amt
var5b = id_loans_group.apply(lambda x: mean(x.Undisbursed_Amt),axis=1)
var5b = var5b.to_frame(name='Mean_Undisbursed_Amt')


#ratio of total Undisbursed_Amt and total loan amount
var6a = pd.concat([var4a,var5a],axis=1)
var6a['Total_Undisbursed_to_Loan'] = var6a.apply(lambda x: x.Total_Undisbursed_Amt/x.Total_Loan_Amt,axis=1)
del var6a['Total_Undisbursed_Amt']
del var6a['Total_Loan_Amt']

#ratio of mean Undisbursed_Amt and mean loan amount
var6b = pd.concat([var4b,var5b],axis=1)
var6b['Mean_Undisbursed_to_Loan'] = var6b.apply(lambda x: x.Mean_Undisbursed_Amt/x.Mean_Loan_Amt,axis=1)
del var6b['Mean_Undisbursed_Amt']
del var6b['Mean_Loan_Amt']


normalization(var4a,'Total_Loan_Amt')
normalization(var4b,'Mean_Loan_Amt')
normalization(var5a,'Total_Undisbursed_Amt')
normalization(var5b,'Mean_Undisbursed_Amt')


#min and max of interest rate of the single customer
var7a = id_loans_group.apply(lambda x: min(x.Interest_Rate),axis=1)
var7a = var7a.to_frame(name='Min_Interest_Rate')
normalization(var7a,'Min_Interest_Rate')

var7b = id_loans_group.apply(lambda x: max(x.Interest_Rate),axis=1)
var7b = var7b.to_frame(name='Max_Interest_Rate')
normalization(var7b,'Max_Interest_Rate')

derived_features = pd.concat([var1,var2,var3a,var3b,var3c,var4a,var4b,var5a,var5b,var6a,var6b,var7a,var7b],axis=1)


#onehot encode the Business_Type_Code，Repay_Way， Interest_Payment，Rural，External_Ind，Credit_Level，Gender
var_onehot_list = ['Business_Type_Code','Repay_Way','Interest_Payment','Rural','External_Ind','Credit_Level','Gender']
for var_onehot in var_onehot_list:
    var_onehot_df = id_loans_group[var_onehot]
    var_onehot_df = var_onehot_df.to_frame(name = var_onehot)
    oneHotEncoding(var_onehot_df, var_onehot)
    derived_features = pd.concat([derived_features,var_onehot_df],axis = 1)

'''
Use the Kmeans clustering method
'''
#just for demonstration
M = 1000
dataset = np.matrix(derived_features)[:M,]
cost = []
for k in range(2,7):
    result = KmeansAlgo(dataset, k)
    cost.append(result['cost'])

plt.plot(range(2,7),cost[:5])
plt.xlabel("number of clusters")
plt.ylabel("cost of clustering")
plt.title("Elbow method")
plt.show()

#3 clusters are suitable for the dataset
result = KmeansAlgo(dataset, 3)
featureCompared = np.matrix(np.zeros(dataset.shape[1]))
for l in range(3):
    groupIndex = [i for i in range(M) if result['group'][i]==l]
    temp = dataset[groupIndex]
    featureMean = np.mean(temp, axis = 0)
    featureCompared = np.row_stack((featureCompared,featureMean))
np.delete(featureCompared,0,0)

cols = ['g','b','r']
for l in range(3):
    groupIndex = [i for i in range(M) if result['group'][i]==l]
    x1 = dataset[groupIndex,2].getA()
    x2 = dataset[groupIndex,11].getA()
    p=plt.scatter(x1,x2,color = cols[l])

### Hierarchy Clustering
disMat = sch.distance.pdist(dataset,'euclidean')
Z=sch.linkage(disMat,method='average')
P=sch.dendrogram(Z)
cluster= sch.fcluster(Z, t=1, criterion='inconsistent')
