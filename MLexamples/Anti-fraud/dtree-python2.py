import pandas as pd
import operator
import numpy as np
import numbers

def GiniScore(df, attribute, y):
    '''
    :param df: the dataframe containing the attribute and target
    :param attribute: the attribute that need to be calculated the gini score and is type 1
    :param y: the target in the classification problem with value 0 or 1
    :return: the gini score of the attribute
    '''
    values = list(set(df[attribute]))
    GiniScore = 0
    for v in values:
        subset = df.loc[df[attribute]==v]
        w = subset.shape[0]*1.0/df.shape[0]
        y1 = sum(subset[y])*1.0/subset.shape[0]
        y2 = 1-y1
        gini = 1-(y1**2+y2**2)
        GiniScore += gini*w
    return GiniScore

def GiniScoreContinuous(df, attribute, y):
    '''
    :param df: the dataframe containing the attriute and target
    :param attribute: the continuous attribute that need to be calculated the gini score
    :param y: the target in the classification problem with value 0 or 1
    :return: the gini score of the continuous attribute
    '''
    vals = list(set(df[attribute]))
    vals.sort()
    split_point = [(vals[i]+vals[i+1])*0.5 for i in range(len(vals)-1)]
    split_point_gini = {}
    for s in split_point:
        df['temp'] = df[attribute].apply(lambda x: x<=s)
        split_point_gini[s] = GiniScore(df, 'temp', y)
    #return the best split point corresponding to the minimum gini
    best_point = min(split_point_gini.iteritems(), key=operator.itemgetter(1))[0]
    del df['temp']
    return {best_point:split_point_gini[best_point]}



def AttributeType(x):
    '''
    :param x: the Serie or list of which the type need to be checked
    :return: the type of the Serie or list
    '''
    vals = set(x)
    if len(vals)<=4:
        #return the discrete variable with fewer values
        return 1
    elif isinstance(list(vals)[0],str):
        # return the discrete variable with more values
        return 2
    elif isinstance(list(vals)[0],numbers.Real):
        # return the continuous variables
        return 3
    else:
        # return the unknown type
        return 4



def TerminateTree(df,attributes,target):
    '''
    :param df: decision tree input dataset
    :param attributes: list of attributes, excluding target variable
    :param target: label of the classes
    :return: if sub-tree is leaf then return the label, otherwise return subtree
    '''
    #if no candidate attribute or all instances are the same, return leaf node with the majority class
    if attributes == [] or df[attributes].drop_duplicates().shape[0] == 1:
        positive = sum(df[target])
        zero = len(df[target]) - positive
        if positive>= zero:
            return 1
        else:
            return 0
    # if all instances belong to the same class, return leaf node with the unique class
    elif len(set(df[target])) == 1:
        return df.iloc[0][target]
    else:
        return 'substree'


def CreateTree(df, attributes, target):
    '''
    :param df: decision tree input dataset
    :param attributes: the list of attributes by which the df is splitted
    :param target: label of the classes
    :return: the splitted nodes and splitting rule
    '''
    giniscore_attribute_dict = {}   #the dictionary containing each attribute with its Gini
    child = TerminateTree(df, attributes, target)
    #if current node can be determined as leaf, then return
    if child in [0, 1]:
        return child
    type2BestPoint = {}
    type3BestPoint = {}
    for a in attributes:
        type_a = AttributeType(df[a])
        if type_a == 1:
            giniscore_attribute_dict[a] = GiniScore(df,a,target)
        if type_a == 2:
            string_to_float = {}
            for v in set(df[a]):
                subset = df.loc[df[a] == v]
                pcnt = sum(subset[target]) * 1.0 / subset.shape[0]
                string_to_float[v] = pcnt
            df['a2'] = df[a].map(string_to_float)
            gini_d = GiniScoreContinuous(df,'a2',target)
            giniscore_attribute_dict[a] = gini_d.values()[0]
            del df['a2']
            type2BestPoint[a] = gini_d.keys()[0]
        if type_a == 3:
            gini_d = GiniScoreContinuous(df, a, target)
            giniscore_attribute_dict[a] = gini_d.values()[0]
            type3BestPoint[a] = gini_d.keys()[0]
    # select splitted node as the node with min Gini score
    splitNode = min(giniscore_attribute_dict.iteritems(), key=operator.itemgetter(1))[0]
    type_node = AttributeType(df[splitNode])
    subAttributes = [i for i in attributes if i != splitNode]
    remainCol = [i for i in subAttributes]
    remainCol.append(target)
    grownTree = {splitNode:{}}
    if type_node == 1:
        for v in set(df[splitNode]):
            subTreeDf = df.loc[df[splitNode]==v][remainCol]
            grownTree[splitNode][str(v)] = CreateTree(subTreeDf,subAttributes,target)
    if type_node == 2:
        vals = list(set(df[splitNode]))
        bestPoint = type2BestPoint[splitNode]
        groupOfVals1 = [i for i in vals if string_to_float[i]<=bestPoint]
        groupOfVals2 = list(set(vals) - set(groupOfVals1))
        groupOfVals2 = list(set(vals) - set(groupOfVals1))
        subTreeDf1 = df[df[splitNode].isin(groupOfVals1)][remainCol]
        grownTree[splitNode][str(groupOfVals1)] = CreateTree(subTreeDf1, subAttributes, target)
        subTreeDf2 = df[df[splitNode].isin(groupOfVals2)][remainCol]
        grownTree[splitNode][str(groupOfVals2)] = CreateTree(subTreeDf2, subAttributes, target)
    if type_node == 3:
        bestPoint = type3BestPoint[splitNode]
        subTreeDf1 = df[df[splitNode]<=bestPoint][remainCol]
        grownTree[splitNode]["<="+str(bestPoint)] = CreateTree(subTreeDf1, subAttributes, target)
        subTreeDf2 = df[df[splitNode]>bestPoint][remainCol]
        grownTree[splitNode][">"+str(bestPoint)] = CreateTree(subTreeDf2, subAttributes, target)
    return grownTree


#print the decision tree recursively
def PrintTreeDict(obj, indent=' '):
    def _recurse(obj, indent):
        for i, tup in enumerate(obj.items()):
            k, v = tup
            # if string then concatinating with""
            if isinstance(k, basestring): k = '"%s"' % k
            if isinstance(v, basestring): v = '"%s"' % v
            # if dict then recurse
            if isinstance(v, dict):
                v = ''.join(_recurse(v, indent + ' ' * len(str(k) + ': {')))
            if i == 0:
                if len(obj) == 1:
                    yield '{%s: %s}'%(k, v)
                else:
                    yield '{%s: %s,\n'%(k, v)
            elif i == len(obj) - 1:
                yield '%s%s: %s}' % (indent, k, v)
            else:
                yield '%s%s: %s,\n' % (indent, k, v)
    print ''.join(_recurse(obj, indent))

if __name__  == '__main__':
    AllData = pd.read_csv("input/bank.csv",header = 0)
    AllData['y2'] = AllData['y'].apply(lambda x: int(x=='yes'))

    AllData.columns
    attributes = [u'age', u'job', u'marital', u'education', u'default', u'spending',u'housing', u'cash_loan',
                  u'contact_number_type', u'maturity',u'app_channel', u'max_late_charge', u'cash_withdraw_freq', u'poutcome']
    myTree = CreateTree(AllData, attributes, 'y2')
    PrintTreeDict(myTree)

    fewerAttributes = [u'age', u'job', u'marital', u'education', u'default']
    smallTree = CreateTree(AllData, fewerAttributes, 'y2')
    PrintTreeDict(smallTree)
