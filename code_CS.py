# -*- coding: utf-8 -*-
"""
Created on Tue Nov  2 12:09:38 2021

@author: Mathilde
"""

import json
import numpy as np
import pandas as pd
import re
import copy
import string
import random
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.cluster import AgglomerativeClustering
from strsimpy.qgram import QGram
from strsimpy.normalized_levenshtein import NormalizedLevenshtein

## Retrieve the data set

with open(r"C:\Users\Mathilde\Documents\Pre-master Econometrie\Computer Science\Paper\TVs-all-merged\TVs-all-merged.json") as f:
    data = json.load(f)
    
    
## Funtion to obtain the unique items of a certain key in the list of features
def uniqueItems(obj):
    unique_Items = []
    for i in obj.items():
        for j in range(len(i[1])):
            l = i[1][j]["featuresMap"].keys()
            res = list(l)
            for match in res:
                if match not in unique_Items:
                    unique_Items.append(match)
    count = np.zeros(len(unique_Items))
    for i in obj.items():
        for j in range(len(i[1])):
            l = i[1][j]["featuresMap"].keys()
            res = list(l)
            for match in res:
                for j in range(len(unique_Items)):
                    if unique_Items[j] is match:
                        count[j] = count[j] + 1

    return (unique_Items, count)

## Function to retrieve all keys and all values corresponding to the keys
def featureItems(obj, feature):
    feature_Items = []
    feature_keys = []
    for i in obj.items():
        for j in range(len(i[1])):
            l = i[1][j]["featuresMap"].keys()
            res = list(l)
            feature_keys.append(res)
            if feature in res:
                value = i[1][j]["featuresMap"][feature]
                if value not in feature_Items:
                    feature_Items.append(value)

    return (feature_Items, feature_keys)


## Funtion to obtain all values of the key-value pairs
def key_values(obj):
    feature_values = []
    for i in obj.items():
        for j in range(len(i[1])):
            l = i[1][j]["featuresMap"].values()
            res = list(l)
            feature_values.append(res)

    return feature_values

## Function to obtain all the values corresponding to either: title, modelID, url, shop or featuresMap 
def label(obj, string):
    label_values = []
    for i in obj.items():
        for j in range(len(i[1])):
            label = i[1][j][string]
            label_values.append(label)
    return label_values

## Funtion to obtain all product titles of a data set and all the model words from the product titles
def MW_title(obj, string):
    titles = []
    MW = []
    for i in obj.items():
        for j in range(len(i[1])):
            title = i[1][j][string]
            titles.append(title)

    titles = title_cleaning(titles)
    for title in titles:
        for s in re.findall("([a-zA-Z0-9]*(([0-9]+[ˆ0-9, ]+)|([ˆ0-9, ]+[0-9]+))[a-zA-Z0-9]*)", title):
            if s[0] not in MW:
                MW.append(s[0])

    return titles, MW

## Function to generate binary vectors for each product.
## Output: matrix containing all binary vectors, the nr of product, nr of model words considered
def binary_vectors(obj, MW_tot, MW_title):
    nr_titles = len(MW_title[0])
    nr_MW = len(MW_tot)
    b = np.zeros((nr_MW, nr_titles))
    for i in range(nr_titles):
        title = MW_title[0][i]
        values = key_values(obj)[i]
        values = title_cleaning(key_values(obj)[i])
        for j in range(nr_MW):
            if MW_tot[j] in title or MW_tot[j] in values:
                b[j][i] = 1
            else:
                b[j][i] = 0

    return (b, nr_titles, nr_MW)

## Function that determines is a value is a prime number
def isPrime(x):
    for j in range(2, int(x ** 0.5) + 1):
        if x % j == 0:
            return False
    return True

## Function that finds the nearest prime number
def findPrimeNum(num):
    for i in range(num - 1, 1, -1):
        if isPrime(i):
            return i

## Function that generates the coefficients "a" and "b" of "k" the random hash functions for k 
def hash_function(k, rows):
    a = []
    b = []
    for j in range(k):
        a.append(random.randint(0, rows))
        b.append(random.randint(0, rows))
    return a, b

## Function that computes the hash values of the random hash function for each row of the matrix of binary vectors
def modulo(x, k, rows, a, b):
    h = []
    prime = findPrimeNum(rows + 1)
    for j in range(k):
        h_ij = (a[j] + b[j] * x) % prime
        h.append(h_ij)
    return h


## Function that constructs the signature matrix based on the binary vector matrix
def signature(vector):
    nr_rows = vector[2]
    nr_titles = vector[1]

    k = round(nr_rows / 2)

    signatures = 10000 * np.ones((k, nr_titles))

    a = hash_function(k, nr_rows)[0]
    b = hash_function(k, nr_rows)[1]

    for i in range(nr_rows):
        v = i + 1
        h = modulo(v, k, nr_rows, a, b)
        for j in range(nr_titles):
            if vector[0][i][j] == 1:
                for p in range(len(h)):
                    if h[p] < signatures[p][j]:
                        signatures[p][j] = h[p]

    return signatures


## Function that determins the number of bands an rows of the signature matrix given a certain dimension and similarity threshold
def b_r(n, threshold):
    options = []
    thres = []
    rows = np.arange(start=1, stop=n)

    for i in range(len(rows)):
        bands = n / rows[i]
        t = (1 / bands) ** (1 / rows[i])
        thres.append(abs(threshold - t))
        options.append([rows[i], int(bands)])
    index_t = np.where(thres == np.amin(thres))

    b_r_comb = options[index_t[0][0]]
    t = (1 / b_r_comb[1]) ** (1 / b_r_comb[0])

    return b_r_comb, t

## Function that places products in similar buckets based on a hash function that merges the strings of the rows of a certain band
## First output: matrix of dimension (# bands x # products) where entry > 0 indicates that products are hashed to the same bucket for a certain band
## Second output: matrix of dimension (# product x # products), where entry of 1 shows that products are hashed to the same bucket at least ones and are candidate duplicate pairs
## Third output: signature matrix
def LSH(vector, t):
    titles = vector[1]
    sign_matrix = signature(vector)
    n = len(sign_matrix)

    bands = b_r(n, t)[0][1]
    rows = b_r(n, t)[0][0]
    buckets = np.zeros((bands, titles))

    for i in range(bands):
        for t in range(titles):
            name = []
            for j in range(rows):
                number = int(sign_matrix[j + i * rows][t])
                name.append(number)
            str1 = ''.join(str(e) for e in name)
            buckets[i][t] = str1

    compare = np.zeros((titles, titles))
    for b in range(bands):
        for i in range(titles):
            for j in np.arange(start=i + 1, stop=titles):
                if buckets[b][i] == buckets[b][j] and buckets[b][i] != np.inf:
                    compare[i][j] += 1

    return buckets, compare, sign_matrix

## Function to retrieve a list of comparisons that need to be made when they are in the same cluster 
def position_comparisons(dfIn):
    positions = []

    for i in range(len(dfIn)):
        for j in np.arange(start = i+1, stop = len(dfIn)):
            if dfIn.values[i][j]>=1:
                positions.append([i,j])
    return positions

## Function that splits strings consisting of numerical, non-numerical and special characters 
## Output: only non-numerical characters, only numerical characters, only special characters
def splitString(str):
    alpha = ""
    num = ""
    special = ""
    for i in range(len(str)):
        if (str[i].isdigit()):
            num = num + str[i]
        elif ((str[i] >= 'A' and str[i] <= 'Z') or
              (str[i] >= 'a' and str[i] <= 'z')):
            alpha += str[i]
        else:
            special += str[i]

    return (alpha, num, special)

## Function that removes special characters of a string and makes text lower case
def clean_data(text):
    text = ''.join([ele for ele in text if ele not in string.punctuation])
    text = text.lower()
    return text

## Function to retrieve model words of a specific title, based on the index of the data object
## Output: model words in the title; splitted model words in numeric, non-numeric and special characters
def MW_T(title, obj):
    mw_title = []
    for s in re.findall("([a-zA-Z0-9]*(([0-9]+[ˆ0-9, ]+)|([ˆ0-9, ]+[0-9]+))[a-zA-Z0-9]*)", title):
        mw_title.append(s[0])

    splits_title = []

    for i in range(len(mw_title)):
        split = splitString(mw_title[i])
        splits_title.append(split)
    return mw_title, splits_title


## Function that computes the average normalized levenshtein similarity of two titles
def avgLvSim(X, Y):
    normalized_levenshtein = NormalizedLevenshtein()

    avgLVSIM = 0
    totalsum = 0
    for i in range(len(X)):
        for j in range(len(Y)):
            lengthx = len(X[i][0])
            lengthy = len(Y[j][0])
            totalsum = totalsum + (lengthx + lengthy)
            lv = normalized_levenshtein.distance(X[i][0], Y[j][0])
            avgLVSIM = avgLVSIM + (1 - lv) * (lengthx + lengthy)
    avgLVSIM = avgLVSIM / totalsum
    return avgLVSIM

## Function that computes the average normalized levenshtein similarity of a set of model words
def avgLvSimMW(close_MW):
    normalized_levenshtein = NormalizedLevenshtein()
    avgLVSIMMW = 0
    totalsum = 0
    for i in range(len(close_MW)):
        word1 = close_MW[i][0]
        word2 = close_MW[i][1]
        lengthx = len(word1)
        lengthy = len(word2)
        totalsum = totalsum + (lengthx + lengthy)
        lv = normalized_levenshtein.distance(word1, word2)

        avgLVSIMMW = avgLVSIMMW + (1 - lv) * (lengthx + lengthy)
    avgLVSIMMW = avgLVSIMMW / totalsum
    return avgLVSIMMW

## Function that computes the cosine similarity between two strings 
def cosine_sim(p1, p2):
    title_text = [p1, p2]
    mapping = list(map(clean_data, title_text))

    vectorizer = CountVectorizer()
    vectorizer.fit(mapping)
    vectors = vectorizer.transform(mapping).toarray()

    cos_sim = cosine_similarity(vectors)

    return (cos_sim)

## TMWM algorithm modified to return:
## 1 when titles are similar based on the cosine similarity measures
## -1 when titles have a similar non-numeric part but different numeric part
## a similarity measure when titles have a similar non-numeric part and similar numeric part
def check_for_similar_titles(p1, p2, obj, alpha, tresh, beta, delta):
    normalized_levenshtein = NormalizedLevenshtein()
    splits_p1 = MW_T(p1, obj)[1]
    splits_p2 = MW_T(p2, obj)[1]
    simTitle = 0

    if cosine_sim(p1, p2)[0][1] > alpha:
        simTitle = 1
    else:
        for i in range(len(splits_p1)):
            for j in range(len(splits_p2)):
                if normalized_levenshtein.distance(splits_p1[i][0], splits_p2[j][0]) < tresh and splits_p1[i][1] != splits_p2[j][1]:
                    simTitle = -1
                else:
                    simTitle = NameSim(p1, p2, obj, beta, delta)
    return simTitle

## Function that computes the name similarity of two titles based ont the cosine similarity and normalized levenshtein similarities
def NameSim(p1, p2, obj, beta, delta):
    normalized_levenshtein = NormalizedLevenshtein()
    splits_p1 = MW_T(p1, obj)
    splits_p2 = MW_T(p2, obj)


    finalNameSim = beta * cosine_sim(p1, p2)[0][1] + (1 - beta) * avgLvSim(splits_p1, splits_p2)
    close_words = []
    for i in range(len(splits_p1[1])):
        for j in range(len(splits_p2[1])):
            if normalized_levenshtein.distance(splits_p1[1][i][0], splits_p2[1][j][0]) < 0.5 and splits_p1[1][i][1] == \
                    splits_p2[1][j][1]:
                close_words.append((splits_p1[0][i], splits_p2[0][j]))


    if close_words == []:
        finalNameSim = finalNameSim
    else:
        update = avgLvSimMW(close_words)
        finalNameSim = delta * finalNameSim + (1 - delta) * update

    return (finalNameSim)


## Function that computes the q-gram similarity between two strings, using tokens of 3 characters
def qGramSim(text1, text2):
    qgram = QGram(3)
    n1 = len(text1)
    n2 = len(text2)

    dist = qgram.distance(text1, text2)
    qgram_sim = (n1 + n2 - dist) / (n1 + n2)
    return qgram_sim

## Function retrieves all model words from tht values of the attributes from product p 
def exMW_(keys_1, index):
    values = []
    for i in range(len(keys_1)):
        value = label(bootstrap_1, "featuresMap")[index][keys_1[i]]
        values.append(value)
    return values

## Function that computes the percentage of matching model words from two sets of model words 
def mwPerc(values_1, values_2):
    count = 0
    total = 0

    for s in re.findall("([a-zA-Z0-9]*(([0-9]+[ˆ0-9,]+)|([ˆ0-9,]+[0-9]+))[a-zA-Z0-9]*)", values_1):
        total = total + 1
    for t in re.findall("([a-zA-Z0-9]*(([0-9]+[ˆ0-9,]+)|([ˆ0-9,]+[0-9]+))[a-zA-Z0-9]*)", values_2):
        total = total + 1
    for s in re.findall("([a-zA-Z0-9]*(([0-9]+[ˆ0-9,]+)|([ˆ0-9,]+[0-9]+))[a-zA-Z0-9]*)", values_1):
        for t in re.findall("([a-zA-Z0-9]*(([0-9]+[ˆ0-9,]+)|([ˆ0-9,]+[0-9]+))[a-zA-Z0-9]*)", values_2):
            if s[0] == t[0]:
                count = count + 1

    return count / (total - count)

## Function that transforms a string containing a version of inch or hz to standardized form
def title_cleaning(titles):
    
    y = ['Inch', 'inches','"', '-inch', 'inch',' inch']
    x = ['Hertz', 'hertz', 'Hz', 'HZ', ' hz', '-hz', 'hz']
    titles_normalized_inch = []
    
    for title in titles:
        count = 0
        new_title = title
        for i in range(len(x)):
            if x[i] in title: 
                count = count + 1
                new_title = new_title.replace(x[i], 'hz')

            if x[i] not in title:
                count = count + 1
        if count == 6:
            titles_normalized_inch.append(title)
        else:
            titles_normalized_inch.append(new_title)
            
    
    titles_normalized_inch_hz = []            
    for title in titles_normalized_inch:
        count = 0
        new_title = title
        for i in range(len(y)):
            if y[i] in title: 
                count = count + 1
                new_title = new_title.replace(y[i], 'inch')
            if y[i] not in title:
                count = count + 1
        if count == 7:
            titles_normalized_inch_hz.append(title)
        else:
            titles_normalized_inch_hz.append(new_title)
    return(titles_normalized_inch_hz)

# Retrieve unnested data if originally you have a nested dictionary
def NestedDictValues(d):
    for v in d.items():
        for j in range(len(v[1])):
            if isinstance(v, dict):
                yield from NestedDictValues(v)
            else:
                yield v

## Create a dissimilarity matrix based on the MSM algorithm
## Output: potential duplicates, dissimilarity matrix
def similarity_matrix(titles, obj, duplicate_list, alpha, tresh, beta, delta, gamma, mu):

    dist = np.ones((nr_titles,nr_titles))*10000
    for i in range(nr_titles):
        for j in range(nr_titles):
            if i == j:
                dist[i][j]=0
    Sim = []
    total_Sim =[]
    found_duplicates = []

    for i in range(len(duplicate_list)):
        sim = 0
        avgSim = 0
        m = 0 #number of matches
        w = 0 #weight of matches
        p1 = duplicate_list[i][0]
        p2 = duplicate_list[i][1]
        keys_p1 = featureItems(obj, " ")[1][p1]
        keys_p2 = featureItems(obj, " ")[1][p2]
        nmk_p1 = copy.copy(keys_p1)
        nmk_p2 = copy.copy(keys_p2)
        
        for j1 in range(len(keys_p1)):
            for j2 in range(len(keys_p2)):
    
                keySim = qGramSim(keys_p1[j1], keys_p2[j2])
                if keySim > gamma:
                    value1 = label(obj,"featuresMap")[p1][keys_p1[j1]]
                    value2 = label(obj,"featuresMap")[p2][keys_p2[j2]]
                    valueSim = qGramSim(value1, value2)
                    weight = keySim
                    sim = sim + weight*valueSim
                    m = m + 1
                    w = w + weight
                    if keys_p1[j1] in nmk_p1:
                        nmk_p1.remove(keys_p1[j1])
                    if keys_p2[j2] in nmk_p2:
                        nmk_p2.remove(keys_p2[j2])           
        if w>0:
            avgSim = sim/w
            Sim.append(avgSim)
        exMW_nmk_p1 = exMW_(nmk_p1,p1)
        exMW_nmk_p2 = exMW_(nmk_p2,p2)
        if exMW_nmk_p1 != [] and exMW_nmk_p2 != []:
            mwperc = mwPerc(str(exMW_nmk_p1), str(exMW_nmk_p2))
        else: 
            mwperc = 0
        title_p1 = label(obj,"title")[p1]
        title_p2 = label(obj,"title")[p2]
        titleSim = check_for_similar_titles(title_p1,title_p2,obj,alpha,tresh,beta,delta)
        if titleSim == -1:
            theta_1 = m/(min(len(keys_p1),len(keys_p2)))
            theta_2 = 1 - theta_1
            hSim = theta_1*avgSim + theta_2*mwperc
            total_Sim.append(hSim)
        else: 
            theta_1 = (1-mu) * m/(min(len(keys_p1),len(keys_p2)))
            theta_2 = 1-mu-theta_1
            hSim = theta_1*avgSim + theta_2*mwperc + mu*titleSim
            total_Sim.append(hSim)
            found_duplicates.append([p1,p2])

        dist[p1][p2] = 1 - hSim
        dist[p2][p1] = 1 - hSim
        
    return(found_duplicates, dist)


## Function that computes the F1 measure of after MSM
def F1_measure(nr_titles, obj, comparisonlist, alpha, tresh, beta, delta, gamma, mu, total_duplicates, dist_thres):
    X = similarity_matrix(nr_titles, obj, comparisonlist, alpha, tresh, beta, delta, gamma, mu)
    clustering = AgglomerativeClustering(n_clusters = None, affinity='precomputed', linkage='complete', distance_threshold= dist_thres)
    clustering.fit(X[1])
    labels = clustering.labels_
    unique_labels = np.unique(labels)

    groups = []
    for i in range(len(unique_labels)):
        group = []
        for j in range(len(labels)):
            if unique_labels[i] == labels[j]:
                group.append(j)
        groups.append(group)
           
    final_comparisons = []
    for i in range(len(groups)):
        if len(groups[i])>1:
            for j in range(len(groups[i])):
                for k in np.arange(start = j+1, stop=len(groups[i])):
                    combi = [groups[i][j], groups[i][k]]
                    final_comparisons.append(combi)
    TP = 0
    FP = 0       
    for z in range(len(final_comparisons)):
        if label(obj, "modelID")[final_comparisons[z][0]]==label(obj, "modelID")[final_comparisons[z][1]]:
            TP = TP + 1
        else:
            FP = FP + 1
      
    precision = TP/(TP + FP)
    recall = TP/total_duplicates
    F1 = (2*precision*recall)/(precision+recall)
    return(F1, TP, FP, precision, recall)


## The fist step of MSM where for the potential duplicates of LSH it is checked wether the products are from different shops and have the different brands
## Output: filtered potential duplicates of LSH that continue in MSM 
def filter1_duplicates(comparisons, obj):
    brands = featureItems(data,"Brand")[0]
    brands.append("Insignia")
    
    potential_duplicates = []
    for i in range(len(comparisons)):
        p1 = comparisons[i][0]
        p2 = comparisons[i][1]
        brandp1 = 0
        brandp2 = 0
        if label(obj, "shop")[p1]!=label(obj, "shop")[p2]:
            if "Brand" in label(obj,"featuresMap")[p1] and "Brand" in label(obj,"featuresMap")[p2]:
                if label(obj,"featuresMap")[p1]["Brand"] != label(obj,"featuresMap")[p2]["Brand"]:
                    continue 
            else:
                for j in range(len(brands)): 
    
                    if brands[j] in label(obj, "title")[p1]:
                        brandp1 = brands[j]
                    if brands[j] in label(obj, "title")[p2]:
                        brandp2 = brands[j]
                        
                if brandp1 != brandp2 and (brandp1 != 0 or brandp2 !=0):
                    continue
                else:
                    potential_duplicates.append([p1,p2])
    return(potential_duplicates) 


### Identify the frequency and TF of key features
unique_words = pd.DataFrame({"words": uniqueItems(data)[0], "count": uniqueItems(data)[1]})
dfObj = unique_words
dfObj = dfObj.sort_values(by="count", ascending=False)

df_mask = dfObj[dfObj['count'] >= 500]
count = df_mask["count"]
max_value = count.max()

TF_scores = df_mask['count'].div(max_value)
df_mask.insert(2, "TF", TF_scores)
TF_words = df_mask["words"].tolist()

### Append the unique values corresponding to these key features
values = []
instances = []
for i in range(len(TF_words)):
    value = featureItems(data, TF_words[i])[0]
    instances.append(len(value))
    values.append(value)

df_mask.insert(3, "Values", values)
df_mask.insert(4, "Unique values", instances)

# Construct a list of model words based on the values of key-value pairs corresponding to the following keys
MW_KVP = []
values_resolution = list(df_mask[df_mask["words"] == "Maximum Resolution"]["Values"])
values_AR = list(df_mask[df_mask["words"] == "Aspect Ratio"]["Values"])
values_Brand = list(df_mask[df_mask["words"] == "Brand"]["Values"])
values_screensize = list(df_mask[df_mask["words"] == "Screen Size Class"]["Values"])
values_RR = list(df_mask[df_mask["words"] == "Mount Bracket/VESA Pattern"]["Values"])
values_SOP = list(df_mask[df_mask["words"] == "Speaker Output Power"]["Values"])


for i in range(len(values_resolution[0])):
    MW_KVP.append(values_resolution[0][i])
for i in range(len(values_AR[0])):
    MW_KVP.append(values_AR[0][i])
for i in range(len(values_Brand[0])):
    MW_KVP.append(values_Brand[0][i])
for i in range(len(values_screensize[0])):
    MW_KVP.append(values_screensize[0][i])
for i in range(len(values_RR[0])):
    MW_KVP.append(values_RR[0][i])
for i in range(len(values_SOP[0])):
    MW_KVP.append(values_SOP[0][i])

## Clean the model words
MW_KVP = title_cleaning(MW_KVP)


## Unnest the entire dataset such that bootstraps can be made
total_data = list(NestedDictValues(data))
n = len(total_data)
total_data = pd.DataFrame(total_data)

## Compute the PC, PQ and F1* measures for 5 bootstraps over a range of threshold values
treshold = np.arange(start = 0.95, stop = 0, step = -0.05)
bootstraps  = 5
fraction_comp = np.zeros((bootstraps, len(treshold)))
LSH_dup = np.zeros((bootstraps, len(treshold)))
PQ_all = np.zeros((bootstraps, len(treshold)))
PC_all = np.zeros((bootstraps, len(treshold)))
F1_diff_all = np.zeros((bootstraps, len(treshold)))

for i in range(bootstraps):
    
    bootstrap_1 = total_data.sample(n=n, replace=True)
    bootstrap_1 = bootstrap_1.iloc[bootstrap_1.astype(str).index.drop_duplicates()]
    bootstrap_1 = bootstrap_1.to_dict()
    bootstrap_1 = dict(zip(bootstrap_1[0].values(), bootstrap_1[1].values()))


    MW_titles = MW_title(bootstrap_1, "title")
    MW_only_titles = MW_titles[0]

    MW_total = MW_titles[1]
    for value in MW_KVP:
        if value not in MW_total:
            MW_total = MW_total + list([value])
    
    
    b_vector = binary_vectors(bootstrap_1, MW_total, MW_titles)
    nr_rows = b_vector[2]
    nr_titles = b_vector[1]
    k = round(nr_rows/2)

    duplicates = []
    products = label(bootstrap_1, "modelID")
    no_products = len(products)
    count = 0
    for p in range(no_products):
        duplicates.append(products[p])

    df_duplicates = pd.DataFrame(duplicates)
    ndf = df_duplicates.apply(pd.Series.value_counts).fillna(0)
    for d in range(len(ndf[0])):
        ndf[0][d] = ndf[0][d]*(ndf[0][d]-1)/2

    total_duplicates = (ndf[ndf[0]>0]).sum()
    total_number_of_comparisons = int(nr_titles*(nr_titles-1)/2)

    for j in range(len(treshold)):

        lsh = LSH(b_vector,treshold[j])
        compare = lsh[1]
        df = pd.DataFrame(compare)
        comparisons = position_comparisons(df)
        total_comparisons_lsh = len(comparisons)
        fraction_comparisons = total_comparisons_lsh/total_number_of_comparisons

        fraction_comp[i][j] = fraction_comparisons

        LSH_duplicates_found = 0
        for k in range(len(comparisons)):
            if label(bootstrap_1, "modelID")[comparisons[k][0]]==label(bootstrap_1, "modelID")[comparisons[k][1]]:
                LSH_duplicates_found = LSH_duplicates_found + 1
                LSH_dup[i][j] = LSH_duplicates_found      
                PQ = 2*LSH_duplicates_found/total_comparisons_lsh
                PC = LSH_duplicates_found/total_duplicates
                F1_diff = (2*PC*PQ)/(PC+PQ)
                PQ_all[i][j] = PQ
                PC_all[i][j] = PC
                F1_diff_all[i][j] = F1_diff

# Take the average of the metrics over the 5 bootstrap values  
PQ_avg = PQ_all.mean(0)
PQ_avg_mod = PQ_avg
PC_avg = PC_all.mean(0)
F1_diff_avg = F1_diff_all.mean(0)
frac_avg = fraction_comp.mean(0)

# Plot the PC, PQ and F1* measures against fraction of comparisons
plt.plot(frac_avg, PQ_avg_mod)
#plt.xlim(-0.001, 0.25)
#plt.ylim(0, 0.2)
plt.xlabel('Fraction of comparisons')
plt.ylabel('Pair quality')

plt.plot(frac_avg, PC_avg)
plt.xlabel('Fraction of comparisons')
plt.ylabel('Pair completeness')

plt.plot(frac_avg, F1_diff_avg)
plt.xlabel('Fraction of comparisons')
plt.ylabel('F1*-measure')

plt.show()



## Calculate F1-measure after MSM 
bootstraps = 5
treshold_2 = [0.95,0.9,0.85,0.8,0.75,0.7,0.65,0.6,0.55,0.5,0.4,0.3]
F1_final = np.zeros((bootstraps, len(treshold_2)))
fraction_comp_final = np.zeros((bootstraps, len(treshold_2)))
TP_final = np.zeros((bootstraps, len(treshold_2)))
FP_final = np.zeros((bootstraps, len(treshold_2)))


for i in range(bootstraps):
    
    bootstrap_1 = total_data.sample(n=n, replace=True)
    bootstrap_1 = bootstrap_1.iloc[bootstrap_1.astype(str).index.drop_duplicates()]
    bootstrap_1 = bootstrap_1.to_dict()
    bootstrap_1 = dict(zip(bootstrap_1[0].values(), bootstrap_1[1].values()))

    MW_titles = MW_title(bootstrap_1, "title")
    MW_only_titles = MW_titles[1]

    MW_total = MW_titles[1]
    for value in MW_KVP:
        if value not in MW_total:
            MW_total = MW_total + list([value])


    b_vector = binary_vectors(bootstrap_1, MW_total, MW_titles)
    nr_rows = b_vector[2]
    nr_titles = b_vector[1]
    k = round(nr_rows/2)

    duplicates = []
    products = label(bootstrap_1, "modelID")
    no_products = len(products)
    count = 0
    for p in range(no_products):
        duplicates.append(products[p])

    df_duplicates = pd.DataFrame(duplicates)
    ndf = df_duplicates.apply(pd.Series.value_counts).fillna(0)
    for d in range(len(ndf[0])):
        ndf[0][d] = ndf[0][d]*(ndf[0][d]-1)/2

    total_duplicates = (ndf[ndf[0]>0]).sum()
    total_number_of_comparisons = int(nr_titles*(nr_titles-1)/2)

    ## Code that can be used to run optimization for the MSM parameters.
    ##
    
    # lsh = LSH(b_vector, 0.75)
    # compare = lsh[1]
    # df = pd.DataFrame(compare)
    # comparisons = position_comparisons(df)
    # total_comparisons_lsh = len(comparisons)
    # fraction_comparisons = total_comparisons_lsh/total_number_of_comparisons
    # #fraction_comp[i] = fraction_comparisons
    
    # selected_dup_brand_shop = filter1_duplicates(comparisons, bootstrap_1)
    #similarity_all = similarity_matrix(nr_titles, bootstrap_1, selected_dup_brand_shop, alpha, tresh, beta, delta, gamma, mu)
    # alpha = 0.6#np.arange(start = 0, stop = 1, step = 0.5)    
    # tresh = 0.5
    # beta = 0.5#np.arange(start = 0, stop = 1, step = 0.5)
    # delta = 0.5#np.arange(start = 0, stop = 1, step = 0.5)
    # gamma = 0.756#np.arange(start = 0, stop = 1, step = 0.5)
    # mu = 0.650#np.arange(start = 0, stop = 1, step = 0.5)
    
    # final_found_correct = []
    # final_found_wrong = []
    # F1_all = []
    # for i in range(len(alpha)):
    #     for j in range(len(beta)):
    #         for k in range(len(delta)):
    #             for p in range(len(gamma)):
    #                 for e in range(len(mu)):    
    #                     F1_value = F1_measure(nr_titles, bootstrap_1, selected_dup_brand_shop , alpha[i], tresh, beta[j], delta[k], gamma[p], mu[e], total_duplicates)
    #                     F1_all.append([alpha[i],beta[j],delta[k],gamma[p],mu[e],F1_value[0]])
    #                     final_found_correct.append(F1_value[1])
    #                     final_found_wrong.append(F1_value[2])
    # maximum = 0                    
    # for i in range(len(F1_all)):
    #     value = F1_all[i][5][0]
    #     if value>maximum:
    #         maximum = value
    #         indx = F1_all[i]

    ## Fixed optimal parameters based on the MSM paper
    ##
    
    alpha = 0.6 # indx[0]
    tresh = 0.5 
    beta = 0 #indx[1]
    delta = 0.5 # indx[2]
    gamma = 0.756 #indx[3]
    mu = 0.650 #indx[4]
    for j in range(len(treshold_2)):

        lsh = LSH(b_vector,treshold_2[j])
        compare = lsh[1]
        df = pd.DataFrame(compare)
        comparisons = position_comparisons(df)
        total_comparisons_lsh = len(comparisons)
        fraction_comparisons = total_comparisons_lsh/total_number_of_comparisons
        fraction_comp_final[i][j] = fraction_comparisons
        selected_dup_brand_shop = filter1_duplicates(comparisons, bootstrap_1)

        F1_value = F1_measure(nr_titles, bootstrap_1, selected_dup_brand_shop , alpha, tresh, beta, delta, gamma, mu, total_duplicates, 0.5)
        F1_final[i][j] = F1_value[0]
        TP_final[i][j] = F1_value[1]
        FP_final[i][j] = F1_value[2]

## Plot the F1-measure against fraction of comparisons averages over 5 bootstraps
F1_final_avg = F1_final.mean(0)
fraction_comp_final_avg = fraction_comp_final.mean(0)
plt.plot(fraction_comp_final_avg, F1_final_avg)
plt.axhline(y=0.525, color='r', linestyle='-')
plt.xlabel('Fraction of comparisons')
plt.ylabel('F1-measure')







