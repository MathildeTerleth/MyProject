# Duplicate detection algorithm (MSMP+ variant)
***
This project is about finding a scalable duplicate detection algorithm using data of television products coming from several Web shops. Due to the size of the data as well as the number of comparisons, it tries to find a solution to reduce the long computation time compared to other readily available duplicate detection methods. In order to reduce the number of comparisons, this method used the technique of Locality Sensity Hashing (LSH) and can be seen as a variant of the MSMP+ algorithm proposed by Hartveld, A. et al. (2018)[[1]](#1).

## Table of Contents
1. [General Info](#general-info)
2. [Structure of the code](#structure-of-the-code)
3. [How to use the code](#how-to-use-the-code)

### General Info
***
As already mentioned the aim of the project is to get a more scalable solution for duplicate detection. This method uses the LSH to reduce the number of comparisons to be made by an time consuming duplicated detection like the state-of-the-art MSM. LSH reduces the number of comparisons to be made by finding similar products instead of exact products. In order to do so products are represented by binary vectors, which are constructed using model words. This approach extracts model words from product titles as well as from key-value pairs (in the attribute list) based on the frequency of the keys. Before LSH is applied the matrix of binary vectors is reduced to a signature matrix by means of minhashing, which allows for a dimension reduction. Each signature represents a binary vector of a product, only with a reduced dimension and without much loss of important information.
This signature matrix can subsequently be divided into x bands containing y rows,  if two products are hashed to the same bucket by the LSH hash function for at least one band are considered duplicates. The number of bands and rows are uniquely determined by the size of the signature matrix and the similarity threshold, which represents how similar products have to be in order to be considered duplicates. 

### Structure of the code
***
The structure of the code will be explained from top to bottom using the line numbers of the code. 
 - [8 - 20]: import the necessary packages.
 - [22 - 24]: open the data set.
 - [27 - 584]: define the different functions that will be used in the main code that has to be run. Explanation of each function can be found in the code itself. 

Main running code starting:
 - [587 - 635]: extracts value-based model words from relevant keys of the key-value pairs
 - [638 - 641]: code to unnest a nested dictionary, due to the duplicates
 - [643 - 717]: code to compute the LSH evaluation metrics: pair quality (PQ), pair completeness (PC) and F1* (which is the harmonic mean between the two) and fraction of comparisons made compared to the total number of possible comparisons. These metrics are computed and averaged over 5 bootstrapped samples with replacement for a range of different threshold values. 
 - [719 - 734]: code to plot the PQ, PC and F1*-measure against the fraction of comparisons. 
 - [738 - 849]: code to compute the F1-measure, being the harmonic mean between precision and recall after the potential candidate duplicates of LSH went through a variant of the duplicate detection method MSM. 
 - [850 - 853]: code to plot the F1-measure against the fraction of comparisons.

### How to use the code
***
1. In order to have a running main code, you first need to run the lines 8 till 584. Nothiing will be printed and the data is loaded.
2. Then run the lines [587 - 598]. Here the frequency of the different keys of the key-value pairs are computed. At the moment only the keys with a frequency larger than 500 are retrieved but this can be altered in line 592. The keys can be viewed in df_mask. Subsequently in [601 - 609] the unique values corresponding to each "kept" key are appended to df_mask. Then one needs to open df_mask and inspect the keys and values to determine which value-based information of which key one wants to include. 
3. Running the lines [612 - 635] allows to append model words extracted from the specified keys in lines [613 - 618], which can be altered.
4. In order to get plots of the evaluation metrics of LSH, run the code [639 - 734]. In line 645 the number of bootstraps can be altered and in 644 the range of threshold values. 
5. In order to get plots of the evaltuation metric F1-measure of the entire method on the dataset, after MSM, run the code on lines [739 - 853]. In line 739 the number of bootstraps can be altered and in 740 the range of threshold values. 
  NOTE: at the moment some code within this block is commented out. This code [786 - 820] can be used to run optimization of the MSM parameters for every threshold for every bootstrap. This will be very time-consuming. 

## References
<a id="1">[1]</a> 
Hartveld, A., van Keulen, M., Mathol, D., van Noort, T., Plaatsman, T., Frasincar, F., Schouten, K.: An LSH-based model-words-driven product duplicate detection method, In: 30th International Conference on Advanced Information Systems Engineering (CAiSE 2018). Lecture Notes in Computer Science, vol. 10816, pp. 149-161. Springer (2018)





