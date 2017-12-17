#Matrix Profile Version 1.4.0

#A Python implementation of the matrix profile algorithm described in Chin-Chia Michael Yeh, Yan Zhu, Liudmila Ulanova, Nurjahan Begum, Yifei Ding, Hoang Anh Dau, Diego Furtado Silva, Abdullah Mueen, Eamonn Keogh (2016): 'Matrix Profile I: All Pairs Similarity Joins for Time Series: A Unifying View that Includes Motifs, Discords and Shapelets', available at http://www.cs.ucr.edu/~eamonn/MatrixProfile.html

#Currently, this implementation supports parallel processing and early termination. A planned update will support the updating of the matrix profile when either time series in the comparison is updated. A GPU implementation is also planned.
        
#The latest version of this code can be found at https://github.com/javidlakha/matrix-profile

import pandas as pd
import numpy as np
import itertools
import time
import random
import os
import multiprocessing as mp
from scipy.fftpack import fft, ifft


def sliding_dot_product(time_series, query):                      #Time complexity: O(n log n)
    
    #This function computes the dot products of a 'query' sequence of length M and every contiguous subsequence of
    #length M in the time series. It is used in the distance calculations in MASS (below). The heart of the matrix
    #profile algorithm is the insight that whilst the complexity of calculating the dot product of every 'instigating'
    #subsequence that starts at position 1, 2, ..., N in the time series with every other 'target' subsequence of equal
    #length is O(n^2), the dot product of two vectors is the inverse Fourier transform of the dot product of their
    #Fourier transforms. The time complexity of the Fast Fourier Transform is O(n log n).
    
    #NB. Computational complexity depends only on the length of the time series - not on the length of the 'query'
    #sequence. This is a useful property: short patterns do not take more time to identify than long patterns.
    
    #Based on the pseudocode - Keogh et al (2016): http://www.cs.ucr.edu/~eamonn/PID4481997_extend_Matrix%20Profile_I.pdf  
    
    n = time_series.shape[0]
    m = query.shape[0]
    query = query[::-1]                                           #Reverse query
    query = np.append(query,np.zeros(n-m))                        #Append reversed query with n-m zeroes
    query = fft(query)                                            #Fast Fourier Transform of reversed query
    time_series = fft(time_series)                                #Fast Fourier Transform of time_series
    QT = np.multiply(query, time_series)                          #Element-wise multiplication of time_series and reversed query
    dot_product = np.real(ifft(QT))                               #Inverse Fast Fourier Transform
    return dot_product


def MASS(time_series, query):
    
    #Calculates the normalised distances between every 'query' sequence of length M with every contiguous subsequence
    #of M in the time series. Except for the sliding dot product (which is O(n log n)) the time complexity of this
    #algorithm is O(n).
    
    #Based on the Matlab code - Mueen at al (2015): http://www.cs.unm.edu/~mueen/FastestSimilaritySearch.html
    
    n = time_series.shape[0]
    m = query.shape[0]   
    query_mean = np.mean(query)                                   #Query mean (scalar)
    query_std = np.std(query)                                     #Query standard deviation (scalar)
    time_series_mean = pd.rolling_mean(time_series,m)             #Time series rolling mean; window is the length of the query
    time_series_std = pd.rolling_std(time_series,m,ddof=0)        #Time series rolling standard deviation; window is the length of the query. No degrees of freedom correction.
    dot_product = sliding_dot_product(time_series, query)
    distances = 2 * (m - (dot_product[m-1:n] - m * query_mean * time_series_mean[m-1:n]) / (query_std * time_series_std[m-1:n]))
    distances = np.sqrt(distances + 0j)                           #Normalised Euclidean distance. See page 4 of http://www.cs.ucr.edu/~eamonn/PID4481997_extend_Matrix%20Profile_I.pdf
    return distances


def STAMP_single(target_series, query_series=None, subsequence_length=10, max_time=600, self_join=False, verbose=False):
    
    #THIS IS THE SINGLE-THREADED VERSION OF THE ALGORITHM. IT IS BETTER TO USE 'STAMP_parallel'.
    
    #Based on the pseudocode - Keogh et al (2016): http://www.cs.ucr.edu/~eamonn/PID4481997_extend_Matrix%20Profile_I.pdf  
    
    n_original = query_series.shape[0]
    m = target_series.shape[0]
    
    if n_original > m:
        raise ValueError('Query series should not be larger than target series.')
        
    if m > n_original:
        query_series = np.concatenate([query_series,np.zeros(m - n_original)])    

    n = query_series.shape[0]
    
    #Initialise the matrix profile distances to be very high
    matrix_profile = 999999999 * np.ones(n - subsequence_length + 1)
    matrix_profile_index = np.zeros(n - subsequence_length + 1)

    #Matrix profile is an anytime algorithm: its accuracy improves (at a diminishing rate) the longer it runs, but its
    #output is useful even if it is terminated early. However, if the algorithm is terminated early, it is desirable to
    #have compared (to every other 'target' subsequence in the time series) 'instigating' subsequences starting at
    #random points which are evenly distributed throughout the time series rather than the first M 'instigating'
    #subsequences in the time series. Hence, the indices (the position in the time series from which 'instigating'
    #subsequences begin) are shuffled.
    indices = [i for i in range(0, n_original - subsequence_length + 1)]
    random.shuffle(indices)

    #Matrix profile is an anytime algorithm. Consequently, considerations of computational time and expense mean that
    #for large time series it may be desirable to terminate the algorithm after it has run for a user-specified time.
    start_time = time.time()
    update_time = time.time()
    max_time = time.time() + max_time 
    iteration = 0
        
    for index in indices:
        
        #Stop updating the matrix profile once time is up
        if time.time() > max_time:
            break
            
        #Compute progress update it at most once per second
        if verbose == True:
            if time.time() - update_time > 1:
                os.system('cls')
                print('{}% complete'.format(round(iteration/len(indices)*100,3)))
                print('Elapsed time: {} seconds'.format(round(time.time() - start_time,1)))
                update_time = time.time()
            iteration += 1
        
        #Compute the distances between the subsequence starting at a particular point in the time series and every
        #other sub-sequence of equal length in the time series.
        distances = MASS(target_series, query_series[index : index + subsequence_length])
        
        #Exclude trivial cases where the matrix profile will be very low because the sequence is being matched to
        #itself. These occur when the subsequence being compared is within a distance of (subsequence_length / 2)
        #of the position in the time series.  
        if self_join == True:
            exclusion_range = (int(max(0, index - subsequence_length/2)), int(min(index + subsequence_length/2 + 1, n)))
            distances[exclusion_range[0]:exclusion_range[1]] = 99999
        
        #Update the matrix profile and the matrix profile index if a subsequence which is a closer match is discovered
        matrix_profile_index = np.where(matrix_profile <= distances, matrix_profile_index, index)
        matrix_profile = np.minimum(matrix_profile,distances)
    
    output = pd.DataFrame([np.real(matrix_profile_index), np.real(matrix_profile)]).T
    output.columns = ['Matrix_Profile_Index','Matrix_Profile']

    return output


def STAMP_parallel(target_series, query_series, subsequence_length=10, max_time=600, self_join=False, verbose=False):
       
    #Based on the pseudocode - Keogh et al (2016): http://www.cs.ucr.edu/~eamonn/PID4481997_extend_Matrix%20Profile_I.pdf           
    n_original = query_series.shape[0]
    m = target_series.shape[0]
    
    if n_original > m:
        raise ValueError('Query series should not be larger than target series.')
        
    if m > n_original:
        query_series = np.concatenate([query_series,np.zeros(m - n_original)])    

    n = query_series.shape[0]
    
    processes = mp.cpu_count()
    matrix_profile = {}
    matrix_profile_index = {}
    
    #Matrix profile is an anytime algorithm: its accuracy improves (at a diminishing rate) the longer it runs, but its
    #output is useful even if it is terminated early. However, if the algorithm is terminated early, it is desirable to
    #have compared (to every other 'target' subsequence in the time series) 'instigating' subsequences starting at
    #random points which are evenly distributed throughout the time series rather than the first M 'instigating'
    #subsequences in the time series. Hence, the indices (the position in the time series from which 'instigating'
    #subsequences begin) are shuffled.
    indices = [i for i in range(0, n_original - subsequence_length + 1)]
    random.shuffle(indices)
    
    #The indices are then divided by the number of CPUs. The algorithm is easy to parallelise because each element of
    #the matrix profile is minimum distance between the 'instigating' subsequence (of user-specified length) which
    #starts at that particular position in the time series and every other 'target' subsequence in the time series.
    #Hence, if the 'instigating' time series are divided between CPUs and sub-matrix profiles computed, the overall
    #matrix profile will be the element-wise minimum of the sub-profiles.
    indices = np.array_split(np.array(indices), processes)    
    pool = mp.Pool(processes=processes)
    results = [pool.apply_async(update_matrix_profile, args=(target_series, query_series, self_join, subsequence_length, indices[s], s, n, max_time, verbose)) for s in range(0,processes)]
    output = [p.get() for p in results]
    pool.close()
        
    #The overall matrix profile is the element-wise minimum of each sub-profile, and each element of the overall
    #matrix profile index is the time series position of the corresponding sub-profile.
    s = 0
    for subindices in indices:
        matrix_profile[s] = output[s][0]
        matrix_profile_index[s] = output[s][1]
        if s != 0:
            matrix_profile_index[s] = np.where(matrix_profile[s-1] <= matrix_profile[s], matrix_profile_index[s-1], matrix_profile_index[s])
            matrix_profile[s] = np.minimum(matrix_profile[s-1],matrix_profile[s])
        s += 1
    
    output = pd.DataFrame([np.real(matrix_profile_index[s-1]), np.real(matrix_profile[s-1])]).T
    output.columns = ['Matrix_Profile_Index','Matrix_Profile']
    
    return output


def update_matrix_profile(target_series, query_series, self_join, subsequence_length, subindices, s, n, max_time, verbose=False):    
    
    #Based on the pseudocode - Keogh et al (2016): http://www.cs.ucr.edu/~eamonn/PID4481997_extend_Matrix%20Profile_I.pdf  
    
    #Initialise the matrix profile distances to be very high
    matrix_profile_ = 999999999 * np.ones(n - subsequence_length + 1)
    
    #Initialise the matrix profile index. The matrix profile corresponding to a particular point in the time series
    #is the smallest distance of between the subsequence of user-specified length that starts at this point to every
    #other subsequence of the same length in the time series. The matrix profile index gives the starting position of
    #the subsequence which is the closest match.
    matrix_profile_index_ = np.zeros(n - subsequence_length + 1)
    
    #Matrix profile is an anytime algorithm: its accuracy improves (at a diminishing rate) the longer it runs, but its
    #output is useful even if it is terminated early. Consequently, considerations of computational time and expense
    #mean that for large time series it may be desirable to terminate the algorithm after it has run for a user-
    #specified time.
    start_time = time.time()
    update_time = time.time()
    max_time = time.time() + max_time
    iteration = 0
   
    for index in subindices:
        
        #Stop updating the matrix profile once time is up
        if time.time() > max_time:
            break
            
        #Compute progress based on the first CPU and update it at most once per second
        if verbose == True:
            if s == 0:
                if time.time() - update_time > 1:
                    os.system('cls')
                    print('{}% complete'.format(round(iteration/len(subindices)*100,3)))
                    print('Number of CPUs: {}'.format(mp.cpu_count()))
                    print('Elapsed time: {} seconds'.format(round(time.time() - start_time,1)))
                    update_time = time.time()
                iteration += 1

        #Compute the distances between the subsequence starting at a particular point in the time series and every
        #other sub-sequence of equal length in the time series.
        distances = MASS(target_series, query_series[index : index + subsequence_length])
        
        #Exclude trivial cases where the matrix profile will be very low because the sequence is being matched to
        #itself. These occur when the subsequence being compared is within a distance of (subsequence_length / 2)
        #of the position in the time series.
        if self_join == True:
            exclusion_range = (int(max(0, index - subsequence_length/2)), int(min(index + subsequence_length/2 + 1, n)))
            distances[exclusion_range[0]:exclusion_range[1]] = 99999
        
        #Update the matrix profile and the matrix profile index if a subsequence which is a closer match is discovered
        matrix_profile_index_ = np.where(matrix_profile_ <= distances, matrix_profile_index_, index)
        matrix_profile_ = np.minimum(matrix_profile_,distances)
        
    return matrix_profile_, matrix_profile_index_


def STAMP(target_series, query_series=None, subsequence_length=10, max_time=60, verbose=True, parallel=True):
    
    self_join = False
    
    if type(query_series) == type(None):
        query_series = target_series
        self_join = True
    
    q_shape = query_series.shape[0]
    t_shape = target_series.shape[0]
    
    if t_shape >= q_shape:
        
        if parallel == True:
            matrix_profile = STAMP_parallel(target_series=target_series, query_series=query_series, subsequence_length=subsequence_length, max_time=max_time, self_join=self_join, verbose=verbose)
        else:
            matrix_profile = STAMP_single(target_series=target_series, query_series=query_series, subsequence_length=subsequence_length, max_time=max_time, self_join=self_join, verbose=verbose)

    elif t_shape < q_shape:
        
        #Pad the target series with q_shape - t_shape 0s at the end 
        new_target_series = np.concatenate([target_series, np.zeros(q_shape - t_shape)])

        if parallel == True:
            matrix_profile = STAMP_parallel(target_series=new_target_series, query_series=query_series, subsequence_length=subsequence_length, max_time=max_time, self_join=self_join, verbose=verbose)
        else:
            matrix_profile = STAMP_single(target_series=new_target_series, query_series=query_series, subsequence_length=subsequence_length, max_time=max_time, self_join=self_join, verbose=verbose)
    
        #Delete the q_shape - t_shape entries at the end, whose Matrix Profile values will be 0
        matrix_profile = matrix_profile[0 : t_shape]
    
    return matrix_profile