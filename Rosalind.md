# Bioinformatics Problems from Rosalind Stronghold
http://rosalind.info/problems/list-view/


## Counting DNA Nucleotides

A string is simply an ordered collection of symbols selected from some alphabet and formed into a word; the length of a string is the number of symbols that it contains.

An example of a length 21 DNA string (whose alphabet contains the symbols 'A', 'C', 'G', and 'T') is "ATGCTTCAGAAAGGTCTTACG."
s
Given: A DNA string s of length at most 1000 nt.

Return: Four integers (separated by spaces) counting the respective number of times that the symbols 'A', 'C', 'G', and 'T' occur in s.

```python
from collections import Counter

contents = open("/Downloads/rosalind_dna.txt", "r").readlines()
tout=Counter(contents[0].rstrip())
print(' '.join([str(tout[b]) for  b in 'ACGT']))
```

## Transcribing DNA into RNA 

An RNA string is a string formed from the alphabet containing 'A', 'C', 'G', and 'U'.

Given a DNA string t corresponding to a coding strand, its transcribed RNA string u is formed by replacing all occurrences of 'T' in t with 'U' in u.

Given: A DNA string t having length at most 1000 nt.

Return: The transcribed RNA string of t.

Sample Dataset

```python
contents = open("/Downloads/rosalind_rna.txt", "r").readlines()
tout=contents[0].rstrip()
print(tout.replace('T', 'U'))
```

## Complementing a Strand of DNA

Problem
In DNA strings, symbols 'A' and 'T' are complements of each other, as are 'C' and 'G'.

The reverse complement of a DNA string s is the string sc formed by reversing the symbols of s, then taking the complement of each symbol (e.g., the reverse complement of "GTCA" is "TGAC").

Given: A DNA string s of length at most 1000 bp.

Return: The reverse complement sc of s.

Sample Dataset
AAAACCCGGT
Sample Output
ACCGGGTTTT

```python
tdat='AAAACCCGGT'
tdat = open("/Downloads/rosalind_revc.txt", "r").readlines()[0].rstrip()
#tstring.translate({"A":"T","C":"G", "T":"A","G":"C" })
#assert tstring.translate(tstring.maketrans("ATCG", "TAGC"))[::-1]=='ACCGGGTTTT'
print(tdat.translate(tdat.maketrans("ATCG", "TAGC"))[::-1])

```

## Rabbits and Recurrence Relations solved by 20588
Feb. 22, 2013, 3:50 a.m. by Rosalind TeamTopics: Combinatorics, Dynamic Programming
←→
Wascally Wabbitsclick to expand
Problem
A sequence is an ordered collection of objects (usually numbers), which are allowed to repeat. Sequences can be finite or infinite. Two examples are the finite sequence (π,−2‾√,0,π) and the infinite sequence of odd numbers (1,3,5,7,9,…). We use the notation an to represent the n-th term of a sequence.

A recurrence relation is a way of defining the terms of a sequence with respect to the values of previous terms. In the case of Fibonacci's rabbits from the introduction, any given month will contain the rabbits that were alive the previous month, plus any new offspring. A key observation is that the number of offspring in any month is equal to the number of rabbits that were alive two months prior. As a result, if Fn represents the number of rabbit pairs alive after the n-th month, then we obtain the Fibonacci sequence having terms Fn that are defined by the recurrence relation Fn=Fn−1+Fn−2 (with F1=F2=1 to initiate the sequence). Although the sequence bears Fibonacci's name, it was known to Indian mathematicians over two millennia ago.

When finding the n-th term of a sequence defined by a recurrence relation, we can simply use the recurrence relation to generate terms for progressively larger values of n. This problem introduces us to the computational technique of dynamic programming, which successively builds up solutions by using the answers to smaller cases.

Given: Positive integers n≤40 and k≤5.

Return: The total number of rabbit pairs that will be present after n months, if we begin with 1 pair and in each generation, every pair of reproduction-age rabbits produces a litter of k rabbit pairs (instead of only 1 pair).

Sample Dataset
5 3
Sample Output
19

```python
import copy

def fib_rabs(n, k):

    if n <2 :
        return(1)
    elif n==3:
        return(1+k)
    else:

        track=[1, 1, 1+k]
        print(3, track)
        for i in range(4 , n+1):
            tmp=track[1:3]
            track[2]+=k*track[1]
            track[0:2]=tmp
            
#            print(i, track, newval)
    
    
    return(track[2])

```

```python

import pandas as pd


tdat=pd.read_csv("data/rosalind_fib.txt", delimiter=' ', header=None)
#tdat = open("sample.txt", "r").readlines()[0].rstrip().split(' ')
print(tdat)
print(fib_rabs(tdat[0][0], tdat[0][1]))

```

##Mortal Fibonacci Rabbits solved by 8804
March 8, 2013, 12:36 p.m. by Rosalind TeamTopics: Combinatorics, Dynamic Programming
←→
Wabbit Seasonclick to expand
Problem

Figure 4. A figure illustrating the propagation of Fibonacci's rabbits if they die after three months.
Recall the definition of the Fibonacci numbers from “Rabbits and Recurrence Relations”, which followed the recurrence relation Fn=Fn−1+Fn−2 and assumed that each pair of rabbits reaches maturity in one month and produces a single pair of offspring (one male, one female) each subsequent month.

Our aim is to somehow modify this recurrence relation to achieve a dynamic programming solution in the case that all rabbits die out after a fixed number of months. See Figure 4 for a depiction of a rabbit tree in which rabbits live for three months (meaning that they reproduce only twice before dying).

Given: Positive integers n≤100 and m≤20.

Return: The total number of pairs of rabbits that will remain after the n-th month if all rabbits live for m months.