#Write a function that takes an array of integers sorted in ascending order, and returns an array of the squares of each number, also in sorted ascending order.



Example: tdat=[-4, -2, -1, 0,  3]
Output:  [0, 1, 4,  9, 16]

#Cases
# All negative
# All positive
# Mixed neg and positive
# Identical entries
# Matched pos/neg entries


output=[0]*len(tdat)


def square_sort(input_array):
    
    #Initialize each side 
    sorted_neg_entries=[]
    sorted_pos_entries=[]
    
    for entry in input_array:
        if entry<0:
            sorted_neg_entries=sorted_neg_entries+[entry]
        else:
           sorted_pos_entries=sorted_pos_entries+[entry] 
    
    
    sorted_neg_entries=sorted_neg_entries[ : : -1]
    
    i=0
    j=0
    cur_pos=sorted_pos_entries[i]
    cur_neg=sorted_neg_entries[j]
    if len(sorted_pos_entries)==0:
        return(sorted_neg_entries)
    elif len(sorted_neg_entries)==0:
        return(sorted_neg_entries)
    else:
        output=[0]*len(input_array)
        for k in range(len(input_array)):
            if (cur_pos<=cur_neg):
                output[k]=cur_pos
                i+=1
                if (i<len(sorted_pos_entries)):
                    cur_pos=sorted_pos_entries[i]
                
            else:
                output[k]=cur_neg
                j+=1
                if (j<len(sorted_neg_entries)):
                    cur_neg=sorted_neg_entries[j]
                

    
    return(output)


