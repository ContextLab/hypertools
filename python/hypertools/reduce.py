from .helpers import reduceD, reduceD_list

def reduce(arr,ndims=3):
    if type(arr) is list:
        return reduceD_list(arr,ndims)
    else:
        return reduceD(srr,ndims)
