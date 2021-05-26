import numpy as np
def a2l(arr,ele_dim):
    arr = np.array(arr)
    s = arr.shape
    if s[len(s)-1] != ele_dim:
        raise Exception()
    return np.reshape(arr,[-1,ele_dim])

def l2a(list,sample_arr):
    s1 = sample_arr.shape
    s2 = list.shape
    newshape = np.concatenate([s1[:len(s1)-1],s2[len(s2)-1:]])
    return np.reshape(list,newshape)

class udMatrix:
    def __init__(self,A):
        self.data = np.reshape(A,[-1])
        self.shape = np.array(A.shape)
        self.multiplier = [np.prod(np.concatenate([self.shape[i+1:],np.array([1])])) for i in range(len(self.shape))]
        self.multiplier = np.array(self.multiplier,dtype = np.int64)

        self.vecM = lambda A: np.reshape(A,[np.prod(self.SHAPE['input shape']),1])
        self.vecN = lambda B: np.reshape(B,[np.prod(self.SHAPE['output shape']),1])

        self.devecM = lambda a: np.reshape(a,self.SHAPE['input shape'])
        self.devecN = lambda b: np.reshape(b,self.SHAPE['output shape'])

    def len(self):
        return len(self.data)

    def ij2k(self,ij_arr):
        ij_arr = np.array(ij_arr)
        ij_list = a2l(ij_arr,len(self.shape))
        mul = np.broadcast_to(self.multiplier,[ij_list.shape[0],self.multiplier.shape[0]])
        k_list = np.sum(ij_list * mul, axis = 1, dtype = np.int64, keepdims = True)
        k_arr = l2a(k_list,ij_arr)
        return k_arr

    def k2ij(self,k_arr):
        k_arr = np.array(k_arr)
        k_list = a2l(k_arr,1)
        ij_list = np.zeros([len(k_list),self.multiplier.shape[0]])
        for i,m in enumerate(self.multiplier):
            ij_list[:,i:i+1] = np.array(k_list / m, dtype = np.int64)
            k_list = k_list % m
        ij_arr = l2a(ij_list,k_arr)
        return ij_arr

    def numpy(self):
        return np.reshape(self.data,self.shape)
    
    def copy(self):
        return udMatrix(self.numpy())

    def is_valid(self,ij_arr):
        return np.logical_and(np.array(ij_arr) < self.shape,0 <= np.array(ij_arr))

    def get_by_k_arr(self,k_arr):
        k_arr = np.array(k_arr)
        k_list = a2l(k_arr,1)
        Ak_list = self.data[k_list]
        Ak_arr = l2a(Ak_list,k_arr)
        return Ak_arr

    def set_by_k_arr(self,k_arr,x_arr):
        k_arr,x_arr = np.array(k_arr),np.array(x_arr)
        s = x_arr.shape
        k_list,x_list = a2l(k_arr,1),a2l(x_arr,s[len(s)-1])
        self.data[k_list] = x_list

    def get(self,ij_arr):
        ij_arr = np.array(ij_arr)
        valid_ij = self.is_valid(ij_arr)
        ij_arr = ij_arr * valid_ij
        k_arr = self.ij2k(ij_arr)
        return self.get_by_k_arr(k_arr) * np.prod(valid_ij, axis = len(valid_ij.shape)-1, keepdims=True)
        
    def set(self,ij_arr,x_arr):
        ij_arr,x_arr = np.array(ij_arr),np.array(x_arr)
        k_arr = self.ij2k(ij_arr)
        self.set_by_k_arr(k_arr,x_arr)
