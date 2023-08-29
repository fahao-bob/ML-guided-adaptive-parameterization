# %%
######### rank order centroid method
import numpy as np #requires numpy

# %%
class RankOrderCentroid():
    def __init__(self):
        self.__weight_array_dict = {} # cache weight vectors to avoid recalculation

    def __vector_centroid(self,vec):
        veclen = len(vec)
        temp_weight_array = np.array(range(1,veclen+1))
        inverse_vec = 1./temp_weight_array
        tempsum = 0
        weight_array = []
        for idx, elem in enumerate(reversed(inverse_vec)):
            tempsum += elem
            weight_array.append(tempsum/veclen)
        self.__weight_array_dict[veclen] = weight_array[::-1] #reverse array

    def get_ROC_weight(self, array_length, array_rank):
        """INT array_length: length of vector (number of items)
        INT array_rank: rank of item in the item vector with 1 being highest"""
        if array_length in self.__weight_array_dict:
            return self.__weight_array_dict[array_length][array_rank-1]
        else:
            self.__vector_centroid(np.array(range(1,array_length+1)))
            return self.get_ROC_weight(array_length,array_rank)

    def get_ROC_array(self, array_length):
        """Returns entire ROC weight array for n items
        INT array_length: length of vector (number of items)"""
        if array_length in self.__weight_array_dict:
            return np.round(self.__weight_array_dict[array_length],3)
        else:
            self.__vector_centroid(np.array(range(1,array_length+1)))
            return np.round(self.get_ROC_array(array_length),3)

# %%
# roc = RankOrderCentroid() #instantiate
# print(roc.get_ROC_weight(3,2)) # 0.2778
# print(roc.get_ROC_array(3)) # [0.521, 0.271, 0.145, 0.063]

# %%


# %%
