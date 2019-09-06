# The idea here is to wrap nparray 
# and wrap returns that are ndarray in Tensor 
# this way we can implement the differentiate and compute specifically for arrays, 
# and that'd make the library compatible with numpy

class Tensor:
    nptensor: np.ndarray: None
    def __getattr__(self, attr):
        return getattr(self.nptensor, attr)