import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

class LR:
    MSE_Record=[]
    def gradAscent1(self,dataMat, labelMat):
        dataMat=np.hstack((dataMat,np.ones((dataMat.shape[0],1))))
        nrow, nclo = dataMat.shape
        weights = np.ones((nclo, 1))
        maxCycles = 500
        while maxCycles > 0:
            inZ = dataMat.dot(weights)
            y = self.sigmoid(inZ)
            error = labelMat - y
            # show errors
            MSE=sum(np.square(error))/error.shape[0]
            LR.MSE_Record.append(MSE)
            alpha = 0.001
            weights = weights + alpha * dataMat.transpose().dot(error)  # dataMat.transpose().dot(error) 为梯度
            maxCycles -= 1

        return weights

    def sigmoid(self,inZ):
        return 1.0 / (1.0 + np.exp(-inZ))
    def plotMSE(self):
        plt.plot(LR.MSE_Record)
        plt.show()
