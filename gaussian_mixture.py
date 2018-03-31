import numpy as np
import math

class GaussianMixture:
    def __init__(self):
        pass

    def norm_pdf_multivariate(self, x, mu, sigma):
        det = np.linalg.det(sigma)
        if det == 0:
            raise NameError("Covariance matrix is singular.")
        norm_const = 1.0/(np.math.pow(2*np.pi, float(self.D)/2) * np.math.pow(det, 0.5))
        x_mu = np.matrix(x - mu)
        inv_ = np.linalg.inv(sigma)
        result = np.math.pow(np.math.e, -0.5 * (x_mu.T * inv_ * x_mu))
        return norm_const * result

    def _m_step(self):
        N = np.sum(self.W, 0)
        self.Alpha = N/np.sum(N)

        self.Mu = self.data.T.dot(self.W).dot(np.diag(np.reciprocal(N))) # DxK

        self.Sigma = np.zeros((self.D, self.D, self.K))
        for k in range(self.K):
            data_diff = self.data.T - self.Mu[:,k][None].T.dot(np.ones((1, self.N)))
            self.Sigma[:,:,k] = ((data_diff.dot(np.diag(self.W[:, k]))).dot(data_diff.T))/N[k]

        return

    def _e_step(self):
        for k in range(self.K):
            for i in range(self.N):
                self.W[i, k] = self.Alpha[k] * self.norm_pdf_multivariate(self.data[i, :][None].T, self.Mu[:,k][None].T, self.Sigma[:, :, k])

        self.W = self.W*np.reciprocal(np.sum(self.W, 1)[None].T)

        return

    def log_likelihood(self):
        P = np.zeros((self.N, self.K))
        for k in range(self.K):
            for i in range(self.N):
                P[i, k] = self.norm_pdf_multivariate(self.data[i, :][None].T, self.Mu[:,k][None].T, self.Sigma[:, :, k])

        return np.sum(np.log(P.dot(self.Alpha)))

    def cluster(self, data, K):
        (N, D) = np.shape(data)
        nPerK = N/K

        self.K = K
        self.N = N
        self.D = D
        self.data = data
        self.W = np.zeros([N, K])

        for k in range(K):
            self.W[math.floor(k*nPerK):math.floor((k+1) * nPerK), k] = 1

        self._m_step()

        i = 0
        prevll = -999999

        while True:
            # if (tiedCov):
            #     SigmaSum = np.sum(self.Sigma, 2)
            #     for k in range(K):
            #         Sigma[:, :, k] = SigmaSum
            self._e_step()
            self._m_step()
            ll_train = self.log_likelihood()
            i = i + 1
            if i > 150 or abs(ll_train - prevll) < 0.01:
                break
            prevll = ll_train

        return self.Mu.T