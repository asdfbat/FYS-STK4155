import numpy as np
from KFold_iterator import KFold_iterator
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Lasso

class Regression:
    def __init__(self):
        pass

    def generate_generic_data(self, x, y, func):
        self.xshape, self.yshape = len(x), len(y)
        x_mesh, y_mesh = np.meshgrid(x, y)
        self.set_data(x_mesh, y_mesh, func(x_mesh, y_mesh))

    def load_matrix_data(self, A):
        self.yshape, self.xshape = A.shape
        x = np.linspace(0, 1, self.xshape); y = np.linspace(0, 1, self.yshape)
        x_mesh, y_mesh = np.meshgrid(x, y)
        self.set_data(x_mesh, y_mesh, A)
        
    def set_data(self, x_mesh, y_mesh, f):
        self.f = f
        self.x_mesh, self.y_mesh = x_mesh, y_mesh
        self.x_flat, self.y_flat, self.f_flat = self.x_mesh.flatten(), self.y_mesh.flatten(), self.f.flatten()
        self.nr_datapoints = len(self.x_flat)

    def get_X(self, x, y, poly_order):
        self.poly_order = poly_order
        nr_terms = ((poly_order + 1)*(poly_order + 2))//2
        X = np.zeros((np.size(x), nr_terms))
        X[:,0] = 1

        i = 0
        for ix in range(poly_order+1):
            for iy in range(poly_order+1):
                if 0 < ix + iy < poly_order+1:
                    i += 1
                    X[:,i] = x**ix*y**iy
        print("Cond of XT*X: ", np.linalg.cond(X.T@X))
        return X

    def apply_model(self, beta):
        i = 0
        result = beta[0]
        for ix in range(self.poly_order+1):
            for iy in range(self.poly_order+1):
                if 0 < ix + iy < self.poly_order+1:
                    i += 1
                    result += beta[i]*self.x_mesh**ix*self.y_mesh**iy
        return result

    def get_beta(self, X, f, solver="OLS", lamda=0, max_iter=1e3, tol=1e-4):
        XT = X.T
        if solver=="OLS":
            print("cond XT*X: ", np.linalg.cond(XT@X))
            beta = np.linalg.pinv(XT@X)@XT@f
        elif solver =="OLS_unsafe":
            beta = np.linalg.inv(XT@X)@XT@f
        elif solver=="Ridge":
            beta = np.linalg.inv(XT@X + np.identity(X.shape[1])*lamda)@XT@f
        elif solver=="Lasso":
            _Lasso = Lasso(alpha=lamda,max_iter=max_iter,tol=tol,fit_intercept=True)
            clf = _Lasso.fit(X,f)
            beta = clf.coef_
            #_Lasso = Lasso(alpha=lamda,max_iter=max_iter,tol=tol,fit_intercept=False)
            #clf = _Lasso.fit(X[1:,:],f[1:,:])
        else:
            print("Dust")
            raise NotImplementedError
        return beta
    
    def solveCoefficients(self, poly_order=5, solver="OLS", lamda=1e-4, max_iter=1e3, tol=1e-4):
        X = self.get_X(self.x_flat, self.y_flat, poly_order)
        beta = self.get_beta(X, self.f_flat, solver=solver, lamda=lamda, max_iter=1e3, tol=1e-4)
        return beta

    def solveTrainTest(self, poly_order=5, test_fraction=0.25, solver="OLS", lamda=1e-4):
        x_flat, y_flat, f_flat = self.x_flat, self.y_flat, self.f_flat
        x_train, x_test, y_train, y_test, output_train, output_test = train_test_split(x_flat, y_flat, f_flat, test_size=test_fraction)
        X = self.get_X(x_train, y_train, poly_order)
        beta = self.get_beta(X, output_train, solver=solver, lamda=lamda)
        output_test_pred = self.apply_model(beta, x_test, y_test, poly_order)
        return output_test, output_test_pred
        
    def solveKFold(self, K=5, solver="OLS", poly_order=5, lamda=1e-4, max_iter=1e3, tol=1e-4):
        x_flat, y_flat, f_flat = self.x_flat, self.y_flat, self.f_flat
        output_pred = np.zeros(self.nr_datapoints)
        kf = KFold_iterator(self.nr_datapoints, K)
        for train_index, test_index in kf:
            x_train, x_test, y_train, y_test = x_flat[train_index], x_flat[test_index], y_flat[train_index], y_flat[test_index]
            output_train, output_test = f_flat[train_index], f_flat[test_index]
            X = self.get_X(x_train, y_train, poly_order)
            beta = self.get_beta(X, output_train, solver=solver, lamda=lamda)
            output_test_pred = self.apply_model(beta, x_test, y_test, poly_order)
            output_pred[test_index] = output_test_pred
        output_pred_stacked = np.zeros((self.yshape, self.xshape))
        for i in range(self.yshape):
            output_pred_stacked[i] = output_pred[i*self.xshape : (i+1)*self.xshape]
        return output_pred_stacked