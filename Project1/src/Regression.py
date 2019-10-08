import numpy as np
from KFold_iterator import KFold_iterator
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Lasso
from sklearn.preprocessing import StandardScaler

class Regression:
    """ Class for performing polynomial regression on a chosen dataset of two explanatory variables."""
    def __init__(self):
        pass

    def generate_generic_data(self, x, y, func):
        # Generates generic data from a chosen function of two explanatory variables, x and y.
        self.xshape, self.yshape = len(x), len(y)
        x_mesh, y_mesh = np.meshgrid(x, y)
        self.set_data(x_mesh, y_mesh, func(x_mesh, y_mesh))

    def load_matrix_data(self, A, x=None, y=None):
        # Loads data from a provided matrix. x and y are set to span [-1, 1] unless they are provided as additional arguments.
        self.yshape, self.xshape = A.shape
        if x is None and y is None:
            x = np.linspace(-1, 1, self.xshape); y = np.linspace(-1, 1, self.yshape)
        x_mesh, y_mesh = np.meshgrid(x, y)
        self.set_data(x_mesh, y_mesh, A)
        
    def set_data(self, x_mesh, y_mesh, f):
        # Internal method for flattening and setting up data.
        self.f = f
        self.x_mesh, self.y_mesh = x_mesh, y_mesh
        self.x_flat, self.y_flat, self.f_flat = self.x_mesh.flatten(), self.y_mesh.flatten(), self.f.flatten()
        self.nr_datapoints = len(self.x_flat)

    def ravel_data(self, x):
        # Internal method for raveling data that has been flattened.
        temp = np.zeros((self.yshape, self.xshape))
        for i in range(self.yshape):
            temp[i] = x[i*self.xshape : (i+1)*self.xshape]
        return temp

    def get_X(self, x, y, poly_order):
        # Function for generating the design matrix X, assuming polynomial regression of order poly_order, from two explanatory variables x and y.
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
        return X

    def apply_model(self, beta, x, y, ravel_xy = True):
        # Given a vector of coefficients, beta, applies the model corresponding to these coefficients to some explantory variables x and y.
        i = 0
        result = beta[0]
        for ix in range(self.poly_order+1):
            for iy in range(self.poly_order+1):
                if 0 < ix + iy < self.poly_order+1:
                    i += 1
                    result += beta[i]*x**ix*y**iy
        if len(x.shape) > 1:
            result = result.flatten()
        if ravel_xy:
            return self.ravel_data(result)
        else:
            return result

    def get_beta(self, X, f, solver="OLS", lamda=1e-4, max_iter=1e8, tol=1e-3):
        # Performs the regression of some type, given by the solver argument, on the data given by the design matrix X and the output data f.
        # For Ridge, a lambda value may also be given, as well as tolerance and max iteration for Lasso, which is solved by sklearn library.
        XT = X.T
        if solver=="OLS":
            beta = np.linalg.pinv(XT@X)@XT@f
        elif solver=="OLS_unsafe":
            beta = np.linalg.inv(XT@X)@XT@f
        elif solver=="Ridge":
            beta = np.linalg.pinv(XT@X + np.identity(X.shape[1])*lamda)@XT@f
        elif solver=="Lasso":
            _Lasso = Lasso(alpha=lamda,max_iter=max_iter,tol=tol,fit_intercept=False)
            clf = _Lasso.fit(X,f)
            beta = clf.coef_
        else:
            print("Dust")
            raise NotImplementedError
        return beta
    
    def solveCoefficients(self, poly_order=5, solver="OLS", lamda=1e-4, max_iter=1e8, tol=1e-3):
        # Solves the system and returns the betas for whatever data is preloaded. Solver and polynomial order may be specified.
        X = self.get_X(self.x_flat, self.y_flat, poly_order)
        beta = self.get_beta(X, self.f_flat, solver=solver, lamda=lamda, max_iter=max_iter, tol=tol)
        return beta

    def solveTrainTest(self, poly_order=5, test_fraction=0.25, solver="OLS", lamda=1e-4):
        # Performs a train test split of the preloaded dataset, and returns the true and predicted output on the test data.
        x_flat, y_flat, f_flat = self.x_flat, self.y_flat, self.f_flat
        x_train, x_test, y_train, y_test, output_train, output_test = train_test_split(x_flat, y_flat, f_flat, test_size=test_fraction)
        X = self.get_X(x_train, y_train, poly_order)
        beta = self.get_beta(X, output_train, solver=solver, lamda=lamda)
        output_test_pred = self.apply_model(beta, x_test, y_test, ravel_xy = False)
        return output_test, output_test_pred
        
    def solveKFold(self, K=10, solver="OLS", poly_order=5, lamda=1e-4, max_iter=1e8, tol=1e-3, store_beta=False):
        # Performs a K fold cross validation on the preloaded data. Splits the data in K folds, trains on K-1 folds, and tests on one fold,
        # until all folds have been tested on. 
        if store_beta:
            self.beta = np.zeros(((poly_order + 1)*(poly_order + 2))//2)
        x_flat, y_flat, f_flat = self.x_flat, self.y_flat, self.f_flat
        output_pred = np.zeros(self.nr_datapoints)
        kf = KFold_iterator(self.nr_datapoints, K)
        for train_index, test_index in kf:
            x_train, x_test, y_train, y_test = x_flat[train_index], x_flat[test_index], y_flat[train_index], y_flat[test_index]
            output_train, output_test = f_flat[train_index], f_flat[test_index]
            X = self.get_X(x_train, y_train, poly_order)
            beta = self.get_beta(X, output_train, solver=solver, lamda=lamda, max_iter=max_iter, tol=tol)
            if store_beta:
                self.beta += beta
            output_test_pred = self.apply_model(beta, x_test, y_test, ravel_xy = False)
            output_pred[test_index] = output_test_pred
        output_pred_stacked = np.zeros((self.yshape, self.xshape))
        for i in range(self.yshape):
            output_pred_stacked[i] = output_pred[i*self.xshape : (i+1)*self.xshape]
        if store_beta:
            self.beta /= K
        return output_pred_stacked