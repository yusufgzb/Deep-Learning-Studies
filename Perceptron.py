import numpy as np

class Perceptron():
    def __init__(self,learning_reate=0.01,n_iters=1000):
        self.lr=learning_reate
        self.n_iters=n_iters
        self.actication_func=self._unit_step_func
        self.weights=None
        self.bias=None


    def fit(self,X,y):
        n_samples,n_features=X.shape

        self.weights=np.zeros(n_features)
        self.bias=0

        #y değeri 0 yada 1 lerden oluşmalıdır 
        #   bizde buradaki değerleri 0 la 1 yapıyoruz
        y_=np.array([1 if i>0 else 0 for i in y])


        for _ in range(self.n_iters):
            for idx,x_i in enumerate(X):
                #predict methodundaki yöndetmi yapıyoruz
                linear_output=np.dot(x_i,self.weights)+self.bias
                y_predict=self.actication_func(linear_output)
                
                # update için formülümüzü yazalım
                update=self.lr * (y_[idx]-y_predict)
                self.weights +=update*x_i
                self.bias +=update


    #tahmin etme için predict methodunu yazalım
    def predict(self,X):
        #numpy ile doğrusal bir formül yazıyoruz
        #linear_output=g(f(w,h))=g(w^(f)*x+b)
        linear_output=np.dot(X,self.weights)+self.bias
        #sonucu actication_func sokarak y_predict değerini buluyoruz
        y_predict=self.actication_func(linear_output)
        return y_predict
    
    def _unit_step_func(self,x):
        return np.where(x>=0,1,0)#x 0 dan farklıysa 1 yap değilse hep 0 olsun