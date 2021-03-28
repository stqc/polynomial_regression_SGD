class PolynomialRegression:
    '''
    Polynomial Regression with Stochastic Gradient Descent
    
    data_x = Independent Variables
    data_y = Dependent Variables
    alpha = Learning Rate
    degree = Polynomial Degree
    
    train(iterations = 10)
    --------------------------------------------
    Train method takes one argument i.e number of epochs
    Default = 10
    
    predict(data)
    --------------------------------------------
    Predict method takes one argument i.e data
    data should be either an array or a list
    
    '''
    def __init__(self,data_x,data_y,alpha,degree=1):
        self.degree = degree
        self.x = data_x
        self.y = data_y
        self.alpha = alpha
        self.cols = data_x.shape[1]
        self.theta1 = 0
        self.theta2 = [[0 for i in range(degree)] for j in range(self.cols)]
        
    def train_update(self,x):
        mx = 0
        for i in range(len(self.theta2)):
            for j in range(self.degree):
                mx += self.theta2[i][j]*x**(j+1)
        return self.theta1 + mx
        
    def update(self,epoch):
        total_error = 0.0
        for i in range(len(self.x)):
            error = (self.train_update(self.x[i]) - self.y[i])
            self.theta1 = self.theta1 - error*self.alpha
            for j in range(len(self.theta2)):
                for k in range(self.degree):
                    self.theta2[j][k] = self.theta2[j][k] - self.alpha*self.x[i]**(k+1)*error
            total_error+=error
        try:
                print(f'{epoch} : {total_error**2}, {error}')
        except Exception as e:
                print(e)

    def train(self,iterations=10):
        for i in range(iterations):
            self.update(i)
        
    def predict(self,data):
        output =[]
        for i in range(len(data)):
            output.append(self.train_update(data[i]))
        return output
