import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
x=np.array([2,3,4]).reshape([-1,1])
y=np.array([5,6,15]).reshape([-1,1])
model=LinearRegression()
model.fit(x,y)
xtest=np.array([1,2,4]).reshape([-1,1])
ypredict=model.predict(xtest)
plt.scatter(x,y,c='b')

plt.plot(xtest,ypredict,color='red')
plt.show()