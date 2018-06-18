Attribute Information:
                                                                                        faturtes
    1. CRIM      per capita crime rate by town -- got                               = x + log x + x^0.36
    2. ZN        proportion of residential land zoned for lots over                 = x + x^(1e-4) + x^1.4 + x^1.5----
                 25,000 sq.ft.
    #3. INDUS     proportion of non-retail business acres per town                  = x + x^1e-5 ------
    4. CHAS      Charles River dummy variable (= 1 if tract bounds                  = x
                 river; 0 otherwise)
    #5. NOX       nitric oxides concentration (parts per 10 million)                = x +log(x) -----
    #6. RM        average number of rooms per dwelling                              = x + x^0.5-----
    #7. AGE       proportion of owner-occupied units built prior to 1940            = x + x^1.1-----
    8. DIS       weighted distances to five Boston employment centres               = x +log x + x^(0.001) + x^(0.01) -----
    #9. RAD       index of accessibility to radial highways                         = logx + x + (1/x)8 + (1/x)^10 + (/x)^12 ------ 
    10. TAX      full-value property-tax rate per $10,000                           = log x + x + x^(0.1)
    #11. PTRATIO  pupil-teacher ratio by town                                       = x + x^1.5       
    12. B        1000(Bk - 0.63)^2 where Bk is the proportion of blacks             = x + e^(2/x) + x^1.2 + x^1.1
                 by town
    #13. LSTAT    % lower status of the population                                  = x +x^0.1 + log x
    14. MEDV     Median value of owner-occupied homes in $1000's


*for learning rate used back propogation alogorithm
  where learning rate is at start is instailized to max of 2*min learning rate maxruncal function  

*Used 4-fold cross validation for finding/tuning lamda.  cost matrix = sum {(X.W - Y)^2} /numdata
    from lamde array

*using closed form we get the global optimum  but using gradient descent we get loacl one W
 So but closed form is not available for every expression so we use gradient descent as it converges to it.
 but our implementation uses iterations and epsilon wrt cost above defined to terminate the graient descent

