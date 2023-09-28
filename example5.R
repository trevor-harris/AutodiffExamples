# install.packages('torch')
rm(list = ls())
gc()
library(torch)

x = matrix(rnorm(100 * 10, 0, 1), 100, 10)
beta = seq(10, 1, length.out = 10)/5
y = x %*% beta + rnorm(100, 0, 0.1)

model = function(params, x) {
  w1 = params[[1]]
  b1 = params[[2]]
  w2 = params[[3]]
  b2 = params[[4]]
  
  x = torch_mm(x, w1) + b1
  x = torch_relu(x)
  x = torch_mm(x, w2) + b2
  return(x)
}

w1 = matrix(rnorm(10 * 10, 0, 1/100), 10, 10)
b1 = rnorm(10, 0, 1/10)

w2 = matrix(rnorm(10, 0, 1/10), 10, 1)
b2 = 0

x = torch_tensor(x)
w1 = torch_tensor(w1, requires_grad = TRUE)
b1 = torch_tensor(b1, requires_grad = TRUE)
w2 = torch_tensor(w2, requires_grad = TRUE)
b2 = torch_tensor(b2, requires_grad = TRUE)
params = list(w1, b1, w2, b2)

yhat = model(params, x)
loss = function(params, x, y) {
  yhat = model(params, x)
  out = torch_mean((yhat-y)**2)
  return(out)
}

alpha = 0.01
for (i in 1:10000) {
  l = loss(params, x, y)
  l$backward()
  
  with_no_grad({
    w1 = w1$sub_(alpha * w1$grad)
    b1 = b1$sub_(alpha * b1$grad)
    w2 = w2$sub_(alpha * w2$grad)
    b2 = b2$sub_(alpha * b2$grad)
    
    w1$grad$zero_()
    b1$grad$zero_()
    w2$grad$zero_()
    b2$grad$zero_()
    params = c(w1, b1, w2, b2)
  })
}
beta_hat = as_array(params[[1]])

yhat = as_array(model(params, x))
mean((y - yhat)**2)
