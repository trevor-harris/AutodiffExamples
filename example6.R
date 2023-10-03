rm(list = ls())
gc()
library(torch)

model_fn = function(params, x) {
  w1 = params[[1]]
  w2 = params[[2]]
  
  x = x * w1
  x = torch_relu(x)
  x = x + torch_normal(0, 1, 1)
  x = x * w2
  return(x)
}

loss_fn = function(params, x, y) {
  yhat = model_fn(params, x)
  out = torch_mean((yhat-y)**2)
  return(out)
}

x = torch_tensor(1.0)
w1 = torch_tensor(1.2, requires_grad = TRUE)
w2 = torch_tensor(0.2, requires_grad = TRUE)
y = 2.0 * x
params = c(w1, w2)

loss = loss_fn(params, x, y)
loss$backward()
w1$grad
w1$grad$zero_()
