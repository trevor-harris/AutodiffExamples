rm(list = ls())
gc()
library(torch)

model = function(params, x) {
  w = params[[1]]
  b = params[[2]]
  c = torch_normal(0, 1, 1)
  return(x*w + b + c)
}

x = torch_tensor(0.0, requires_grad = TRUE)
w = torch_tensor(1.2)
b = torch_tensor(2.0)
params = c(w, b)

y = x*w + b + torch_normal(0, 1, 1)
f = model(params, x)
f
f$backward()
x$grad
