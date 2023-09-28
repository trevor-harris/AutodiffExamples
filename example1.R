# install.packages('torch')
rm(list = ls())
gc()
library(torch)

x = torch_tensor(2.9)
w = torch_tensor(1.0, requires_grad = TRUE)
b = torch_tensor(2.0, requires_grad = TRUE)

y = 1.5 * x + 0.2

params = c(w, b)
model = function(params, x) {
  w = params[[1]]
  b = params[[2]]
  return(x*w + b)
}

loss = function(params, x, y) {
  yhat = model(params, x)
  out = torch_mean((yhat - y)**2)
  return(out)
}

l = loss(params, x, y)
l$backward()
w$grad

w$grad$zero_()
b$grad$zero_()
params = c(w, b)

alpha = 0.001
for (i in 1:100) {
  l = loss(params, x, y)
  l$backward()

  with_no_grad({
    w = w$sub_(alpha * w$grad)
    b = b$sub_(alpha * b$grad)
    w$grad$zero_()
    b$grad$zero_()
    params = c(w, b)
  })
}