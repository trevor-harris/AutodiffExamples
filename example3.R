# install.packages('torch')

library(torch)

x = matrix(rnorm(100 * 10, 0, 1), 100, 10)
beta = seq(10, 1, length.out = 10)/5
y = x %*% beta + rnorm(100, 0, 0.1)

model = function(params, x) {
  w = params[[1]]
  b = params[[2]]
  torch_mm(x, w) + b
}

w = matrix(rep(0, 10), 10, 1)
b = 0

x = torch_tensor(x)
w = torch_tensor(w, requires_grad = TRUE)
b = torch_tensor(b, requires_grad = TRUE)
params = list(w, b)

loss = function(params, x, y) {
  yhat = model(params, x)
  out = torch_mean((yhat-y)**2)
  return(out)
}

alpha = 0.1
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
beta_hat = as_array(params[[1]])
plot(beta - beta_hat, ylim = c(-1, 1))

t(beta_hat)
beta
