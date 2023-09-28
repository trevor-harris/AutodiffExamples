# logistic regression

rm(list = ls())
gc()

library(torch)

x = matrix(rnorm(500 * 10, 0, 1), 500, 10)
beta = seq(10, -10, length.out = 10)/5
logit_p = x %*% beta
p = 1/(1 + exp(-logit_p))
y = rbinom(dim(x)[1], 1, p)


model = function(params, x) {
  w = params[[1]]
  b = params[[2]]
  logit_p = torch_mm(x, w) + b
  p = 1/(1 + exp(-logit_p))
}

loss = function(params, x, y) {
  phat = model(params, x)[,1]
  phat = torch_clamp(phat, 1e-4, 1-1e-4)
  out = -torch_mean(y * log(phat) + (1-y)*log(1-phat))
  return(out)
}

w = matrix(rep(0, 10), 10, 1)
b = 0

x = torch_tensor(x)
w = torch_tensor(w, requires_grad = TRUE)
b = torch_tensor(b, requires_grad = TRUE)
params = list(w, b)

alpha = 0.1
for (i in 1:1000) {
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
plot(beta - beta_hat, ylim = c(-4, 4))

t(beta_hat)
beta
