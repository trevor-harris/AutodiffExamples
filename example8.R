# install.packages('deSolve')

library(deSolve)

params = c(a = 2.25, b = 0.25, c = 0.2, d = 2)
state = c(y1 = 2, y2 = 2)

lv = function(t, state, params) {
  with(as.list(c(state, params)),{
    dy1 = a * y1 - b * y1 * y2
    dy2 = c * y1 * y2 - d * y2
    list(c(dy1, dy2))
  })
}

times = seq(0, 10, by = 0.01)
y = ode(y = state, times = times, func = lv, parms = params)

plot(y[,1], y[,2], type = 'l')
lines(y[,1], y[,3], col = 'red')

y1_obs = y[,2] + torch_normal(0, 2, 1001)
y2_obs = y[,3] + torch_normal(0, 2, 1001)
plot(y[,1], y1_obs, type = 'l')
lines(y[,1], y2_obs, col = 'red')

idx = ((1:length(times)) %% 15) == 0
t_obs = times[idx]
y_obs = torch_stack(list(y1_obs, y2_obs), dim = 2)[idx]
plot(t_obs, y_obs[,1], type = 'l')
lines(t_obs, y_obs[,2], col = 'red')

t_obs = torch_tensor(t_obs, requires_grad = TRUE)
model = function(x, theta) {
  x * theta
}

diff_operator = function(x0, theta) {
  f1 = model(x0, theta)[1]
  f1$backward()
  df1 = x0$grad$clone()
  x0$grad$zero_()
  f1 = f1$clone()
  
  f2 = model(x0, theta)[2]
  f2$backward()
  df2 = x0$grad$clone()
  x0$grad$zero_()
  f2 = f2$clone()
  
  f = c(f1, f2)
  df = c(df1, df2)
  
  return(list(f, df))
}
lv_loss = function(x0, theta, params) {
  all_f = diff_operator(x0, theta)
  f = all_f[[1]]
  df = all_f[[2]]
  
  f1 = f[[1]]
  f2 = f[[2]]
  df1 = df[[1]]
  df2 = df[[2]]
  
  r1 = df1 - (params['a'] * f1 - params['b'] * f1 * f2)
  r2 = df2 - (params['c'] * f1 * f2 - params['d'] * f2)
  return(r1**2 + r2**2)
}

theta = torch_tensor(c(1.0, 1.0))
x0 = torch_tensor(t_obs[1], requires_grad = TRUE)

all_f = diff_operator(x0, theta)
lv_loss(x0, theta, params)
