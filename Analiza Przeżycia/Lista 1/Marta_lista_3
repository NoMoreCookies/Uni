# Zadanie 1
# a)
X_A <- c(0.03345514, 0.08656403, 0.08799947, 0.24385821, 0.27755032,
         0.40787247, 0.58825664, 0.64125620, 0.90679161, 0.94222208)
X_B <- c(0.03788958, 0.12207257, 0.20319983, 0.24474299, 0.30492413,
         0.34224462, 0.42950144, 0.44484582, 0.63805066, 0.69119721)
t <- 1
n <- 20
R_A <- sum(X_A <= t)
R_B <- sum(X_B <= t) 
T_A <- sum(X_A) + t*(n-R_A)
T_B <- sum(X_B) + t*(n-R_B)

theta_hat_A <- R_A / T_A
theta_wave_A <- - log(1-R_A/n)/t
theta_hat_B <- R_B / T_B
theta_wave_B <- - log(1-R_B/n)/t

mi_hat_A <- 1/theta_hat_A
mi_wave_A <- 1/theta_wave_A
mi_hat_B <- 1/theta_hat_B
mi_wave_B <- 1/theta_wave_B

# b)
library(binom)
T.L.1 <- binom.confint(R, n, methods = 'exact', conf.level = 1-0.05)$lower
T.U.1 <- binom.confint(R, n, methods = 'exact', conf.level = 1-0.05)$upper

T.L.A1 <- - log(1-T.L.1)/t
T.U.A1 <- - log(1-T.U.1)/t

T.L.2 <- binom.confint(R, n, methods = 'exact', conf.level = 1-0.01)$lower
T.U.2 <- binom.confint(R, n, methods = 'exact', conf.level = 1-0.01)$upper

T.L.A2 <- - log(1-T.L.2)/t
T.U.A2 <- - log(1-T.U.2)/t

#Zadanie 2
fun <- function(X, m, n, alpha) {
  Xs <- sort(X)
  T2 <- sum(Xs[1:m]) + (n - m)*Xs[m]
  theta.hat <- m/T2
  mi.hat <- 1/theta.hat
  
  q.m.1 <- qgamma(alpha / 2, shape = m, rate = m)
  q.m.2 <- qgamma(1 - alpha / 2, shape = m, rate = m)
  
  T.L <- m*q.m.1/T2
  T.U <- m*q.m.2/T2
  
  return(list(
    theta.hat = theta.hat,
    mi.hat = mi.hat,
    TL = T.L, TU = T.U
  ))
}

A_al1 <- fun(X_A, 10, 20, 0.05)
A_al2 <- fun(X_A, 10, 20, 0.01)
B_al1 <- fun(X_B, 10, 20, 0.05)
B_al2 <- fun(X_B, 10, 20, 0.01)

A_al1
A_al2
B_al1
B_al2


# Zadanie 3
M <- 10000
theta <- 1
t.values <- c(0.5, 1, 2)
n.values <- c(10, 30)

gen1 <- function(n, alpha=1, lambda=theta, t0){
  y <- runif(n)
  x <- - (1/lambda)*log(1- y^(1/alpha))
  
  delta <- ifelse(x <= t0, 1, 0)
  x_obs <- pmin(x, t0)
  
  return(data.frame(x = x_obs, delta = delta))
}

symulacja <- function(n, t0, theta){
  repeat {
    X <- gen1(n, t0=t0, lambda=theta)
    R <- sum(X$delta)
    if (R < n) break  # akceptujemy tylko próby, gdzie jest cenzurowanie
  }
  T1 <- sum(X$x) + t0*(n - R)
  
  theta.hat <- R/T1
  theta.wave <- - log(1 - R/n)/t0
  
  return(c(theta.hat, theta.wave))
}

results <- data.frame(
  n = numeric(),
  t0 = numeric(),
  Bias_hat = numeric(),
  MSE_hat = numeric(),
  Bias_wave = numeric(),
  MSE_wave = numeric()
)

for (n in n.values) {
  for (t0 in t.values) {
    
    est <- replicate(M, symulacja(n, t0, theta))
    
    theta.hat <- est[1, ]
    theta.wave <- est[2, ]
    
    bias.hat <- mean(theta.hat - theta)
    mse.hat  <- mean((theta.hat - theta)^2)
    
    bias.wave <- mean(theta.wave - theta)
    mse.wave  <- mean((theta.wave - theta)^2)
    
   
    results <- rbind(results, data.frame(
      n = n,
      t0 = t0,
      Bias_hat = bias.hat,
      MSE_hat = mse.hat,
      Bias_wave = bias.wave,
      MSE_wave = mse.wave
    ))
  }
}

results
