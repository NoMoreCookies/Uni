# Zadanie 1

gen1 <- function(n, alpha, lambda, t0){
  y <- runif(n)
  x <- - (1/lambda)*log(1- y^(1/alpha))
  
  delta <- ifelse(x <= t0, 1, 0)
  x_obs <- pmin(x, t0)
  
  return(data.frame(x = x_obs, delta = delta))
}

gen2 <- function(n, alpha, lambda, m) {
  y <- runif(n)
  x <- - (1/lambda)*log(1- y^(1/alpha))
  
  x <- sort(x)
  x_cenz <- x[m]
  
  X <- c(x[1:m], rep(x_cenz, n-m))
  delta <- c(rep(1,m), rep(0, n-m))
  
  return(data.frame(X, delta))
}


gen3 <-  function(n, alpha, lambda, eta){
  y <- runif(n)
  x <- - (1/lambda)*log(1- y^(1/alpha))
  c <- rexp(n, rate=1/eta)
  x <- pmin(x, c)
  delta <- ifelse(x==c, 0, 1)
  
  return(data.frame(x, delta))
}


# Zadanie 2
podstawowe_statystyki <- function(x) {
  min <- min(x)
  q1 <- quantile(x, 0.25)
  med <- median(x)
  q3 <- quantile(x, 0.75)
  max <- max(x)
  n <- length(x)
  
  wynik <- c(
    Min = min,
    Q1 = q1,
    Mediana = med,
    Q3 = q3,
    Max = max,
    n = n
  )
  
  return(round(wynik, 2))
}

dane_1 <- gen1(n=20, alpha = 1.2, lambda = 1, t0=7)
kompletne_1 <- sum(dane_1$delta == 1)
dane_2 <- gen2(n = 20, alpha = 1.2, lambda = 1, m = 12)
kompletne_2 <- sum(dane_2$delta == 1)
dane_3 <- gen3(n = 20, alpha = 1.2, lambda = 1, eta = 1.8)
kompletne_3 <- sum(dane_3$delta == 1)

podstawowe_statystyki(dane_1$x)
kompletne_1
podstawowe_statystyki(dane_2$X)
kompletne_2
podstawowe_statystyki(dane_3$x)
kompletne_3


#Zadanie 3
czasy_A <- c(0.03345514, 0.08656403, 0.08799947, 0.24385821, 0.27755032,
              0.40787247, 0.58825664, 0.64125620, 0.90679161, 0.94222208)
delta_A <- c(rep(1,10), rep(0,10))

dane_A <- data.frame(czasy = c(czasy_A, rep(1,10)), delta = delta_A)

czasy_B <- c(0.03788958, 0.12207257, 0.20319983, 0.24474299, 0.30492413,
             0.34224462, 0.42950144, 0.44484582, 0.63805066, 0.69119721)
delta_B <- c(rep(1,10), rep(0,10))

dane_B <- data.frame(czasy = c(czasy_B, rep(1,10)), delta = delta_A)


statystyki_A <- podstawowe_statystyki(dane_A$czasy)
kompletne_A <- sum(dane_A$delta == 1)  


statystyki_B <- podstawowe_statystyki(dane_B$czasy)
kompletne_B <- sum(dane_B$delta == 1)

statystyki_A
kompletne_A
statystyki_B
kompletne_B
