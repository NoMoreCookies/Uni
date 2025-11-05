# Zadanie 1

density_weib <- function(x, alpha, beta, gamma) {
  dens <- ifelse(
    x > 0,
    (alpha * gamma / beta) *
      (x / beta)^(alpha - 1) *
      (1 - exp(-(x / beta)^alpha))^(gamma - 1) *
      exp(-(x / beta)^alpha),
    0
  )
  return(dens)
}

distrib_weib <- function(x, alpha, beta, gamma){
  dist <- (1-exp(-(x/beta)^alpha))^gamma
  return(dist)
}

quant_weib <- function(p, alpha, beta, gamma){
  quant <- beta * (-log(1 - p^(1 / gamma)))^(1 / alpha)
  return(quant)
}

haz_weib <- function(x, alpha, beta, gamma){
  haz <- density_weib(x, alpha, beta, gamma) / (1 - distrib_weib(x, alpha, beta, gamma))
  return(haz)
}


# Zadanie 2
library(ggplot2)
xvalue <- seq(0.1, 10, 0.05)
yvalue1 <- haz_weib(xvalue, 1/2, 2, 3/4)
yvalue2 <- haz_weib(xvalue, 1, 2, 1)
yvalue3 <- haz_weib(xvalue, 3/2, 3, 3)
yvalue4 <- haz_weib(xvalue, 3/2, 4, 1/8)
yvalue5 <- haz_weib(xvalue, 3/4, 1, 2)

data1 <- data.frame(xvalue, yvalue1) 
data2 <- data.frame(xvalue, yvalue2) 
data3 <- data.frame(xvalue, yvalue3) 
data4 <- data.frame(xvalue, yvalue4) 
data5 <- data.frame(xvalue, yvalue5) 

hazard_plot <- ggplot() + geom_line(data=data1, aes(x=xvalue, y=yvalue1, color='A')) + 
  geom_line(data=data2, aes(x=xvalue, y=yvalue2, color='B')) + 
  geom_line(data=data3, aes(x=xvalue, y=yvalue3, color='C')) + 
  geom_line(data=data4, aes(x=xvalue, y=yvalue4, color='D')) + 
  geom_line(data=data5, aes(x=xvalue, y=yvalue5, color='E')) +
  scale_color_manual(values=c('A'='yellow', 'B'='blue', 'C'='magenta', 'D'='green', 'E'='red')) +
  theme(legend.position = "right") + xlim(0,10) + ylim(0,1) + theme_minimal() + 
  labs(y="hazard", x="x", title = "Wykresy funkcji hazardu")

print(hazard_plot)

# Zadanie 3
rweib_exp <- function(n, alpha, beta, gamma){
  u <- runif(n)  # losujemy z U(0,1)
  x <- quant_weib(u, alpha, beta, gamma)  
  return(x)
}


# Zadanie 4
mala_p <- rweib_exp(50, 1, 2, 1)
mala_d <- rweib_exp(50, 1/2, 2, 3/8)
duza_p <- rweib_exp(100, 1, 2, 1)
duza_d <- rweib_exp(100, 1/2, 2, 3/8)

par(mfrow = c(2, 2))

hist(mala_p, probability = TRUE, col = "lightblue", main = "α=1, β=2, γ=1, n=50")
lines(density(mala_p), col = "deeppink", lwd = 2)

hist(mala_d, probability = TRUE, col = "lightblue", main = "α=1/2, β=2, γ=3/8, n=50")
lines(density(mala_d), col = "deeppink", lwd = 2)

hist(duza_p, probability = TRUE, col = "lightblue", main = "α=1, β=2, γ=1, n=100")
lines(density(duza_p), col = "deeppink", lwd = 2)

hist(duza_d, probability = TRUE, col = "lightblue", main = "α=1/2, β=2, γ=3/8, n=100")
lines(density(duza_d), col = "deeppink", lwd = 2)



# Zadanie 5
statystyki_weib <- function(dane, alpha, beta, gamma, nazwa){
  srednia <- mean(dane)
  mediana_emp <- median(dane)
  sd_emp <- sd(dane)
  kw1_emp <- quantile(dane, 0.25)
  kw3_emp <- quantile(dane, 0.75)
  min_emp <- min(dane)
  max_emp <- max(dane)
  rozstep <- max_emp - min_emp
  
  mediana_teor <- quant_weib(0.5, alpha, beta, gamma)
  kw1_teor <- quant_weib(0.25, alpha, beta, gamma)
  kw3_teor <- quant_weib(0.75, alpha, beta, gamma)
  
  data.frame(
    Próba = nazwa,
    Średnia = srednia,
    Mediana_empiryczna = mediana_emp,
    Mediana_teoretyczna = mediana_teor,
    SD = sd_emp,
    Q1_emp = kw1_emp,
    Q1_teor = kw1_teor,
    Q3_emp = kw3_emp,
    Q3_teor = kw3_teor,
    Min = min_emp,
    Max = max_emp,
    Rozstęp = rozstep,
    stringsAsFactors = FALSE
  )
}

tab_stat <- rbind(
  statystyki_weib(mala_p, 1, 2, 1, "α=1, β=2, γ=1, n=50"),
  statystyki_weib(mala_d, 1/2, 2, 3/8, "α=1/2, β=2, γ=3/8, n=50"),
  statystyki_weib(duza_p, 1, 2, 1, "α=1, β=2, γ=1, n=100"),
  statystyki_weib(duza_d, 1/2, 2, 3/8, "α=1/2, β=2, γ=3/8, n=100")
)

num_cols <- sapply(tab_stat, is.numeric)
tab_stat[num_cols] <- lapply(tab_stat[num_cols], function(x) round(x, 4))

print(tab_stat)
