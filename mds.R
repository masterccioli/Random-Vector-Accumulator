library(MASS)
library(BiDimRegression)
library(vegan)
rm(list = ls())

#converts similarities to dissimilarities/distances
simtodist <- function(sims, gamma = 1) {
  base <- exp(-gamma * sims)
  base/sum(base)
}

Word_list <- c("financial","savings","finance","pay","invested",
               "loaned","borrow","lend","invest","investments",
               "bank","spend","save","astronomy","physics",
               "chemistry","psychology","biology","scientific",
               "mathematics","technology","scientists","science",
               "research","sports","team","teams","football",
               "coach","sport","players","baseball","soccer",
               "tennis","basketball")

file = 'cosine_ww_matrix.csv'
data = read.csv(file, header = FALSE,check.names=F) # load data

d = simtodist(data.matrix(data),gamma=1)


fit <- cmdscale(d,eig=TRUE, k=2) # k is the number of dim
fit # view results

# plot solution
x <- fit$points[,1]
y <- fit$points[,2]

# save plot
jpeg(paste('ww_matrix.jpg'))
plot(x, y, xlab="Coordinate 1", ylab="Coordinate 2",
     main="RVA", type="n")
text(x, y, labels = Word_list, cex=.8)
dev.off()

