# Invoked by compare_solvers.py --rscript after install.packages("pliable").
args <- commandArgs(trailingOnly = TRUE)
data <- as.matrix(read.csv(args[1], header = FALSE))
p <- as.integer(args[3])
k <- ncol(data) - p - 2L
x <- data[, seq_len(p), drop = FALSE]
z <- data[, p + seq_len(k), drop = FALSE]
y <- data[, p + k + 1L]
weights <- data[, p + k + 2L]
penalties <- as.numeric(args[-seq_len(3L)])
timing <- system.time(
  fit <- pliable::pliable(x, z, y, lambda = penalties, alpha = 0.5,
                          family = "gaussian", w = weights, thr = 1e-10,
                          zlinear = TRUE, maxinter = p * k)
)
stopifnot(isTRUE(all.equal(fit$lambda, penalties)), fit$jerr == 0)
prediction <- predict(fit, x, z)
rows <- lapply(seq_along(penalties), function(i) {
  c(fit$beta[, i], fit$betaz[, i], as.vector(t(fit$theta[, , i])),
    prediction[, i])
})
write.table(do.call(rbind, rows), args[2], sep = ",", row.names = FALSE,
            col.names = FALSE)
cat(as.numeric(timing[["elapsed"]]), as.character(packageVersion("pliable")),
    as.character(getRversion()))
