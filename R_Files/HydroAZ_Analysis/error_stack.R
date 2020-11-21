library(raster)
library(rgdal)
library(colorRamps)
library(rasterVis)
library(viridisLite)
library(RColorBrewer)

err.raster.list <- list()
pred.raster.list <- list()
actual.raster.list <- list()
years <- seq(2011, 2018)
k <- 1
for (i in years) {
  pred.raster <- raster(paste("../../Outputs/Output_AZ_Apr_Sept/Predicted_Rasters/pred_", i, ".tif", sep=""))
  actual.raster <- raster(paste("../../Inputs/Files_AZ_Apr_Sept/RF_Data/GW_", i, ".tif", sep=""))
  
  wgs84 = "+proj=longlat +ellps=WGS84 +datum=WGS84 +no_defs"
  actual.raster = projectRaster(actual.raster, crs = wgs84, method = "ngb")
  pred.raster = projectRaster(pred.raster, crs = wgs84, method = "ngb")
  err.raster.list[[k]] <- actual.raster - pred.raster
  pred.raster.list[[k]] <- pred.raster
  actual.raster.list[[k]] <- actual.raster
  k <- k + 1
}
err.raster.stack <- stack(err.raster.list)
pred.raster.stack <- stack(pred.raster.list)
actual.raster.stack <- stack(actual.raster.list)

err.mean.raster <- mean(err.raster.stack)
actual.mean.raster <- mean(actual.raster.stack)
pred.mean.raster <- mean(pred.raster.stack)

min_value_actual  <- round(min(minValue(actual.raster.stack)))
min_value_pred  <- round(min(minValue(pred.raster.stack)))
min_value <- min(min_value_actual, min_value_pred)

max_value_actual <- round(max(maxValue(actual.raster.stack)))
max_value_pred <- round(max(maxValue(pred.raster.stack)))
max_value <- max(max_value_actual, max_value_pred)
max_value <- ceiling(max_value / 100) * 100
breaks <- seq(min_value, max_value, by=300)
col <- topo.colors(length(breaks) - 1)
col <- rev(brewer.pal(n=length(breaks) - 1, name='RdYlBu'))

plot_ext <- extent(-114, -109, 31, 35)

min_value_error  <- round(min(minValue(err.raster.stack)))
max_value_error <- round(max(maxValue(err.raster.stack)))
min_value_error <- floor(min_value_error / 100) * 100
max_value_error <- ceiling(max_value_error / 100) * 100
breaks_error <- seq(min_value_error, max_value_error, by=300)
col_error <- brewer.pal(n=length(breaks_error) - 1, name='Reds')

n <- 8
plot(actual.raster.list[[n]], xlab='Longitude (Degree)', ylab='Latitude (Degree)', legend.args=list(text='Actual GW Pumping (mm)', side = 2, font = 0.5, cex = 1), breaks=breaks, zlim=c(min_value, max_value), col=col, box=F, axes=F, ext=plot_ext)
axis(side=2, at=c(31:35))
axis(side=1, at=c(-114:-109))

plot(pred.raster.list[[n]], xlab='Longitude (Degree)', ylab='Latitude (Degree)', legend.args=list(text='Predicted GW Pumping (mm)', side = 2, font = 0.5, cex = 1), breaks=breaks, zlim=c(min_value, max_value), col=col, box=F, axes=F, ext=plot_ext)
axis(side=2, at=c(31:35))
axis(side=1, at=c(-114:-109))

plot(err.raster.list[[n]], xlab='Longitude (Degree)', ylab='Latitude (Degree)', legend.args=list(text='Error (mm)', side = 2, font = 0.5, cex = 1), breaks=breaks_error, zlim=c(min_value_error, max_value_error), col=col_error, box=F, axes=F, ext=plot_ext)
axis(side=2, at=c(31:35))
axis(side=1, at=c(-114:-109))


min_value_mean  <- round(min(minValue(actual.mean.raster), minValue(pred.mean.raster)))
max_value_mean <- round(max(maxValue(actual.mean.raster), maxValue(pred.mean.raster)))
max_value_mean <- ceiling(max_value_mean / 100) * 100
breaks_mean <- seq(min_value_mean, max_value_mean, by=300)
col_mean <- rev(brewer.pal(n=length(breaks_mean) - 1, name='RdYlBu'))

plot(actual.mean.raster, xlab='Longitude (Degree)', ylab='Latitude (Degree)', legend.args=list(text='Actual Mean GW Pumping (mm)', side = 2, font = 0.55, cex = 0.8), breaks=breaks_mean, zlim=c(min_value_mean, max_value_mean), col=col_mean, box=F, axes=F, ext=plot_ext)
plot(pred.mean.raster, xlab='Longitude (Degree)', ylab='Latitude (Degree)', legend.args=list(text='Predicted Mean GW Pumping (mm)', side = 2, font = 0.55, cex = 0.8), breaks=breaks_mean, zlim=c(min_value_mean, max_value_mean), col=col_mean, box=F, axes=F, ext=plot_ext)
axis(side=2, at=c(31:35))
axis(side=1, at=c(-114:-109))

min_value_mean_error  <- round(minValue(err.mean.raster))
max_value_mean_error  <- round(maxValue(err.mean.raster))
min_value_mean_error <- floor(min_value_mean_error / 100) * 100
max_value_mean_error <- ceiling(max_value_mean_error / 100) * 100
breaks_error_mean <- seq(min_value_mean_error, max_value_mean_error, by=200)
col_error_mean <- brewer.pal(n=length(breaks_error_mean) - 1, name='Reds')

plot(err.mean.raster, col = col_error_mean, breaks=breaks_error_mean, ylab='Latitude (Degree)', xlab='Longitude (Degree)', yaxt='n',
     legend.args=list(text='Mean Error (mm)', side = 2, font = 0.5, cex = 1), ext=plot_ext, box=F, axes=F)
axis(side=2, at=c(31:35))
axis(side=1, at=c(-114:-109))





err.df <- as.data.frame(err.mean.raster, na.rm = T)
err <- err.df$layer
err.mean <- mean(err)
err.sd <- sd(err)
std.err <- err / err.sd
std.err.df <- as.data.frame(std.err)
names(std.err.df) <- c('STD.ERR')
hist(std.err.df$STD.ERR, freq = F, main="", xlab='Standardized Residuals')
x <- seq(min(std.err.df$STD.ERR), max(std.err.df$STD.ERR), length.out=length(std.err.df$STD.ERR))
dist <- dnorm(x, mean(std.err.df$STD.ERR), sd(std.err.df$STD.ERR))
lines(x, dist, col = 'red')

plot(pred.mean.raster, actual.mean.raster, xlab="Mean Predicted GW Pumping (mm)", ylab="Mean Actual GW Pumping (mm)")
legend(0, 1500, bty = 'n', legend = c("1:1 relationship"),
       col = c("red"), lty = 1, cex = 0.8)
abline(a=0, b=1, col='red')

pred.raster.df <- as.data.frame(mean(pred.raster.stack),na.rm=T)
names(pred.raster.df) <- c('pred')
plot(pred.raster.df$pred, std.err, xlab = 'Mean Predicted GW Pumping (mm)', ylab = 'Standardized Residuals')
abline(h = 0, col = "red")
qqnorm(std.err, main = "")
qqline(std.err, col = "red")
