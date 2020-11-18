library(raster)
library(rgdal)
library(colorRamps)
library(rasterVis)
library(viridisLite)

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

min_value_actual  <- round(min(minValue(actual.raster.stack)))
min_value_pred  <- round(min(minValue(pred.raster.stack)))
min_value <- min(min_value_actual, min_value_pred)

max_value_actual <- round(max(maxValue(actual.raster.stack)))
max_value_pred <- round(max(maxValue(pred.raster.stack)))
max_value <- max(max_value_actual, max_value_pred)
max_value <- ceiling(max_value / 100) * 100
breaks <- seq(min_value, max_value, by=200)
col <- rev(terrain.colors(length(breaks) - 1))

plot_ext <- extent(-114, -109, 31, 35)
plot(actual.raster.list[[1]], ylab='Longitude (Degree)', legend.args=list(text='Actual GW Pumping (mm)', side = 2, font = 0.5, cex = 0.7), breaks=breaks, zlim=c(min_value, max_value), col=col, box=F, axes=F, ext=plot_ext)
plot(pred.raster.list[[1]], legend.args=list(text='Predicted GW Pumping (mm)', side = 2, font = 0.5, cex = 0.7), breaks=breaks, zlim=c(min_value, max_value), col=col, box=F, axes=F, ext=plot_ext)
axis(side=2, at=c(31:35))
axis(side=1, at=c(-114:-109))
min_value_error  <- round(min(minValue(err.raster.stack)))
max_value_error <- round(max(maxValue(err.raster.stack)))
min_value_error <- floor(min_value_error / 100) * 100
max_value_error <- ceiling(max_value_error / 100) * 100
breaks_error <- seq(min_value_error, max_value_error, by=150)
col_error <- rev(matlab.like2(length(breaks_error) - 1))
plot(err.raster.list[[1]], legend.args=list(text='Error (mm)', side = 2, font = 0.5, cex = 0.8), breaks=breaks_error, zlim=c(min_value_error, max_value_error), col=col_error, box=F, axes=F)




plot(err.mean.raster, col = matlab.like2(255), ylab='Latitude (Degree)', xlab='Longitude (Degree)', yaxt='n',
     legend.args=list(text='Error (mm)', side = 2))
axis(side=2, at=c(37, 38, 39, 40))
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

pred.raster.df <- as.data.frame(mean(pred.raster.stack),na.rm=T)
names(pred.raster.df) <- c('pred')
plot(pred.raster.df$pred, std.err, xlab = 'Mean Predicted GW Pumping (mm)', ylab = 'Standardized Residuals')
abline(h = 0, col = "red")
qqnorm(std.err, main = "")
qqline(std.err, col = "red")
