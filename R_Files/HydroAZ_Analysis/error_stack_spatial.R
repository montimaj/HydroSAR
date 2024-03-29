library(raster)
library(rgdal)
library(colorRamps)
library(rasterVis)
library(viridisLite)
library(usmap)
library(RColorBrewer)

err.raster.list <- list()
pred.raster.list <- list()
actual.raster.list <- list()
years <- seq(2002, 2020)
k <- 1
for (i in years) {
  pred.raster <- raster(paste("../../Outputs/Output_AZ_Annual/Pred_GW_Rasters/HAR/pred_", i, ".tif", sep=""))
  actual.raster <- raster(paste("../../Outputs/Output_AZ_Annual/Actual_GW_Rasters/HAR/GW_", i, ".tif", sep=""))
  
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

writeRaster(actual.mean.raster, 'Actual_HAR_S.tif')
writeRaster(pred.mean.raster, 'Pred_HAR_S.tif')
writeRaster(err.mean.raster, 'Error_HAR_S.tif')

min_value_mean  <- round(min(minValue(actual.mean.raster), minValue(pred.mean.raster)))
max_value_mean <- round(max(maxValue(actual.mean.raster), maxValue(pred.mean.raster)))
max_value_mean <- ceiling(max_value_mean / 100) * 100
breaks_mean <- seq(min_value_mean, max_value_mean, by=170)
max_value_mean <- 700
breaks_mean <- seq(min_value_mean, max_value_mean, by=70)
col_mean <- rev(brewer.pal(n=length(breaks_mean), name='BrBG'))
col_mean <- c("#1D7480", "#35978F", "#80CDC1", "#C7EAE5", "#F5F5F5", "#F6E8C3", "#DFC27D", "#BF812D", "#8C510A", "#543005")
actual.mean.raster[actual.mean.raster < min_value_mean] <- min_value_mean
actual.mean.raster[actual.mean.raster > max_value_mean] <- max_value_mean
pred.mean.raster[pred.mean.raster > max_value_mean] <- max_value_mean
pred.mean.raster[pred.mean.raster < min_value_mean] <- min_value_mean
plot_ext <- extent(-113.7, -112.9, 33, 34)

tiff("D:/HydroMST/Paper2/Figures_New/Spatial/Actual_HAR_S.tif", width=6, height=6, units='in', res=600)
plot(actual.mean.raster, xlab=list('Longitude (Degree)', cex=1.5), ylab=list('Latitude (Degree)', cex=1.5), legend=T, legend.args=list(text='Actual GW Pumping (mm/yr)', side = 2, font = 1, cex = 1.5), breaks=breaks_mean, zlim=c(min_value_mean, max_value_mean), col=col_mean, ext=plot_ext, box=F, axes=F, legend.shrink=0.8)
axis(side=2, at=c(33.2, 33.6, 33.9), cex.axis=1.5)
axis(side=1, at=c(-113.7, -113.4, -113), cex.axis=1.5)
dev.off()

tiff("D:/HydroMST/Paper2/Figures_New/Spatial/Pred_HAR_S.tif", width=6, height=6, units='in', res=600)
plot(pred.mean.raster, xlab=list('Longitude (Degree)', cex=1.5), ylab=list('Latitude (Degree)', cex=1.5), legend.args=list(text='Predicted GW Pumping (mm/yr)', side = 2, font = 1, cex = 1.5), breaks=breaks_mean, zlim=c(min_value_mean, max_value_mean), col=col_mean, box=F, axes=F, ext=plot_ext, legend.shrink=0.8)
axis(side=2, at=c(33.2, 33.6, 33.9), cex.axis=1.5)
axis(side=1, at=c(-113.7, -113.4, -113), cex.axis=1.5)
dev.off()

min_value_mean_error  <- round(minValue(err.mean.raster))
max_value_mean_error  <- round(maxValue(err.mean.raster))
min_value_mean_error <- floor(min_value_mean_error / 100) * 100
max_value_mean_error <- ceiling(max_value_mean_error / 100) * 100
min_value_mean_error <- -500
max_value_mean_error <- 700
breaks_error_mean <- seq(min_value_mean_error, max_value_mean_error, by=120)
col_error_mean <- brewer.pal(n=length(breaks_error_mean), name='Spectral')
err.mean.raster[err.mean.raster > max_value_mean_error] <- max_value_mean_error
err.mean.raster[err.mean.raster < min_value_mean_error] <- min_value_mean_error

tiff("D:/HydroMST/Paper2/Figures_New/Spatial/Error_HAR_S.tif", width=6, height=6, units='in', res=600)
plot(err.mean.raster, col = rev(col_error_mean), breaks=breaks_error_mean, xlab=list('Longitude (Degree)', cex=1.5), ylab=list('Latitude (Degree)', cex=1.5), yaxt='n',
     legend.args=list(text='Mean Error (mm/yr)', side = 2, font = 0.5, cex = 1.5), ext=plot_ext, box=F, axes=F, legend.shrink=0.8, zlim=c(min_value_mean_error, max_value_mean_error))
axis(side=2, at=c(33.2, 33.6, 33.9), cex.axis=1.5)
axis(side=1, at=c(-113.7, -113.4, -113), cex.axis=1.5)
dev.off()

err.mean.raster <- mean(err.raster.stack)
actual.mean.raster <- mean(actual.raster.stack)
pred.mean.raster <- mean(pred.raster.stack)

tiff("D:/HydroMST/Paper2/Figures_New/Spatial/AP_HAR_S.tif", width=7, height=5, units='in', res=600)
plot(pred.mean.raster, actual.mean.raster, xlab='Predicted GW Pumping (mm/yr)',
     ylab='Actual GW Pumping (mm/yr)', cex=1.5, cex.axis=1.5, cex.lab=1.5)
legend(600, 400, bty = 'n', legend = c("1:1 relationship"), col = c("red"), lty = 1, cex = 1.5)
segments(x0=0,y0=0,x1=maxValue(actual.mean.raster),
         y1=maxValue(actual.mean.raster),col="red")
dev.off()

err.df <- as.data.frame(err.mean.raster, na.rm = T)
err <- err.df$layer
err.mean <- mean(err)
err.sd <- sd(err)
std.err <- err / err.sd
std.err.df <- as.data.frame(std.err)
names(std.err.df) <- c('STD.ERR')

num_2sig <- length(std.err.df$STD.ERR[std.err.df$STD.ERR >= -2 & std.err.df$STD.ERR <= 2])
p_2sig <- num_2sig * 100/ length(std.err.df$STD.ERR)


std.err.df$STD.ERR[std.err.df$STD.ERR < -2] <- NA
std.err.df$STD.ERR[std.err.df$STD.ERR > 2] <- NA

std.err.df <- na.omit(std.err.df)
tiff("D:/HydroMST/Paper2/Figures_New/Spatial/SR_HAR_S.tif", width=6, height=4.5, units='in', res=600)
breaks <- seq(min(std.err.df$STD.ERR),max(std.err.df$STD.ERR),l=12)
hist(std.err.df$STD.ERR, freq = F, main="", xlab='Standardized Residuals', breaks=breaks, cex=1.5, cex.axis=1.5, cex.lab=1.5)
x <- seq(min(std.err.df$STD.ERR), max(std.err.df$STD.ERR), length.out=length(std.err.df$STD.ERR))
dist <- dnorm(x, mean(std.err.df$STD.ERR), sd(std.err.df$STD.ERR))
lines(x, dist, col = 'red')
dev.off()



pred.raster.df <- as.data.frame(pred.mean.raster)
names(pred.raster.df) <- c('pred')
err.df <- as.data.frame(err.mean.raster)
names(err.df) <- c('error')
pred.raster.df$pred[is.na(err.df$error) == T] <- NA
pred.raster.df <- na.omit(pred.raster.df)


tiff("D:/HydroMST/Paper2/Figures_New/Spatial/SRP_HAR_S.tif", width=6, height=4.5, units='in', res=600)
plot(pred.raster.df$pred, std.err.df$STD.ERR, xlab = 'Predicted GW Pumping (mm/yr)', ylab = 'Standardized Residuals', cex=1.5, cex.lab=1.5, cex.axis=1.5)
abline(h = 0, col = "red")
dev.off()

tiff("D:/HydroMST/Paper2/Figures_New/Spatial/QQ_HAR_S.tif", width=6, height=4.5, units='in', res=600)
qqnorm(std.err.df$STD.ERR, main = "", cex=1.5, cex.lab=1.5, cex.axis=1.5)
qqline(std.err.df$STD.ERR, col = "red")
dev.off()

caret::postResample(pred.mean.raster, actual.mean.raster)
