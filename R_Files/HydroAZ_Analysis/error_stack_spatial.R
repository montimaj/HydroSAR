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
years <- seq(2002, 2019)
k <- 1
for (i in years) {
  pred.raster <- raster(paste("../../Outputs/Output_AZ_Apr_Sept_2K_S/Pred_GW_Rasters/HAR/pred_", i, ".tif", sep=""))
  actual.raster <- raster(paste("../../Outputs/Output_AZ_Apr_Sept_2K_S/Actual_GW_Rasters/HAR/GW_", i, ".tif", sep=""))
  # actual.raster <- raster(paste("../../Inputs/Files_AZ_Apr_Sept_2K/RF_Data/GW_", i, ".tif", sep=""))
  
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

writeRaster(actual.mean.raster, 'Actual_Mean_HAR.tif')
writeRaster(pred.mean.raster, 'Pred_Mean_HAR.tif')

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

min_value_error  <- round(min(minValue(err.raster.stack)))
max_value_error <- round(max(maxValue(err.raster.stack)))
min_value_error <- floor(min_value_error / 100) * 100
max_value_error <- ceiling(max_value_error / 100) * 100
breaks_error <- seq(min_value_error, max_value_error, by=600)
col_error <- brewer.pal(n=length(breaks_error), name='Reds')

az_map <- readOGR('../../../HydroNet/Inputs/Data/Arizona_GW/Arizona/Arizona.shp')
ama_map <- readOGR('../../Inputs/Data/Arizona_GW/Boundary/AMA_and_INA.shp')
plot_ext <- extent(-113.7, -112.9, 33, 34)
plot_ext <- extent(-114.99, -109, 31, 37)
az_map <- crop(az_map, plot_ext)
n <- 1
plot(az_map, col='grey', border='NA', xlab='Longitude (Degree)', ylab='Latitude (Degree)')
axis(side=2, at=c(37:40))
axis(side=1, at=c(-102:-94))
plot(actual.raster.list[[n]], xlab='Longitude (Degree)', ylab='Latitude (Degree)', legend.args=list(text='Actual GW Pumping (mm)', side = 2, font = 1, cex = 1), breaks=breaks, zlim=c(min_value, max_value), col=col, box=F, axes=F, ext=plot_ext, add=T)
plot(ama_map, col=NA, border='coral', add=T)

#plot(az_map, col='NA', border='NA')
plot(pred.raster.list[[n]], xlab='Longitude (Degree)', ylab='Latitude (Degree)', legend.args=list(text='Predicted GW Pumping (mm)', side = 2, font = 0.5, cex = 1), breaks=breaks, zlim=c(min_value, max_value), col=col, box=F, axes=F, ext=plot_ext)
axis(side=2, at=c(31:37))
axis(side=1, at=c(-115:-109))

plot(az_map, col='grey', border='NA')
plot(err.raster.list[[n]], xlab='Longitude (Degree)', ylab='Latitude (Degree)', legend.args=list(text='Error (mm)', side = 2, font = 0.5, cex = 1), breaks=breaks_error, zlim=c(min_value_error, max_value_error), col=col_error, box=F, axes=F, ext=plot_ext)
axis(side=2, at=c(31:37))
axis(side=1, at=c(-115:-109))


min_value_mean  <- round(min(minValue(actual.mean.raster), minValue(pred.mean.raster)))
max_value_mean <- round(max(maxValue(actual.mean.raster), maxValue(pred.mean.raster)))
max_value_mean <- ceiling(max_value_mean / 100) * 100
breaks_mean <- seq(min_value_mean, max_value_mean, by=170)
col_mean <- rev(brewer.pal(n=length(breaks_mean) - 1, name='RdYlBu'))


# min_value_mean  <- 0
# max_value_mean <- 400
# breaks_mean <- seq(min_value_mean, max_value_mean, by=50)
# col_mean <- rev(brewer.pal(n=length(breaks_mean) - 1, name='RdYlBu'))

png("D:/HydroMST/Paper2/Figures/Spatial/Pred_HAR_S.png", width=6, height=6, units='in', res=600)
plot(az_map, col='grey', border='NA', xlab='Longitude (Degree)', ylab='Latitude (Degree)')
plot(actual.mean.raster, xlab='Longitude (Degree)', ylab='Latitude (Degree)', legend=T, legend.args=list(text='Actual GW Pumping (mm/yr)', side = 2, font = 1, cex = 1), breaks=breaks_mean, zlim=c(min_value_mean, max_value_mean), col=col_mean, ext=plot_ext, box=F, axes=F)
plot(ama_map, col=NA, border='coral', add=T)

plot(pred.mean.raster, xlab='Longitude (Degree)', ylab='Latitude (Degree)', legend.args=list(text='Predicted GW Pumping (mm/yr)', side = 2, font = 1, cex = 1), breaks=breaks_mean, zlim=c(min_value_mean, max_value_mean), col=col_mean, box=F, axes=F, ext=plot_ext)
axis(side=2, at=c(33.2, 33.6, 33.9))
# axis(side=2, at=c(31:37))
#axis(side=1, at=c(-103:-94))
axis(side=1, at=c(-113.7, -113.4, -113))
dev.off()

min_value_mean_error  <- round(minValue(err.mean.raster))
max_value_mean_error  <- round(maxValue(err.mean.raster))
min_value_mean_error <- floor(min_value_mean_error / 100) * 100
max_value_mean_error <- ceiling(max_value_mean_error / 100) * 100
breaks_error_mean <- seq(min_value_mean_error, max_value_mean_error, by=220)
col_error_mean <- brewer.pal(n=length(breaks_error_mean) - 1, name='Spectral')

png("D:/HydroMST/Paper2/Figures/Spatial/Error_HAR_S.png", width=6, height=6, units='in', res=600)
plot(az_map, col='grey', border='NA', xlab='Longitude (Degree)', ylab='Latitude (Degree)')
plot(err.mean.raster, col = rev(col_error_mean), breaks=breaks_error_mean, ylab='Latitude (Degree)', xlab='Longitude (Degree)', yaxt='n',
     legend.args=list(text='Mean Error (mm/yr)', side = 2, font = 0.5, cex = 1), ext=plot_ext, box=F, axes=F)
plot(ama_map, col=NA, border='black', add=T)
axis(side=2, at=c(31:37))
axis(side=1, at=c(-115:-109))
dev.off()


png("D:/HydroMST/Paper2/Figures/Spatial/AP_HAR_S.png", width=6, height=4.5, units='in', res=600)
plot(pred.mean.raster, actual.mean.raster, xlab='Predicted GW Pumping (mm/yr)',
     ylab='Actual GW Pumping (mm/yr)')
legend(600, 1000, bty = 'n', legend = c("1:1 relationship"),
       col = c("red"), lty = 1, cex = 0.8)
segments(x0=0,y0=0,x1=maxValue(pred.mean.raster),
         y1=maxValue(actual.mean.raster),col="red")
dev.off()

err.df <- as.data.frame(err.mean.raster, na.rm = T)
err <- err.df$layer
err.mean <- mean(err)
err.sd <- sd(err)
std.err <- err / err.sd
std.err.df <- as.data.frame(std.err)
names(std.err.df) <- c('STD.ERR')
std.err.df$STD.ERR[std.err.df$STD.ERR < -3] <- NA
std.err.df$STD.ERR[std.err.df$STD.ERR > 3] <- NA

std.err.df <- na.omit(std.err.df)
png("D:/HydroMST/Paper2/Figures/Spatial/SR_HAR_S.png", width=6, height=4.5, units='in', res=600)
breaks <- seq(min(std.err.df$STD.ERR),max(std.err.df$STD.ERR),l=35)
hist(std.err.df$STD.ERR, freq = F, main="", xlab='Standardized Residuals', breaks=breaks)
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


png("D:/HydroMST/Paper2/Figures/Spatial/SRP_HAR_S.png", width=6, height=4.5, units='in', res=600)
plot(pred.raster.df$pred, std.err.df$STD.ERR, xlab = 'Predicted GW Pumping (mm/yr)', ylab = 'Standardized Residuals')
abline(h = 0, col = "red")
dev.off()

png("D:/HydroMST/Paper2/Figures/Spatial/QQ_HAR_S.png", width=6, height=4.5, units='in', res=600)
qqnorm(std.err.df$STD.ERR, main = "")
qqline(std.err.df$STD.ERR, col = "red")
dev.off()
