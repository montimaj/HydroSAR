library(raster)
library(colorRamps)
library(RColorBrewer)

tpgw_raster1 <- raster('C:/Users/sayan/PycharmProjects/HydroSAR/Outputs/Output_AZ_Apr_Sept/Subsidence_Analysis/TPGW/TPGW_2004_2010.tif')
ls_raster1 <- raster('C:/Users/sayan/PycharmProjects/HydroSAR/Outputs/Output_AZ_Apr_Sept/Subsidence_Analysis/Subsidence_GW/2004_2010/TS_2004_2010.tif')

tpgw_raster2 <- raster('C:/Users/sayan/PycharmProjects/HydroSAR/Outputs/Output_AZ_Apr_Sept/Subsidence_Analysis/TPGW/TPGW_2010_2018.tif')
ls_raster2 <- raster('C:/Users/sayan/PycharmProjects/HydroSAR/Outputs/Output_AZ_Apr_Sept/Subsidence_Analysis/Subsidence_GW/2010_2018/TS_2010_2018.tif')


wgs84 <- "+proj=longlat +ellps=WGS84 +datum=WGS84 +no_defs"
tpgw_raster1 <- projectRaster(tpgw_raster1, crs = wgs84, method = "ngb")
ls_raster1 <- projectRaster(ls_raster1, crs = wgs84, method = "ngb")

tpgw_raster2 <- projectRaster(tpgw_raster2, crs = wgs84, method = "ngb")
ls_raster2 <- projectRaster(ls_raster2, crs = wgs84, method = "ngb")


tpgw_raster1 <- tpgw_raster1 / 9
tpgw_raster2 <- tpgw_raster2 / 9
ls_raster1 <- ls_raster1 / 9
ls_raster2 <- ls_raster2 / 9

min_value  <- round(min(minValue(tpgw_raster1), minValue(tpgw_raster2)))
max_value <- round(max(maxValue(tpgw_raster1), maxValue(tpgw_raster2)))
max_value <- ceiling(max_value / 100) * 100
breaks <- seq(min_value, max_value, by=10)
col <- rev(brewer.pal(n=length(breaks) - 1, name='RdYlBu'))


col <- rev(brewer.pal(n=11, name='RdYlBu'))
plot_ext <- extent(-115, -108.9, 31.98, 37.5)
plot(tpgw_raster2, xlab='Longitude (Degree)', ylab='Latitude (Degree)', legend.args=list(text='Predicted GW Pumping (cm)', side = 2, font = 0.5, cex = 1), col=col, box=F, axes=F, ext=plot_ext)
axis(side=2, at=c(31:37))
axis(side=1, at=c(-115:-109))

col <- rev(brewer.pal(n=9, name='Reds'))
plot(ls_raster2, xlab='Longitude (Degree)', ylab='Latitude (Degree)', legend.args=list(text='Land Subsidence (cm)', side = 2, font = 0.5, cex = 1), col=col, box=F, axes=F, ext=plot_ext)
axis(side=2, at=c(31:37))
axis(side=1, at=c(-115:-109))


plot_ext <- extent(-115, -108.9, 31.98, 37.5)
plot(tpgw_raster1, xlab='Longitude (Degree)', ylab='Latitude (Degree)', legend.args=list(text='Predicted GW Pumping (cm)', side = 2, font = 0.5, cex = 1), breaks=breaks, zlim=c(min_value, max_value), col=col, box=F, axes=F, ext=plot_ext)
plot(tpgw_raster2, xlab='Longitude (Degree)', ylab='Latitude (Degree)', legend.args=list(text='Predicted GW Pumping (cm)', side = 2, font = 0.5, cex = 1), breaks=breaks, zlim=c(min_value, max_value), col=col, box=F, axes=F, ext=plot_ext)
axis(side=2, at=c(31:37))
axis(side=1, at=c(-115:-109))



col <- rev(brewer.pal(n=7, name='Blues'))

plot(ls_raster1, xlab='Longitude (Degree)', ylab='Latitude (Degree)', legend.args=list(text='Land Subsidence (cm)', side = 2, font = 0.5, cex = 1), col=col, box=F, axes=F, ext=plot_ext)
plot(ls_raster2, xlab='Longitude (Degree)', ylab='Latitude (Degree)', legend.args=list(text='Land Subsidence (cm)', side = 2, font = 0.5, cex = 1), col=col, box=F, axes=F, ext=plot_ext)
axis(side=2, at=c(31:37))
axis(side=1, at=c(-115:-109))



tpgw1.df <- as.data.frame(tpgw_raster1, na.rm=T)
ls1.df <- as.data.frame(ls_raster1, na.rm=T)
plot(tpgw1.df$TPGW_2004_2010, abs(ls1.df$TS_2004_2010))

pos <- which(ls1.df$TS_2004_2010 >= 0)
tpgw1.df$TPGW_2004_2010[pos] <- NA
ls1.df$TS_2004_2010[ls1.df$TS_2004_2010 >= 0] <- NA
tpgw1.df <- na.omit(tpgw1.df)
ls1.df <- na.omit(ls1.df)
ls1.df$TS_2004_2010 <- abs(ls1.df$TS_2004_2010)
cor(tpgw1.df$TPGW_2004_2010, ls1.df$TS_2004_2010)
plot(tpgw1.df$TPGW_2004_2010, ls1.df$TS_2004_2010)

tpgw2.df <- as.data.frame(tpgw_raster2, na.rm=T)
ls2.df <- as.data.frame(ls_raster2, na.rm=T)
cor(tpgw2.df$TPGW_2010_2018, abs(ls2.df$TS_2010_2018))
plot(tpgw2.df$TPGW_2010_2018, abs(ls2.df$TS_2010_2018))


pos <- which(ls2.df$TS_2010_2018 >= 0)
tpgw2.df$TPGW_2010_2018[pos] <- NA
ls2.df$TS_2010_2018[ls2.df$TS_2010_2018 >= 0] <- NA
tpgw2.df <- na.omit(tpgw2.df)
ls2.df <- na.omit(ls2.df)
ls2.df$TS_2010_2018 <- abs(ls2.df$TS_2010_2018)
cor(tpgw2.df$TPGW_2010_2018, ls2.df$TS_2010_2018)
plot(tpgw2.df$TPGW_2010_2018, ls2.df$TS_2010_2018)

names(tpgw1.df) <- c('tpgw')
names(tpgw2.df) <- c('tpgw')
tpgw.df <- rbind(tpgw1.df, tpgw2.df)
names(ls1.df) <- c('ls')
names(ls2.df) <- c('ls')
ls1.df$ls <- abs(ls1.df$ls)
ls2.df$ls <- abs(ls2.df$ls)
ls.df <- rbind(ls1.df, ls2.df)

plot(tpgw.df$tpgw, ls.df$ls)
cor(tpgw.df$tpgw, ls.df$ls)
m <- lm(ls.df$ls ~ tpgw.df$tpgw)
summary(m)

