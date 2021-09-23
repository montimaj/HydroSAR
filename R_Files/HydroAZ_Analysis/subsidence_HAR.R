library(raster)
library(rgdal)
library(colorRamps)
library(rasterVis)
library(viridisLite)
library(usmap)
library(RColorBrewer)

ts_raster <- raster('../../Outputs/Output_AZ_Annual_2K_T_Full/Subsidence_Analysis/TS_HAR.tif')
tpgw_raster <- raster('../../Outputs/Output_AZ_Annual_2K_T_Full/Subsidence_Analysis/TPGW_HAR.tif')
sed_raster <- raster('../../Outputs/Output_AZ_Annual_2K_T_Full/Subsidence_Analysis/Sed_HAR.tif')

wgs84 = "+proj=longlat +ellps=WGS84 +datum=WGS84 +no_defs"
ts_raster <- projectRaster(ts_raster, crs = wgs84, method = "ngb")
tpgw_raster <- projectRaster(tpgw_raster, crs = wgs84, method = "ngb")
sed_raster <- projectRaster(sed_raster, crs = wgs84, method = "ngb")

plot_ext <- extent(-113.7, -112.9, 33, 34)

tiff("D:/HydroMST/Paper2/Figures_New/Subsidence/TS_HAR_2010_2020.tif", width=6, height=6, units='in', res=600)
plot(abs(ts_raster), xlab='Longitude (Degree)', ylab='Latitude (Degree)', legend=T, legend.args=list(text='Total Subsidence (mm)', side = 2, font = 1, cex = 1), ext=plot_ext, box=F, axes=F)
axis(side=2, at=c(33.2, 33.6, 33.9))
axis(side=1, at=c(-113.7, -113.4, -113))
dev.off()

tiff("D:/HydroMST/Paper2/Figures_New/Subsidence/TPGW_HAR_2010_2020.tif", width=6, height=6, units='in', res=600)
plot(tpgw_raster, xlab='Longitude (Degree)', ylab='Latitude (Degree)', legend=T, legend.args=list(text='Total Predicted GW Pumping (mm)', side = 2, font = 1, cex = 1), ext=plot_ext, box=F, axes=F)
axis(side=2, at=c(33.2, 33.6, 33.9))
axis(side=1, at=c(-113.7, -113.4, -113))
dev.off()

tiff("D:/HydroMST/Paper2/Figures_New/Subsidence/Sed_HAR_2010_2020.tif", width=6, height=6, units='in', res=600)
plot(sed_raster, xlab='Longitude (Degree)', ylab='Latitude (Degree)', legend=T, legend.args=list(text='Sediment Thickness (m)', side = 2, font = 1, cex = 1), ext=plot_ext, box=F, axes=F)
axis(side=2, at=c(33.2, 33.6, 33.9))
axis(side=1, at=c(-113.7, -113.4, -113))
dev.off()
