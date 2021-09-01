year <- c(2008, 2010, 2015, 2020)
al_km <- c(44.096, 43.973, 70.655, 80.885)
al_acre <- c(10896.4, 10866, 17459.3, 19987.1)

png("D:/HydroMST/Paper2/Figures/Spatial/Alfalfa.png", width=6, height=4, units='in', res=600)
plot(year, al_acre, type='l', xlab='Year', ylab='Alfalfa Acreage')
dev.off()
