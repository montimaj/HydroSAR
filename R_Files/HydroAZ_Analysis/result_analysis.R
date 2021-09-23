library(ggplot2)

mytheme = list(
  theme_classic()+
    theme(panel.background = element_blank(),strip.background = element_rect(colour=NA, fill=NA),panel.border = element_rect(fill = NA, color = "black"),
          legend.title = element_text(face="plain"),legend.position="bottom", strip.text = element_text(face="plain", size=9),
          axis.text=element_text(face="plain"),axis.title = element_text(face="plain"),plot.title = element_text(face = "bold", hjust = 0.5,size=13))
)


merged_df <- read.csv('D:/HydroMST/Paper2/Results_New/Scale/Spatial/merged.csv')
scale <- as.factor(merged_df$Scale)
tiff("D:/HydroMST/Paper2/Figures_New/SA/SA_S.tif", width=6, height=4, units='in', res=600)

ggplot(data = merged_df, aes(x=Window, y=Test_Score)) + 
  mytheme + 
  geom_point(aes(colour=scale)) + 
  geom_line(aes(colour=scale)) + 
  labs(x=bquote(~sigma~'(pixel)'), y=bquote(~R^2), color='scale (km)') + 
  scale_x_continuous(breaks=seq(1, 10, by=1))

dev.off()


merged_df <- read.csv('D:/HydroMST/Paper2/Results_New/Scale/ST/merged.csv')
scale <- as.factor(merged_df$Scale)
tiff("D:/HydroMST/Paper2/Figures_New/SA/SA_ST.tif", width=6, height=4, units='in', res=600)

ggplot(data = merged_df, aes(x=Window, y=Test_Score)) + 
  mytheme + 
  geom_point(aes(colour=scale)) + 
  geom_line(aes(colour=scale)) + 
  labs(x=bquote(~sigma~'(pixel)'), y=bquote(~R^2), color='scale (km)') + 
  scale_x_continuous(breaks=seq(1, 10, by=1))

dev.off()



merged_df <- read.csv('D:/HydroMST/Paper2/Results_New/Scale/Temporal/merged.csv')
scale <- as.factor(merged_df$Scale)
tiff("D:/HydroMST/Paper2/Figures_New/SA/SA_T.tif", width=6, height=4, units='in', res=600)

ggplot(data = merged_df, aes(x=Window, y=Test_Score)) + 
  mytheme + 
  geom_point(aes(colour=scale)) + 
  geom_line(aes(colour=scale)) + 
  labs(x=bquote(~sigma~'(pixel)'), y=bquote(~R^2), color='scale (km)') + 
  scale_x_continuous(breaks=seq(1, 10, by=1))

dev.off()

