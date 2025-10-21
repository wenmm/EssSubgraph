library(dplyr)
library(ggplot2)

#PC compare
pc_compare <- read.delim("pc_compare.txt",sep = " ")

sbg <- pc_compare %>% 
  group_by(PC) %>% 
  summarise(count = n(),
            mAUC = mean(AUC), 
            sdAUC = sd(AUC), 
            mAUPR = mean(AUPR), 
            sdAUPR = sd(AUPR))

library(RColorBrewer)

p1<- ggplot(pc_compare, aes(PC, AUC)) +
  geom_point(size = 3) +
  geom_smooth(size = 3) +
  ggtitle("") +
  scale_x_continuous(breaks = c(10, 50, 100, 150, 200,250,300)) +
  theme(
    legend.position = 'top',
    legend.title = element_blank(),
    legend.text = element_text(size = 9), # Fixed typo
    axis.text = element_text(size = 10, color = "black"),
    axis.title = element_text(color = "black")
  ) +
  guides(colour = guide_legend(override.aes = list(size = 2))) +
  scale_color_brewer(palette = "Dark2") +
  geom_point(data = sbg,      # mean points per x
             aes(x = PC, y = mAUC), 
             color = "red", size = 4) +
  theme_classic() +
  theme(legend.position = c(0.2, 0.2),
        legend.title = element_blank(),
        legend.text = element_text(size = 9),
        axis.text = element_text(size = 10, color = "black"),
        axis.title = element_text(color = "black")) +
  theme(
    axis.line = element_line(linewidth = 1),
    axis.ticks.length = unit(0.2, "cm"),  # 刻度长度
    axis.ticks = element_line(linewidth = 1),  # 刻度粗细
    axis.text = element_text(size = 12),  # 刻度标签大小
    axis.title = element_text(size = 14)  # 坐标轴标题大小
  ) +
  ylab("AUROC") + xlab("Vector Size")


p2 <- ggplot(pc_compare, aes(PC, AUPR)) +
  geom_point(size = 3) +
  geom_smooth(size = 3) +
  ggtitle("") +
  scale_x_continuous(breaks = c(10, 50, 100, 150, 200,250,300)) +
  theme(
    legend.position = 'top',
    legend.title = element_blank(),
    legend.text = element_text(size = 9), # Fixed typo
    axis.text = element_text(size = 10, color = "black"),
    axis.title = element_text(color = "black")
  ) +
  guides(colour = guide_legend(override.aes = list(size = 2))) +
  scale_color_brewer(palette = "Dark2")+
  geom_point(data = sbg,      # mean points per x
             aes(x = PC, y = mAUPR), 
             color = "red", size = 4) +
  theme_classic() +
  theme(legend.position = c(0.2, 0.2),
        legend.title = element_blank(),
        legend.text = element_text(size = 9),
        axis.text = element_text(size = 10, color = "black"),
        axis.title = element_text(color = "black")) +
  theme(
    axis.line = element_line(linewidth = 1),
    axis.ticks.length = unit(0.2, "cm"),  # 刻度长度
    axis.ticks = element_line(linewidth = 1),  # 刻度粗细
    axis.text = element_text(size = 12),  # 刻度标签大小
    axis.title = element_text(size = 14)  # 坐标轴标题大小
  )+
  ylab("AUPRC") + xlab("Vector Size")


library(ggpubr)
arranged_plot <- ggarrange(p2,p1, labels = c("A", "B"))
arranged_plot

library(ComplexHeatmap)
library(colorRamp2)



library("RColorBrewer")

col<- colorRampPalette(c("red", "white", "blue"))(256)



methods_compare_heatmap <- read.delim("methods_prauc_compare.txt", row.names=1)

df <- as.matrix(methods_compare_heatmap)

col_fun <- colorRampPalette(c("blue", "white", "red"))(256)


row_group = factor(c(rep("Network + Node feature", 5), "DNN",rep("Node feature", 2), "Network only"), levels = c("Network + Node feature", "DNN", "Node feature", "Network only"))
Heatmap(df, name = "mat", 
        cluster_rows = FALSE,
        cluster_columns = FALSE,
        row_split = row_group,
        cell_fun = function(j, i, x, y, width, height, fill) {
          grid.text(sprintf("%.2f", df[i, j]), x, y, gp = gpar(fontsize = 10, col = "black"))
        },
        col = col_fun)



