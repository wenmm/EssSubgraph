library(ggplot2)
library(dplyr)
library(gridExtra)
library(ggpubr)



simulation_network_gpu_result <- read.csv("simulation_network_gpu_result.txt", sep="")

df = simulation_network_gpu_result

df = subset(df, edges <= 1800000)
nodes_groups <- unique(df$nodes)

# 创建空列表存储图形
plot_list <- list()

method_colors <- c(
  "EssSubgraph" = "#377EB8",  # Blue
  "EMOGI" = "#E41A1C",  # Orange
  "MTGCN" = "saddlebrown",  # Green
  "GAT" = "#4DAF4A",  # Red
  "SVM" = "#FF7F00",  # Purple
  "GCN" = "#984EA3"   # Brown
  # Add more colors if you have additional methods
)

for (i in 1:length(nodes_groups)) {
  group <- nodes_groups[i]
  p <- ggplot(df[df$nodes == group, ], aes(as.numeric(edges)/100000, memory_usage, color = methods))+
    geom_point(position = position_dodge(width = .5), size=3.5) + 
    theme_classic() +
    #theme(text = element_text(size=14, color = "black")) +
    xlab("Numbe of Edges (100k)") +
    ylab("Memory Usage (G)") +
    ggtitle("nodes") +
    theme(axis.text.y = element_text(colour = 'black', size = 14),
          axis.title.y = element_text(size = 14),axis.ticks.length = unit(0.2, "cm")) + 
    theme(axis.text.x = element_text(colour = 'black', size = 14),
          axis.title.x = element_text(size = 14)) +
    theme(axis.line=element_line(linetype=1,color="black",size=1))+
    theme(axis.ticks=element_line(color="black",size=1,lineend = 12)) +
    scale_colour_manual(values = method_colors) +
    scale_x_continuous(breaks = unique(as.numeric(df$edges)/100000))
  plot_list[[i]] <- p
}

# 显示 5 张图（按2列排列）
ggarrange(plotlist = plot_list, 
          ncol = 3, nrow = 2,  # 2 列排 5 张图
          common.legend = TRUE, # 启用共享图例
          legend = "bottom", labels = c("A", "B","C","D","E"))  # 图例放在底部



