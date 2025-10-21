library(dplyr)
library(ggplot2)

plot_feature_perturbation <- function(filename) {
  # 读取数据
  feature_pertubation <- read.csv(filename, sep="")
  
  # 计算统计量
  sbg <- feature_pertubation %>% 
    group_by(pertubation_type, percent) %>% 
    summarise(count = n(),
              mAUC = mean(AUC), 
              sdAUC = sd(AUC), 
              mAUPR = mean(AUPR), 
              sdAUPR = sd(AUPR),
              .groups = 'drop')
  
  # 画图
  p <- ggplot() + 
    geom_point(data = feature_pertubation, aes(x = percent, y = AUPR, colour = factor(pertubation_type))) +
    geom_line(data = sbg, aes(x = percent, y = mAUPR, color = factor(pertubation_type)), size = 2) +
    geom_errorbar(data = sbg, aes(x = percent, ymin = mAUPR - sdAUPR, ymax = mAUPR + sdAUPR), width = 0.05) +
    xlab("Percent Edited") + ylab("AUPRC") +
    ggtitle("") +
    theme_classic() +
    theme(legend.position = c(0.2, 0.2),
          legend.title = element_blank(),
          legend.text = element_text(size = 9),
          axis.text = element_text(size = 10, color = "black"),
          axis.title = element_text(color = "black")) +
    guides(colour = guide_legend(override.aes = list(size = 6))) +
    scale_color_manual(values = c("#F87189", "#CE9031", "#A48CF5")) +
    coord_cartesian(xlim = c(0, 1), ylim = c(0, 1)) +
    theme(
      axis.line = element_line(linewidth = 1),
      axis.ticks.length = unit(0.2, "cm"),  # 刻度长度
      axis.ticks = element_line(linewidth = 1),  # 刻度粗细
      axis.text = element_text(size = 12),  # 刻度标签大小
      axis.title = element_text(size = 14)  # 坐标轴标题大小
    )
  
  
  return(p)
}



p1 <- plot_feature_perturbation("EMOGI_pc50_pertubation.txt")
p2 <- plot_feature_perturbation("GAT_pc50_pertubation.txt")
p3 <- plot_feature_perturbation("GCN_pc50_pertubation.txt")
p4 <- plot_feature_perturbation("MTGCN_pc50_pertubation.txt")
p5 <- plot_feature_perturbation("EssSubgraph2_pc50_pertubation.txt")

library(ggpubr)
arranged_plot <- ggarrange(p5, p4, p1, p3, p2, common.legend = TRUE, legend="bottom", labels = c("A", "B","C","D"))
arranged_plot


df_EMOGI <- read.csv("EMOGI_pc50_pertubation.txt", sep="")
df_GAT <- read.csv("GAT_pc50_pertubation.txt", sep="")
df_GCN <- read.csv("GCN_pc50_pertubation.txt", sep="")
df_MTGCN <- read.csv("MTGCN_pc50_pertubation.txt", sep="")
df_EssSubgraph <- read.csv("EssSubgraph2_pc50_pertubation.txt", sep="")

df_EMOGI$Method <- "EMOGI"
df_GAT$Method <- "GAT"
df_GCN$Method <- "GCN"
df_MTGCN$Method <- "MTGCN"
df_EssSubgraph$Method <- "EssSubgraph"

library(ggplot2)
library(dplyr)

# 假设你的5个数据框分别叫 df1, df2, df3, df4, df5
# 合并
df_all <- bind_rows(df_EMOGI, df_GAT, df_GCN, df_MTGCN, df_EssSubgraph)

# 设置颜色
method_colors <- c(
  "EMOGI" = "#E41A1C",
  "EssSubgraph" = "#377EB8",
  "GAT" = "#4DAF4a",
  "GCN" = "#984EA3",
  "MTGCN" = "#8B4513"
)

# 计算 mean ± sd
df_summary <- df_all %>%
  group_by(pertubation_type, Method, percent) %>%
  summarise(
    mean_AUPR = mean(AUPR, na.rm = TRUE),
    sd_AUPR   = sd(AUPR, na.rm = TRUE),
    .groups = "drop"
  )

# 作图函数（含误差线）
plot_func <- function(ptype) {
  df_raw <- df_all %>% filter(pertubation_type == ptype)
  df_sum <- df_summary %>% filter(pertubation_type == ptype)
  
  ggplot() +
    # 散点（点更大）
    geom_point(data = df_raw,
               aes(x = percent, y = AUPR, color = Method),
               position = position_jitter(width = 0.02, height = 0),
               alpha = 0.6, size = 3.5) +
    
    # 误差条
    geom_errorbar(data = df_sum,
                  aes(x = percent,
                      ymin = mean_AUPR - sd_AUPR,
                      ymax = mean_AUPR + sd_AUPR,
                      color = Method),
                  width = 0.03, size = 1.2) +
    
    # 平均曲线
    geom_line(data = df_sum,
              aes(x = percent, y = mean_AUPR, color = Method, group = Method),
              size = 2) +
    
    scale_color_manual(values = method_colors) +
    theme_classic(base_size = 16) +
    labs(x = "Perturbed Fraction", y = "AUPRC", color = "Method",
         title = ptype) +
    theme(
      text        = element_text(color = "black"),               # 全局文字黑色
      plot.title  = element_text(hjust = 0.5, face = "bold", size = 18),
      axis.ticks  = element_line(color = "black", size = 1.2),
      axis.ticks.length = unit(0.3, "cm"),
      axis.line   = element_line(color = "black", size = 1.2),
      axis.text   = element_text(size = 15, face = "bold", color = "black"),
      axis.title  = element_text(size = 17, face = "bold", color = "black"),
      legend.title= element_text(size = 16, face = "bold", color = "black"),
      legend.text = element_text(size = 15, color = "black"),
      panel.border= element_blank()
    )
}

# 举例画 network 类型
p1 <- plot_func("network")
p2 <- plot_func("feature")
p3 <- plot_func("networkfeature")

library(ggpubr)
ggarrange(p1, p2, p3, 
          nrow = 1, ncol = 3, 
          common.legend = TRUE, legend = "right",
          labels = c("A", "B", "C"))
