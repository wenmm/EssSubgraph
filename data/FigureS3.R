library(ggplot2)
library(dplyr)

# 读入数据
df <- read.csv("FigureS3_simulation_network_gpu_result_with_time", sep = " ", header = TRUE)

# 转换边数单位（100k）
df <- df %>%
  filter(edges <= 1800000 , methods != "SVM") %>%
  mutate(edges_100k = edges / 100000)

# 设置 methods 颜色
method_colors <- c(
  "EMOGI" = "#E41A1C",
  "EssSubgraph" = "#377EB8",
  "GAT" = "#4DAF4a",
  "GCN" = "#984EA3",
  "MTGCN" = "#8B4513",
  "SVM" = "#FF7F00"
)

# 画图
p <- ggplot(df, aes(x = edges_100k, y = avg_epoch_time, color = methods)) +
  geom_point(size = 3) +
  scale_color_manual(values = method_colors) +
  facet_wrap(~nodes, scales = "free", labeller = labeller(nodes = function(x) paste0(x, " nodes"))) +
  labs(x = "Number of Edges (100k)", y = "Average Epoch Time (s)", color = "Methods") +
  theme_bw(base_size = 14) +
  theme(
    panel.grid = element_blank(),
    strip.text = element_text(size = 14, face = "bold"),
    legend.position = "right"
  )

# 保存图片
ggsave("epoch_time_plot.png", p, width = 10, height = 8)
