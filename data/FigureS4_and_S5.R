# 加载必要的库
library(ggplot2)
library(dplyr)
library(gridExtra)

# 读取 CSV 文件

#Figure S4
data <- read.csv("nodes_25000_edges_1000000_log_gpu_metrics_summary.csv")

#Figure S4
data <- read.csv("subgraph_features1500_simulation_network_nodes_25000_edges_1000000_log_gpu_metrics_summary.csv")

# 确保 batch_size 和其他列为数值型
data$batch_size <- as.numeric(data$batch_size)
data$gpu_used_reserved_avg <- as.numeric(data$gpu_used_reserved_avg)
data$epoch_time_avg <- as.numeric(data$epoch_time_avg)

data <- data[order(-data$batch_size), ]

# 获取唯一 size（假设有9个size）
sizes <- unique(data$size)
if (length(sizes) != 9) {
  warning(paste("Found", length(sizes), "sizes instead of 9. Proceeding with available data."))
}

# 存储每个 size 的图表
plots <- list()

desired_order <- c(
  "30_25_10", "60_50_20", "120_100_40",
  "240_200_80", "480_400_160", "960_800_320",
  "1920_1600_640", "3840_3200_1280", "7680_6400_2560"
)

# 为每个 size 绘制图表
for (s in sizes) {
  # 过滤当前 size 的数据
  sub_data <- data[data$size == s, ]
  
  # 格式化标题：下划线改成逗号，加方括号
  title_str <- paste0("Size: [", gsub("_", ",", s), "]")
  
  # 创建图表
  p <- ggplot(sub_data, aes(x = factor(batch_size))) +
    # 左 y 轴：gpu_used_reserved_avg (转换为 MB)
    geom_line(aes(y = gpu_used_reserved_avg * 10 , color = "GPU Usage"), size = 1) +
    geom_point(aes(y = gpu_used_reserved_avg * 10 , color = "GPU Usage"), size = 2) +
    # 右 y 轴：epoch_time_avg
    geom_line(aes(y = epoch_time_avg , color = "Epoch Time"), size = 1, linetype = "dashed") +
    geom_point(aes(y = epoch_time_avg, color = "Epoch Time"), size = 2) +
    # 设置双 y 轴
    scale_y_continuous(name = "GPU Usage (GB/10)", 
                       sec.axis = sec_axis(~ . * 1, name = "Epoch Time (s)")) +
    # x 轴标签
    scale_x_discrete(name = "Batch Size") +
    # 颜色和图例
    scale_color_manual(values = c("GPU Usage" = "#1f77b4", "Epoch Time" = "#ff7f0e"),
                       name = "Metric") +
    # 主题：白色背景，仅四根线
    theme_classic() +
    theme(axis.text.x = element_text(angle = 45, hjust = 1),
          legend.position = "top",
          axis.title.y = element_text(color = "#1f77b4"),
          axis.title.y.right = element_text(color = "#ff7f0e"),
          plot.title = element_text(size = 12, face = "bold")) +   # 调整标题大小 + 加粗
    ggtitle(title_str)
  
  # 存储图表
  plots[[s]] <- p
}

ordered_plots <- list()
for (s in desired_order) {
  if (s %in% sizes) {
    ordered_plots[[s]] <- plots[[s]]
  } else {
    # 若数据中无该 size，添加空图（标题也改成方括号形式 + 小标题 + 加粗）
    ordered_plots[[s]] <- ggplot() + theme_void() + 
      ggtitle(paste0("Size: [", gsub("_", ",", s), "] (No Data)")) +
      theme(plot.title = element_text(size = 12, face = "bold"))
  }
}

# 确保有9个图
while (length(ordered_plots) < 9) {
  ordered_plots[[paste0("placeholder_", length(ordered_plots) + 1)]] <- ggplot() + theme_void()
}

# 将所有子图按 3x3 网格拼接
if (length(ordered_plots) >= 9) {
  combined_plot <- do.call(grid.arrange, c(ordered_plots[1:9], ncol = 3))  # 3x3 排列
  ggsave("size_analysis_combined_3x3.png", plot = combined_plot, 
         width = 18, height = 18, dpi = 300)
  print("Combined plot saved as 'size_analysis_combined_3x3.png'")
} else {
  print("Not enough plots generated. Check data.")
}
