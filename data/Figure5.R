library(org.Hs.eg.db)
BiocManager::install("org.Hs.eg.db")

subgraph_string_predicted_essential_gene <- read.table("esssubgraph_human_pc50_string.predict_essential_gname_cutoff0.1954", quote="\"", comment.char="")

gene.df <- bitr(subgraph_string_predicted_essential_gene$V1, fromType = "SYMBOL",
                toType = c("ENTREZID"),
                OrgDb = org.Hs.eg.db)

data(geneList)


ego <- enrichGO(gene          = gene.df$ENTREZID,
                OrgDb         = org.Hs.eg.db,
                ont           = "BP",
                pAdjustMethod = "BH",
                pvalueCutoff  = 0.01,
                qvalueCutoff  = 0.05)
barplot(ego, showCategory=10)
write.csv(data.frame(ego), "subgraph_essential_BP.csv")



essential_gene <- read.table("Essential_genes", quote="\"", comment.char="")

gene.df <- bitr(essential_gene$V1, fromType = "SYMBOL",
                toType = c("ENTREZID"),
                OrgDb = org.Hs.eg.db)


ego <- enrichGO(gene          = gene.df$ENTREZID,
                OrgDb         = org.Hs.eg.db,
                ont           = "BP",
                pAdjustMethod = "BH",
                pvalueCutoff  = 0.01,
                qvalueCutoff  = 0.05)
barplot(ego, showCategory=10)
write.csv(data.frame(ego), "essential_BP.csv")



essential_BP <- read.csv("top_essential_BP.csv")
subgraph_essential_BP <- read.csv("top_subgraph_essential_BP.csv")

essential_BP <- essential_BP[order(-essential_BP$Count), ]  
essential_BP$index <- seq_len(nrow(essential_BP)) 

subgraph_essential_BP <- subgraph_essential_BP[order(-subgraph_essential_BP$Count), ]  
subgraph_essential_BP$index <- seq_len(nrow(subgraph_essential_BP)) 



custom_theme <- theme(
  panel.background = element_blank(),
  panel.border = element_rect(color = "black", fill = NA, size = 1.2),
  axis.line = element_line(size = 1.2, color = "black"),
  axis.title = element_text(size = 18, face = "bold"),
  axis.text = element_text(size = 16, color = "black"),  # 坐标轴数字大且黑色
  axis.ticks = element_line(size = 1.2, color = "black"), # 粗刻度线
  axis.ticks.length = unit(6, "pt")                       # 长刻度线
)


top10_1 <- head(essential_BP, 10)
p1 <- ggplot(top10_1, aes(x = factor(index, levels = 1:10), y = Count)) +
  geom_bar(stat = "identity", width = 0.8, fill = "#51c4d3") +
  labs(title = "Ground Truth", x = "Biological Process", y = "Count") +
  custom_theme

top10 <- head(subgraph_essential_BP, 10)
p2 <- ggplot(top10, aes(x = factor(index, levels = 1:10), y = Count)) +
  geom_bar(stat = "identity", width = 0.8, fill = "#9589C1") +
  labs(title = "Predicted Essential Genes", x = "Biological Process", y = "Count") +
  custom_theme

ggarrange(p1, p2, 
          labels = c("A", "B"))


# Figure S6B
library(VennDiagram)
venn.plot <- draw.pairwise.venn(
  area1 = 2299,    # 集合A大小
  area2 = 3521,    # 集合B大小
  cross.area = 1881, # A和B的交集大小
  category = c("Ground Truth","Predicted Essential Genes"),
  fill = c("#51c4d3","#9589C1"),
  lty = "blank",
  cex = 2,
  cat.cex = 2,
  cat.pos = c(-20, 20),
  cat.dist = 0.05
)

# Figure 5C
library(VennDiagram)
venn.plot <- draw.pairwise.venn(
  area1 = 888,    # 集合A大小
  area2 = 601 ,    # 集合B大小
  cross.area = 589, # A和B的交集大小








  category = c("Predicted Essential Genes","Ground Truth"),
  fill = c("#9589C1","#51c4d3"),
  lty = "blank",
  cex = 2,
  cat.cex = 2,
  cat.pos = c(-20, 20),
  cat.dist = 0.05
)

dev.off()


