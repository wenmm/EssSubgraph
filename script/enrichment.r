#BiocManager::install("org.Hs.eg.db")
#BiocManager::install("clusterProfiler")

library(org.Hs.eg.db)
library(clusterProfiler)

subgraph_string_predicted_essential_gene <- read.table("subgraph_string_predicted_essential_gene", quote="\"", comment.char="")

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