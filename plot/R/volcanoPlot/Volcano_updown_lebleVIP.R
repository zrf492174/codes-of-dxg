VIP<- c("SIX1")
library(tidyverse)
library(ggthemes)
library(ggrepel)
data <- read.table("volcano-Asp-vs-LPS.csv", sep = ",", header = T)
head(DEPdata)

mydata1 <- DEPdata %>%
  mutate(expression = case_when(PG_vs_CON_ratio>=0.263034406&-log10(PG_vs_CON_p.val)>=1.301 ~ 'Significant-up',  
                                PG_vs_CON_ratio<=-0.263034406&-log10(PG_vs_CON_p.val)>=1.301 ~ 'Significant-down',
                                PG_vs_CON_ratio>0&-log10(PG_vs_CON_p.val)>=1.301 ~ 'Up',
                                PG_vs_CON_ratio<0&-log10(PG_vs_CON_p.val)>=1.301 ~ 'Down', 
                                TRUE ~ "Non-significant")) 

DEP_resul <-bind_rows(
  mydata1 %>%
    filter(expression == 'Significant-up'),
  mydata1 %>%
    filter(expression == 'Significant-down')
)
write.csv(DEP_resul,"DEP_result.csv") 

# 统计符合条件的数据点数量
significant_up_count <- sum(mydata1$expression == "Significant-up")
significant_down_count <- sum(mydata1$expression == "Significant-down")

mydata1 <- DEPdata %>%
  mutate(expression = case_when(name%in%VIP ~ 'VIP',
                                PG_vs_CON_ratio>=0.263034406&-log10(PG_vs_CON_p.val)>=1.301 ~ 'Significant-up',  
                                PG_vs_CON_ratio<=-0.263034406&-log10(PG_vs_CON_p.val)>=1.301 ~ 'Significant-down',
                                PG_vs_CON_ratio>0&-log10(PG_vs_CON_p.val)>=1.301 ~ 'Up',
                                PG_vs_CON_ratio<0&-log10(PG_vs_CON_p.val)>=1.301 ~ 'Down', 
                                TRUE ~ "Non-significant")) 


VIP <-bind_rows(mydata1 %>% filter(expression == 'VIP'))
head(VIP)

top_20_UP <-bind_rows(
  mydata1 %>%
    filter(expression == 'Significant-up') %>%
    arrange(desc(-log10(PG_vs_CON_p.val))) %>%
    head(19))
top_20_Down <-bind_rows(
  mydata1 %>%
    filter(expression == 'Significant-down') %>%
    arrange(desc(-log10(PG_vs_CON_p.val))) %>%
    head(20)
)
head(top_20)

volc_plot1 <- ggplot(mydata1, aes(PG_vs_CON_ratio, -log10(PG_vs_CON_p.val))) +
  geom_point(data = mydata1[mydata1$expression != 'VIP',],alpha = 0.4, size = 2,  aes(color = expression)) +
  geom_point(data = mydata1[mydata1$expression == 'VIP',], alpha = 0.6, size = 3, color = "purple") +
  scale_color_manual(values = c("Up" = "#FFCCCC", "Down" = "#99CCFF", 
                                "Significant-up" = "red", "Significant-down" = "#6699CC",  
                                "Non-significant" = "grey",'VIP'="purple")) +
  geom_hline(yintercept = 1.301, linetype = "dashed", color = "black") +
  geom_vline(xintercept = c(-0.263034406,0.263034406), linetype = "dashed", color = "black") +
  annotate("text", x = Inf, y = 1.1, hjust = 1, label = "p-value = 0.05", color = "black") +
  annotate("text", x = 6.5, y = 6, hjust = 1, vjust = 1, 
           label = paste("Sig:", significant_up_count, "\np < 0.05\nFC >1.2 \nUp"), 
           color = "black",
           size = 3.5) + 
  annotate("text", x = -6.5, y = 6, hjust = 0, vjust = 1, 
           label = paste("Sig:", significant_down_count, "\np < 0.05\nFC < 0.833\nDown"), 
           color = "black",
           size = 3.5) +
  xlab(expression("log"[2]*" fold change")) +
  ylab(expression("-log"[10]*" p-value")) +
  theme_bw(base_size = 12, base_family = "Times") +
  geom_text_repel(data = VIP,
                  aes(PG_vs_CON_ratio, -log10(PG_vs_CON_p.val), label = name),
                  size = 3) +  # 设置 box.padding 为 0 或其他小的值
  geom_text_repel(data = top_20_Down,
                  aes(PG_vs_CON_ratio, -log10(PG_vs_CON_p.val), label = name),
                  size = 3,nudge_x=-2,direction="y", hjust=0) +  # 设置 box.padding 为 0 或其他小的值 
  geom_text_repel(data = top_20_UP,
                  aes(PG_vs_CON_ratio, -log10(PG_vs_CON_p.val), label = name),
                  size = 3,nudge_x=1,direction="y", hjust=0) +  # 设置 box.padding 为 0 或其他小的值 
  theme(legend.position="none",
        panel.grid=element_blank(),
        legend.title = element_blank())+
  coord_cartesian(xlim = c(-6.5, 6.5), clip = "on")
volc_plot1
ggsave("volc_top20_VIP_20.pdf",plot = volc_plot1)
ggsave("volc_top20_VIP_20.png",plot = volc_plot1)
ggsave("volc_top20_VIP_20.svg",plot = volc_plot1)
