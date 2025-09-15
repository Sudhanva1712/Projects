#%%
library(dplyr)
library(stringr)
tcr <- "/home/skshastry/Project/rosmap_brain_blood/data/TCR"
#listing files
#read csv
tcr_files <- list.files(tcr, pattern = "\\.csv$", full.names = TRUE)

#processing all files

tcr_data <- lapply(tcr_files, function(file) {
    df <- read.csv(file,stringsAsFactors = FALSE)
    sample_id <- tools::file_path_sans_ext(basename(file))
    sample_id_cleaned <- str_extract(sample_id,"DA-[0-9]+")
    df$sample_id <- sample_id_cleaned
    df
}) %>% bind_rows()

tcr_data <- tcr_data %>% filter(productive=="true")
tcr_valid_trb <- tcr_data %>%
    filter(chain=="TRB") %>%
    mutate(cdr3=paste(cdr3))%>%
    select(cdr3,chain,sample_id,barcode,contig_id)

write.csv(tcr_valid_trb, "tcr_valid_input_trb.csv", row.names = FALSE)

#%%