remove_empty <- function(dir_name){
  imgs <- dir(dir_name)
  annotations <- read.csv("data_full_annotations.csv")
  not_in_annotations <- !(imgs %in% annotations$filename)
  if (any(not_in_annotations)) {
    file.remove(paste0(dir_name, "/", imgs[not_in_annotations]))
  }
  empty_imgs <- annotations %>% 
    dplyr::filter(class == "empty") %>% 
    pull(filename)
  in_empty <- imgs %in% empty_imgs
  file.remove(paste0(dir_name, "/", imgs[in_empty]))
}

