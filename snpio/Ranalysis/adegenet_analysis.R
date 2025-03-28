library("adegenet")
library("vcfR")
library("poppr")
library("hierfstat")
library("ggplot2")
library("pheatmap")


# Load the data
vcf <- read.vcfR("../example_data/vcf_files/nremover_test.vcf")

# Convert the vcfR object to a genlight object
gen <- vcfR2genind(vcf)
popmap <- read.csv("../example_data/popmaps/nremover_test.popmap", header=FALSE, sep=",", stringsAsFactors = TRUE)

gen@pop <- popmap$V2
genpop <- genind2genpop(gen)

# Run the adegenet analysis
nei <- dist.genpop(genpop, method = "pairwise")

boot.ppfst(dat=gen, nboot = 100)
wcfst <- pairwise.WCfst(gen)

hf <- genind2hierfstat(gen)

nei87 <- hierfstat::genet.dist(hf, method="Nei87")

nei87df <- read.csv("../Ranalysis/nei87_genet_dist_hierfstat_results.csv")

pheatmap(nei87, display_numbers = TRUE)
pheatmap(wcfst, display_numbers = TRUE)
