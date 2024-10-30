import kagglehub

# Download latest version
path = kagglehub.dataset_download("wanderdust/skin-lesion-analysis-toward-melanoma-detection")

print("Path to dataset files:", path)