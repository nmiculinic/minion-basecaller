from mincall.bioinf_utils import error_rates_for_sam
import sys

df = error_rates_for_sam(sys.argv[1])
print(df.describe(percentiles=[]).transpose())
