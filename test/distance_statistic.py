import pandas as pd

df = pd.read_csv('../out/net2_tanh_margin11/net2_tanh_margin11_merge_200000_test_dist.txt', header=-1, sep=' ')
print(df.describe(percentiles=[0.025,0.05,0.1,0.2,0.5,0.8,0.9,0.95,0.975]))
