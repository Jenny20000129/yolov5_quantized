with open('runs/train//success42_limit4_per10til60_800epoch_useOldWeights//quant.txt', 'r') as f1:
    lines = f1.readlines()[-57:]
sum_col1 = 0
sum_col2 = 0
count1 = 0
count2 = 0
for line in lines:
    item = line.strip().split()
    col1 = float(item[1])
    col2 = float(item[2])
    wt1 = float(item[5])
    wt2 = float(item[6])
    sum_col1 += col1 * wt1
    sum_col2 += col2 * wt2
    count1 += wt1
    count2 += wt2
avg_col1 = sum_col1 / count1
avg_col2 = sum_col2 / count2
with open('runs/train//success42_limit4_per10til60_800epoch_useOldWeights//quant_average_new.txt', 'w') as f2:
    f2.write(f'Average Weight bit:{avg_col1}\n')
    f2.write(f'Average Activation bit:{avg_col2}\n')
