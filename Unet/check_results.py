import re

output_file="/media/talhassid/Elements/haimTal/stage2_solution.csv"
results_file ='/media/talhassid/Elements/haimTal/run/test_results.csv'


import csv

labels = []
results = []
with open(output_file, 'r') as f:
    reader = csv.reader(f)
    for row in reader:
        labels.append(row)


with open(results_file, 'r') as f:
    reader = csv.reader(f)
    for row in reader:
        results.append(row)

res_dict = {}
for patient in results:
    patient[0] = re.sub(r'\s*(\S+)\s*',r'\1',patient[0])
    if patient[0] != 'id':
        patient[1] = int(float(re.sub(r'\s*(\S+)\s*',r'\1',patient[1])))
        res_dict[patient[0]] = patient[1]

labels_dict = {}
for patient in labels:
    patient[0] = re.sub(r'\s*(\S+)\s*',r'\1',patient[0])
    if patient[0] != 'id':
        patient[1] = int(float(re.sub(r'\s*(\S+)\s*',r'\1',patient[1])))
        labels_dict[patient[0]] = patient[1]

count = 0
for id, classify in res_dict.items():
    if id in labels_dict.keys() and classify == labels_dict[id]:
        count += 1

print(count*100/(len(res_dict.keys())))
print("finish")
