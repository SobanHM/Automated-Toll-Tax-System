from dataset_balancer import DatasetBalancer

DATASET = r"C:/Users/soban/PycharmProjects/Smart-Toll-Tax-System/Merged_Dataset"

# classes read from 'data.yaml' inside the Merged_Dataset
db = DatasetBalancer(DATASET)

print("\n Before Balancing:")
db.print_report()

db.balance_dataset(output_multiplier=1.5)
print("\n After Balancing:")
db.print_report()