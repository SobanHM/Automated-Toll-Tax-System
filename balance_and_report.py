from dataset_balancer import DatasetBalancer

DATASET = r"C:/Users/soban/PycharmProjects/Smart-Toll-Tax-System/Merged_Dataset"
CLASSES = ["car", "truck", "van", "bus"]

db = DatasetBalancer(DATASET, CLASSES)

print("\n BEFORE BALANCING:")
db.print_report()

db.balance_dataset(output_multiplier=1.5)

print("\n AFTER BALANCING:")
db.print_report()
