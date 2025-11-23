
import csv, random, argparse
from datetime import datetime, timedelta
from pathlib import Path
from ..utils.helper import ensure_dir

MERCHANTS = [
    ("Starbucks","Coffee and snacks"),
    ("Amazon","Online shopping"),
    ("Flipkart","Online shopping"),
    ("Shell","Fuel station"),
    ("BigBasket","Grocery order"),
    ("Uber","Cab ride"),
    ("Netflix","Streaming subscription"),
    ("Dominos","Pizza restaurant"),
    ("Walmart","Grocery store"),
    ("HP Petrol","Fuel station"),
    ("Zomato","Food delivery"),
    ("Swiggy","Food delivery"),
    ("Paytm Mall","Online shopping")
]
LABEL_MAP = {m: l for m,l in [
    ("Starbucks","Food"),("Amazon","Shopping"),("Flipkart","Shopping"),
    ("Shell","Fuel"),("BigBasket","Groceries"),("Uber","Travel"),
    ("Netflix","Entertainment"),("Dominos","Food"),("Walmart","Groceries"),
    ("HP Petrol","Fuel"),("Zomato","Food"),("Swiggy","Food"),("Paytm Mall","Shopping")
]}

def random_date(start, end):
    return start + timedelta(seconds=random.randint(0, int((end-start).total_seconds())))

def generate(out='data/synthetic_transactions.csv', n=3000):
    ensure_dir(Path(out).parent)
    start = datetime.now() - timedelta(days=365)
    end = datetime.now()
    rows=[]
    for i in range(n):
        merch, desc = random.choice(MERCHANTS)
        amount = round(random.uniform(10,5000),2)
        ts = random_date(start,end).isoformat()
        variants = [
            f"{merch} {desc}",
            f"{merch} - {desc} Store #{random.randint(1,9999)}",
            f"{merch}.com purchase",
            f"{merch} PAYMENT",
            f"{merch} POS {random.randint(100,999)}",
            f"{merch} online order {random.randint(1,999)}"
        ]
        description = random.choice(variants)
        label = LABEL_MAP.get(merch,"Others")
        rows.append({
            "merchant": merch,
            "description": description,
            "amount": amount,
            "timestamp": ts,
            "label": label
        })
    header = ["merchant","description","amount","timestamp","label"]
    with open(out,'w',newline='',encoding='utf8') as f:
        writer = csv.DictWriter(f, fieldnames=header)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Generated {n} rows -> {out}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', default='data/synthetic_transactions.csv')
    parser.add_argument('--n', default=3000, type=int)
    args = parser.parse_args()
    generate(args.out,args.n)
