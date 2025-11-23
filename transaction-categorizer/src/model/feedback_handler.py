
import argparse, csv, os
FEEDBACK_PATH = 'data/feedback.csv'

def init_feedback():
    os.makedirs(os.path.dirname(FEEDBACK_PATH), exist_ok=True)
    if not os.path.exists(FEEDBACK_PATH):
        with open(FEEDBACK_PATH,'w',newline='',encoding='utf8') as f:
            writer = csv.DictWriter(f, fieldnames=['merchant','description','predicted','correct'])
            writer.writeheader()

def save_feedback(merchant, description, predicted, correct):
    init_feedback()
    with open(FEEDBACK_PATH,'a',newline='',encoding='utf8') as f:
        writer = csv.DictWriter(f, fieldnames=['merchant','description','predicted','correct'])
        writer.writerow({'merchant':merchant,'description':description,'predicted':predicted,'correct':correct})
    print("Saved feedback")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--merchant', required=True)
    parser.add_argument('--description', required=True)
    parser.add_argument('--predicted', required=True)
    parser.add_argument('--correct', required=True)
    args = parser.parse_args()
    save_feedback(args.merchant, args.description, args.predicted, args.correct)
