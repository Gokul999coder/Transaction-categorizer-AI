# -----------------------------

# &nbsp;	README FIRST

# -----------------------------
used python version : 3.11.4

**COMMANDS TO:**

---

	**To create a virtual environment:
		.\\venv\\Scripts\\activate** 


**Then run the following commands:**



* **Train the model:**  python -m src.model.train_baseline --data data/synthetic_transactions.csv --models models/




* **Predict the text description:** python -m src.model.predict --text "your text"   



* **Correct the wrong prediction(feedback):** python src/model/feedback_handler.py --merchant "merchant name" --description "text decription predicted wrong" --predicted "Enter the wrong prediction" --correct "Enter the correct one"
  
* **Retrain the model with feedback:** python -m src.model.retrain_from_feedback --data data/synthetic_transactions.csv --feedback data/feedback.csv --models models/ 
  
