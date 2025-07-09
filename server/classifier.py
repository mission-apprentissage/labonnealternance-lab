from transformers import AutoTokenizer, AutoModel
import torch
import joblib
import pandas as pd

class Classifier:
    def __init__(self, joblib_path):
        # Load language model
        model_name = "almanach/camembertav2-base"
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = AutoModel.from_pretrained(model_name).to(self.device)
        print(f"- Loaded '{model_name}' model on device: {self.device}")

        # Load classifier model
        self.rf_pipeline = joblib.load(joblib_path)
        print(f"- Loaded '{joblib_path.split('/')[-1]}' model on device: {self.device}")

    # Embedder function
    def encoding(self, text):
        inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=False).to(self.device)

        # Step 3: Generate embeddings
        with torch.no_grad():
            outputs = self.model(**inputs)

        # Use the embeddings from the last hidden state
        embeddings = outputs.last_hidden_state

        # Step 4: Normalize the embeddings
        # Average the embeddings across the sequence length dimension
        sentence_embedding = torch.mean(embeddings, dim=1)

        # Normalize to unit length
        normalized_embedding = sentence_embedding / torch.norm(sentence_embedding, p=2, dim=1, keepdim=True)

        return normalized_embedding.cpu().tolist()

    # Classifier function
    def score(self, text):
        x = pd.DataFrame(self.encoding([text])).add_prefix('emb_')
        y_label = self.rf_pipeline.predict(x)[0]
        y_prob = self.rf_pipeline.predict_proba(x)[0].tolist()
        y_prob = [round(i, 4) for i in y_prob]
        return {'text': text, 'label': y_label, 
                'scores': {'cfa': y_prob[0], 'entreprise': y_prob[1], 'entreprise_cfa': y_prob[2]}}
