from transformers import AutoTokenizer, AutoModel
import torch
import pickle
import pandas as pd

class Classifier:
    """
    A classifier class that uses a pre-trained language model for text encoding
    and a pre-trained classifier model for prediction.

    Attributes:
        tokenizer: A tokenizer for the language model.
        device (torch.device): The device (CPU or GPU) where the model is loaded.
        model: The pre-trained language model.
        rf_pipeline: The pre-trained classifier model loaded from a joblib file.
    """
    def __init__(self, model_path):
        """
        Initializes the Classifier with a pre-trained language model and a classifier model.

        Args:
            model_path (str): The file path to the pre-trained classifier model in pickle format.
        """
        # Load language model
        model_name = "almanach/camembertav2-base"
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = AutoModel.from_pretrained(model_name).to(self.device)
        print(f"- Loaded '{model_name}' model on device: {self.device}")

        # Load classifier model
        with open(model_path, 'rb') as file:
            self.classifier = pickle.load(file)
        print(f"- Loaded '{model_path.split('/')[-1]}' model on device: {self.device}")

    # Embedder function
    def encoding(self, text):
        """
        Encodes the input text into a normalized embedding using the language model.

        Args:
            text (str): The input text to be encoded.

        Returns:
            list: A list containing the normalized embedding of the input text.
        """
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
        """
        Predicts the label and scores for the input text using the classifier model.

        Args:
            text (str): The input text to be classified.

        Returns:
            dict: A dictionary containing the input text, predicted label, and scores for each class.
        """
        x = pd.DataFrame(self.encoding([text])).add_prefix('emb_')
        y_label = self.classifier.predict(x)[0]
        y_prob = self.classifier.predict_proba(x)[0].tolist()
        y_prob = [round(i, 4) for i in y_prob]
        return {'text': text, 'label': y_label, 
                'scores': {'cfa': y_prob[0], 'entreprise': y_prob[1], 'entreprise_cfa': y_prob[2]}}
