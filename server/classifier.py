from transformers import AutoTokenizer, AutoModel
import torch
import pickle
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from datasets import Dataset, load_dataset
import pickle as pickle
from pathlib import Path
from tempfile import mkdtemp, mkstemp
from huggingface_hub import hf_hub_download, ModelCard, ModelCardData, EvalResult
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix, accuracy_score, f1_score
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from huggingface_hub import HfApi
from tqdm import tqdm
import logging
import numpy as np
import requests
tqdm.pandas()
logger = logging.getLogger(__name__)

class Classifier:
    """
    A classifier class that uses a pre-trained language model for text encoding
    and a trained classifier model for prediction.

    Attributes:
        tokenizer: A tokenizer for the language model.
        device (torch.device): The device (CPU or GPU) where the model is loaded.
        llm: The pre-trained language model.
        version (str): The version of the model.
        model_file (str): The filename of the model.
        repo_id (str): The repository ID on HuggingFace Hub.
        token (str): The HuggingFace token.
        classifier: The trained classifier model.
        dataset: The dataset used for training.
    """
    def __init__(self, version="2025-08-06", 
                 lang_model="almanach/camembertav2-base",
                 token=""):
        """
        Initializes the Trainer with a pre-trained language model.

        Args:
            version (str): The version of the model.
            lang_model (str): The huggingface path of the pre-trained language model.
        """
        # Load language model
        self.tokenizer = AutoTokenizer.from_pretrained(lang_model)
        self.device = torch.device(f"cuda" if torch.cuda.is_available() else "cpu")
        self.llm = AutoModel.from_pretrained(lang_model).to(self.device)
        self.version = version
        self.model_file = f"svc-clf-offer-{version}.pkl"
        self.repo_id = f"la-bonne-alternance/{version}"
        self.token = token
        self.classifier = None
        self.dataset = None

        # Set model to evaluation mode for faster inference
        self.llm.eval()

        # Enable optimizations if available
        if hasattr(torch, 'set_float32_matmul_precision'):
            torch.set_float32_matmul_precision('high')
        device = f"cuda - {torch.cuda.get_device_name(0)}" if torch.cuda.is_available() else "cpu"
        logger.info(f"Loaded '{lang_model}' model on device: {device}")

    # Embedder function
    def encoding(self, text):
        """
        Encodes the input text into a normalized embedding using the language model.

        Args:
            text (str or list): The input text(s) to be encoded.

        Returns:
            list: A list containing the normalized embedding(s) of the input text(s).
        """
        # Handle both single text and batch of texts
        if isinstance(text, str):
            texts = [text]
        else:
            texts = text

        inputs = self.tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=512).to(self.device)

        # Generate embeddings
        with torch.no_grad():
            outputs = self.llm(**inputs)

        # Use the embeddings from the last hidden state
        embeddings = outputs.last_hidden_state

        # Normalize the embeddings
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
        embeddings = self.encoding([text])
        x = pd.DataFrame([embeddings[0]])
        y_label = self.classifier.predict(x)[0]
        y_prob = self.classifier.predict_proba(x)[0].tolist()
        y_prob = [round(i, 4) for i in y_prob]
        return {'model': self.version,
                'text': text, 'label': y_label, 
                'scores': {'cfa': y_prob[0], 'entreprise': y_prob[1], 'entreprise_cfa': y_prob[2]}}

    # Batch classifier function
    def score_batch(self, texts):
        """
        Predicts the labels and scores for multiple texts using batch processing.

        Args:
            texts (list): List of input texts to be classified.

        Returns:
            list: List of dictionaries, each containing text, predicted label, and scores.
        """
        # Remove chunking for GPU - process all at once for better GPU utilization
        # GPU benefits from larger batches, not smaller chunks
        
        # Generate embeddings for all texts in one batch
        embeddings = self.encoding(texts)
        
        # Create DataFrame with embeddings
        x = pd.DataFrame(embeddings)
        
        # Batch predict labels and probabilities
        y_labels = self.classifier.predict(x)
        y_probs = self.classifier.predict_proba(x)
        
        # Format results
        results = []
        for text, label, probs in zip(texts, y_labels, y_probs):
            prob_rounded = [round(p, 4) for p in probs.tolist()]
            results.append({
                'model': self.version,
                'text': text,
                'label': label,
                'scores': {'cfa': prob_rounded[0], 'entreprise': prob_rounded[1], 'entreprise_cfa': prob_rounded[2]}
            })
        
        return results

    def evaluate(self, texts, labels):
        # Generate embeddings for all texts in one batch
        embeddings = self.encoding(texts)
        
        # Create DataFrame with embeddings
        x = pd.DataFrame(embeddings)
        
        # Batch predict labels
        y_preds = self.classifier.predict(x)

        # Compute scores
        accuracy = accuracy_score(labels, y_preds)
        f1 = f1_score(labels, y_preds, average="weighted")
        return {"preds": y_preds.tolist(), "accuracy": round(accuracy,4), "f1": round(f1,4)}

    # Dataset create and encode function from payload
    def create_dataset_local(self, version, ids, texts, labels, batch_size=20):
        """
        Create a pandas dataset from the given ids, texts, and labels.
        Args:
            ids (list): List of ids.
            texts (list): List of texts.
            labels (list): List of labels.

        Returns:
            dataset: The created dataset with embeddings
        """
        # Create dataset
        dataset = pd.DataFrame({'_id': ids, 'text': texts, 'label': labels})

        # Batch encoding texts
        embeddings = []
        for i in tqdm(range(0, len(dataset), batch_size), desc=f"- Creating dataset '{self.version}'"):
            batch = dataset['text'][i:i+batch_size]
            batch_embeddings = self.encoding(list(batch))
            embeddings.extend(batch_embeddings)
        dataset['embeddings'] = embeddings

        # Update dataset and model version
        self.dataset = dataset
        self.version = version
        logger.info(f"Dataset '{self.version}' created: {dataset.shape}")
        return dataset

    # Dataset create and encode function from API
    def create_dataset_online(self, version, endpoint="https://labonnealternance.apprentissage.beta.gouv.fr/api/classification", batch_size=20):
        """
        Create a pandas dataset from the given API endpoint.
        Args:
            version (str): Version of the dataset
            endpoint (str) : API endpoint

        Returns:
            dataset: The created dataset with embeddings
        """
        # Make a get request
        response = requests.get(endpoint)

        # Test if response = 200
        if response.status_code == 200:
            # Get the list response
            data = response.json()
        else:
            logger.info(f"Request failed with status code: {response.status_code}")
            return None

        if len(data) == 0:
            logger.info("Dataset is empty!")
            return None

        # Create dataset
        dataset = pd.DataFrame(data)

        dataset['text'] = ['']*len(dataset)
        
        for col in ['workplace_name', 'workplace_description', 'offer_title', 'offer_description']:
            dataset['text'] += dataset[col].fillna('') + '\n'

        dataset = dataset[['_id', 'text', 'label']].dropna().reset_index(drop=True)

        # Batch encoding texts
        embeddings = []
        for i in tqdm(range(0, len(dataset), batch_size), desc=f"- Creating dataset '{self.version}'"):
            batch = dataset['text'][i:i+batch_size]
            batch_embeddings = self.encoding(list(batch))
            embeddings.extend(batch_embeddings)
        dataset['embeddings'] = embeddings

        # Update dataset and model version
        self.dataset = dataset
        self.version = version
        logger.info(f"Dataset '{self.version}' created: {dataset.shape}")
        return dataset

    # Dataset save function
    def save_dataset(self):
        """
        Upload a pandas dataset to the HuggingFace Hub.

        Returns:
            url: The URL of the saved dataset.
        """

        # Save dataset to HF
        hf_dataset = Dataset.from_pandas(self.dataset)
        hf_dataset.push_to_hub(self.repo_id, private=True, token=self.token)
        url = f"https://huggingface.co/datasets/{self.repo_id}"
        logger.info(f"Dataset exported to: {url}.")
        return url

    # Dataset loader function
    def load_dataset(self, split="all"):
        """
        Load a dataset from the HuggingFace Hub.

        Args:
            split (str, optional): The split of the dataset to load. Defaults to "all".

        Returns:
            dataset: The loaded dataset.
        """
        self.dataset = load_dataset(self.repo_id, token=self.token, split=split).to_pandas().reset_index(drop=True)
        logger.info(f"Dataset loaded from https://huggingface.co/datasets/{self.repo_id}: {self.dataset.shape}")
        return self.dataset

    # Classifier trainer function
    def train_model(self):
        """
        Train a SVC classifier on the given dataset.

        Returns:
            classifier: Trained SVC model.
            train_score: Training score of the model.
            test_score: Testing score of the model.
        """
        # Create training dataset
        label_df = self.dataset['label']
        feat_df = self.dataset['embeddings'].apply(pd.Series)
        X_train, X_test, y_train, y_test = train_test_split(feat_df, label_df, test_size=0.2, random_state=42, shuffle=True, stratify=label_df)

        # PCA optimization
        # print("- PCA features optimization...")
        pca = PCA()
        sc = StandardScaler()
        X_train_std = sc.fit_transform(X_train)
        X_pca = pca.fit_transform(X_train_std)

        # Find optimal PCA features
        threshold = 0.9999
        features = 0
        v = 0
        exp_var_pca = pca.explained_variance_ratio_
        cum_sum_eigenvalues = np.cumsum(exp_var_pca)
        while v < threshold:
            v = cum_sum_eigenvalues[features]
            features+=1

        # Pipeline configuration
        # print("- Configure training pipeline...")
        numeric_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler()),
            ('pca', PCA(n_components=features))
            ])

        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, feat_df.columns),
            ],
            verbose_feature_names_out=False)

        # SVM classifier
        clf = SVC(random_state=42, kernel='rbf', probability=True)
        classifier = make_pipeline(preprocessor, clf)
        classifier.fit(X_train, y_train)

        # Evaluate model
        train_score = classifier.score(X_train, y_train)
        test_score = classifier.score(X_test, y_test)
        logger.info(f"SVM classifier trained on {features} PCA features: Train={round(train_score,4)} / Test={round(test_score,4)}")        
        
        # Update classifier
        self.classifier = classifier
        logger.info(f"SVM classifier {self.version} updated.")        
        return (classifier, train_score, test_score)
    
    def save_model(self):
        """
        Save a classifier model to the HuggingFace Hub.

        Returns:
            url: The URL of the saved model.
        """
        logger.info(f"Save model locally...")
        local_repo = mkdtemp(prefix="lba-")
        with open(Path(local_repo) / self.model_file, mode="bw") as f:
            pickle.dump(self.classifier, file=f)

        """
        # Create model card
        logger.info(f"- Create model card...")
        card_data = ModelCardData(
            language='fr',
            license='mit',
            library_name='la-bonne-alternance/2025-08-06',
            tags=['text-classification', 'camembert'],
            datasets=['la-bonne-alternance/2025-08-06'],
            metrics=['f1-score'],
        )
        card = ModelCard.from_template(
            card_data,
            model_description='This model does x + y...'
        )

        # Add metrics to model card
        print(f"- Add metrics to model card...")
        y_pred = model.predict(X_test)

        eval_descr = (
            "The model is evaluated on test data using accuracy and F1-score with "
            "weighted average."
        )
        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average="weighted")

        cm = confusion_matrix(y_test, y_pred, labels=model.classes_)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=model.classes_).plot()
        disp.figure_.savefig(Path(local_repo) / "confusion_matrix.png")

        # Save model card
        print(f"- Save model card...")
        card.save(Path(local_repo) / "README.md")
        """
        api = HfApi()

        # Delete previous repo with the same name
        try:
            logger.info(f"Deleting existing repo: {self.repo_id}")
            api.delete_repo(repo_id=self.repo_id, token=self.token)
        except:
            pass

        # Create repo
        logger.info(f"Creating repo: {self.repo_id}")
        api.create_repo(repo_id=self.repo_id, token=self.token, repo_type="model", private=True)

        # Upload model
        logger.info(f"Uploading model: {local_repo}")
        out = api.upload_folder(
            folder_path=local_repo,
            repo_id=self.repo_id,
            token=self.token,
            repo_type="model",
            commit_message=f"pushing model '{self.version}' SVC with camembert v2 embeddings",
        )
        url = f"https://huggingface.co/{self.repo_id}"
        logger.info(f"Model ready on: {url}")
        return url

    # Classifier loader function
    def load_model(self):
        """
        Load a classifier model from the HuggingFace Hub.

        Returns:
            model: The loaded classifier model.
        """
        # Download model
        logger.info(f"Downloading model: {self.repo_id}")
        model_dump = hf_hub_download(repo_id=self.repo_id, filename=self.model_file, token=self.token)
        # print(f"- Model downloaded to: {model_dump}")

        # Reload pickle model
        with open(model_dump, 'rb') as f:
            self.classifier = pickle.load(f)
        logger.info(f"Classifier model ready.")
