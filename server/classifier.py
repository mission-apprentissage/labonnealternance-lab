# from transformers import AutoTokenizer, AutoModel
from sentence_transformers import SentenceTransformer
import torch
import pickle
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import BaggingClassifier
from sklearn.linear_model import SGDClassifier
from datasets import Dataset, load_dataset
import joblib
from pathlib import Path
from tempfile import mkdtemp, mkstemp
from huggingface_hub import hf_hub_download, HfApi, ModelCard, ModelCardData, EvalResult
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix, accuracy_score, f1_score
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
import logging
import numpy as np
import requests
from config import MODEL_VERSION, LANG_MODEL, LBA_API_TOKEN
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
    def __init__(self, version=MODEL_VERSION,
                 lang_model=LANG_MODEL,
                 token=""):
        """
        Initializes the Trainer with a pre-trained language model.

        Args:
            version (str): The version of the model.
            lang_model (str): The huggingface path of the pre-trained language model.
        """
        # Load language model
        #self.tokenizer = AutoTokenizer.from_pretrained(lang_model)
        #self.device = torch.device(f"cuda" if torch.cuda.is_available() else "cpu")
        #self.llm = AutoModel.from_pretrained(lang_model).to(self.device)
        self.llm = SentenceTransformer(lang_model)
        self.version = version
        self.model_file = f"model.joblib"
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
    def encoding(self, texts, batch_size=32):
        """
        Encodes the texts using the language model.

        Args:
            texts (list(dict)): list of dict texts to encode with fields :
                          'workplace_name', 'workplace_description', 
                          'offer_title', 'offer_description'

        Returns:
            DataFrame: A dataframe containing the concat embedding(s) of the input text(s).
        """
        
        # Prepare dataset
        dataset = pd.DataFrame(texts)
        
        # Batch encoding texts
        features = pd.DataFrame()
        for col in ['workplace_name', 'workplace_description', 'offer_title', 'offer_description']:
            embeddings = []
            for i in tqdm(range(0, len(dataset), batch_size), desc=f"- Encoding {col}"):
                batch = dataset[col][i:i+batch_size]
                batch_embeddings = self.llm.encode(list(batch))
                embeddings.extend(batch_embeddings)
            embeddings = pd.DataFrame(embeddings).add_prefix(col+'_emb_')
            features = pd.concat([features, embeddings], axis=1)
        return features

    # Classifier function
    def score(self, texts):
        """
        Predicts the label and scores for the input text using the classifier model.

        Args:
            texts (list(dict)): List of dict texts to be classified.

        Returns:
            dict: A dictionary containing the input text, predicted label, and scores for each class.
        """
        features = self.encoding(texts)
        print(f"Features ready: {features.shape}")

        if len(features.columns) != 4*self.llm.get_sentence_embedding_dimension():
            logger.warning(f"Features size {len(features.columns)} incompatible on score function")
            return {'error': 'Feature size incompatible', 'status_code': 400}
        
        # Batch predict labels and probabilities
        y_labels = self.classifier.predict(features)
        y_probs = self.classifier.predict_proba(features)
        
        # Format results
        results = []
        for label, probs in zip(y_labels, y_probs):
            prob_rounded = [round(p, 4) for p in probs.tolist()]
            results.append({
                'model': self.version,
                'label': label,
                'scores': {'publish': prob_rounded[0], 'unpublish': prob_rounded[1]}
            })        
        return results

    def evaluate(self, texts, labels):
        # Generate embeddings for all texts in one batch
        features = self.encoding(texts)
        
        # Batch predict labels
        y_preds = self.classifier.predict(features)

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
    def create_dataset_online(self, version, endpoint="https://labonnealternance.apprentissage.beta.gouv.fr/api/classification", batch_size=32):
        """
        Create a pandas dataset from the given API endpoint.
        Args:
            version (str): Version of the dataset
            endpoint (str) : API endpoint

        Returns:
            dataset: The created dataset with embeddings
        """
        # Load data
        if not LBA_API_TOKEN:
            raise ValueError("LBA_API_TOKEN is required to load the online classification dataset")

        response = requests.get(
            endpoint,
            headers={"Authorization": LBA_API_TOKEN},
            timeout=(5, None),
        )
        response.raise_for_status()
        dataset = pd.DataFrame(response.json())
        dataset.fillna('', inplace=True)
        
        # Batch encoding texts
        features = self.encoding(dataset.to_dict(orient='records'))
        dataset = pd.concat([dataset, features], axis=1)

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
            classifier: Trained classifier model.
            train_score: Training score of the model.
            test_score: Testing score of the model.
        """
        # Extract features
        features = pd.DataFrame()
        labels = self.dataset['label']
        emb_cols = [col for col in self.dataset.columns if col.endswith("_emb")]
        for col in emb_cols:
            print(f"Adding {col} embeddings...")
            embeddings = self.dataset[col].progress_apply(pd.Series).add_prefix(col+'_')
            features = pd.concat([features, embeddings], axis=1)

        # Create training dataset
        X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42, shuffle=True, stratify=labels)

        # Pipeline configuration
        # print("- Configure training pipeline...")
        numeric_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler()),
            ])

        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, features.columns),
            ],
            verbose_feature_names_out=False)

        smote = SMOTE(random_state=42, sampling_strategy='minority')

        # Bagging SGD
        base_estimator = SGDClassifier(class_weight='balanced',
                                    loss='log_loss',
                                    penalty='l2',
                                    alpha=0.0001,
                                    random_state=42,
                                    n_jobs=-1)
        model = BaggingClassifier(
                    estimator=base_estimator, # Changed base_estimator to estimator
                    n_estimators=100,
                    max_samples=0.2,  # Use 20% of samples per estimator
                    n_jobs=-1,
                    verbose=0
                )
        # Classifier pipeline
        classifier = ImbPipeline(steps=[
            ('preprocessor', preprocessor),
            ('smote', smote),
            ('model', model)
        ])

        classifier.fit(X_train, y_train)

        # Evaluate model
        train_score = classifier.score(X_train, y_train)
        test_score = classifier.score(X_test, y_test)
        logger.info(f"Classifier trained on {len(features.columns)} features: Train={round(train_score,4)} / Test={round(test_score,4)}")        
        
        # Update classifier
        self.classifier = classifier
        logger.info(f"Classifier {self.version} updated.")        
        return (classifier, train_score, test_score)
    
    def save_model(self):
        """
        Save a classifier model to the HuggingFace Hub.

        Returns:
            url: The URL of the saved model.
        """
        logger.info(f"Save model locally...")
        local_repo = mkdtemp(prefix="tmp-")
        model_file = f"model.joblib"
        joblib.dump(self.classifier, Path(local_repo) / self.model_file)

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
            commit_message=f"pushing model '{self.version}'",
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

        # Reload joblib model
        self.classifier = joblib.load(model_dump)
        logger.info(f"Classifier model ready.")
