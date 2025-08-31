import re
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.pipeline import Pipeline
import joblib
import emoji
from collections import Counter
import pytesseract
from PIL import Image
import cv2
import os
from datetime import datetime, timedelta
import random

class TrustThreatModel:
    def __init__(self):
        # Initialize factors with weights
        self.factors = {
            'content_risk': {'weight': 0.35, 'value': 0},
            'source_credibility': {'weight': 0.3, 'value': 0},
            'bot_likeness': {'weight': 0.25, 'value': 0},
            'virality_factor': {'weight': 0.1, 'value': 0}
        }

    def calculate_content_risk(self, classification, confidence):
        """Calculate risk based on content classification"""
        risk_map = {
            "Neutral": 0.1,
            "Anti-India": 0.7,
            "Misinformation": 0.9
        }

        base_risk = risk_map.get(classification, 0.5)
        # Adjust based on confidence
        adjusted_risk = base_risk * (confidence / 100)
        return min(1.0, adjusted_risk * 1.2)  # Cap at 1.0

    def assess_source_credibility(self, source_info):
        """Assess source credibility based on available information"""
        # Simulate source assessment (in real implementation, use actual source data)
        credibility = 0.7  # Default

        # Check if verified account
        if source_info.get('verified', False):
            credibility += 0.2

        # Check account age (older accounts more credible)
        if 'account_age_days' in source_info:
            if source_info['account_age_days'] > 365:
                credibility += 0.1
            elif source_info['account_age_days'] < 30:
                credibility -= 0.2

        # Check follower count (moderate following is best)
        if 'followers_count' in source_info:
            if 1000 <= source_info['followers_count'] <= 100000:
                credibility += 0.1
            elif source_info['followers_count'] > 1000000:  # Very high could be suspicious
                credibility -= 0.1

        return max(0.1, min(1.0, credibility))  # Ensure between 0.1 and 1.0

    def assess_bot_likeness(self, user_behavior):
        """Assess likelihood that the account is a bot"""
        bot_score = 0.3  # Default

        # Check posting frequency
        if 'avg_posts_per_day' in user_behavior:
            if user_behavior['avg_posts_per_day'] > 50:  # Very high frequency
                bot_score += 0.4
            elif user_behavior['avg_posts_per_day'] > 20:  # High frequency
                bot_score += 0.2

        # Check content similarity
        if 'content_similarity_score' in user_behavior:
            bot_score += user_behavior['content_similarity_score'] * 0.3

        # Check follower-following ratio
        if 'follower_following_ratio' in user_behavior:
            if user_behavior['follower_following_ratio'] < 0.1:  # Many followings, few followers
                bot_score += 0.2

        return max(0.1, min(1.0, bot_score))  # Ensure between 0.1 and 1.0

    def calculate_virality_factor(self, engagement_metrics):
        """Calculate virality factor based on engagement"""
        virality = 0.3  # Default

        if 'retweet_count' in engagement_metrics:
            if engagement_metrics['retweet_count'] > 1000:
                virality += 0.4
            elif engagement_metrics['retweet_count'] > 100:
                virality += 0.2

        if 'like_count' in engagement_metrics:
            if engagement_metrics['like_count'] > 5000:
                virality += 0.3
            elif engagement_metrics['like_count'] > 500:
                virality += 0.15

        return max(0.1, min(1.0, virality))  # Ensure between 0.1 and 1.0

    def calculate_trust_score(self, classification, confidence, source_info, user_behavior, engagement_metrics):
        """Calculate overall trust score (0-100)"""
        # Calculate individual factors
        self.factors['content_risk']['value'] = self.calculate_content_risk(classification, confidence)
        self.factors['source_credibility']['value'] = self.assess_source_credibility(source_info)
        self.factors['bot_likeness']['value'] = self.assess_bot_likeness(user_behavior)
        self.factors['virality_factor']['value'] = self.calculate_virality_factor(engagement_metrics)

        # Calculate weighted trust score (invert some factors)
        trust_score = 0
        for factor, data in self.factors.items():
            if factor in ['content_risk', 'bot_likeness']:
                # Invert these factors (higher value means lower trust)
                trust_score += (1 - data['value']) * data['weight'] * 100
            else:
                trust_score += data['value'] * data['weight'] * 100

        return round(trust_score, 1)

    def determine_threat_level(self, trust_score, classification):
        """Determine threat level based on trust score and content classification"""
        if trust_score >= 70:
            base_level = "Low"
        elif trust_score >= 40:
            base_level = "Medium"
        else:
            base_level = "High"

        # Adjust based on content classification
        if classification == "Misinformation" and base_level != "High":
            # Upgrade threat level for misinformation
            if base_level == "Low":
                return "Medium"
            elif base_level == "Medium":
                return "High"
        elif classification == "Anti-India" and base_level == "Low":
            return "Medium"

        return base_level

    def generate_threat_assessment(self, trust_score, threat_level, classification):
        """Generate a detailed threat assessment report"""
        assessment = {
            "trust_score": trust_score,
            "threat_level": threat_level,
            "classification": classification,
            "factors": self.factors,
            "recommendation": self.generate_recommendation(trust_score, threat_level)
        }

        return assessment

    def generate_recommendation(self, trust_score, threat_level):
        """Generate recommendation based on trust score and threat level"""
        if threat_level == "High":
            return "Immediate review required. Consider content removal and account suspension."
        elif threat_level == "Medium":
            return "Close monitoring recommended. Flag for further analysis."
        else:
            return "Routine monitoring. Low priority action."

    def visualize_trust_gauge(self, trust_score):
        """Generate a visual representation of trust score (text-based)"""
        # Create a simple text-based gauge
        gauge_length = 20
        filled = int(trust_score / 100 * gauge_length)
        gauge = "[" + "=" * filled + " " * (gauge_length - filled) + "]"

        return f"Trust Score: {trust_score}/100 {gauge}"

class EnhancedTextClassifier:
    def __init__(self, n_features=10000):
        self.n_features = n_features
        self.pipeline = self._build_pipeline()
        self.trust_threat_model = TrustThreatModel()

    def _build_pipeline(self):
        """Build the SVM pipeline with efficient preprocessing"""
        vectorizer = TfidfVectorizer(
            max_features=self.n_features,
            ngram_range=(1, 2),
            sublinear_tf=True,
            min_df=2,
            max_df=0.95
        )

        svm = LinearSVC(
            C=1.0,
            class_weight='balanced',
            max_iter=1000,
            random_state=42
        )

        # Reduced cv to 2 as a temporary fix for insufficient data
        calibrated_svm = CalibratedClassifierCV(svm, cv=2)

        return Pipeline([
            ('vectorizer', vectorizer),
            ('classifier', calibrated_svm)
        ])

    def preprocess_text(self, text):
        """Efficient text preprocessing"""
        if not isinstance(text, str):
            return ""

        # Convert to lowercase
        text = text.lower()

        # Remove URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)

        # Remove user mentions and hashtags
        text = re.sub(r'@\w+', '', text)
        text = re.sub(r'#', '', text)

        # Remove emojis
        text = emoji.replace_emoji(text, replace='')

        # Remove punctuation and digits
        text = re.sub(r'[^a-z\s]', '', text)

        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()

        return text

    def preprocess_batch(self, texts):
        """Preprocess a batch of texts efficiently"""
        return [self.preprocess_text(text) for text in texts]

    def fit(self, X, y):
        """Train the model"""
        X_processed = self.preprocess_batch(X)
        self.pipeline.fit(X_processed, y)
        return self

    def predict(self, X):
        """Predict classes for input texts"""
        X_processed = self.preprocess_batch(X)
        return self.pipeline.predict(X_processed)

    def predict_proba(self, X):
        """Predict probabilities for input texts"""
        X_processed = self.preprocess_batch(X)
        return self.pipeline.predict_proba(X_processed)

    def extract_keywords(self, text, top_n=3):
        """Extract important keywords for explanation"""
        processed_text = self.preprocess_text(text)

        # Get the vectorizer and classifier from the pipeline
        vectorizer = self.pipeline.named_steps['vectorizer']
        classifier = self.pipeline.named_steps['classifier']

        # Transform the text to feature vector
        features = vectorizer.transform([processed_text])
        feature_names = vectorizer.get_feature_names_out()

        # Get the coefficients for the predicted class
        probas = self.predict_proba([text])[0]
        predicted_class_idx = np.argmax(probas)

        # Get coefficients for the predicted class
        # Check if the classifier has 'coef_' (for LinearSVC) or 'calibrated_estimators_' (for CalibratedClassifierCV)
        if hasattr(classifier, 'coef_'):
            coef = classifier.coef_[predicted_class_idx]
        elif hasattr(classifier, 'calibrated_estimators_'):
             # Access coef_ from the underlying estimator within CalibratedClassifierCV
             # Assuming the underlying estimator is a LinearSVC with a single coef_ array
             coef = classifier.calibrated_estimators_[0].estimator.coef_[predicted_class_idx]
        else:
            # Fallback or handle other classifier types if needed
            print("Warning: Could not access classifier coefficients.")
            return []


        # Get feature importance scores
        feature_scores = features.multiply(coef).toarray()[0]

        # Get indices of top features
        top_indices = np.argsort(feature_scores)[-top_n:][::-1]

        # Return the corresponding feature names
        keywords = [feature_names[i] for i in top_indices if feature_scores[i] > 0]

        # If no keywords found with positive coefficients, use most frequent words
        if not keywords:
            words = processed_text.split()
            word_counts = Counter(words)
            keywords = [word for word, count in word_counts.most_common(top_n)]

        return keywords

    def predict_with_trust_assessment(self, text, source_info=None, user_behavior=None, engagement_metrics=None):
        """Make prediction with trust and threat assessment"""
        # Get prediction and probability
        processed_text = self.preprocess_text(text)
        prediction = self.predict([text])[0]
        probability = np.max(self.predict_proba([text])[0]) * 100

        # Get explanatory keywords
        keywords = self.extract_keywords(text)

        # Map prediction to category name
        category_map = {0: "Neutral", 1: "Anti-India", 2: "Misinformation"}
        category = category_map.get(prediction, "Unknown")

        # Default values if not provided
        if source_info is None:
            source_info = {
                'verified': False,
                'account_age_days': random.randint(1, 1000),
                'followers_count': random.randint(100, 10000)
            }

        if user_behavior is None:
            user_behavior = {
                'avg_posts_per_day': random.randint(1, 20),
                'content_similarity_score': random.uniform(0.1, 0.7),
                'follower_following_ratio': random.uniform(0.1, 2.0)
            }

        if engagement_metrics is None:
            engagement_metrics = {
                'retweet_count': random.randint(0, 500),
                'like_count': random.randint(0, 1000)
            }

        # Calculate trust score and threat level
        trust_score = self.trust_threat_model.calculate_trust_score(
            category, probability, source_info, user_behavior, engagement_metrics
        )

        threat_level = self.trust_threat_model.determine_threat_level(trust_score, category)

        # Generate threat assessment
        threat_assessment = self.trust_threat_model.generate_threat_assessment(
            trust_score, threat_level, category
        )

        # Generate trust gauge visualization
        trust_gauge = self.trust_threat_model.visualize_trust_gauge(trust_score)

        return {
            'text': text,
            'processed_text': processed_text,
            'category': category,
            'confidence': f"{probability:.2f}%",
            'keywords': keywords,
            'explanation': f"Keywords: {', '.join(keywords)}",
            'trust_score': trust_score,
            'threat_level': threat_level,
            'trust_gauge': trust_gauge,
            'threat_assessment': threat_assessment
        }

    def save(self, path):
        """Save the model to disk"""
        joblib.dump(self, path)

    @staticmethod
    def load(path):
        """Load a model from disk"""
        return joblib.load(path)

# Example usage and demonstration
def demonstrate_trust_threat_model():
    """Demonstrate the trust and threat assessment model"""
    # Initialize the enhanced classifier
    classifier = EnhancedTextClassifier()

    # Train with sample data (in real scenario, use actual training data)
    training_data = [
        ("India is a great country with diverse culture", "Neutral"),
        ("The weather is nice today", "Neutral"), # Added another neutral example
        ("Indian government is suppressing human rights", "Anti-India"),
        ("India bans free speech and silences all journalists", "Misinformation"),
        ("According to constitution Sikh can demand for Khalistan", "Anti-India"),
        ("This is false information about the government policies", "Misinformation") # Added another misinformation example
    ]

    texts = [item[0] for item in training_data]
    labels = [item[1] for item in training_data]

    classifier.fit(texts, labels)

    # Test with the text from the image
    test_text = "According to constitution Sikh can demand for Khalistan than how it's wrong if u want to stop this issue try to change the act otherwise shut th furl up"

    # Simulate source information, user behavior, and engagement metrics
    source_info = {
        'verified': False,
        'account_age_days': 150,  # About 5 months
        'followers_count': 1200
    }

    user_behavior = {
        'avg_posts_per_day': 15,  # Quite active
        'content_similarity_score': 0.65,  # High content similarity (possible bot)
        'follower_following_ratio': 0.3  # Following more than followed
    }

    engagement_metrics = {
        'retweet_count': 45,
        'like_count': 120
    }

    # Get prediction with trust assessment
    result = classifier.predict_with_trust_assessment(
        test_text, source_info, user_behavior, engagement_metrics
    )

    # Display results
    print("="*60)
    print("SECURITY-GRADE THREAT ASSESSMENT")
    print("="*60)
    print(f"Text: {result['text']}")
    print(f"Classification: {result['category']}")
    print(f"Confidence: {result['confidence']}")
    print(f"Keywords: {', '.join(result['keywords'])}")
    print(f"\nTrust Assessment: {result['trust_gauge']}")
    print(f"Threat Level: {result['threat_level']}")

    print("\nDetailed Factor Analysis:")
    for factor, data in result['threat_assessment']['factors'].items():
        print(f"  {factor.replace('_', ' ').title()}: {data['value']:.2f} (weight: {data['weight']})")

    print(f"\nRecommendation: {result['threat_assessment']['recommendation']}")
    print("="*60)

if __name__ == "__main__":
    demonstrate_trust_threat_model()