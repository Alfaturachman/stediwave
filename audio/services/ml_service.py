"""
ML Service - Machine Learning operations for lung sound analysis
Handles model loading, feature extraction, and predictions
"""

import os
import logging
import numpy as np
import joblib
import librosa
import librosa.display
from collections import Counter
from scipy.signal import butter, filtfilt
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from django.conf import settings

logger = logging.getLogger(__name__)

# Model paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MODEL_XGB_PATH = os.path.join(BASE_DIR, 'ml_model', 'xgb_model_optimized_v6.pkl')
SCALER_PATH = os.path.join(BASE_DIR, 'ml_model', 'scaler_v6.pkl')
LABEL_ENCODER_PATH = os.path.join(BASE_DIR, 'ml_model', 'label_encoder_v6.pkl')

SAMPLE_RATE = 22050

# Disease to sound mapping
DISEASE_TO_SOUND_STRICT = {
    "ASMA": "Mengi",
    "BRONKITIS": "Krepitasi",
    "COPD": "Mengi",
    "GAGAL JANTUNG": "Crackles",
    "NORMAL": "Normal",
    "PNEUMONIA": "Bron"
}

# Global model instances
xgb_model = None
scaler = None
label_encoder = None


class MLService:
    """Service class for ML operations"""
    
    def __init__(self):
        self.xgb_model = None
        self.scaler = None
        self.label_encoder = None
    
    def load_models(self):
        """Load ML models from disk"""
        if self.xgb_model is not None:
            return True
        
        try:
            logger.info("="*60)
            logger.info("LOADING MODELS")
            logger.info("="*60)
            
            if not os.path.exists(MODEL_XGB_PATH):
                logger.error(f"Model not found: {MODEL_XGB_PATH}")
                return False
            
            self.xgb_model = joblib.load(MODEL_XGB_PATH)
            self.scaler = joblib.load(SCALER_PATH)
            self.label_encoder = joblib.load(LABEL_ENCODER_PATH)
            
            logger.info(f"✓ Model loaded successfully")
            logger.info(f"✓ Classes: {list(self.label_encoder.classes_)}")
            logger.info("="*60)
            
            return True
        
        except Exception as e:
            logger.error(f"✗ Error loading models: {e}", exc_info=True)
            return False
    
    def extract_features(self, y, sr):
        """
        Extract EXACTLY 121 features - matching trained model
        KEY: Do NOT use n_bands parameter for spectral_contrast!
        """
        features = []
        
        try:
            # 1. MFCC (60 features)
            mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
            mfccs_mean = np.mean(mfccs.T, axis=0)  # 20
            mfccs_std = np.std(mfccs.T, axis=0)    # 20
            mfccs_delta = librosa.feature.delta(mfccs)
            mfccs_delta_mean = np.mean(mfccs_delta.T, axis=0)  # 20
            features.extend([mfccs_mean, mfccs_std, mfccs_delta_mean])
            
            # 2. Chroma (24 features)
            chroma = librosa.feature.chroma_stft(y=y, sr=sr)
            chroma_mean = np.mean(chroma.T, axis=0)  # 12
            chroma_std = np.std(chroma.T, axis=0)    # 12
            features.extend([chroma_mean, chroma_std])
            
            # 3. Spectral (6 features)
            spectral_centroids = librosa.feature.spectral_centroid(y=y, sr=sr)
            spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
            spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)
            features.extend([
                np.mean(spectral_centroids), np.std(spectral_centroids),
                np.mean(spectral_rolloff), np.std(spectral_rolloff),
                np.mean(spectral_bandwidth), np.std(spectral_bandwidth)
            ])
            
            # 4. Spectral contrast (14 features) - NO n_bands parameter!
            # Using librosa default: produces 7 bands
            try:
                contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
                contrast_mean = np.mean(contrast.T, axis=0)  # 7
                contrast_std = np.std(contrast.T, axis=0)    # 7
                features.extend([contrast_mean, contrast_std])
            except Exception as e:
                logger.warning(f"Spectral contrast failed: {e}, using zeros")
                features.extend([np.zeros(7), np.zeros(7)])
            
            # 5. Tonnetz (12 features)
            y_harmonic = librosa.effects.harmonic(y)
            tonnetz = librosa.feature.tonnetz(y=y_harmonic, sr=sr)
            tonnetz_mean = np.mean(tonnetz.T, axis=0)  # 6
            tonnetz_std = np.std(tonnetz.T, axis=0)    # 6
            features.extend([tonnetz_mean, tonnetz_std])
            
            # 6. Zero crossing rate (2 features)
            zcr = librosa.feature.zero_crossing_rate(y)
            features.extend([np.mean(zcr), np.std(zcr)])
            
            # 7. RMS Energy (2 features)
            rms = librosa.feature.rms(y=y)
            features.extend([np.mean(rms), np.std(rms)])
            
            # 8. Tempo (1 feature)
            try:
                tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
                features.append(tempo)
            except:
                features.append(0)
            
            # Total: 60 + 24 + 6 + 14 + 12 + 2 + 2 + 1 = 121 features
            
            final_features = np.hstack(features)
            
            if len(final_features) != 121:
                logger.error(f"Feature count mismatch: {len(final_features)} (expected 121)")
                # Pad or trim to exactly 121
                if len(final_features) < 121:
                    final_features = np.pad(final_features, (0, 121 - len(final_features)))
                else:
                    final_features = final_features[:121]
            
            logger.debug(f"Features extracted: {len(final_features)} dims")
            
            return final_features
        
        except Exception as e:
            logger.error(f"Feature extraction error: {e}", exc_info=True)
            return np.zeros(121)
    
    def predict_disease(self, y, sr):
        """Predict disease from audio segment"""
        try:
            if not self.load_models():
                return "Unknown", 0.0, "Unknown", None
            
            features = self.extract_features(y, sr)
            features_scaled = self.scaler.transform(features.reshape(1, -1))
            
            pred_proba = self.xgb_model.predict_proba(features_scaled)[0]
            pred_idx = np.argmax(pred_proba)
            pred_disease = str(self.label_encoder.classes_[pred_idx])
            pred_confidence = float(pred_proba[pred_idx])
            
            # Map disease to sound
            predicted_sound = DISEASE_TO_SOUND_STRICT.get(pred_disease, "Unknown")
            
            return pred_disease, pred_confidence, predicted_sound, pred_proba
        
        except Exception as e:
            logger.error(f"Prediction error: {e}", exc_info=True)
            return "Unknown", 0.0, "Unknown", None
    
    def sliding_window_analysis(self, y, sr, win_dur=1.0, hop_dur=0.5):
        """Analyze audio with sliding window"""
        try:
            if not self.load_models():
                return {
                    'windows': [],
                    'primary_disease': 'Unknown',
                    'primary_sound': 'Unknown',
                    'confidence': 0.0
                }
            
            win_len = int(win_dur * sr)
            hop_len = int(hop_dur * sr)
            n_frames = max(1, (len(y) - win_len) // hop_len + 1)
            
            window_results = []
            disease_votes = Counter()
            sound_votes = Counter()
            
            logger.info(f"\n📊 Analyzing {min(n_frames, 25)} windows...")
            
            for i in range(min(n_frames, 25)):
                start_idx = i * hop_len
                end_idx = start_idx + win_len
                
                if end_idx > len(y):
                    seg = y[start_idx:]
                else:
                    seg = y[start_idx:end_idx]
                
                if len(seg) < sr * 0.1:
                    continue
                
                pred_disease, pred_confidence, pred_sound, pred_proba = self.predict_disease(seg, sr)
                
                if pred_disease == "Unknown":
                    continue
                
                rms_val = np.sqrt(np.mean(seg**2))
                level_db = 20 * np.log10(rms_val + 1e-10) + 60
                level_db = float(max(0, min(120, level_db)))
                
                disease_votes[pred_disease] += 1
                sound_votes[pred_sound] += 1
                
                all_disease_probs = []
                if pred_proba is not None:
                    for idx, disease in enumerate(self.label_encoder.classes_):
                        all_disease_probs.append({
                            'disease': str(disease),
                            'probability': float(pred_proba[idx]),
                            'sound': DISEASE_TO_SOUND_STRICT.get(str(disease), "Unknown")
                        })
                    all_disease_probs.sort(key=lambda x: x['probability'], reverse=True)
                
                window_result = {
                    "window": f"Window {i+1}",
                    "window_number": int(i + 1),
                    "predicted_disease": pred_disease,
                    "disease_confidence": float(pred_confidence),
                    "predicted_sound": pred_sound,
                    "level_db": level_db,
                    "time_range": f"{start_idx/sr:.1f}s - {min(end_idx, len(y))/sr:.1f}s",
                    "all_disease_probabilities": all_disease_probs[:3]
                }
                
                window_results.append(window_result)
                
                if (i + 1) % 5 == 0 or i == 0:
                    logger.info(f"  Win {i+1}: {pred_disease} ({pred_confidence:.2f}) → {pred_sound} | {level_db:.1f}dB")
            
            if not window_results:
                return {
                    'windows': [],
                    'primary_disease': 'Unknown',
                    'primary_sound': 'Unknown',
                    'confidence': 0.0
                }
            
            primary_disease = disease_votes.most_common(1)[0][0]
            primary_sound = sound_votes.most_common(1)[0][0]
            
            total_confidence = sum([
                w['disease_confidence'] for w in window_results
                if w['predicted_disease'] == primary_disease
            ])
            primary_confidence = total_confidence / sum([
                1 for w in window_results
                if w['predicted_disease'] == primary_disease
            ])
            
            total_windows = len(window_results)
            disease_distribution = {
                disease: {
                    'count': int(count),
                    'percentage': float(round(count / total_windows * 100, 1)),
                    'sound': DISEASE_TO_SOUND_STRICT.get(disease, "Unknown")
                }
                for disease, count in disease_votes.items()
            }
            disease_distribution = dict(sorted(
                disease_distribution.items(),
                key=lambda x: x[1]['count'],
                reverse=True
            ))
            
            logger.info(f"\n✓ Analysis complete:")
            logger.info(f"   Primary disease: {primary_disease}")
            logger.info(f"   Primary sound: {primary_sound}")
            logger.info(f"   Confidence: {primary_confidence:.2%}")
            
            return {
                'windows': window_results,
                'primary_disease': primary_disease,
                'primary_sound': primary_sound,
                'primary_confidence': float(primary_confidence),
                'disease_distribution': disease_distribution,
                'sound_distribution': dict(sound_votes),
                'total_windows': total_windows
            }
        
        except Exception as e:
            logger.error(f"✗ Window error: {e}", exc_info=True)
            return {
                'windows': [],
                'primary_disease': 'Unknown',
                'primary_sound': 'Unknown',
                'confidence': 0.0
            }
    
    def generate_waveform(self, y, sr):
        """Generate waveform visualization"""
        try:
            fig = plt.figure(figsize=(10, 2.5))
            ax = fig.add_subplot(111)
            time_axis = np.linspace(0, len(y) / sr, len(y))
            ax.plot(time_axis, y, color='#2E86AB', linewidth=0.8)
            ax.set_title("Waveform Audio Paru", fontsize=14, fontweight='bold')
            ax.set_xlabel("Time (s)")
            ax.set_ylabel("Amplitude")
            ax.grid(True, alpha=0.3)
            
            temp_dir = os.path.join(settings.MEDIA_ROOT, "temp")
            os.makedirs(temp_dir, exist_ok=True)
            
            import uuid
            path = os.path.join(temp_dir, f"waveform_{uuid.uuid4()}.png")
            fig.savefig(path, bbox_inches="tight", dpi=150)
            plt.close(fig)
            
            return path
        except Exception as e:
            logger.error(f"Waveform error: {e}")
            raise
    
    def generate_spectrogram(self, y, sr):
        """Generate spectrogram visualization"""
        try:
            fig = plt.figure(figsize=(10, 3))
            ax = fig.add_subplot(111)
            S = librosa.feature.melspectrogram(y=y, sr=sr)
            S_dB = librosa.power_to_db(S, ref=np.max)
            img = librosa.display.specshow(S_dB, sr=sr, x_axis="time", y_axis="mel", ax=ax, cmap='viridis')
            ax.set_title('Spectrogram Audio Paru', fontsize=14, fontweight='bold')
            plt.colorbar(img, ax=ax, format='%+2.0f dB')
            
            temp_dir = os.path.join(settings.MEDIA_ROOT, "temp")
            os.makedirs(temp_dir, exist_ok=True)
            
            import uuid
            path = os.path.join(temp_dir, f"spectrogram_{uuid.uuid4()}.png")
            fig.savefig(path, bbox_inches="tight", dpi=150)
            plt.close(fig)
            
            return path
        except Exception as e:
            logger.error(f"Spectrogram error: {e}")
            raise
    
    def basic_filter(self, y, sr):
        """Apply basic bandpass filter to audio"""
        try:
            nyquist = sr / 2
            low_cutoff = 80 / nyquist
            high_cutoff = 3000 / nyquist
            
            low_cutoff = np.clip(low_cutoff, 0.001, 0.999)
            high_cutoff = np.clip(high_cutoff, low_cutoff + 0.001, 0.999)
            
            b, a = butter(4, [low_cutoff, high_cutoff], btype='band')
            y_filtered = filtfilt(b, a, y)
            
            return librosa.util.normalize(y_filtered)
        except Exception as e:
            logger.warning(f"Filter failed: {e}")
            return y


# Singleton instance
ml_service = MLService()
