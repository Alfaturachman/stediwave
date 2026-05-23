"""
Firestore Service - Firebase Firestore operations
Centralizes all Firestore reads/writes with error handling
"""

import logging
from datetime import date, datetime
from decimal import Decimal
from firebase_admin import firestore

logger = logging.getLogger(__name__)

# Initialize Firestore client
db = firestore.client()


class FirestoreService:
    """Service class for Firestore operations"""
    
    def __init__(self):
        self.db = db
    
    def get_user_role(self, email):
        """Get user role from Firestore by email"""
        try:
            users_ref = self.db.collection('users')
            query = users_ref.where('email', '==', email).limit(1).stream()
            
            for doc in query:
                user_data = doc.to_dict()
                return user_data.get('role')
            
            return None
        except Exception as e:
            logger.error(f"Error getting user role: {e}")
            return None
    
    def get_pasien_list(self):
        """Get all pasien from Firestore"""
        try:
            pasien_ref = self.db.collection("pasien").stream()
            pasien_list = []
            
            for doc in pasien_ref:
                data = doc.to_dict()
                pasien_list.append({
                    "id": doc.id,
                    "namaLengkap": data.get("namaLengkap") or data.get("nama_lengkap"),
                    "tempatLahir": data.get("tempatLahir") or data.get("tempat_lahir"),
                    "tanggalLahir": data.get("tanggalLahir").strftime("%Y-%m-%d") if data.get("tanggalLahir") else (data.get("tanggal_lahir").strftime("%Y-%m-%d") if data.get("tanggal_lahir") else None),
                    "usia": self.hitung_usia(data.get("tanggalLahir") or data.get("tanggal_lahir")),
                    "tanggalPeriksa": data.get("tanggalPeriksa").strftime("%Y-%m-%d") if data.get("tanggalPeriksa") else (data.get("tanggal_periksa").strftime("%Y-%m-%d") if data.get("tanggal_periksa") else None),
                    "jenisKelamin": data.get("jenisKelamin") or data.get("jenis_kelamin"),
                    "tinggiBadan": data.get("tinggiBadan") or data.get("tinggi_badan"),
                    "beratBadan": data.get("beratBadan") or data.get("berat_badan"),
                    "riwayatPenyakit": data.get("riwayatPenyakit") or data.get("riwayat_penyakit"),
                    "status": data.get("status", "pending"),
                })
            
            return pasien_list
        except Exception as e:
            logger.error(f"Error getting pasien list: {e}")
            return []
    
    def get_pasien_by_id(self, patient_id):
        """Get single pasien by ID"""
        try:
            pasien_doc = self.db.collection("pasien").document(patient_id).get()
            
            if not pasien_doc.exists:
                return None
            
            pasien_data = pasien_doc.to_dict()
            return {
                "id": pasien_doc.id,
                "namaLengkap": pasien_data.get("nama_lengkap"),
                "tempatLahir": pasien_data.get("tempat_lahir"),
                "tanggalLahir": pasien_data.get("tanggal_lahir").strftime("%Y-%m-%d") if pasien_data.get("tanggal_lahir") else None,
                "usia": self.hitung_usia(pasien_data.get("tanggal_lahir")),
                "tanggalPeriksa": pasien_data.get("tanggal_periksa").strftime("%Y-%m-%d") if pasien_data.get("tanggal_periksa") else None,
                "jenisKelamin": pasien_data.get("jenis_kelamin"),
                "tinggiBadan": pasien_data.get("tinggi_badan"),
                "beratBadan": pasien_data.get("berat_badan"),
                "riwayatPenyakit": pasien_data.get("riwayat_penyakit"),
                "status": pasien_data.get("status", "pending"),
            }
        except Exception as e:
            logger.error(f"Error getting pasien by ID: {e}")
            return None
    
    def get_analisis_for_pasien(self, patient_id):
        """Get all analisis records for a specific patient"""
        try:
            # Try multiple collection names
            possible_collections = ["analisisDokter", "analisis_dokter", "analisis_audio"]
            analisis_list = []
            collection_found = None
            
            for collection_name in possible_collections:
                try:
                    test_ref = self.db.collection(collection_name).limit(1).stream()
                    if any(True for _ in test_ref):
                        collection_found = collection_name
                        break
                except Exception:
                    continue
            
            if not collection_found:
                return []
            
            analisis_ref = self.db.collection(collection_found).stream()
            
            for doc in analisis_ref:
                data = doc.to_dict()
                
                # Try all possible field names for patient ID
                doc_patient_id = (
                    data.get("pasien") or
                    data.get("patient_id") or
                    data.get("pasienId") or
                    data.get("patientId") or
                    data.get("id_pasien")
                )
                
                if doc_patient_id != patient_id:
                    continue
                
                # Format timestamp
                created_at = data.get("createdAt") or data.get("created_at") or data.get("timestamp")
                tanggal = self.format_timestamp(created_at)
                
                analisis_list.append({
                    "id": doc.id,
                    "tanggal": tanggal,
                    "diagnosa": data.get("preDiagnosis", "-"),
                    "tindakan": data.get("rujukan", "-"),
                    "catatan": data.get("catatan", "-"),
                    "analisisDokter": data.get("analisisDokter", "-"),
                    "jenisSuara": data.get("jenisSuara", "-"),
                    "intensitasSuara": data.get("intensitasSuara", "-"),
                    "audioFile": data.get("audioFile"),
                    "spectrogramFile": data.get("spectrogramFile"),
                    "waveformFile": data.get("waveformFile"),
                    "preDiagnosis": data.get("preDiagnosis", "-"),
                })
            
            # Sort by date descending
            analisis_list.sort(key=lambda x: x['tanggal'] or '', reverse=True)
            
            return analisis_list
        except Exception as e:
            logger.error(f"Error getting analisis for pasien: {e}")
            return []
    
    def get_all_analisis_grouped(self):
        """Get all analisis records grouped by patient ID"""
        try:
            possible_collections = ["analisisDokter", "analisis_dokter", "analisis_audio"]
            analisis_grouped = {}
            collection_found = None
            
            for collection_name in possible_collections:
                try:
                    test_ref = self.db.collection(collection_name).limit(1).stream()
                    if any(True for _ in test_ref):
                        collection_found = collection_name
                        break
                except Exception:
                    continue
            
            if not collection_found:
                return {}
            
            analisis_ref = self.db.collection(collection_found).stream()
            
            for doc in analisis_ref:
                data = doc.to_dict()
                
                patient_id = (
                    data.get("pasien") or
                    data.get("patient_id") or
                    data.get("pasienId") or
                    data.get("patientId") or
                    data.get("id_pasien")
                )
                
                if not patient_id:
                    continue
                
                if patient_id not in analisis_grouped:
                    analisis_grouped[patient_id] = []
                
                created_at = data.get("createdAt") or data.get("created_at") or data.get("timestamp")
                tanggal = self.format_timestamp(created_at)
                
                analisis_grouped[patient_id].append({
                    "id": doc.id,
                    "tanggal": tanggal,
                    "diagnosa": data.get("preDiagnosis", "-"),
                    "tindakan": data.get("rujukan", "-"),
                    "catatan": data.get("catatan", "-"),
                    "analisisDokter": data.get("analisisDokter", "-"),
                    "jenisSuara": data.get("jenisSuara", "-"),
                    "intensitasSuara": data.get("intensitasSuara", "-"),
                    "audioFile": data.get("audioFile"),
                    "spectrogramFile": data.get("spectrogramFile"),
                    "waveformFile": data.get("waveformFile"),
                    "preDiagnosis": data.get("preDiagnosis", "-"),
                })
            
            return analisis_grouped
        except Exception as e:
            logger.error(f"Error getting all analisis grouped: {e}")
            return {}
    
    def create_analisis(self, payload):
        """Create new analisis record"""
        try:
            doc_ref = self.db.collection("analisis_audio").add(payload)
            logger.info(f"Document created with ID: {doc_ref[1].id}")
            return doc_ref[1].id
        except Exception as e:
            logger.error(f"Error creating analisis: {e}")
            raise
    
    def update_pasien_status(self, pasien_id, status="completed"):
        """Update pasien status"""
        try:
            # Try direct update by ID
            try:
                self.db.collection("pasien").document(pasien_id).update({
                    "status": status
                })
                logger.info(f"Updated pasien {pasien_id} status to {status}")
                return True
            except Exception:
                # Try lookup by namaLengkap
                q = self.db.collection("pasien").where("namaLengkap", "==", pasien_id).limit(1).stream()
                for doc in q:
                    doc.reference.update({"status": status})
                    logger.info(f"Updated pasien via namaLengkap: {doc.id}")
                    return True
                
                logger.warning(f"Pasien not found: {pasien_id}")
                return False
        except Exception as e:
            logger.error(f"Error updating pasien status: {e}")
            return False
    
    def save_diagnosis_result(self, diagnosis_data):
        """Save diagnosis result to Firestore"""
        try:
            doc_ref = self.db.collection('diagnosis_results').add(diagnosis_data)
            logger.info(f"✓ Diagnosis saved for: {diagnosis_data.get('pasien_nama')}")
            return doc_ref[1].id
        except Exception as e:
            logger.error(f"Error saving diagnosis: {e}")
            raise
    
    def get_diagnosis_history(self):
        """Get all diagnosis results"""
        try:
            diagnosis_ref = self.db.collection('diagnosis_results').stream()
            results = []
            
            for doc in diagnosis_ref:
                data = doc.to_dict()
                results.append({
                    'id': doc.id,
                    'pasien_nama': data.get('pasien_nama'),
                    'timestamp': data.get('timestamp').strftime("%Y-%m-%d %H:%M:%S") if data.get('timestamp') else None,
                    'disease': data.get('disease_detection', {}).get('primary_disease'),
                    'confidence': data.get('disease_detection', {}).get('confidence'),
                    'status': data.get('status')
                })
            
            return results
        except Exception as e:
            logger.error(f"Error fetching diagnosis history: {e}")
            raise
    
    def delete_riwayat(self, exam_id):
        """Delete examination record"""
        try:
            self.db.collection('analisis_audio').document(exam_id).delete()
            return True
        except Exception as e:
            logger.error(f"Error deleting riwayat: {e}")
            raise
    
    def get_pasien_counts(self):
        """Get pending and completed pasien counts"""
        try:
            pasien_ref = self.db.collection("pasien")
            
            pending_count = len(list(
                pasien_ref.where("status", "==", "pending").stream()
            ))
            completed_count = len(list(
                pasien_ref.where("status", "==", "completed").stream()
            ))
            
            return pending_count, completed_count
        except Exception as e:
            logger.error(f"Error getting pasien counts: {e}")
            return 0, 0
    
    @staticmethod
    def hitung_usia(tanggal_lahir):
        """Calculate age from birth date"""
        if not tanggal_lahir:
            return None
        today = date.today()
        return f"{today.year - tanggal_lahir.year - ((today.month, today.day) < (tanggal_lahir.month, tanggal_lahir.day))} tahun"
    
    @staticmethod
    def format_timestamp(timestamp):
        """Format timestamp to string"""
        if not timestamp:
            return None
        
        try:
            if hasattr(timestamp, 'strftime'):
                return timestamp.strftime("%Y-%m-%d %H:%M:%S")
            elif hasattr(timestamp, 'isoformat'):
                return timestamp.isoformat()
            else:
                return str(timestamp)
        except Exception:
            return str(timestamp)
    
    @staticmethod
    def decimal_datetime_data(data):
        """Convert Decimal and datetime objects in dict"""
        for key, value in data.items():
            if isinstance(value, Decimal):
                data[key] = float(value)
            elif isinstance(value, date) and not isinstance(value, datetime):
                data[key] = datetime.combine(value, datetime.time())
        return data


# Singleton instance
firestore_service = FirestoreService()
