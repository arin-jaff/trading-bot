"""Google Drive integration for syncing training data and model artifacts.

Supports two auth modes:
  1. OAuth user credentials (recommended) — uses your personal Drive quota
     - Set GOOGLE_OAUTH_CREDENTIALS_PATH to the OAuth client JSON (from GCP console)
     - On first run, opens a browser for consent; token is cached in secrets/drive_token.json
  2. Service account — only works with Shared Drives (Google Workspace)
     - Set GOOGLE_SERVICE_ACCOUNT_KEY_PATH to the service account JSON key

Both modes require GOOGLE_DRIVE_FOLDER_ID set to the target folder ID.
"""

import io
import os
import json
import tempfile
from datetime import datetime
from typing import Optional
from loguru import logger


EXPORT_DIR = os.path.join('data', 'exports')
PREDICTIONS_DIR = os.path.join('data', 'predictions')


class DriveSync:
    """Syncs training data and predictions between local storage and Google Drive."""

    SCOPES = ['https://www.googleapis.com/auth/drive']

    # Subfolder names inside the shared Drive folder
    DATA_SUBFOLDER = 'data'
    MODELS_SUBFOLDER = 'models'
    PREDICTIONS_SUBFOLDER = 'predictions'
    TRIGGERS_SUBFOLDER = 'triggers'

    # Project root (one level up from src/)
    _PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    def __init__(self):
        self.root_folder_id = os.getenv('GOOGLE_DRIVE_FOLDER_ID', '')

        # OAuth credentials (preferred — uses your personal Drive quota)
        _raw_oauth_path = os.getenv('GOOGLE_OAUTH_CREDENTIALS_PATH', '')
        if _raw_oauth_path and not os.path.isabs(_raw_oauth_path):
            self.oauth_path = os.path.join(self._PROJECT_ROOT, _raw_oauth_path)
        else:
            self.oauth_path = _raw_oauth_path

        # Token cache path
        self.token_path = os.path.join(self._PROJECT_ROOT, 'secrets', 'drive_token.json')

        # Service account credentials (fallback — only works with Shared Drives)
        _raw_key_path = os.getenv('GOOGLE_SERVICE_ACCOUNT_KEY_PATH', '')
        if _raw_key_path and not os.path.isabs(_raw_key_path):
            self.key_path = os.path.join(self._PROJECT_ROOT, _raw_key_path)
        else:
            self.key_path = _raw_key_path
        self.credentials_json = os.getenv('GOOGLE_DRIVE_CREDENTIALS_JSON', '')

        self._service = None
        self._subfolder_cache: dict[str, str] = {}

    @property
    def is_configured(self) -> bool:
        """Check if Drive integration has valid configuration."""
        has_oauth = bool(self.oauth_path and os.path.exists(self.oauth_path))
        has_service_acct = (
            bool(self.key_path and os.path.exists(self.key_path))
            or bool(self.credentials_json)
        )
        has_token = bool(os.path.exists(self.token_path))
        return bool(self.root_folder_id and (has_oauth or has_token or has_service_acct))

    def _get_service(self):
        """Lazily initialize the Google Drive API service."""
        if self._service:
            return self._service

        try:
            from googleapiclient.discovery import build
        except ImportError:
            raise ImportError(
                "Google API client not installed. Run: "
                "pip install google-api-python-client google-auth google-auth-oauthlib"
            )

        creds = self._get_oauth_credentials()
        if creds is None:
            creds = self._get_service_account_credentials()

        if creds is None:
            raise RuntimeError(
                "No Google Drive credentials found. Set GOOGLE_OAUTH_CREDENTIALS_PATH "
                "(recommended) or GOOGLE_SERVICE_ACCOUNT_KEY_PATH."
            )

        self._service = build('drive', 'v3', credentials=creds)
        logger.info("Google Drive API service initialized")
        return self._service

    def _get_oauth_credentials(self):
        """Get OAuth user credentials (opens browser on first run)."""
        try:
            from google.oauth2.credentials import Credentials
            from google_auth_oauthlib.flow import InstalledAppFlow
            from google.auth.transport.requests import Request
        except ImportError:
            return None

        creds = None

        # Load cached token
        if os.path.exists(self.token_path):
            creds = Credentials.from_authorized_user_file(self.token_path, self.SCOPES)

        # Refresh or get new token
        if creds and creds.expired and creds.refresh_token:
            try:
                creds.refresh(Request())
            except Exception:
                creds = None

        if not creds or not creds.valid:
            if not self.oauth_path or not os.path.exists(self.oauth_path):
                return None
            flow = InstalledAppFlow.from_client_secrets_file(self.oauth_path, self.SCOPES)
            creds = flow.run_local_server(port=0)

        # Save token for next time
        os.makedirs(os.path.dirname(self.token_path), exist_ok=True)
        with open(self.token_path, 'w') as f:
            f.write(creds.to_json())

        logger.info("Using OAuth user credentials for Google Drive")
        return creds

    def _get_service_account_credentials(self):
        """Get service account credentials (fallback)."""
        try:
            from google.oauth2 import service_account
        except ImportError:
            return None

        if self.credentials_json:
            info = json.loads(self.credentials_json)
            return service_account.Credentials.from_service_account_info(
                info, scopes=self.SCOPES,
            )
        elif self.key_path and os.path.exists(self.key_path):
            return service_account.Credentials.from_service_account_file(
                self.key_path, scopes=self.SCOPES,
            )
        return None

    # ------------------------------------------------------------------
    # Subfolder management
    # ------------------------------------------------------------------

    def _get_or_create_subfolder(self, subfolder_name: str) -> str:
        """Get or create a subfolder inside the root Drive folder."""
        if subfolder_name in self._subfolder_cache:
            return self._subfolder_cache[subfolder_name]

        service = self._get_service()

        query = (
            f"name = '{subfolder_name}' and "
            f"'{self.root_folder_id}' in parents and "
            f"mimeType = 'application/vnd.google-apps.folder' and "
            f"trashed = false"
        )
        results = service.files().list(
            q=query, fields='files(id, name)',
            supportsAllDrives=True, includeItemsFromAllDrives=True,
        ).execute()
        files = results.get('files', [])

        if files:
            folder_id = files[0]['id']
        else:
            metadata = {
                'name': subfolder_name,
                'mimeType': 'application/vnd.google-apps.folder',
                'parents': [self.root_folder_id],
            }
            folder = service.files().create(
                body=metadata, fields='id',
                supportsAllDrives=True,
            ).execute()
            folder_id = folder['id']
            logger.info(f"Created Drive subfolder: {subfolder_name} ({folder_id})")

        self._subfolder_cache[subfolder_name] = folder_id
        return folder_id

    # ------------------------------------------------------------------
    # Core file operations
    # ------------------------------------------------------------------

    def upload_file(self, local_path: str,
                    drive_filename: Optional[str] = None,
                    subfolder: Optional[str] = None) -> dict:
        """Upload a local file to Google Drive.

        If a file with the same name already exists in the target folder,
        it is *updated* in-place (no duplicates created).
        """
        from googleapiclient.http import MediaFileUpload

        service = self._get_service()
        filename = drive_filename or os.path.basename(local_path)

        parent_id = self.root_folder_id
        if subfolder:
            parent_id = self._get_or_create_subfolder(subfolder)

        # Check for existing file
        query = (
            f"name = '{filename}' and "
            f"'{parent_id}' in parents and "
            f"trashed = false"
        )
        existing = service.files().list(
            q=query, fields='files(id)',
            supportsAllDrives=True, includeItemsFromAllDrives=True,
        ).execute()
        existing_files = existing.get('files', [])

        media = MediaFileUpload(local_path, resumable=True)

        if existing_files:
            file_id = existing_files[0]['id']
            updated = service.files().update(
                fileId=file_id,
                media_body=media,
                fields='id, name, webViewLink',
                supportsAllDrives=True,
            ).execute()
            logger.info(f"Updated '{filename}' on Drive ({file_id})")
            return {
                'id': updated['id'],
                'name': updated['name'],
                'link': updated.get('webViewLink', ''),
                'action': 'updated',
            }

        metadata = {
            'name': filename,
            'parents': [parent_id],
        }
        created = service.files().create(
            body=metadata,
            media_body=media,
            fields='id, name, webViewLink',
            supportsAllDrives=True,
        ).execute()
        logger.info(f"Uploaded '{filename}' to Drive ({created['id']})")
        return {
            'id': created['id'],
            'name': created['name'],
            'link': created.get('webViewLink', ''),
            'action': 'created',
        }

    def download_file(self, drive_filename: str, local_path: str,
                      subfolder: Optional[str] = None) -> bool:
        """Download a file from Google Drive to local storage."""
        from googleapiclient.http import MediaIoBaseDownload

        service = self._get_service()
        parent_id = self.root_folder_id
        if subfolder:
            parent_id = self._get_or_create_subfolder(subfolder)

        query = (
            f"name = '{drive_filename}' and "
            f"'{parent_id}' in parents and "
            f"trashed = false"
        )
        results = service.files().list(
            q=query, fields='files(id, name)',
            supportsAllDrives=True, includeItemsFromAllDrives=True,
        ).execute()
        files = results.get('files', [])

        if not files:
            logger.debug(f"File '{drive_filename}' not found on Drive")
            return False

        file_id = files[0]['id']
        request = service.files().get_media(fileId=file_id, supportsAllDrives=True)

        os.makedirs(os.path.dirname(local_path) or '.', exist_ok=True)
        with open(local_path, 'wb') as fh:
            downloader = MediaIoBaseDownload(fh, request)
            done = False
            while not done:
                _, done = downloader.next_chunk()

        logger.info(f"Downloaded '{drive_filename}' -> {local_path}")
        return True

    def _delete_drive_file(self, filename: str,
                           subfolder: Optional[str] = None):
        """Delete a file from Drive (used for trigger/completion cleanup)."""
        try:
            service = self._get_service()
            parent_id = self.root_folder_id
            if subfolder:
                parent_id = self._get_or_create_subfolder(subfolder)

            query = (
                f"name = '{filename}' and "
                f"'{parent_id}' in parents and "
                f"trashed = false"
            )
            results = service.files().list(
                q=query, fields='files(id)',
                supportsAllDrives=True, includeItemsFromAllDrives=True,
            ).execute()
            for f in results.get('files', []):
                service.files().delete(fileId=f['id'], supportsAllDrives=True).execute()
                logger.debug(f"Deleted '{filename}' from Drive")
        except Exception as e:
            logger.debug(f"Error deleting '{filename}' from Drive: {e}")

    # ------------------------------------------------------------------
    # High-level sync operations
    # ------------------------------------------------------------------

    def upload_training_data(self) -> dict:
        """Upload all training export files to Drive ``data`` subfolder."""
        if not self.is_configured:
            return {
                'error': (
                    'Google Drive not configured. Set GOOGLE_DRIVE_FOLDER_ID '
                    'and GOOGLE_OAUTH_CREDENTIALS_PATH (or GOOGLE_SERVICE_ACCOUNT_KEY_PATH).'
                ),
            }

        uploaded = []
        errors = []

        files_to_upload = [
            'trump_speeches.jsonl',
            'trump_speeches_metadata.json',
            'term_context.json',
            'event_speech_pairs.json',
            'scenario_context.json',
        ]

        for filename in files_to_upload:
            local_path = os.path.join(EXPORT_DIR, filename)
            if not os.path.exists(local_path):
                logger.debug(f"Skipping {filename} (not found locally)")
                continue
            try:
                result = self.upload_file(
                    local_path, subfolder=self.DATA_SUBFOLDER,
                )
                uploaded.append(result)
            except Exception as e:
                logger.error(f"Failed to upload {filename}: {e}")
                errors.append({'file': filename, 'error': str(e)})

        return {
            'uploaded': uploaded,
            'errors': errors,
            'timestamp': datetime.utcnow().isoformat(),
        }

    def download_predictions(self) -> dict:
        """Download latest predictions from Drive ``predictions`` subfolder."""
        if not self.is_configured:
            return {'error': 'Google Drive not configured'}

        local_path = os.path.join(PREDICTIONS_DIR, 'predictions_latest.json')

        try:
            found = self.download_file(
                'predictions_latest.json', local_path,
                subfolder=self.PREDICTIONS_SUBFOLDER,
            )
            if found:
                with open(local_path) as f:
                    data = json.load(f)
                return {
                    'downloaded': True,
                    'predictions_count': len(data.get('term_predictions', [])),
                    'generated_at': data.get('generated_at', 'unknown'),
                    'local_path': local_path,
                }
            return {'downloaded': False, 'reason': 'File not found on Drive'}
        except Exception as e:
            return {'error': str(e)}

    # ------------------------------------------------------------------
    # Trigger / completion handshake with Colab
    # ------------------------------------------------------------------

    def write_trigger_file(self, trigger_type: str = 'full_pipeline',
                           extra_data: Optional[dict] = None) -> dict:
        """Write a trigger file to Drive to signal Colab to start training."""
        if not self.is_configured:
            return {'error': 'Google Drive not configured'}

        trigger_data = {
            'trigger_type': trigger_type,
            'triggered_at': datetime.utcnow().isoformat(),
            'request': extra_data or {},
        }

        trigger_filename = 'training_trigger.json'
        tmp_path = os.path.join(tempfile.gettempdir(), trigger_filename)

        try:
            with open(tmp_path, 'w') as f:
                json.dump(trigger_data, f, indent=2)

            result = self.upload_file(
                tmp_path,
                drive_filename=trigger_filename,
                subfolder=self.TRIGGERS_SUBFOLDER,
            )
            logger.info(f"Training trigger written to Drive: {trigger_type}")
            return {'triggered': True, 'trigger_data': trigger_data, **result}
        except Exception as e:
            return {'error': str(e)}
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

    def check_completion(self) -> Optional[dict]:
        """Check if Colab has written a completion signal to Drive."""
        if not self.is_configured:
            return None

        completion_filename = 'training_complete.json'
        tmp_path = os.path.join(tempfile.gettempdir(), completion_filename)

        try:
            found = self.download_file(
                completion_filename, tmp_path,
                subfolder=self.TRIGGERS_SUBFOLDER,
            )
            if not found:
                return None

            with open(tmp_path) as f:
                data = json.load(f)

            # Clean up handshake files
            self._delete_drive_file(
                completion_filename, self.TRIGGERS_SUBFOLDER,
            )
            self._delete_drive_file(
                'training_trigger.json', self.TRIGGERS_SUBFOLDER,
            )

            return data
        except Exception as e:
            logger.debug(f"Error checking completion: {e}")
            return None
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

    # ------------------------------------------------------------------
    # Status
    # ------------------------------------------------------------------

    def get_status(self) -> dict:
        """Get Drive sync status and configuration info."""
        status = {
            'configured': self.is_configured,
            'folder_id': (
                self.root_folder_id[:10] + '...'
                if self.root_folder_id else None
            ),
            'has_credentials': bool(
                self.oauth_path or self.key_path or self.credentials_json
            ),
        }

        if self.is_configured:
            try:
                service = self._get_service()
                folder = service.files().get(
                    fileId=self.root_folder_id,
                    fields='name, webViewLink',
                    supportsAllDrives=True,
                ).execute()
                status['folder_name'] = folder.get('name')
                status['folder_link'] = folder.get('webViewLink')
                status['connected'] = True
            except Exception as e:
                status['connected'] = False
                status['error'] = str(e)

        return status
