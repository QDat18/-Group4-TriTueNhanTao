import os

from dotenv import load_dotenv
from supabase import create_client

load_dotenv()

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

if not SUPABASE_URL or not SUPABASE_KEY:
    raise EnvironmentError(
        "Missing SUPABASE_URL or SUPABASE_KEY.\n"
        "Please create a .env file based on .env.example "
        "and fill in your Supabase credentials."
    )

supabase = create_client(
    SUPABASE_URL,
    SUPABASE_KEY
)

def upload_inhouse_file(employee_id: str, filename: str, filepath: str) -> bool:
    """Uploads a single local in-house file to Supabase Storage."""
    try:
        if not os.path.exists(filepath):
            print(f"[Supabase Storage] File not found: {filepath}")
            return False
            
        with open(filepath, "rb") as f:
            content_type = "image/jpeg" if filename.lower().endswith((".jpg", ".jpeg")) else \
                           "image/png" if filename.lower().endswith(".png") else "application/octet-stream"
            
            remote_path = f"{employee_id}/{filename}"
            supabase.storage.from_("inhouse").upload(
                path=remote_path,
                file=f,
                file_options={"content-type": content_type, "upsert": "true"}
            )
            print(f"[Supabase Storage] Uploaded: {remote_path}")
            return True
    except Exception as e:
        print(f"[Supabase Storage] Error uploading {filepath} to Supabase: {e}")
        return False

def download_inhouse_file(employee_id: str, filename: str, filepath: str) -> bool:
    """Downloads a single in-house file from Supabase Storage to local filepath."""
    try:
        remote_path = f"{employee_id}/{filename}"
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        res = supabase.storage.from_("inhouse").download(remote_path)
        with open(filepath, "wb") as f:
            f.write(res)
        print(f"[Supabase Storage] Downloaded: {remote_path} -> {filepath}")
        return True
    except Exception as e:
        print(f"[Supabase Storage] Error downloading {remote_path} from Supabase: {e}")
        return False

def delete_inhouse_folder(employee_id: str) -> bool:
    """Deletes all files in Supabase Storage for the specified employee_id under inhouse bucket."""
    try:
        # List files in the employee folder
        files = supabase.storage.from_("inhouse").list(employee_id)
        if not files:
            print(f"[Supabase Storage] No files found in Supabase for employee: {employee_id}")
            return True
            
        file_paths = [f"{employee_id}/{f['name']}" for f in files if 'name' in f]
        if file_paths:
            supabase.storage.from_("inhouse").remove(file_paths)
            print(f"[Supabase Storage] Deleted folder/files in Supabase: {file_paths}")
        return True
    except Exception as e:
        print(f"[Supabase Storage] Error deleting Supabase files for {employee_id}: {e}")
        return False

def sync_inhouse_to_supabase() -> int:
    """Scans local dataset/in-house folder and uploads all files to Supabase Storage."""
    local_root = os.path.join("dataset", "in-house")
    if not os.path.exists(local_root):
        print(f"[Supabase Storage] Local root directory {local_root} does not exist.")
        return 0
        
    count = 0
    for employee_id in os.listdir(local_root):
        emp_dir = os.path.join(local_root, employee_id)
        if not os.path.isdir(emp_dir):
            continue
            
        for filename in os.listdir(emp_dir):
            if filename.lower().endswith((".jpg", ".jpeg", ".png")):
                filepath = os.path.join(emp_dir, filename)
                if upload_inhouse_file(employee_id, filename, filepath):
                    count += 1
    print(f"[Supabase Storage] Synced {count} files to Supabase.")
    return count

def sync_supabase_to_inhouse() -> int:
    """Downloads all in-house folders and files from Supabase Storage to local dataset/in-house."""
    try:
        resp = supabase.table("employees").select("employee_id").eq("is_active", True).execute()
        employee_ids = [emp["employee_id"] for emp in resp.data] if resp.data else []
        
        count = 0
        local_root = os.path.join("dataset", "in-house")
        for employee_id in employee_ids:
            files = supabase.storage.from_("inhouse").list(employee_id)
            if not files:
                continue
                
            for f in files:
                filename = f.get('name')
                if not filename:
                    continue
                filepath = os.path.join(local_root, employee_id, filename)
                if not os.path.exists(filepath):
                    if download_inhouse_file(employee_id, filename, filepath):
                        count += 1
        print(f"[Supabase Storage] Downloaded {count} missing files from Supabase.")
        return count
    except Exception as e:
        print(f"[Supabase Storage] Error syncing from Supabase: {e}")
        return 0