import os
import tempfile
import zipfile
import logging
from typing import List, Dict
from sarvamai import SarvamAI

logger = logging.getLogger(__name__)

def process_pdf_sarvam(file_path: str, language: str = "hi-IN") -> List[Dict]:
    logger.info(f"Creating Sarvam Document Intelligence job for {file_path}")
    api_key = "sk_o40rv7ob_cp6C474NyfSRhFdw9qaQN6zA"
    client = SarvamAI(api_subscription_key=api_key)
    
    job = client.document_intelligence.create_job(
        language=language,
        output_format="md"
    )
    job.upload_file(file_path)
    job.start()
    job.wait_until_complete()
    
    output_zip = tempfile.NamedTemporaryFile(delete=False, suffix=".zip").name
    job.download_output(output_zip)
    
    pages = []
    with tempfile.TemporaryDirectory() as td:
        with zipfile.ZipFile(output_zip, 'r') as zf:
            zf.extractall(td)
        for root, _, files in os.walk(td):
            for file in sorted(files):
                if file.endswith('.md'):
                    page_num = 1
                    try:
                        name_part = file.split('.')[0]
                        if 'page' in name_part:
                            page_num = int(name_part.split('page')[-1].split('_')[0])
                        else:
                            page_num = int(name_part)
                    except:
                        pass
                    with open(os.path.join(root, file), 'r', encoding='utf-8') as f:
                        text = f.read()
                        if text.strip():
                            pages.append({"page_num": page_num, "text": text})
    
    if os.path.exists(output_zip):
        os.remove(output_zip)
        
    return pages
