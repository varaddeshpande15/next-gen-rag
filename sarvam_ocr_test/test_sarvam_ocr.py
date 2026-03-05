import requests
from sarvamai import SarvamAI

API_KEY = "sk_o40rv7ob_cp6C474NyfSRhFdw9qaQN6zA"

print("Downloading sample PDF...")

pdf_url = "https://www.w3.org/WAI/ER/tests/xhtml/testfiles/resources/pdf/dummy.pdf"
pdf_path = "sample.pdf"

r = requests.get(pdf_url)

with open(pdf_path, "wb") as f:
    f.write(r.content)

print("Sample PDF downloaded.")

print("Connecting to Sarvam API...")

client = SarvamAI(api_subscription_key=API_KEY)

print("Creating Document Intelligence job...")

job = client.document_intelligence.create_job(
    language="en-IN",
    output_format="md"
)

print("Uploading document...")

job.upload_file(pdf_path)

print("Starting OCR processing...")

job.start()

print("Waiting for job completion...")

status = job.wait_until_complete()

print("Job status:", status.job_state)

print("Downloading OCR output...")

job.download_output("./output.zip")

print("OCR completed!")

print("Output saved as output.zip")
