from backend.app.core.config import settings
from backend.app.infrastructure.storage.ozone_client import get_s3_client

s3, bucket = get_s3_client(), settings.OZONE_BUCKET


def show(prefix=""):
    resp = s3.list_objects_v2(Bucket=bucket, Prefix=prefix, Delimiter="/", MaxKeys=200)
    print(f"\n== {bucket}/{prefix}")
    for p in resp.get("CommonPrefixes", []):
        print("  [folder]", p["Prefix"])
    for o in resp.get("Contents", [])[:20]:
        print("  [file]  ", o["Key"], o["Size"])


show("")
show("t1/")
show("ml-workflow/")
show("landing/")
